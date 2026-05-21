from typing import Optional, Tuple
import itertools

import torch
import torch.nn.functional as F
from rdkit import Chem

from src.utils.misc import normalize_weights, mask_E, mask_from_sizes
from src.utils.chem import make_molecules_from_batch, make_molecules_from_outputs
from src.utils.evaluations import unpermute_graph

def count_node_types(X: torch.Tensor) -> torch.Tensor:
    """
    Count occurrences of each node type in a batch of graphs.

    Args:
        X (torch.Tensor): one hot tensor of shape (batch_size, num_nodes) with integer labels. Last class index is assumed to be "no_node".
    Returns:
        torch.Tensor: Tensor of shape (num_classes,) with counts of each node type.
    """
    device = X.device
    counts = X.sum(dim=0)[:-1]  # [C]
    return counts.to(device)


def count_edge_types(E: torch.Tensor, node_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Count occurrences of each edge type in a batch of graphs.
    Args:
        E (torch.Tensor): one hot tensor of shape (batch_size, num_nodes, num_nodes, num_edge_types) with integer labels. Last class index is assumed to be "no_edge".
    Returns:
        torch.Tensor: Tensor of shape (num_classes,) with counts of each edge type.
    """
    device = E.device
    if node_mask is not None:
        edges_mask = mask_E(node_mask).unsqueeze(-1).to(device)
        E = E * edges_mask  # padded edge are zeroed
        counts = E.sum(dim=(0, 1, 2))  # [C]
    else:
        counts = E.sum(dim=(0, 1, 2)) // 2  # [C] (undirected edges counted twice)
    return counts.to(device)


def make_class_weights(
    class_counts, ratio_negative_edges: float = None
) -> torch.Tensor:
    class_counts = class_counts.clone()
    if ratio_negative_edges is not None:
        total_pos = class_counts[:-1].sum()
        desired_neg = total_pos * ratio_negative_edges
        class_counts[-1] = min(class_counts[-1], desired_neg)

    total = class_counts.sum()
    class_number = len(class_counts)
    weights = total / (class_number * class_counts)
    weights = torch.where(class_counts == 0, 0, weights)
    return weights


def weights_from_loader(
    loader: torch.utils.data.DataLoader,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute class weights for nodes and edges from a dataloader.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader to be used for calculating weights.

    Returns:
        Tuple[torch.Tensor,torch.Tensor]:
            Calculated weights for nodes and edges.
    """
    class_counts_nodes, class_counts_edges = None, None

    size_graph = loader.dataset[0].x.shape[0]

    for batch in loader:
        X, E = batch.x, batch.edge_attr_adj

        if X.dim() == 2 and X.sum(dim=-1).eq(1).all():
            X_one_hot = X
            X = X.argmax(dim=-1)
        else:
            X_one_hot = F.one_hot(X, num_classes=X.max().item() + 1).float()

        if E.dim() == 2:
            E = E
            E_one_hot = F.one_hot(
                E, num_classes=E.max().item() + 1
            ).float()  # [num_nodes, num_nodes, C], ignore padding
        else:
            E_one_hot = E
            E = E.argmax(dim=-1)

        num_class_nodes = X_one_hot.shape[-1] - 1  # padding is not a class
        num_class_edges = E_one_hot.shape[-1]  # negative edges are a class
        node_mask = X != num_class_nodes  # Ignore padding nodes

        # Count nodes
        node_counts = count_node_types(X_one_hot)
        if class_counts_nodes is None:
            class_counts_nodes = node_counts.clone()
        else:
            class_counts_nodes += node_counts

        # Count edges
        edge_counts = count_edge_types(
            E_one_hot.reshape(
                batch.num_graphs, size_graph, size_graph, num_class_edges
            ),
            node_mask.reshape(batch.num_graphs, size_graph),
        )
        if class_counts_edges is None:
            class_counts_edges = edge_counts.clone()
        else:
            class_counts_edges += edge_counts

    # Weights calculation
    node_weights = make_class_weights(class_counts_nodes)
    edge_weights = make_class_weights(class_counts_edges)

    node_weights = normalize_weights(node_weights)
    edge_weights = normalize_weights(edge_weights)

    return node_weights.detach(), edge_weights.detach()


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if alpha is None:
        alpha = torch.ones(logits.shape[-1], device=logits.device)

    """ Focal loss for multi-class classification. """
    # source: https://github.com/itakurah/Focal-loss-PyTorch/
    # Convert logits to probabilities with softmax
    probs = F.softmax(logits, dim=1)
    probs = probs.clamp_min(1e-20)  # avoid log(0) -> -inf -> NaN in one-hot multiplication
    # One-hot encode the targets
    num_classes = logits.shape[-1]
    targets_one_hot = F.one_hot(labels, num_classes=num_classes).float()
    # Compute cross-entropy for each class
    ce_loss = -targets_one_hot * torch.log(probs)
    # Compute focal weight
    p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
    focal_weight = (1 - p_t) ** gamma
    # Apply alpha if provided (per-class weighting)
    if alpha is not None:
        alpha_t = alpha.gather(0, labels)
        ce_loss = alpha_t.unsqueeze(1) * ce_loss
    # Apply focal loss weight
    loss = focal_weight.unsqueeze(1) * ce_loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss.sum(-1)
    return loss

def recon_loss_nodes(
    labels: torch.Tensor,
    logits: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    use_focal: bool = False,
    soft_perm_matrix = None,
) -> torch.Tensor:

    """
    Compute node reconstruction loss using cross entropy (or focal), ignoring non-nodes.
    Args:
        labels: (B, N) long
        logits: (B, N, C)
        node_mask: (B, N) bool, True for real node, False for padded/non-node
        weights: (C,) optional class weights
        use_focal: use focal loss if True, else cross-entropy
        soft_perm_matrix: (B, N, N) optional soft permutation weights for pairwise loss (GRALE)
    Returns:
        Scalar loss tensor
    """

    labels = labels.long()
    bs, nmax = labels.shape
    node_mask_expanded = None
    if soft_perm_matrix is None:
        reduction = "mean"
        if node_mask is not None:
            logits_flat = logits.view(-1, logits.shape[-1])  # [B*N, C]
            labels_flat = labels.view(-1)  # [B*N]
            node_mask_flat = node_mask.view(-1).bool()  # [B*N]
            selected_logits = logits_flat[node_mask_flat]  # [num_real_nodes, C]
            selected_labels = labels_flat[node_mask_flat]  # [num_real_nodes]
        else:
            selected_labels = labels.view(-1)
            selected_logits = logits.view(-1, logits.shape[-1])
    else:
        if node_mask is None:
            node_mask = torch.ones((bs, nmax), device=logits.device).bool()
        reduction = "none"
        selected_logits = logits.unsqueeze(2).expand(-1,-1,nmax,-1).reshape(-1,logits.shape[-1])
        selected_labels = labels.unsqueeze(1).expand(-1,nmax,-1).reshape(-1)
        node_mask_expanded = node_mask.unsqueeze(1).expand(-1,nmax,-1).bool()
        node_mask_expanded_flat = node_mask_expanded.reshape(-1)
        selected_logits = selected_logits[node_mask_expanded_flat]
        selected_labels = selected_labels[node_mask_expanded_flat]

    # Compute loss
    ## No node is present
    if selected_labels.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    ## Focal Loss
    if use_focal:
        loss = focal_loss(
            logits=selected_logits,
            labels=selected_labels,
            alpha=weights,
            gamma=2.0,
            reduction=reduction,
        )
    ## Cross-entropy Loss
    else:
        loss = F.cross_entropy(
            input=selected_logits,
            target=selected_labels,
            weight=weights,
            reduction=reduction,
            label_smoothing=0.01,
        )
    if soft_perm_matrix is not None:
        selected_perms = soft_perm_matrix.reshape(-1)[node_mask_expanded_flat]
        loss = loss*selected_perms
        loss = loss.mean()
    return loss

def recon_loss_log_sizes(
        target_sizes: torch.Tensor,
        predicted_log_sizes: torch.Tensor
    ):
    return torch.nn.functional.smooth_l1_loss(predicted_log_sizes, target_sizes.float().log())


def recon_loss_formal_charges(
    formal_charges : torch.Tensor,
    logits : torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    use_focal: bool = False,
    soft_perm_matrix: Optional[torch.Tensor] = None
):
    """Compute formal charge reconstruction loss using cross entropy, ignoring non-nodes.
    Args:
        formal_charges (torch.Tensor): True formal charge labels (batch_size, num_nodes)
        logits (torch.Tensor): Predicted logits for formal charges (batch_size, num_nodes, num_formal_charge_classes)
        node_mask (torch.Tensor | None): Optional tensor of shape (batch_size, num_nodes) indicating non-nodes (True for node, False for non-node). If provided, non-nodes are ignored in loss computation.
        weights (torch.Tensor | None): Optional tensor of shape (num_formal_charge_classes,) for class weighting.
        use_focal (bool, default=False): Use Focal Loss (otherwise cross-entropy).
    Returns:
        torch.Tensor: mean negative log-likelihood loss over all nodes
    """

    return recon_loss_nodes(
        labels=formal_charges,
        logits=logits,
        node_mask=node_mask,
        weights=weights,
        use_focal=use_focal,
        soft_perm_matrix=soft_perm_matrix
    )

def recon_loss_hydrogens(
    hydrogen_counts : torch.Tensor,
    logits : torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    use_focal: bool = False,
    soft_perm_matrix: Optional[torch.Tensor] = None
):
    """Compute hydrogen count reconstruction loss using cross entropy, ignoring non-nodes.
    Args:
        hydrogen_counts (torch.Tensor): True hydrogen count labels (batch_size, num_nodes)
        logits (torch.Tensor): Predicted logits for hydrogen counts (batch_size, num_nodes, num_hydrogen_classes)
        node_mask (torch.Tensor | None): Optional tensor of shape (batch_size, num_nodes) indicating non-nodes (True for node, False for non-node). If provided, non-nodes are ignored in loss computation.
        weights (torch.Tensor | None): Optional tensor of shape (num_hydrogen_classes,) for class weighting.
        use_focal (bool, default=False): Use Focal Loss (otherwise cross-entropy).
    Returns:
        torch.Tensor: mean negative log-likelihood loss over all nodes
    """

    return recon_loss_nodes(
        labels=hydrogen_counts,
        logits=logits,
        node_mask=node_mask,
        weights=weights,
        use_focal=use_focal,
        soft_perm_matrix=soft_perm_matrix
    )

def recon_loss_soft_node_mask(
    real_node_mask: torch.Tensor,
    soft_node_mask_logits : torch.Tensor,
    use_focal: bool = False,
    soft_perm_matrix: Optional[torch.Tensor] = None
):
    """Compute soft_mask reconstruction loss using cross entropy, ignoring non-nodes.
    Args:
        real_node_mask (torch.Tensor): True maskk (batch_size, num_nodes)
        logits (torch.Tensor): Predicted logits for hydrogen counts (batch_size, num_nodes, num_hydrogen_classes)
        node_mask (torch.Tensor | None): Optional tensor of shape (batch_size, num_nodes) indicating non-nodes (True for node, False for non-node). If provided, non-nodes are ignored in loss computation.
        weights (torch.Tensor | None): Optional tensor of shape (num_hydrogen_classes,) for class weighting.
        use_focal (bool, default=False): Use Focal Loss (otherwise cross-entropy).
    Returns:
        torch.Tensor: mean negative log-likelihood loss over all nodes
    """

    return recon_loss_nodes(
        labels=real_node_mask,
        logits=soft_node_mask_logits,
        node_mask=None,
        use_focal=use_focal,
        soft_perm_matrix=soft_perm_matrix
    )

def recon_loss_edges(
    labels: torch.Tensor,
    logits: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    use_focal: bool = False,
    ratio_negative_edges: float = -1.0,
    soft_perm_matrix: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute edge reconstruction loss using cross entropy or focal loss with negative sampling,
    ignoring edges connected to non-nodes.

    Args:
        labels (torch.Tensor): True edge labels (batch_size, num_nodes, num_nodes). Last class index is assumed to be "no_edge".
        logits (torch.Tensor): Predicted logits for edges (batch_size, num_nodes, num_nodes, num_classes)
        node_mask (torch.Tensor | None): Optional tensor of shape (batch_size, num_nodes) indicating non-nodes (False for non-node, True for node). If provided, edges connected to non-nodes are ignored in loss computation.
        weights (torch.Tensor | None): Optional tensor of shape (num_classes,) for class weighting.
        use_focal (bool, default=False): Use Focal Loss (otherwise cross-entropy).
        ratio_negative_edges (float): Number of negatives to sample per positive edge.
    Returns:
        torch.Tensor: mean negative log-likelihood loss over sampled edges
    """
    B, N, _, C = logits.shape
    device = logits.device

    if node_mask is not None:
        edge_mask = mask_E(node_mask)  # [B, N, N]
    else:
        edge_mask = torch.ones((B, N, N), dtype=torch.bool, device=device)

    # Flatten everything
    logits_flat = logits.view(-1, C)  # [B*N*N, C]
    labels_flat = labels.view(-1)  # [B*N*N]
    edge_mask_flat = edge_mask.view(-1)  # [B*N*N]

    real_mask = (labels_flat != (C - 1)) & edge_mask_flat
    neg_mask = (labels_flat == (C - 1)) & edge_mask_flat

    pos_idx = torch.where(real_mask)[0]
    neg_idx = torch.where(neg_mask)[0]

    num_pos = pos_idx.size(0)

    num_neg = int(ratio_negative_edges * num_pos)
    if num_neg > 0 and neg_idx.size(0) > 0:
        sampled_neg_idx = neg_idx[torch.randperm(neg_idx.size(0))[:num_neg]]
        selected_idx = torch.cat([pos_idx, sampled_neg_idx], dim=0)
    else:
        selected_idx = pos_idx

    selected_logits = logits_flat[selected_idx]  # [num_pos + num_neg, C]
    selected_labels = labels_flat[selected_idx]  # [num_pos + num_neg]

    # Compute loss
    ## No edge is present
    if selected_labels.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    ## Focal Loss
    if use_focal:
        loss = focal_loss(
            logits=selected_logits,
            labels=selected_labels,
            gamma=2.0,
            alpha=weights,
            reduction="mean",
        )

    ## Cross-entropy Loss
    else:
        loss = F.cross_entropy(
            input=selected_logits,
            target=selected_labels,
            weight=weights,
            reduction="mean",
            label_smoothing=0.01,
        )
    return loss

def replace_by_padding_nodes(pred_X, sizes):
    classes = pred_X.argmax(-1)
    _, _, padding_value = pred_X.shape
    mask = mask_from_sizes(sizes, max_size=pred_X.shape[1])
    classes = classes.masked_fill(~mask, padding_value)
    return classes, mask

def replace_by_padding_edges(pred_E, sizes):
    classes = pred_E.argmax(-1)
    _, _, _, padding_value = pred_E.shape
    mask_nodes = mask_from_sizes(sizes, max_size=pred_E.shape[1])
    mask_edges = mask_E(mask_nodes)
    classes = classes.masked_fill(~mask_edges, padding_value-1)
    return classes, mask_edges

def accuracy_nodes(X, pred_X, sizes):
    node_classes, node_mask = replace_by_padding_nodes(pred_X, sizes)
    return (node_classes==X).all(-1)

def accuracy_edges(E, pred_E, sizes):
    edge_classes, edge_mask = replace_by_padding_edges(pred_E, sizes)
    return (edge_classes*edge_mask==E*edge_mask).all((1,2))

def accuracy_sizes(sizes, predicted_sizes):
    return predicted_sizes==sizes

def all_accuracies(X, pred_X, E, pred_E, sizes, pred_sizes):
    return accuracy_nodes(X, pred_X, pred_sizes),accuracy_edges(E, pred_E, pred_sizes),accuracy_sizes(sizes, pred_sizes)

def accuracy_hydrogens(hydrogens, pred_hydrogens, hydrogens_mask):
    return (hydrogens*hydrogens_mask==pred_hydrogens.argmax(-1)*hydrogens_mask).all(-1)

def accuracy_formal_charges(formal_charges, pred_formal_charges, formal_charges_mask):
    return (formal_charges*formal_charges_mask==pred_formal_charges.argmax(-1)*formal_charges_mask).all(-1)

def loss_kl(x_mu, x_log_sigma):
    kl = torch.sum(
        1 - x_mu.pow(2) - torch.exp(x_log_sigma).pow(2) + 2 * x_log_sigma,
        dim=1,
    )
    kl = -0.5 * torch.mean(kl)
    return kl

def make_hash_graphs(
        X,
        E,
        hydrogens=None,
        formal_charges=None
):
    assert X.dim()==1
    assert E.dim()==3
    edge_counts_per_node = E.sum(1)
    hashes = torch.cat((X.unsqueeze(-1), edge_counts_per_node), -1)
    if hydrogens is not None:
        hashes = torch.cat((hashes, hydrogens.unsqueeze(-1)), -1)
    if formal_charges is not None:
        hashes = torch.cat((hashes, formal_charges.unsqueeze(-1)), -1)
    return hashes

def make_hash_batch_idx(
    batch,
    idx,
    num_edge_types
):
    node_mask = batch['node_mask'][idx]
    hydrogens = None
    formal_charges = None
    if batch.get('hydrogens') is not None:
        hydrogens = batch['hydrogens'][idx][node_mask]
    if batch.get('formal_charges') is not None:
        formal_charges = batch['formal_charges'][idx][node_mask]
    hash_graph = make_hash_graphs(
        X = batch['X'][idx][node_mask],
        E = torch.nn.functional.one_hot(batch['E'][idx][node_mask][:,node_mask], num_edge_types),
        hydrogens = hydrogens * (batch['aromatic_mask'][idx][node_mask]),
        formal_charges=formal_charges * (batch['aromatic_mask'][idx][node_mask])
    )
    return hash_graph

def make_hash_outputs_idx(
    outputs,
    idx,
    num_edge_types
):
    hydrogens = None
    formal_charges = None
    node_mask = outputs['used_node_filter'][idx]
    Es = outputs['E'][idx][node_mask][:,node_mask].argmax(-1)
    Es.fill_diagonal_(outputs['E'].shape[-1]-1)
    aromatic_mask = (Es == 3).any(-1)
    if outputs.get('predicted_hydrogens') is not None:
        hydrogens = outputs['predicted_hydrogens'][idx][node_mask].argmax(-1)*aromatic_mask
    if outputs.get('predicted_formal_charges') is not None:
        formal_charges = outputs['predicted_formal_charges'][idx][node_mask].argmax(-1)*aromatic_mask

    hash_graph = make_hash_graphs(
        X = outputs['X'][idx][node_mask].argmax(-1),
        E = torch.nn.functional.one_hot(Es, num_edge_types),
        hydrogens = hydrogens,
        formal_charges=formal_charges
    )
    return hash_graph

def make_hash_batch_outputs(
        batch,
        outputs,
        not_matching_idx
):
    num_edge_types = outputs['E'].shape[-1]
    hash_outputs = [make_hash_outputs_idx(outputs, idx, num_edge_types) for idx in not_matching_idx]
    hash_batch = [make_hash_batch_idx(batch, idx, num_edge_types) for idx in not_matching_idx]
    return hash_batch, hash_outputs, not_matching_idx

def groups_via_unique(A: torch.Tensor, B: torch.Tensor):
    """
    Return groups derived from exact row equality:
      groups = [(A_indices, B_indices), ...]
    where A_indices and B_indices are 1D LongTensors listing rows that share the same value.
    """
    assert A.ndim == 2 and B.ndim == 2 and A.size(1) == B.size(1)
    nA, nB = A.size(0), B.size(0)

    AB = torch.cat([A, B], dim=0)
    # NOTE: exact equality; for floats, consider rounding/tolerance before this step
    _, inv = torch.unique(AB, dim=0, return_inverse=True)
    invA, invB = inv[:nA], inv[nA:]

    # Build key -> list of A indices, list of B indices
    # We keep it simple (Python dict), but invA/invB are small 1D tensors so this is O(n)
    key_to_A = {}
    key_to_B = {}
    for i, k in enumerate(invA.tolist()):
        key_to_A.setdefault(k, []).append(i)
    for j, k in enumerate(invB.tolist()):
        key_to_B.setdefault(k, []).append(j)

    groups = []
    for k, Aidx in key_to_A.items():
        Bidx = key_to_B.get(k, [])
        groups.append( (torch.tensor(Aidx, dtype=torch.long),
                        torch.tensor(Bidx, dtype=torch.long)) )
    return groups


def enumerate_permutations_from_matches(
    A: torch.Tensor,
    B: torch.Tensor,
):
    groups = groups_via_unique(B, A)
    n = A.size(0)
    # Feasibility
    for Aidx, Bidx in groups:
        if len(Aidx) != len(Bidx):
            return

    # Precompute per-group permutations over Bidx
    per_group_perms = []
    for _, Bidx in groups:
        J = Bidx.tolist()
        per_group_perms.append(list(itertools.permutations(J)))

    for combo in itertools.product(*per_group_perms):
        # j_from_i default
        j_from_i = [None] * n
        for (Aidx, _), permJ in zip(groups, combo):
            for i_idx, j_val in zip(Aidx.tolist(), permJ):
                j_from_i[i_idx] = j_val

        # Inverse
        i_from_j = [None] * n
        for i_idx, j_val in enumerate(j_from_i):
            i_from_j[j_val] = i_idx

        yield (torch.tensor(j_from_i, dtype=torch.long),
                   torch.tensor(i_from_j, dtype=torch.long))


def perm_vector_to_matrix(j_from_i, dtype=torch.float32, device=None):
    j_from_i = torch.as_tensor(j_from_i, dtype=torch.long)
    n = j_from_i.numel()

    if device is None:
        device = j_from_i.device

    P = torch.zeros((n, n), dtype=dtype, device=device)
    cols = torch.arange(n, device=device)
    rows = j_from_i.to(device)

    P[rows, cols] = 1.0
    return P


def check_permutation_accuracy(batch, outputs, not_matching_idx):
    hashes_batch, hashes_outputs, _ = make_hash_batch_outputs(batch, outputs, not_matching_idx)
    ret_match = torch.zeros(len(not_matching_idx)).bool()
    for i in range(len(not_matching_idx)):
        idx_in_batch = not_matching_idx[i]
        hb = hashes_batch[i]
        ho = hashes_outputs[i]
        if (hb.shape!=ho.shape) or torch.all(hb.sum(0)!=ho.sum(0)):
            ret_match[i] = False
            continue
        node_mask = batch['node_mask'][idx_in_batch]
        gen_perms = enumerate_permutations_from_matches(hb, ho)
        keep = {'X', 'E', 'hydrogens', 'formal_charges', 'aromatic_mask', 'node_mask'}
        one_batch = {
            k: (batch[k][idx_in_batch][node_mask] if batch[k] is not None else None)
            for k in batch
            if k in keep
        }
        one_batch['graph_sizes'] = batch['graph_sizes'][idx_in_batch]
        one_batch = {k:one_batch[k].unsqueeze(0) for k in one_batch}
        one_batch['E'] = one_batch['E'][:,:,node_mask]

        for perms in gen_perms:
            perm_matrix = perm_vector_to_matrix(perms[1])
            one_output = {
                k: (outputs[k][idx_in_batch].unsqueeze(0) if outputs[k] is not None else None)
                for k in outputs
            }
            one_output['predicted_perm_matrix'] = perm_matrix.unsqueeze(0)
            unp_outputs = unpermute_graph(one_output)
            if accuracies_all_metrics_batch_outputs(one_batch, unp_outputs)[-1].sum()>0:
                ret_match[i] = True
                break
    return ret_match


def accuracies_all_metrics(X: torch.Tensor,
                           E: torch.Tensor,
                           sizes: torch.Tensor,
                           pred_X: torch.Tensor,
                           pred_E: torch.Tensor,
                           pred_sizes: torch.Tensor,
                           max_size: int,
                           hydrogens: Optional[torch.Tensor] = None,
                           formal_charges: Optional[torch.Tensor] = None,
                           pred_hydrogens: Optional[torch.Tensor] = None,
                           hydrogens_mask: Optional[torch.Tensor] = None,
                           pred_formal_charges: Optional[torch.Tensor] = None,
                           formal_charges_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    accX, accE, accS = all_accuracies(X,
                                    pred_X,
                                    E,
                                    pred_E,
                                    sizes,
                                    pred_sizes)
    accH = accuracy_hydrogens(hydrogens, pred_hydrogens, hydrogens_mask) if hydrogens is not None and pred_hydrogens is not None else torch.ones_like(accS)
    accFC = accuracy_formal_charges(formal_charges, pred_formal_charges, formal_charges_mask) if formal_charges is not None and pred_formal_charges is not None else torch.ones_like(accS)
    accuracy_per_graph = accX*accE*accS*accH*accFC
    counts = torch.bincount(sizes, minlength=max_size+1)
    corrects = torch.bincount(sizes[accuracy_per_graph], minlength=max_size+1)
    bs = accX.shape[-1]
    accuracy_list = (accX*accE*accS*accH*accFC)
    acc_with_size = accuracy_list.sum()/bs
    acc_without_size = (accX*accE*accH*accFC).sum()/bs
    return acc_with_size, acc_without_size, corrects, counts, accuracy_list

def accuracies_all_metrics_batch_outputs(
        batch, outputs, check_with_permutations: bool=False
):
    hydrogens_mask = batch.get("node_mask") * batch.get("aromatic_mask")
    formal_charges_mask = batch.get("node_mask") * batch.get("aromatic_mask")
    pred_X = outputs.get('X')
    pred_E = outputs.get('E')
    pred_H = outputs.get('predicted_hydrogens')
    pred_FC = outputs.get('predicted_formal_charges')
    pred_sizes = outputs.get('predicted_sizes_log').exp().round().long()

    acc_with_size, acc_without_size, corrects, counts, accuracy_list = accuracies_all_metrics(
        X=batch.get('X'),
        E=batch.get('E'),
        sizes=batch.get('graph_sizes'),
        pred_X=pred_X,
        pred_E=pred_E,
        pred_sizes=pred_sizes,  # Convert log-sizes back to sizes
        max_size=batch.get('X').shape[1],
        hydrogens=batch.get("hydrogens"),
        pred_hydrogens=pred_H,
        hydrogens_mask=hydrogens_mask,
        formal_charges=batch.get("formal_charges"),
        pred_formal_charges=pred_FC,
        formal_charges_mask=formal_charges_mask)

    if check_with_permutations:
        not_matching_idx = torch.where(~accuracy_list)[0]
        if not_matching_idx.numel() > 0:
            match_perm = check_permutation_accuracy(batch, outputs, not_matching_idx=not_matching_idx)
            new_matching_idx = not_matching_idx[match_perm]
            acc_with_size += match_perm.sum()/(pred_X.shape[0])
            correct_sizes = batch.get('graph_sizes')[new_matching_idx]
            corrects += torch.bincount(correct_sizes, minlength=batch.get('X').shape[1]+1)
            accuracy_list[new_matching_idx] = True
    return acc_with_size, acc_without_size, corrects, counts, accuracy_list

def edit_distance_graph_batch_outputs(batch, outputs):
    node_mask = batch.get("node_mask").bool()
    node_targets = batch.get("X")
    edge_targets = batch.get("E")
    node_preds = outputs.get("X").argmax(-1)
    edge_preds = outputs.get("E").argmax(-1)

    node_mismatches = ((node_preds != node_targets) & node_mask).to(torch.float32)
    edge_mask = node_mask[:, :, None] & node_mask[:, None, :]
    edge_mismatches = ((edge_preds != edge_targets) & edge_mask).to(torch.float32)

    edit_graph = node_mismatches.sum(dim=-1) + edge_mismatches.sum(dim=(1, 2)) / 2
    acc_graph = torch.where(
        edit_graph < 1e-5,
        torch.ones_like(edit_graph),
        torch.zeros_like(edit_graph),
    )
    return edit_graph, acc_graph

def accuracies_molecules(
        batch, outputs
):
    molecules_gt = make_molecules_from_batch(batch)
    molecules_out = make_molecules_from_outputs(outputs)
    smiles_gt = [Chem.MolToSmiles(mol) for mol in molecules_gt]
    smiles_out = [Chem.MolToSmiles(mol) for mol in molecules_out]
    acc = torch.tensor([a==b for a,b in zip(smiles_gt,smiles_out)])
    acc = acc.sum()/acc.size(0)
    return acc


def accuracies_permutations_one_graph(
        batch, outputs, idx_graph
):
    hydrogens_mask = batch.get("node_mask") * batch.get("aromatic_mask")
    formal_charges_mask = batch.get("node_mask") * batch.get("aromatic_mask")

    pred_X = outputs.get('X')
    pred_E = outputs.get('E')
    pred_H = outputs.get('predicted_hydrogens')
    pred_FC = outputs.get('predicted_formal_charges')
    pred_sizes = outputs.get('predicted_sizes_log').exp().round().long()

    return(
        accuracies_all_metrics(
            X=batch.get('X'),
            E=batch.get('E'),
            sizes=batch.get('graph_sizes'),
            pred_X=pred_X,
            pred_E=pred_E,
            pred_sizes=pred_sizes,  # Convert log-sizes back to sizes
            max_size=batch.get('X').shape[1],
            hydrogens=batch.get("hydrogens"),
            pred_hydrogens=pred_H,
            hydrogens_mask=hydrogens_mask,
            formal_charges=batch.get("formal_charges"),
            pred_formal_charges=pred_FC,
            formal_charges_mask=formal_charges_mask)
    )
