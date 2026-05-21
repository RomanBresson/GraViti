from typing import Any, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import networkx as nx
from networkx.algorithms import isomorphism

from rdkit import Chem

import matplotlib.pyplot as plt
import seaborn as sn
import PIL


from ..data.treat_data import treat_batch
from ..data.constants import DatasetName
from .chem import (
    molecular_weight,
    compute_penalized_logP,
    get_largest,
    make_molecule_from_xe,
    make_molecules,
)


def plot_graph(
    a: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    non_node_index: Optional[int] = None,
):
    """
    Plot a graph from an adjacency matrix using NetworkX.

    Args:
        a (torch.Tensor):
            Adjacency matrix (N, N)
        colors (torch.Tensor, optional):
            Tensor of colors for each node (node labels)
        non_node_index (int, optional):
            Node index indicating non-existent nodes (padding nodes)

    NOTE: If `colors` is provided and `non_node_index` is set, nodes with that
    label are removed (padding nodes).
    """
    g = nx.from_numpy_array(a.cpu().numpy())
    c = None

    if (colors is not None) and (non_node_index is not None):
        to_remove = torch.where(colors == non_node_index)[0].tolist()
        to_stay = [k for k in range(len(a)) if not k in to_remove]
        g.remove_nodes_from(to_remove)
        c = colors[to_stay]

    nx.draw(g, node_size=100, node_color=c)


def plot_from_model_output(
    model_output: Tuple[torch.Tensor, torch.Tensor],
    node_mask: torch.Tensor,
):
    """
    Plot reconstructed graphs from model output using a node mask.

    Args:
        model_output (tuple): (pred_X, pred_E) where
            pred_X (torch.Tensor): Node logits, shape (B, N, C_nodes).
            pred_E (torch.Tensor): Edge logits, shape (B, N, N, C_edges).
        node_mask (torch.Tensor): Boolean mask of shape (B, N), where True indicates valid node.
    """
    pred_X, pred_Adj_labs = model_output
    X_labels = pred_X.argmax(-1).cpu().detach()
    E_labels = pred_Adj_labs.argmax(-1).cpu().detach()
    non_edge_label = pred_Adj_labs.shape[-1] - 1
    # Binary adjacency: 1 if edge exists, 0 otherwise
    for i, (x_labs, e_labs, mask) in enumerate(zip(X_labels, E_labels, node_mask)):
        mask = mask.cpu()
        x_labs = x_labs[mask]
        adj = (e_labs[mask][:, mask] != non_edge_label).long()
        adj = adj * (1 - torch.eye(adj.shape[-1]))

        # Mask out invalid nodes in adjacency labels
        e_labs = e_labs.clone()
        e_labs[~mask] = non_edge_label
        e_labs[torch.where(torch.eye(e_labs.shape[0]))] = non_edge_label
        e_labs[:, ~mask] = non_edge_label

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        sn.heatmap(e_labs.detach().cpu(), vmax=non_edge_label, cmap="gist_stern")
        plt.title("Edge Labels")

        plt.subplot(1, 3, 2)
        plot_graph(adj, x_labs)
        plt.title("Predicted Graph")

        plt.subplot(1, 3, 3)
        sn.heatmap(x_labs.unsqueeze(1), vmin=0, cmap="gist_stern")
        plt.title("Node Labels")

        plt.tight_layout()
        plt.show()


def plot_batch_output(
    model_output: Tuple[torch.Tensor, torch.Tensor], batch: dict, max_nb: int = 8
):
    """
    Plot several graphs from a batch and their model reconstructions.

    Args:
        model_output (Tuple): (pred_X, pred_E) predicted node and edge logits.
        batch (dict): Batch dictionary containing `tensors`.
        max_nb (int, default=8): Maximum number of graphs to plot.
    """
    i = 0
    for x, e, px, pe, mask in zip(
        batch["X"], batch["E"], model_output[0], model_output[1], batch["node_mask"]
    ):
        print("Ground truth")
        plot_from_model_output(
            model_output=(
                F.one_hot(x.unsqueeze(dim=0), px.shape[-1] + 1),
                F.one_hot(e.unsqueeze(dim=0), pe.shape[-1]),
            ),
            node_mask=mask.unsqueeze(0),
        )
        print("Predicted")
        plot_from_model_output(
            model_output=(px.unsqueeze(dim=0), pe.unsqueeze(dim=0)),
            node_mask=mask.unsqueeze(0),
        )

        # Max `max_nb` plots
        i += 1
        if i == max_nb:
            break


def sample_with_guidance(
    latent_dim: int,
    guidance: torch.Tensor,
    U: torch.nn.Module,
    steps: int,
    coef: float = 0.5,
    z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Evolve latent vectors using a velocity field with guidance.

    Args:
        latent_dim (int): Dimension of latent space.
        guidance (torch.Tensor): Conditioning tensor, shape (batch, cond_dim).
        U (torch.nn.Module): Velocity field function U(z, guidance, t).
        steps (int): Number of integration steps.
        coef (float, default=0.5): Interpolation coefficient between guided and unguided updates.
        z (torch.Tensor, optional): Initial latent (batch, latent_dim). If None, sampled from N(0, I).

    Returns:
        torch.Tensor: Final latent after guided integration, shape (batch, latent_dim).
    """
    device = guidance.device
    U.eval()
    if z is None:
        z = torch.randn(len(guidance), latent_dim).to(device)

    nb_steps = steps
    dt = 1 / nb_steps
    for i in range(nb_steps):
        t = i / nb_steps
        ug = U(z, guidance, t * torch.ones(size=(z.shape[0], 1), device=device))
        uu = U(z, guidance * 0.0, t * torch.ones(size=(z.shape[0], 1), device=device))
        z = z + ((1 - coef) * uu + coef * ug) * dt
    return z


def one_example(X: torch.Tensor, E: torch.Tensor, smiles_gt: str) -> PIL.Image:
    """
    Render two molecules as an image for comparison:
    * Largest connected fragment of the generated molecule from (X, E)
    * Original molecule from SMILES notation

    Args:
        X (torch.Tensor): Node labels for predicted molecule.
        E (torch.Tensor): Edge labels for predicted molecule.
        smiles_gt (str): Ground truth SMILES string.

    Returns:
        PIL.Image: Grid image of [ground truth, predicted molecule].
    """
    mols = []
    mols.append(Chem.MolFromSmiles(smiles_gt))
    mols.append(get_largest(make_molecule_from_xe(X, E)))
    img = Chem.Draw.MolsMatrixToGridImage([mols], subImgSize=(200, 200))
    return img


def sample_latent(nb_examples: int, latent_dim: int = 128) -> torch.Tensor:
    """
    Sample standard normal latent vectors.

    Args:
        nb_examples (int): Number of samples.
        latent_dim (int): Latent dimension.

    Returns:
        torch.Tensor: Latent samples, shape (nb_examples, latent_dim).
    """
    return torch.randn(nb_examples, latent_dim)


def molecules_from_latent(z: torch.Tensor, decoder: nn.Module):
    """
    Decode latent vectors into molecules.

    Args:
        z (torch.Tensor): Latent vectors, shape (batch, latent_dim).
        decoder (nn.Module): Decoder returning (X, E) tensors.

    Returns:
        list[Mol]: List of RDKit molecules.
    """
    X, E = decoder(z)
    mols = [make_molecule_from_xe(x, e) for x, e in zip(X, E)]
    return mols


def get_valids(list_of_mols):
    total = len(list_of_mols)
    valids = [get_largest(mol) for mol in list_of_mols]
    return [v for v in valids if not v is None]


def logP_MAE(list_of_mols: List[Chem.Mol], expected: torch.Tensor) -> torch.Tensor:
    """
    Compute mean absolute error for penalized logP property.

    Args:
        list_of_mols (List[rdkit.Chem.Mol]): Molecules to evaluate.
        expected (torch.Tensor or float): Target logP value.

    Returns:
        torch.Tensor: Scalar MAE.
    """
    logps = torch.tensor([compute_penalized_logP(mol) for mol in list_of_mols])
    return (logps - expected).abs().mean()


def MW_MAE(list_of_mols: List[Chem.Mol], expected: torch.Tensor) -> torch.Tensor:
    """
    Compute mean absolute error for molecular weight property.

    Args:
        list_of_mols (list[Mol]): Molecules to evaluate.
        expected (torch.Tensor or float): Target molecular weight.

    Returns:
        torch.Tensor: Scalar MAE.
    """
    mws = torch.tensor([molecular_weight(mol) for mol in list_of_mols])
    return (mws - expected).abs().mean()


def sample_with_different_steps(
    z: torch.Tensor,
    decoder: nn.Module,
    vel_field: nn.Module,
    guidance: torch.Tensor,
    nb_steps_list: List[int],
    guidance_coef: float,
    solver: str = "midpoint",
) -> List[List[Chem.Mol]]:
    """
    Generate molecules with different integration step counts.

    Args:
        z (torch.Tensor): Initial latent vectors (batch, latent_dim).
        decoder (torch.nn.Module): Decoder mapping z -> (X, E).
        vel_field (torch.nn.Module): Velocity field with .sample() method.
        guidance (torch.Tensor): Conditioning tensor.
        nb_steps_list (list[int]): List of step counts to try.
        guidance_coef (float): Weight of guidance vs unguided updates.
        solver (str, default="midpoint"): Integration solver (passed to vel_field.sample).

    Returns:
        List[List[rdkit.Chem.Mol]]: For each step count, list of molecules generated.
    """
    with torch.inference_mode():
        lists_mols = []
        for steps in nb_steps_list:
            z1 = vel_field.sample(z, guidance, steps, guidance_coef, solver)
            mols = molecules_from_latent(z1, decoder)
            lists_mols.append(mols)
    return lists_mols


def sample_with_different_guidances(
    z: torch.Tensor,
    decoder: nn.Module,
    vel_field: nn.Module,
    guidance: torch.Tensor,
    nb_steps: int,
    guidance_coef_list: List[float],
) -> List[List[Chem.Mol]]:
    """
    Generate molecules with different guidance strengths.

    Args:
        z (torch.Tensor): Initial latent vectors (batch, latent_dim).
        decoder (torch.nn.Module): Decoder mapping z -> (X, E).
        vel_field (torch.nn.Module): Velocity field with .sample() method.
        guidance (torch.Tensor): Conditioning tensor.
        nb_steps (int): Number of integration steps.
        guidance_coef_list (list[float]): List of guidance coefficients to try.

    Returns:
        list[list[Mol]]: For each coef, list of molecules generated.
    """
    with torch.inference_mode():
        lists_mols = []
        for guidance_coef in guidance_coef_list:
            z1 = vel_field.sample(z, guidance, nb_steps, guidance_coef)
            mols = molecules_from_latent(z1, decoder)
            lists_mols.append(mols)
    return lists_mols


def build_graph(
    node_labels: torch.Tensor,
    edge_labels: torch.Tensor,
    to_delete_node: int,
    to_delete_edge: int,
) -> nx.Graph:
    """
    Build a NetworkX graph from node and edge labels.

    Args:
        node_labels (torch.Tensor): Node labels (n,) (already discrete, not logits).
        edge_labels (torch.Tensor): Edge labels (n, n).
        to_delete_node (int, optional): Node label to skip (placeholder).
        to_delete_edge (int, optional): Edge label to skip (e.g., "no edge").

    Returns:
        networkx.Graph: Constructed undirected graph with node/edge labels.
    """
    G = nx.Graph()
    node_labels, edge_labels = delete_placeholder_nodes(
        lab_X=node_labels, lab_E=edge_labels, to_delete=to_delete_node
    )

    node_labels = node_labels.cpu().numpy()
    edge_labels = edge_labels.cpu().numpy()

    n = len(node_labels)

    # Construct graph nodes
    for i in range(n):
        G.add_node(i, label=node_labels[i])

    # Construct graph edges
    for i in range(n):
        for j in range(i + 1, n):
            label = edge_labels[i][j]
            if (
                label != to_delete_edge
            ):  # Add this condition according to your edge presence criteria
                G.add_edge(i, j, label=label)
    return G


def are_isomorphic(
    X1: torch.Tensor,
    E1: torch.Tensor,
    X2: torch.Tensor,
    E2: torch.Tensor,
    no_node_class: int,
    no_edge_class: int,
) -> Tuple[float, List[nx.Graph], List[nx.Graph], List[bool]]:
    """
    Check if two graphs are isomorphic to ground truth graphs.

    Args:
        X1 (torch.Tensor): Graph 1 node labels (batch_size, num_nodes).
        E1 (torch.Tensor): Graph 1 edge labels (batch_size, num_nodes, num_nodes).
        X2 (torch.Tensor): Graph 2 node labels (batch_size, num_nodes).
        E2 (torch.Tensor): Graph 2 edge labels (batch_size, num_nodes, num_nodes).
        no_node_class (int): Class index for placeholder/padding nodes.
        no_edge_class (int): Class index for placeholder/padding edges.

    Returns:
        Tuple[float,List[networkx.Graph],List[networkx.Graph],List[bool]]:
            Tuple of:
            * ratio_ok (float): Fraction of graphs that are exactly isomorphic.
            * graphs_gt (List[nx.Graph]): Ground-truth graphs.
            * graphs_pred (List[nx.Graph]): Predicted graphs.
            * are_ok (List[bool]): Per-graph isomorphism results.
    """
    total_graphs = X1.shape[0]

    # Build graph objects from labels
    graphs_gt_1 = [
        build_graph(x, e, no_node_class, no_edge_class) for x, e in zip(X1, E1)
    ]
    graphs_gt_2 = [
        build_graph(x, e, no_node_class, no_edge_class) for x, e in zip(X2, E2)
    ]

    # Check graph isomorphism using node/edge labels
    are_ok = [
        isomorphism.is_isomorphic(
            G1=G1,
            G2=G2,
            node_match=lambda n1, n2: n1["label"] == n2["label"],
            edge_match=lambda e1, e2: e1["label"] == e2["label"],
        )
        for G1, G2 in zip(graphs_gt_1, graphs_gt_2)
    ]

    return sum(are_ok) / total_graphs, graphs_gt_1, graphs_gt_2, are_ok


def are_isomorphic_loader(
    model: nn.Module, loader: torch.utils.data.DataLoader
) -> Tuple[float, List[List[bool]]]:
    """
    Evaluate a model over a dataset loader by checking isomorphism graph-by-graph.

    Args:
        model (torch.nn.Module): Graph generative model returning node/edge logits.
        loader (torch.utils.data.DataLoader): Loader yielding graph batches.

    Returns:
        Tuple[float,List[List[bool]]]:
            Tuple of:
            * ratio_ok (float): Overall fraction of isomorphic predictions.
            * are_ok_all (list[list[bool]]): Per-batch per-graph isomorphism results.
    """
    model = model.eval()
    device = next(model.parameters()).device
    are_ok_all = []

    if len(loader) == 0:
        return 0.0, are_ok_all

    with torch.inference_mode():
        total_graphs = 0.0
        total_good = 0.0

        for b in loader:
            batch = treat_batch(b, device)
            X, E = batch["X"], batch["E"]

            # Forward pass - Predicted node/edge logits and classes
            outputs = model(batch)
            raise NotImplementedError
            no_node_label = pred_X.shape[-1]
            no_edge_label = pred_E.shape[-1]

            pred_X = torch.argmax(pred_X, dim=-1)
            pred_E = torch.argmax(pred_E, dim=-1)

            # Compare Ground-truth/prediction and accumulate counts
            _, _, _, are_ok = are_isomorphic(
                X1=X,
                E1=E,
                X2=pred_X,
                E2=pred_E,
                no_node_class=no_node_label,
                no_edge_class=no_edge_label,
            )
            total_good += sum(are_ok)
            total_graphs += len(are_ok)

            are_ok_all.append(are_ok)

    return total_good / total_graphs, are_ok_all


def delete_placeholder_nodes(
    lab_X: torch.Tensor, lab_E: torch.Tensor, to_delete: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove nodes with a placeholder label (e.g., 'no_node') and their corresponding edges.

    Args:
        lab_X (torch.Tensor): Node labels (n,).
        lab_E (torch.Tensor): Edge labels (n, n).
        to_delete (int): Node label id to remove.

    Returns:
        Tuple[torch.Tensor,torch.Tensor]:
            * lab_X (torch.Tensor): Filtered node labels.
            * lab_E (torch.Tensor): Filtered edge labels (square matrix).
    """
    indices = torch.where(lab_X != to_delete)[0]
    lab_X = lab_X[indices]
    lab_E = lab_E[indices][:, indices]
    return lab_X, lab_E


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, negative_samples_ratio=1.0):
        super().__init__()
        self.negative_samples_ratio = negative_samples_ratio

    def forward(self, tensor1, tensor2):
        batch_size = tensor1.size(0)
        positive_distances = F.cosine_similarity(tensor1, tensor2)
        mask = (1 - torch.eye(batch_size, batch_size, device=tensor1.device)).bool()
        num_negative_samples = int(self.negative_samples_ratio * batch_size)
        negative_indices_all = torch.stack(torch.where(mask))
        negative_indices = torch.randint(
            0, negative_indices_all.shape[1], (num_negative_samples,)
        )
        print(negative_indices)
        elements = negative_indices_all[:, negative_indices]
        print(elements)
        t1, t2 = tensor1[elements[0]], tensor2[elements[1]]
        print(t1, t2)
        negative_distances = F.cosine_similarity(t1, t2)
        pd = torch.sum(positive_distances)
        nd = torch.sum(negative_distances)
        print(pd, nd, num_negative_samples)
        loss = pd - nd
        loss = loss / (batch_size + num_negative_samples)
        return loss


class DyT(nn.Module):
    """
    From Transformers without normalization (Zhu et al. 2025)
    """

    def __init__(self, C):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.zeros(C))
        self.gamma = nn.Parameter(torch.ones(C))

    def forward(self, x):
        x = F.tanh(self.alpha * x)
        return self.gamma * x + self.beta


def all_molecules_in_batch(batch: Any, dataset: DatasetName):
    treated_batch = treat_batch(batch, "cpu")
    return make_molecules(treated_batch["X"], treated_batch["E"], dataset)


def molecules_batch_model(model: torch.nn.Module, batch: Any, dataset: DatasetName):
    real_molecules = all_molecules_in_batch(batch)
    treated_batch = treat_batch(batch, list(model.parameters())[0].device)
    with torch.no_grad():
        outs = model(treated_batch)
        reconstd_molecules = make_molecules(*outs, dataset)
    return real_molecules, reconstd_molecules


def plot_molecule_lists(list_of_molecules_lists: list, max_numbers: int = 10):
    mols_all = []
    rows = min(max_numbers, len(list_of_molecules_lists[0]))
    for r in range(rows):
        mols = []
        for lom in list_of_molecules_lists:
            mols.append(lom[r])
        mols_all.append(mols)
    img = Chem.Draw.MolsMatrixToGridImage(mols_all, subImgSize=(200, 200))
    return img


def forensics(AE, train_loader):
    with torch.inference_mode():
        device = next(AE.parameters()).device
        batch = next(iter(train_loader))
        batch = treat_batch(batch, device)

        X, E = batch["X"], batch["E"]
        pred_X, pred_E = AE(batch)

        return X, E, pred_X, pred_E


def normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    Normalize class weights (or any tensor) by the mean value
    of its elements (helps balance node/edge classification).

    Args:
        weights (torch.Tensor): The tensor to be normalized

    Returns:
        torch.Tensor: The normalized tensor
    """
    return weights / weights.mean()


def is_ddp() -> bool:
    """
    Check whether PyTorch Distributed Data Parallel (DDP) is active.

    Returns:
        bool: True if torch.distributed is available and the process group
        has been initialized (i.e., DDP is active), False otherwise.
    """
    return dist.is_available() and dist.is_initialized()

def mask_from_sizes(sizes, max_size=None):
    """
    Create a boolean mask from graph sizes.

    Args:
        sizes (torch.Tensor): Tensor of shape (B,) containing sizes of each graph in the batch.
        max_size (int, optional): Maximum size to create the mask for. If None, uses max in sizes.
    Returns:
        torch.Tensor: Boolean mask of shape (B, max_size), where True indicates valid node.
    """
    sizes = sizes.round().long()
    if max_size is None:
        max_size = sizes.max().item()
    batch_size = sizes.size(0)
    mask = torch.arange(max_size, device=sizes.device).expand(batch_size, max_size) < sizes.unsqueeze(1)
    return mask


def mask_E(node_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the mask for edges based on the node mask.

    Args:
        node_mask (torch.Tensor): A boolean tensor of shape (batch_size, num_nodes) where True indicates a real node and False indicates a non-node.
    Returns:
        torch.Tensor: A boolean tensor of shape (batch_size, num_nodes, num_nodes)
                        where True indicates a valid edge (both nodes are real) and False indicates an invalid edge.
    """
    # Create edge mask based on node_mask
    N = node_mask.shape[-1]
    edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)  # [B, N, N]

    triu = torch.triu(
        torch.ones((N, N), dtype=torch.bool, device=edge_mask.device), diagonal=1
    )  # (N,N)
    edge_mask = edge_mask & triu.unsqueeze(0)  # (B, N, N)

    return edge_mask