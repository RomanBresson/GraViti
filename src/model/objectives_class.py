from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from src.model.grale_losses.grale_losses import QuadraticCE, LinearCE, LinearBCE, MarginalKL
from src.utils.evaluations import unpermute_graph
from .losses import (
    recon_loss_nodes,
    recon_loss_edges,
    recon_loss_hydrogens,
    recon_loss_formal_charges,
    loss_kl,
    recon_loss_log_sizes,
    accuracies_all_metrics_batch_outputs,
    edit_distance_graph_batch_outputs,
)
from src.data.treat_data import treat_batch
from src.utils.misc import is_ddp

class LossStack():
    def __init__(self, 
        node_class_weights: Optional[torch.Tensor] = None,
        edge_class_weights: Optional[torch.Tensor] = None,
        use_focal: bool = False,
        use_grale_loss: bool = False,
        ratio_negative_edges: float = 1.0,
        weight_edge_loss: float = 1.0):
    
        self.node_class_weights = node_class_weights
        self.edge_class_weights = edge_class_weights
        self.use_focal = use_focal
        self.ratio_negative_edges = ratio_negative_edges
        self.weight_edge_loss = weight_edge_loss

        self.loss_nodes_fn = recon_loss_nodes if not use_grale_loss else LinearCE()
        self.loss_edges_fn = recon_loss_edges if not use_grale_loss else QuadraticCE(mask_self_loops=True)
        self.loss_hydrogens_fn = recon_loss_hydrogens if not use_grale_loss else LinearCE()
        self.loss_formal_charges_fn = recon_loss_formal_charges if not use_grale_loss else LinearCE()
        self.loss_kl_fn = loss_kl
        self.loss_sizes_fn = recon_loss_log_sizes
        self.loss_soft_mask = None if not use_grale_loss else LinearBCE()
        self.marginal_objective = None if not use_grale_loss else MarginalKL()
        self.use_grale_loss = use_grale_loss

    def forward(self, batch, outputs):
        LH = torch.zeros((), device=batch.get('X').device)
        LFC = torch.zeros((), device=batch.get('X').device)
        Lsize = torch.zeros((), device=batch.get('X').device)
        Ly = outputs.get('y').sum() * 0.0
        Lmask = torch.zeros((), device=batch.get('X').device)
        Lmargin = torch.zeros((), device=batch.get('X').device)
        LReg = torch.zeros((), device=batch.get('X').device)
        if not self.use_grale_loss:
            LE = self.loss_edges_fn(
                labels=batch.get('E'),
                logits=outputs.get('E'),
                weights=self.edge_class_weights,
                use_focal=self.use_focal,
                ratio_negative_edges=self.ratio_negative_edges,
            )
            LX = self.loss_nodes_fn(
                labels=batch.get('X'),
                logits=outputs.get('X'),
                node_mask=batch.get('node_mask'),
                weights=self.node_class_weights,
                use_focal=self.use_focal,
                soft_perm_matrix=outputs.get('predicted_perm_matrix') if self.use_grale_loss else None
            )
            if outputs.get('predicted_hydrogens') is not None:
                aromatic_mask = batch.get("aromatic_mask")
                hydrogen_labels = batch.get("hydrogens")
                LH = self.loss_hydrogens_fn(
                    hydrogen_counts=hydrogen_labels,
                    logits=outputs.get('predicted_hydrogens'),
                    node_mask=batch.get("node_mask") * aromatic_mask,
                    use_focal=True,
                    weights=None,
                    soft_perm_matrix=outputs.get('predicted_perm_matrix')  if self.use_grale_loss else None
                )
            if outputs.get('predicted_formal_charges') is not None:
                aromatic_mask = batch.get("aromatic_mask")
                formal_charge_labels = batch["formal_charges"]
                LFC = self.loss_formal_charges_fn(
                    formal_charges=formal_charge_labels,
                    logits=outputs.get('predicted_formal_charges'),
                    node_mask=batch.get("node_mask") * aromatic_mask,
                    use_focal=True,
                    weights=None,
                    soft_perm_matrix=outputs.get('predicted_perm_matrix')  if self.use_grale_loss else None
                )
        else:
            permutation_matrices = outputs['predicted_perm_matrix']
            targets_E = torch.nn.functional.one_hot(batch.get('E'), num_classes=outputs.get('E').size(-1)).float()
            targets_X = torch.nn.functional.one_hot(torch.clamp(batch.get('X'),min=0, max=outputs.get('X').size(-1)-1), num_classes=outputs.get('X').size(-1)).float()
            targets_mask = batch.get('node_mask').float()

            size_targets = targets_mask.sum(dim=-1).clamp_min(1.0)
            size_max = float(targets_mask.size(-1))

            Lmask = self.loss_soft_mask.forward(T = permutation_matrices, F1 = outputs['predicted_node_mask'], F2 = targets_mask)
            Lmask = (Lmask / size_max).mean()
            LX = self.loss_nodes_fn.forward(T = permutation_matrices, F1 = outputs['X'], F2 = targets_X, weight_2 = batch['node_mask'])
            LX = (LX / size_targets).mean()
            LE = self.loss_edges_fn.forward(T = permutation_matrices, C1 = outputs['E'], C2 = targets_E, weight_2 = batch['node_mask'])
            LE = (LE / (size_targets**2)).mean()
            Lmargin = self.marginal_objective.forward(permutation_matrices).mean()

        Lsize = self.loss_sizes_fn(
            batch.get('graph_sizes'),
            outputs.get('predicted_sizes_log').squeeze(-1)
        )

        # KL divergence term for VAE
        LKL = torch.zeros((), device=batch.get('X').device)
        if outputs.get('latent_mu') is not None:
            LKL = self.loss_kl_fn(outputs.get('latent_mu'), outputs.get('latent_log_sigma'))
        losses = {
            'X': LX,
            'E': LE,
            'KL': LKL,
            'SIZE': Lsize,
            'Y': Ly,
            'H': LH,
            'FC': LFC,
            'REG': LReg,
            'MASK': Lmask,
            'MARGIN': Lmargin
        }
        return losses

class Evaluator():
    def __init__(self,
            loss_stack: Optional[LossStack] = None,
            check_with_permutations: Optional[bool] = False
    ):
        self.loss_stack = loss_stack
        self.check_with_permutations = check_with_permutations

    def forward(self,
        loader,
        model,
        override_treat = None
    ):
        with torch.no_grad():
            model.eval()
            device = next(model.parameters()).device
            base_model = model.module if isinstance(model, DDP) else model
            # Totalizers
            # Reconstruction Loss, Accuracy, KL Loss, Size Loss, Size Accuracy,
            # Edit Distance, Zero Edit-Distance Rate, # of Elements
            metrics_sum = torch.zeros(8, device=device)
            counts_sum = None
            corrects_sum = None
            edit_sum_by_size = None
            with torch.inference_mode():
                for b in loader:
                    if isinstance(b, dict):
                        batch = b
                    elif override_treat is not None:
                        batch = override_treat(b.to(device))
                    else:
                        batch = treat_batch(b, device)
                    batch_size = batch.get('X').shape[0]
                    max_size = b.x.shape[0]//batch_size #before pruning

                    # Forward pass & reconstruction loss/accuracy
                    if getattr(self.loss_stack, "use_grale_loss", False):
                        # GRALE Loss is used:
                        # Loss calculation aligned with training (soft matching)
                        soft_outputs = base_model(batch, hard_matching=False, override_size_padding=batch['X'].size(1))
                        if self.loss_stack is not None:
                            losses = self.loss_stack.forward(batch, soft_outputs)
                        else:
                            losses = {k:torch.zeros((), device=device) for k in ['X', 'E', 'KL', 'SIZE']}
                        # Strict reconstruction accuracy (hard matching)
                        outputs = base_model(batch, hard_matching=True, override_size_padding=batch['X'].size(1))
                    else:
                        # Otherwise: Strict loss and reconstruction accuracy
                        outputs = base_model(batch, hard_matching=True, override_size_padding=batch['X'].size(1))
                        if self.loss_stack is not None:
                            losses = self.loss_stack.forward(batch, outputs)
                        else:
                            losses = {k:torch.zeros((), device=device) for k in ['X', 'E', 'KL', 'SIZE']}

                    pred_matrix = outputs.get('predicted_perm_matrix') # will be hard
                    if pred_matrix is not None:
                        outputs = unpermute_graph(outputs)

                    edit_graph, acc_graph = edit_distance_graph_batch_outputs(batch, outputs)

                    acc_with_size, acc_without_size, correct_batch, count_batch, accuracy_list = accuracies_all_metrics_batch_outputs(batch=batch, outputs=outputs, check_with_permutations=self.check_with_permutations)
                    # Update total loss & accuracy counters (weighed by batch size)
                    if counts_sum is None:
                        counts_sum = torch.zeros(max_size+1, device=device)
                        corrects_sum = torch.zeros(max_size+1, device=device)
                        edit_sum_by_size = torch.zeros(max_size+1, device=device)
                    counts_sum += count_batch
                    corrects_sum += correct_batch
                    edit_sum_by_size += torch.bincount(
                        batch.get('graph_sizes'),
                        weights=edit_graph,
                        minlength=max_size+1,
                    )
                    metrics_sum[0] += (losses['X'] + losses['E']) * batch_size
                    metrics_sum[1] += acc_without_size * batch_size
                    metrics_sum[2] += losses['KL'] * batch_size
                    metrics_sum[3] += losses['SIZE'] * batch_size
                    metrics_sum[4] += acc_with_size * batch_size
                    metrics_sum[5] += edit_graph.sum()
                    metrics_sum[6] += acc_graph.sum()
                    metrics_sum[7] += batch_size

            if is_ddp():
                dist.all_reduce(metrics_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(counts_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(corrects_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(edit_sum_by_size, op=dist.ReduceOp.SUM)

            # Compute Averages
            total_elements = metrics_sum[7]
            avg_metrics = metrics_sum[:7] / total_elements
            counts_sum = torch.clip(counts_sum, min=1.)
            acc_by_size = (corrects_sum / counts_sum)
            edit_by_size = edit_sum_by_size / counts_sum
            return tuple(avg_metrics.cpu().tolist()), acc_by_size, edit_by_size
