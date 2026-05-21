import math
from typing import Any, Tuple

import torch

from torch.nn.parallel import DistributedDataParallel as DDP

from src.model.objectives_class import LossStack

class EarlyStopper:
    """
    Implements early stopping during training to prevent overfitting.

    Attributes:
        patience (int): Number of epochs to wait without improvement before stopping.
        min_delta (float): Minimum change in validation loss to qualify as an improvement.
    """

    def __init__(self, patience: int = 1, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> Tuple[bool, bool]:
        """
        Check if training should stop early based on validation loss. If a model
        improvement is detected, suggest saving the model. If model is not improving
        for `patience` epochs, suggest stopping training.

        Args:
            validation_loss (float): Current validation loss (or 1 - accuracy).

        Returns:
            Tuple[bool,bool]: (should_stop, should_save) - decided based on model's improvement
                as described above.
        """

        # No improvement - increment counter
        if validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            # if `patience` is exceeded, advice to stop
            return self.counter >= self.patience, False

        # Improvement detected - reset counter & advice to save
        if validation_loss <= self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            return False, True

        # Default case - return False, False
        return False, False

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the `EarlyStopper` as a :class:`dict`."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "counter": self.counter,
            "min_validation_loss": self.min_validation_loss,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load the `EarlyStopper`'s state.

        Args:
            state_dict (dict): EarlyStopper state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.patience = state_dict["patience"]
        self.min_delta = state_dict["min_delta"]
        self.counter = state_dict["counter"]
        self.min_validation_loss = state_dict["min_validation_loss"]


def annealing_scheduler_beta(
    epoch: int,
    beta: float,
    cycle_length: int,
    plateau: bool = True,
) -> float:
    """
    Beta annealing scheduler for variational models.

    The schedule follows a half-sinusoidal warm-up curve for KL-divergence
    weight (`beta`) over training epochs:
    - Starts at 0 at the beginning of a cycle.
    - Gradually increases to `beta` by the end of the cycle.
    - Resets every `cycle_length` epochs.

    Args:
        epoch (int): Current training epoch (0-indexed).
        beta (float): Maximum value for beta (per cycle)
        cycle_length (int): Scheduler cycle (in epochs) - reset beta on this interval
        plateau (bool, default=True): If True, beta remains at max value after first cycle.
                                     If False, beta resets every cycle.

    Returns:
        float: The current beta value in [0, beta].
    """
    # Current step [0, cycle_length - 1]
    if plateau and epoch >= cycle_length:
        return beta

    step_in_cycle = epoch % cycle_length

    # Calculate current value of beta [0, beta]
    cycle_ratio = step_in_cycle / (cycle_length - 1)  # [0, 1]
    curr_beta = beta * 0.5 * (math.sin(math.pi * (cycle_ratio - 0.5)) + 1)

    return min(curr_beta, beta)

def train_one_from_batched(
    model: torch.nn.Module,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    objective: LossStack,
) -> Tuple[float, float, float, float, float]:
    """
    Perform a single training step on a batched graph dataset.

    Args:
        model (torch.nn.Module): Autoencoder/VariationalAutoencoder instance.
        batch (torch.Tensor): Dictionary with batched tensors (nodes, edges, etc.).
        optimizer (torch.optim.Optimizer): Optimizer used for model update.
        node_class_weights (torch.Tensor | None): Optional tensor of class weights for nodes.
        edge_class_weights (torch.Tensor | None): Optional tensor of class weights for edges.
        use_focal (bool, default=False): Use Focal Loss (otherwise cross-entropy).
        use_grale_loss (bool, default=False): Use GRALE loss for edges (otherwise standard loss).
        ratio_negative_edges (float, default=1): Ratio of negative edges sampled for reconstruction.
        weight_edge_loss (float, default=1): Weight of the reconstruction loss for edge.

    Returns:
        Tuple[float,float,float,float,float]:
            (node reconstruction loss, edge reconstruction loss, KL loss, regression loss, size prediction loss)
    """
    model.train()
    optimizer.zero_grad()
    base_model = model.module if isinstance(model, DDP) else model
    # Decode latent embeddings to reconstruct node and edge features

    outputs = base_model(batch)
    losses = objective.forward(
        batch=batch,
        outputs=outputs
    )
    beta = getattr(base_model, "beta", 0.0)
    # Total loss + backpropagation
    loss = (
            losses["X"]
            + losses["E"] * objective.weight_edge_loss
            + losses["KL"] * beta
            + losses["SIZE"]
            + losses["Y"]
            + 0.1 * (losses["H"] + losses["FC"])
            + losses["MASK"]
            + losses["MARGIN"]
            + losses["REG"] * 0.0
        )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    optimizer.step()
    # Return scalar values of all losses

    return (
        losses['X'].detach().item(),
        losses['E'].detach().item(),
        losses['KL'].detach().item(),
        losses['REG'].detach().item(),
        losses['SIZE'].detach().item(),
        losses['H'].detach().item(),
        losses['FC'].detach().item(),
    )

def summarize_model(model):
    print(f"{'Submodule':30} {'# Parameters':>15}")
    print("-" * 50)
    for name, module in model.named_modules():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{name:30} {num_params:15}")
