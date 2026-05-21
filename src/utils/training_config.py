from pathlib import Path
from dataclasses import dataclass, field, asdict
import json
from typing import Optional, List, Any
import inspect

from ..data.constants import DatasetName


@dataclass(kw_only=True)
class TrainingConfig:
    """
    Configuration class for training parameters

    Args:
        dataset (DatasetName):
            Any available dataset between the defined dataset names (`data.constants.DatasetName`)
        batch_size (int):
            How many samples to load per batch
        with_aromatic (bool):
            Include aromatic bonds (set to `False` if kekulization will be used)
        latent_size (int):
            Latent space size
        encoder_hidden_size (int):
            Hidden size of the transformers
        encoder_output_size (int):
            Output size of the transformers
        encoder_heads (int):
            Attention heads of the transformers
        encoder_layers (int):
            Number of transformer layers for encoder
        decoder_layers (int):
            Number of transformer layers for decoder
        decoder_sigma (float):
            Noise for the decoder's sequence transformer
        reg_weight (float):
            Regression weight (0 = no regressor)
        variational (bool):
            Use VAE (if True), otherwise deterministic AE
        use_atom_attr (bool):
            Use atoms properties
        lr (float):
            Learning rate (maximum)
        lr_gamma (float):
            Max learning rate decay max factor by cycle
        weight_decay (float):
            Weight decay for regularization
        cycle_length (int):
            Period of LR-scheduler cycles (in sub-epochs/chunks)
        beta (float):
            Maximal value of beta
        beta_cycle (int, default=None):
            Number of scheduler cycles for beta annealing.
            If `None` (default), defaults to 2 LR cycles (see `cycle_length`).
        dropout (float):
            Dropout for transformers
        ratio_negative_edges (float):
            Ratio of negative edges
        use_focal (bool):
            Use Focal Loss (False=Cross-Entropy, True=Focal)
        use_grale_loss (bool):
            Use GRALE loss for edges (True=Use GRALE, False=Don't use GRALE)
        use_classes_weights (bool):
            Use weights for classes in loss computation
            (True=Use weights, False=Don't use weights)
        predict_hydrogens_formal_charges (bool):
            Predict hydrogen counts and formal charges (True=Predict, False=Don't predict)
        reinject_size (bool):
            Re-inject graph size information into the decoder (True=Re-inject, False=Don't re-inject)
        max_epochs (int, default=None):
            Maximum number of epochs (full passes over the dataset) to train for.
            If `None` (default), defaults to 10 LR cycles (see `cycle_length`).
        chunk_size (int, default=None):
            Sub-epoch chunk size, for updating schedulers within a large epoch.
            If `None` (default), full dataset will be used as scheduler step.
        warmup_epochs (int, default=None):
            Number of sub-epochs (chunks) used for LR-scheduler.
            If `None` (default), defaults to 1/50 of a LR cycle (see `cycle_length`).
        early_stopping_patience (int, default=None):
            Number of sub-epochs (chunks) to wait without improvement before
            early stopping.
            If `None` (default), defaults to 3 LR cycles (see `cycle_length`).
        weight_edge_loss (float):
            Weight of the reconstruction loss for edges
        exists_ok (int):
            Behavior if model already exists: 0=Raise error, 1=Re-train and
            overwrite previous model, 2=Continue training from checkpoint
        wandb_project (str):
            Weights & Biases project name
        wandb_entity (str):
            Weights & Biases entity name
        experiment_name (str):
            Experiment name for tracking
        wandb_tags (str):
            Tags for the experiment
        wandb_group (str):
            Group for current experiments
        wandb_notes (str):
            Notes for the experiment
        track_gradients (bool):
            Track gradients in W&B
    """

    # Data parameters
    dataset: DatasetName
    batch_size: int
    with_aromatic: bool
    predict_hydrogens_formal_charges: bool

    # Model architecture
    latent_size: int
    encoder_hidden_size: int
    encoder_output_size: int
    encoder_heads: int
    encoder_layers: int
    decoder_layers: int
    decoder_sigma: float
    reg_weight: float
    variational: bool
    use_atom_attr: bool
    reinject_size: bool

    # Training parameters
    lr: float
    lr_gamma: float = 0.1
    weight_decay: float
    cycle_length: int
    beta: float
    beta_cycle: Optional[int] = None
    dropout: float
    ratio_negative_edges: float
    use_focal: bool
    use_grale_loss: bool
    use_classes_weights: bool

    max_epochs: Optional[int] = None
    chunk_size: Optional[int] = None
    warmup_epochs: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    weight_edge_loss: float

    # Logging and Experiment tracking
    exists_ok: int
    log_interval: int
    wandb_project: str
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_group: Optional[str] = None
    experiment_name: Optional[str] = None
    track_gradients: bool = False
    use_grale_loss: bool = False

    def __post_init__(self):
        """
        Post-initialization processing for optional parameters
        that should default to other parameters' values.
        """
        # Beta scheduler cycle length
        if self.beta_cycle is None:
            self.beta_cycle = 2 * self.cycle_length

        # Max epochs
        if self.max_epochs is None:
            self.max_epochs = 10 * self.cycle_length

        # Warmup epochs
        if self.warmup_epochs is None:
            self.warmup_epochs = self.cycle_length // 50

        # Early stopping patience
        if self.early_stopping_patience is None:
            self.early_stopping_patience = self.cycle_length * 3

    @property
    def checkpoint_name(self) -> str:
        """
        Descriptive name based on configuration, in order to be used for checkpointing.
        """
        return (
            f"{'vae' if self.variational else 'ae'}_{self.dataset}_"
            f"batch{self.batch_size}_arom{int(self.with_aromatic)}_hfc{int(self.predict_hydrogens_formal_charges)}_lat{self.latent_size}_"
            f"enc{self.encoder_hidden_size}_{self.encoder_output_size}_{self.encoder_heads}_{self.encoder_layers}_"
            f"dec{self.decoder_layers}_{self.decoder_sigma}_reinj{int(self.reinject_size)}_regw{self.reg_weight}_attr{int(self.use_atom_attr)}_"
            f"lr{self.lr}_{self.lr_gamma}_wd{self.weight_decay}_"
            f"b{self.beta}_{self.beta_cycle}_cycle{self.cycle_length}_"
            f"drop{self.dropout}_rne{self.ratio_negative_edges}_fc{int(self.use_focal)}_gl{int(self.use_grale_loss)}_"
            f"weloss{self.weight_edge_loss}"
        )

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the `TrainingConfig` as a :class:`dict`."""
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "TrainingConfig":
        """
        Create a new `TrainingConfig` object from a state dictionary, created
        with `state_dict`.

        Args:
            state_dict (dict): TrainingConfig state. Should be an object returned
                from a call to :meth:`state_dict`.

        Returns:
            TrainingConfig: Constructed class object.
        """
        valid_keys = inspect.signature(cls.__init__).parameters
        filtered = {k: v for k, v in state.items() if k in valid_keys}
        if "use_grale_loss" not in filtered:
            filtered["use_grale_loss"] = False
        return cls(**filtered)

    def to_json(self, json_path: Path, mkdir: bool = True):
        """
        Save training configuration to JSON file.

        Args:
            json_path (Path):
                The path where the configuration should be saved (path/to/file.json)
            mkdir (bool, default=True):
                Create the "json_path" directory if it doesn't already exist
        """
        if mkdir:
            json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf8") as f:
            json.dump(self.state_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(json_path: Path) -> "TrainingConfig":
        """
        Load a saved training configuration from a JSON file.

        Args:
            json_path (Path):
                The path where the configuration should be loaded from (path/to/file.json)
        """
        with open(json_path, "r", encoding="utf8") as f:
            data = json.load(f)

        return TrainingConfig(**data)
