from dataclasses import dataclass, fields, asdict
from typing import Optional, Any
import inspect

import torch


@dataclass
class ModelMetadata:
    """
    Metadata needed to reconstruct a model but not included in
    the `TrainingConfig` object.

    Dataset-specific attributes that are calculated only after
    loading and processing the dataset used for training.
    """

    # Dataset characteristics
    max_size: int
    num_node_features: int
    num_edge_features: int
    y_size: int
    pe_lap_size: int
    pe_rw_size: int
    atom_attr_dim: int
    max_hydrogens: int
    formal_charge_bounds: tuple[int, int]
    reinject_size: bool

    # Data preprocessing info (not needed for re-building the
    # model but might be useful for inference)
    node_class_weights: Optional[torch.Tensor]
    edge_class_weights: Optional[torch.Tensor]

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the `ModelMetadata` as a :class:`dict`."""
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
        return cls(**filtered)