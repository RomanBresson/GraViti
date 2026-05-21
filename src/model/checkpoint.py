import sys
from pathlib import Path
import time
from typing import Optional

import torch


class ModelCheckpoint:
    """
    Complete model checkpoint with everything needed for inference
    or continuing training.
    """

    def __init__(
        self,
        model_state_dict: dict,
        config_state_dict: dict,
        metadata_state_dict: dict,
        optimizer_state_dict: Optional[dict] = None,
        scheduler_state_dict: Optional[dict] = None,
        early_stopper_dict: Optional[dict] = None,
        global_step: int = 0,
        save_timestamp=time.time(),
        python_version=sys.version,
        pytorch_version=torch.__version__,
    ):
        self.model_state_dict = model_state_dict
        self.config_state_dict = config_state_dict
        self.metadata_state_dict = metadata_state_dict
        self.optimizer_state_dict = optimizer_state_dict
        self.scheduler_state_dict = scheduler_state_dict
        self.early_stopper_dict = early_stopper_dict
        self.global_step = global_step
        self.save_timestamp = save_timestamp
        self.python_version = python_version
        self.pytorch_version = pytorch_version

    def save(self, path: Path | str):
        """Save complete checkpoint"""
        torch.save(
            {
                "model_state_dict": self.model_state_dict,
                "config_state_dict": self.config_state_dict,
                "metadata_state_dict": self.metadata_state_dict,
                "optimizer_state_dict": self.optimizer_state_dict,
                "scheduler_state_dict": self.scheduler_state_dict,
                "early_stopper_dict": self.early_stopper_dict,
                "global_step": self.global_step,
                "save_timestamp": self.save_timestamp,
                "python_version": self.python_version,
                "pytorch_version": self.pytorch_version,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "ModelCheckpoint":
        """
        Load complete checkpoint.

        Args:
            path (Path | str): Path to checkpoint file

        Returns:
            ModelCheckpoint: The saved checkpoint data, loaded to CPU.
        """
        checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)

        return cls(
            model_state_dict=checkpoint_data["model_state_dict"],
            config_state_dict=checkpoint_data["config_state_dict"],
            metadata_state_dict=checkpoint_data["metadata_state_dict"],
            optimizer_state_dict=checkpoint_data.get("optimizer_state_dict"),
            scheduler_state_dict=checkpoint_data.get("scheduler_state_dict"),
            early_stopper_dict=checkpoint_data.get("early_stopper_dict"),
            global_step=checkpoint_data.get("global_step", 0),
            save_timestamp=checkpoint_data.get("save_timestamp", 0),
            python_version=checkpoint_data.get("python_version", "0.0.0"),
            pytorch_version=checkpoint_data.get("pytorch_version", "0.0.0"),
        )
