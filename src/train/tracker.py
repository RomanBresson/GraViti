import json
from pathlib import Path
from typing import Optional, Any, Dict

import torch.nn as nn
import wandb

from ..utils.training_config import TrainingConfig


class ExperimentTracker:
    """Wrapper for experiment tracking with Weights & Biases"""

    def __init__(self, config: TrainingConfig, tracking_dir: Path):
        self.config = config

        # If experiment name is not defined, use config's name
        if config.experiment_name is not None:
            self.experiment_name = config.experiment_name
        else:
            self.experiment_name = config.checkpoint_name

        self.experiment_id = self._get_run_id(tracking_dir / "wandb")

        try:
            # Initialize wandb
            self.run = wandb.init(
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=str(tracking_dir),
                id=self.experiment_id,
                name=self.experiment_name,
                notes=config.wandb_notes,
                tags=config.wandb_tags,
                config=wandb.helper.parse_config(
                    config.state_dict(),
                    exclude=(
                        "exists_ok",
                        "log_interval",
                        "wandb_project",
                        "wandb_entity",
                        "wandb_tags",
                        "wandb_notes",
                        "wandb_group",
                        "experiment_name",
                        "track_gradients",
                    ),
                ),
                group=config.wandb_group,
                mode="online",
                reinit="default",
                resume="never" if config.exists_ok == 1 else "allow",
            )

        except Exception as e:
            self.run = None
            raise Exception(f"Failed to initialize Weights & Biases: {e}")

    def _get_run_id(self, tracking_dir: Path):
        # Load run ID from checkpoint or generate new one
        run_id_dir = tracking_dir / "ids"
        run_id_dir.mkdir(parents=True, exist_ok=True)
        run_id_file = run_id_dir / self.experiment_name

        # No saved ID or new run is started
        if not run_id_file.exists() or self.config.exists_ok == 1:
            run_id = wandb.util.generate_id()
            with open(run_id_file, "w") as f:
                f.write(run_id)
            return run_id

        # Restore run ID for given experiment
        with open(run_id_file, "r") as f:
            return f.read().strip()

    def log_json(self, data: Dict[str, Any], name: Optional[str] = "data"):
        """Log a data dictionary as json to wandb"""
        try:
            self.run.log({name: wandb.Html(f"<pre>{json.dumps(data, indent=2)}</pre>")})
        except Exception as e:
            print(f"Failed to log data to wandb: {e}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb"""
        try:
            self.run.log(metrics, step=step)
        except Exception as e:
            print(f"Failed to log metrics to wandb: {e}")

    def log_summary(self, metrics: Dict[str, Any]):
        """Log summary metrics to wandb"""
        try:
            wandb.run.summary["final_metrics"] = metrics
        except Exception as e:
            print(f"Failed to log metrics to wandb: {e}")

    def watch_model(self, model: nn.Module):
        """Watch model for gradient and parameter logging"""
        if self.run is not None and self.config.track_gradients:
            try:
                self.run.watch(model, log="all", log_freq=self.config.log_interval)
            except Exception as e:
                print(f"Failed to watch model: {e}")

    def finish(self):
        """Finish the wandb run"""
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as e:
                print(f"Failed to finish wandb run: {e}")
