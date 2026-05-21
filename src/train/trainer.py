import os
from pathlib import Path
import math
from typing import Tuple, Optional

import torch
import torch.optim as optim
from torch.utils.data import Subset

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

from torch_geometric.loader import DataLoader

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from loguru import logger

from ..data.treat_data import load_data, make_loader, treat_batch
from ..data.constants import get_valid_atoms, get_valid_bonds, treat_aromatic_valid_table
from ..model.checkpoint import ModelCheckpoint
from ..model.builder import ModelMetadata, ModelBuilder
from ..model.losses import normalize_weights, weights_from_loader
from ..model.objectives_class import LossStack,Evaluator

from .tracker import ExperimentTracker
from ..utils.training_config import TrainingConfig
from ..utils.misc import is_ddp
from .utils import (
    train_one_from_batched,
    EarlyStopper,
    annealing_scheduler_beta,
    summarize_model,
)

# For DataLoaders with many workers

mp.set_start_method("spawn", force=True)

class Trainer:
    """
    Trainer class.
    Handles data loading and preprocessing, model building and full training loop with
    logging and checkpointing.

    Args:
        config (TrainingConfig): Training configuration
        data_root (str): Root directory where dataset will be stored
        splits (list[float], optional, default=[0.8, 0.1, 0.1]): train/validation/test split ratios -
            only applicable to `QM9` and 'PubChem' datasets (`ZINC` datasets are pre-splitted)
        force_reload (bool): Force re-processing the dataset (needed when pre_transform is changed)
    """

    def __init__(
        self,
        config: TrainingConfig,
        data_root: str = "data",
        splits: list = [0.8, 0.1, 0.1],
        force_reload: bool = False,
        models_root: str = "saved_models"
    ):
        # Base dirs for storing training related data & logs
        self.models_base_dir = Path(models_root)
        self.models_last_dir = self.models_base_dir / "last"
        self.models_last_dir.mkdir(parents=True, exist_ok=True)

        self.logs_base_dir = Path("logs")
        self.logs_base_dir.mkdir(parents=True, exist_ok=True)

        self.tracker_base_dir = Path(".")
        self.tracker_base_dir.mkdir(parents=True, exist_ok=True)

        # Configuration and Dataset inputs
        self.config = config

        self.data_root = data_root
        self.splits = splits
        self.force_reload = force_reload

        # Validate splits input
        assert math.isclose(
            sum(splits), 1.0, rel_tol=1e-6
        ), f"Splits must sum to 1.0 ({splits}={sum(splits)})"
        assert all(s > 0 for s in splits), f"Splits must be positive, got {splits}"

        # Paths to save the best model and the last checkpoint (for continuing training)
        self.model_name = self.config.checkpoint_name
        self.best_checkpoint_filename = self.models_base_dir / self.model_name
        self.last_checkpoint_filename = self.models_last_dir / self.model_name

        self.log_path = self.logs_base_dir / f"{self.model_name}.log"

        # Get rank info and detect GPU
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        # init process group if DDP
        if self.world_size > 1:
            dist.init_process_group("nccl", device_id=self.device)

        # Training info
        self.checkpoint = self._handle_existing_checkpoints()
        self.batch_size = max(1, self.config.batch_size // self.world_size)

        self.global_step = 0
        self.epoch = 0
        self.chunk_index = 0
        self.scheduler_step = 0

        # Initialize experiment tracking and logger
        self.tracker, self.log_id = None, None
        if self.is_main_process:
            self.tracker = ExperimentTracker(config, self.tracker_base_dir)
            self.log_id = logger.add(
                str(self.log_path),
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                colorize=True,
            )

        logger.info(f"Initialized trainer on device: {self.device}")
        if is_ddp():
            logger.info(
                f"DDP info: Rank {self.rank}/{self.world_size} | Local rank: {self.local_rank}"
            )
        if self.is_main_process:
            logger.info(f"Model name: {self.model_name}")
            logger.info(f"Configuration: {self.config}")

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def _handle_existing_checkpoints(self):
        """
        Handle overwriting existing checkpoints or loading them to continue training.
        """
        checkpoint = None
        checkpoint_exists = (
            self.last_checkpoint_filename.exists()
            or self.best_checkpoint_filename.exists()
        )

        if is_ddp():
            dist.barrier()

        # If model already exists follow behavior defined in `config.exists_ok`
        if checkpoint_exists:
            # Overwrite previous trained model
            if self.config.exists_ok == 1:
                # Only main process (in case of DDP)
                if self.is_main_process:
                    self.best_checkpoint_filename.unlink(missing_ok=True)
                    self.last_checkpoint_filename.unlink(missing_ok=True)
                    logger.warning(
                        f"Removed previous trained model from {str(self.models_base_dir)}"
                    )

            # Continue training from checkpoint
            elif self.config.exists_ok == 2:
                checkpoint_path = (
                    self.last_checkpoint_filename
                    if self.last_checkpoint_filename.exists()
                    else self.best_checkpoint_filename
                )
                checkpoint = ModelCheckpoint.load(checkpoint_path)
                if self.is_main_process:
                    logger.info(f"Loading checkpoint from {str(checkpoint_path)}")
                    logger.info(
                        f"Checkpoint info: Timestamp {checkpoint.save_timestamp} | "
                        f"Torch: {checkpoint.pytorch_version} | "
                        f"Python: {checkpoint.python_version}"
                    )

            # Raise error if `exists_ok` is undefined or 0
            else:
                raise Exception(
                    f"Model checkpoint already exists at {str(self.models_base_dir)}. "
                    f"Use `exists_ok=1` to overwrite or `exists_ok=2` to continue training."
                )

        if is_ddp():
            dist.barrier()

        return checkpoint

    def setup_loaders(self):
        """
        Load and preprocess data for the selected dataset, and create the
        validation and test dataloaders  (`self.val_loader`, `self.test_loader`).

        The training dataloader is not created here - the full dataset is stored
        in `self.train_data`.
        """
        if self.is_main_process:
            logger.info(f"Loading dataset: {self.config.dataset}")

        # Get dataset splits
        self.train_data, val_data, test_data = load_data(
            dataset=self.config.dataset,
            splits=self.splits,
            force_reload=self.force_reload,
            with_aromatic=self.config.with_aromatic,
            data_root=self.data_root,
        )

        # Create samplers and loaders
        val_sampler = DistributedSampler(val_data, shuffle=False) if is_ddp() else None
        self.val_loader = make_loader(
            dataset=val_data,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=val_sampler,
        )

        tst_sampler = DistributedSampler(test_data, shuffle=False) if is_ddp() else None
        self.test_loader = make_loader(
            dataset=test_data,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=tst_sampler,
        )

        if self.is_main_process:
            logger.info(
                "# of Data: "
                f"Train={len(self.train_data)} | "
                f"Validation={len(val_data)} | "
                f"Test={len(test_data)}"
            )

    def prepare_metadata(self, checkpoint: Optional[ModelCheckpoint]):
        """
        Prepare metadata required to build the model. Optionally, can
        skip calculations and load metadata from a saved checkpoint.

        Args:
            checkpoint (ModelCheckpoint|None):
                If provided, the metadata will not be calculated but loaded
                from the corresponding metadata dictionary saved in the given
                `ModelCheckpoint`.

        NOTE: Metadata will be saved in `self.metadata`.
        """
        # Load metadata from checkpoint
        if checkpoint is not None:
            metadata = ModelMetadata.from_state_dict(checkpoint.metadata_state_dict)

        # Compute all metadata from loaded dataset
        else:
            weights_loader = make_loader(
                dataset=self.train_data,
                batch_size=self.batch_size,
                shuffle=False,
                workers=0,
            )

            # Parse one batch to get dataset metadata
            batch = treat_batch(next(iter(weights_loader)), device="cpu")

            max_size = batch["X"].shape[1]
            pe_lap_size = batch["PE"]["pe_lap"].shape[-1]
            pe_rw_size = batch["PE"]["pe_rw"].shape[-1]
            y_size = batch["y"].shape[-1] if batch.get("y") is not None else None
            atom_attr_dim = (
                batch["atom_attr"].shape[-1]
                if batch.get("atom_attr") is not None
                else None
            )

            valid_atoms          = get_valid_atoms(self.config.dataset)
            num_node_classes     = len(valid_atoms)
            num_edge_classes     = len(get_valid_bonds(self.config.with_aromatic, self.config.dataset)) + 1
            max_hydrogens = 0
            formal_charge_bounds = (0,0)

            if self.config.predict_hydrogens_formal_charges:
                max_hydrogens, min_fc, max_fc, atoms_must_use_classifier = treat_aromatic_valid_table(self.config.dataset, self.config.with_aromatic)  # for validation during training
                formal_charge_bounds = (min_fc, max_fc)

            # Compute class weights and number of node/edge features
            node_class_weights = torch.ones(num_node_classes)
            edge_class_weights = torch.ones(num_edge_classes)

            if self.config.use_classes_weights:
                node_class_weights, edge_class_weights = weights_from_loader(
                    weights_loader
                )
                node_class_weights = node_class_weights.sqrt()
                edge_class_weights = edge_class_weights.sqrt()
                node_class_weights = normalize_weights(node_class_weights)
                edge_class_weights = normalize_weights(edge_class_weights)

            num_node_features = node_class_weights.shape[0]
            num_edge_features = edge_class_weights.shape[0]

            # Create `ModelMetadata` object
            metadata = ModelMetadata(
                max_size=max_size,
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                y_size=y_size,
                pe_lap_size=pe_lap_size,
                pe_rw_size=pe_rw_size,
                atom_attr_dim=atom_attr_dim,
                node_class_weights=node_class_weights,
                edge_class_weights=edge_class_weights,
                max_hydrogens=max_hydrogens,
                formal_charge_bounds=formal_charge_bounds,
                reinject_size=self.config.reinject_size
            )

            if self.is_main_process:
                logger.info(f"Max size: {metadata.max_size}")
                logger.info(f"Metadata: {metadata}")

        # Finalize and log metadata
        metadata.node_class_weights = metadata.node_class_weights.to(self.device)
        metadata.edge_class_weights = metadata.edge_class_weights.to(self.device)
        self.metadata = metadata

        # Only log to tracker once
        if checkpoint is None and self.is_main_process:
            self.tracker.log_json(
                name="metadata",
                data={
                    **self.metadata.state_dict(),
                    "node_class_weights": self.metadata.node_class_weights.cpu().tolist(),
                    "edge_class_weights": self.metadata.edge_class_weights.cpu().tolist(),
                },
            )

    def build_model(self, checkpoint: Optional[ModelCheckpoint]):
        """
        Build the autoencoder model using `ModelBuilder` with the given configuration and metadata.

        Optionally, load its weights from a saved checkpoint.

        Args:
            checkpoint (ModelCheckpoint|None):
                If provided, the model's weights will be loaded from the
                corresponding model dictionary saved in the given `ModelCheckpoint`.

        NOTE: Model will be saved in `self.model`.
        """
        if checkpoint is None:
            model = ModelBuilder.build_from_config(self.config, self.metadata)
        else:
            model = ModelBuilder.build_from_checkpoint(checkpoint)

        model = model.to(self.device)

        # Wrap in DDP
        if is_ddp():
            self.model = DDP(module=model, device_ids=[self.local_rank])
        else:
            self.model = model

        # Log the model's parameter count (and separately for the encoder/decoder)
        base_model = self.model.module if isinstance(self.model, DDP) else self.model
        enc_params = sum(param.numel() for param in base_model.encoder.parameters())
        dec_params = sum(param.numel() for param in base_model.decoder.parameters())
        model_params = sum(param.numel() for param in base_model.parameters())

        ## Only log model info once - do not bloat the log file
        if checkpoint is None and self.is_main_process:
            logger.info("Model details:")
            logger.info(self.model)
            logger.info(summarize_model(self.model))
            logger.info(
                f"Parameters:\nEncoder: {enc_params:,} parameters | Decoder: {dec_params:,} parameters | Total: {model_params:,} parameters"
            )
            self.tracker.log_json(
                name="model",
                data={
                    "encoder_parameters": enc_params,
                    "decoder_parameters": dec_params,
                    "total_parameters": model_params,
                },
            )

        # Track model
        if self.is_main_process:
            self.tracker.watch_model(base_model)

    def setup_optimizer(self, checkpoint: Optional[ModelCheckpoint]):
        """
        Setup Optimizer, LR Scheduler and Early Stopping detection.

        Optionally, load their states from a saved checkpoint.

        Args:
            checkpoint (ModelCheckpoint|None):
                If provided, the optimizer's, scheduler's and early stopper's
                state will be loaded from the corresponding dictionaries saved
                in the given `ModelCheckpoint`.

        NOTE: Optimizer, scheduler and early stopper will be saved in `self.optimizer`,
        `self.scheduler`, and `self.early_stopper` accordingly.
        """

        # AdamW optimizer
        self.optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        # Cosine annealing for LR
        # NOTE: `sched_needs_val` might be used by other schedulers (do not remove)
        self.scheduler, self.sched_needs_val = (
            CosineAnnealingWarmupRestarts(
                optimizer=self.optimizer,
                first_cycle_steps=self.config.cycle_length,
                cycle_mult=1.5,
                max_lr=self.config.lr,
                min_lr=1e-8,
                warmup_steps=self.config.warmup_epochs,
                gamma=self.config.lr_gamma,
            ),
            False,
        )
        if checkpoint is not None:
            self.scheduler.load_state_dict(checkpoint.scheduler_state_dict)

        # Early stopper - detect no progress in model and instruct to stop
        self.early_stopper = EarlyStopper(
            patience=self.config.early_stopping_patience,
        )
        if checkpoint is not None:
            self.early_stopper.load_state_dict(checkpoint.early_stopper_dict)

        self.objective = LossStack(
            node_class_weights=self.metadata.node_class_weights,
            edge_class_weights=self.metadata.edge_class_weights,
            use_focal=self.config.use_focal,
            ratio_negative_edges=self.config.ratio_negative_edges,
            weight_edge_loss=self.config.weight_edge_loss,
            use_grale_loss=self.config.use_grale_loss
        )

        self.evaluator = Evaluator(
            self.objective
        )

    def _train_chunk(
        self, loader: DataLoader
    ) -> Tuple[float, float, float, float, float]:
        """
        Train model for one chunk (epoch or sub-epoch) and log training data
        every 10 global training steps.

        Returns:
            Tuple[float,float,float,float,float]:
                Last batch's node reconstruction loss, edge reconstruction loss, KL
                loss, regression loss, size prediction loss.
        """
        self.scheduler_step = self.epoch * self.chunks_per_epoch + self.chunk_index

        base_model = self.model.module if isinstance(self.model, DDP) else self.model
        base_model.beta = annealing_scheduler_beta(
            epoch=self.scheduler_step,
            beta=self.config.beta,
            cycle_length=self.config.beta_cycle,
            plateau=True,
        )

        for b in loader:
            batch = treat_batch(b, self.device)

            lnode, ledge, lkl, lreg, lsize, lhydrogens, lformal_charges = train_one_from_batched(
                model=self.model,
                batch=batch,
                optimizer=self.optimizer,
                objective=self.objective
            )
            self.global_step += 1

            # Track batch metrics (index corresponds to total batches counter)
            if self.global_step % 50 == 0:
                if self.is_main_process:
                    self.tracker.log_metrics(
                        metrics={
                            "batch/node_loss": lnode,
                            "batch/edge_loss": ledge,
                            "batch/kl_loss": lkl,
                            "batch/reg_loss": lreg,
                            "batch/size_loss": lsize,
                            "batch/hydrogens_loss": lhydrogens,
                            "batch/formal_charges_loss": lformal_charges,
                        },
                        step=self.global_step,
                    )

        # Average losses across all ranks
        if is_ddp():
            losses = torch.tensor(
                [lnode, ledge, lkl, lreg, lsize, lhydrogens, lformal_charges],
                device=self.device,
            )
            dist.all_reduce(losses, op=dist.ReduceOp.AVG)
            lnode, ledge, lkl, lreg, lsize, lhydrogens, lformal_charges = losses.tolist()

        return lnode, ledge, lkl, lreg, lsize, lhydrogens, lformal_charges

    def _train_epoch(self) -> bool:
        """
        Train the model for all chunks in one epoch.

        Returns:
            bool:
                `True` if early stopping was triggered.
        """
        # Deterministic per-epoch permutation
        g = torch.Generator()
        g.manual_seed(self.epoch)
        train_size = len(self.train_data)
        all_indices = torch.randperm(train_size, generator=g).tolist()

        base_model = self.model.module if isinstance(self.model, DDP) else self.model

        while self.chunk_index < self.chunks_per_epoch:
            start = self.chunk_index * self.chunk_size
            end = min(start + self.chunk_size, train_size)
            chunk_indices = all_indices[start:end]

            # Build chunk loader
            chunk_data = Subset(self.train_data, chunk_indices)

            chunk_sampler = None
            if is_ddp():
                chunk_sampler = DistributedSampler(chunk_data, shuffle=False)
                chunk_sampler.set_epoch(self.scheduler_step)

            chunk_loader = make_loader(
                dataset=chunk_data,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=chunk_sampler,
            )

            # Train
            self._train_chunk(chunk_loader)
            lreg = 0.0  # placeholder for regression loss if not used
            # Evaluate
            (
                val_recon_loss,
                val_acc,
                val_kl_loss,
                val_size_loss,
                val_acc_with_sizes,
                val_edit_distance,
                val_edit_distance_acc,
            ), val_accuracy_by_size, val_edit_distance_by_size = (
                self.evaluator.forward(
                    self.val_loader,
                    self.model
                )
            )

            if self.is_main_process:
                ## Log to file
                logger.info(
                    f"Epoch {self.epoch+1}/{self.config.max_epochs}, "
                    f"chunk {self.chunk_index+1}/{self.chunks_per_epoch} | "
                    f"Global step: {self.global_step:,}"
                )
                logger.info(
                    f"\tValid Reconstruction Loss: {val_recon_loss:.6f} | KL Loss: {val_kl_loss:.6f} | Accuracy: {val_acc:.6f} | Size Loss: {val_size_loss:.6f} | Accuracy with Sizes: {val_acc_with_sizes:.6f} | Edit Distance: {val_edit_distance:.6f}"
                )

                ## Log in tracker
                self.tracker.log_metrics(
                    metrics={
                        "val/reconstruction_loss": val_recon_loss,
                        "val/kl_loss": val_kl_loss,
                        "val/accuracy": val_acc,
                        "val/size_loss": val_size_loss,
                        "val/edit_distance": val_edit_distance,
                        "val/edit_distance_accuracy": val_edit_distance_acc,
                        "hyperparameters/beta": base_model.beta,
                        "hyperparameters/lr": self.optimizer.param_groups[0]["lr"],
                        "epoch": self.epoch,
                        "scheduler_step": self.scheduler_step,
                    },
                    step=self.global_step,
                )

            # Update scheduler
            if self.sched_needs_val:
                self.scheduler.step(val_recon_loss)
            else:
                self.scheduler.step()

            # Periodic logging for training stats - Every `config.log_interval` scheduler steps (chunks)
            if (self.scheduler_step + 1) % self.config.log_interval == 0:
                (
                    (train_recon_loss,
                    train_acc,
                    train_kl_loss,
                    train_size_loss,
                    train_acc_with_sizes,
                    train_edit_distance,
                    train_edit_distance_acc),
                    train_accuracy_by_size,
                    train_edit_distance_by_size
                ) = self.evaluator.forward(
                    model=self.model,
                    loader=chunk_loader,
                )

                if self.is_main_process:
                    # Log to file
                    logger.info(
                        f"\tTrain Reconstruction Loss: {train_recon_loss:.6f} | KL Loss: {train_kl_loss:.6f} | Accuracy: {train_acc:.6f} | Size Loss: {train_size_loss:.6f} | Accuracy with Sizes: {train_acc_with_sizes:.6f} | Edit Distance: {train_edit_distance:.6f}"
                    )
                    if base_model.has_regressor:
                        logger.info(f"\tRegression Loss Train: {lreg:.6f}")

                    # Log in `tracker`
                    self.tracker.log_metrics(
                        metrics={
                            "train/reconstruction_loss": train_recon_loss,
                            "train/kl_loss": train_kl_loss,
                            "train/accuracy": train_acc,
                            "train/regression_loss": lreg,
                            "train/size_loss": train_size_loss,
                            "train/edit_distance": train_edit_distance,
                            "train/edit_distance_accuracy": train_edit_distance_acc,
                        },
                        step=self.global_step,
                    )

            # Early stopping and checkpointing
            chk = ModelCheckpoint(
                model_state_dict=base_model.state_dict(),
                config_state_dict=self.config.state_dict(),
                metadata_state_dict=self.metadata.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                early_stopper_dict=self.early_stopper.state_dict(),
                global_step=self.global_step,
            )

            to_stop, to_save = False, False
            if self.is_main_process:
                to_stop, to_save = self.early_stopper.early_stop(1 - val_acc)

            if is_ddp():
                # broadcast stop/save decisions
                stop_tensor = torch.tensor(
                    [to_stop], device=self.device, dtype=torch.int
                )
                save_tensor = torch.tensor(
                    [to_save], device=self.device, dtype=torch.int
                )
                dist.broadcast(stop_tensor, src=0)
                dist.broadcast(save_tensor, src=0)
                to_stop = bool(stop_tensor.item())
                to_save = bool(save_tensor.item())

            if self.is_main_process:
                if to_save:
                    chk.save(self.best_checkpoint_filename)
                    logger.info("Model checkpoint saved")

                chk.save(self.last_checkpoint_filename)

            if to_stop:
                logger.warning("Early stopping")
                return True

            self.chunk_index += 1

        # Reset chunk index if loop was finished
        else:
            self.chunk_index = 0

        return False

    def train(self, checkpoint: Optional[ModelCheckpoint]):
        """
        Training Loop:
        * Runs for `config.max_epochs` epochs, optionally intializing the starting
        epoch from a saved checkpoint (epoch = 1 full pass over train dataset).
        * Inside each epoch, splits the dataset into chunks of size `config.chunk_size`
        and trains sequentially on each chunk.
        * Logs training and validation metrics
        * Uses early stopping with patience `config.early_stopping_patience` if model
            does not improve on validation accuracy
        * Saves model checkpoints when validation accuracy improves

        Args:
            checkpoint (ModelCheckpoint|None):
                If provided, the training will continue from the last
                global training step saved in the given `ModelCheckpoint`.
        """
        # Dataset info
        train_size = len(self.train_data)
        if self.config.chunk_size is None:
            self.chunk_size = train_size
        else:
            self.chunk_size = min(self.config.chunk_size, train_size)
        steps_per_chunk = math.ceil(self.chunk_size / self.config.batch_size)
        last_chunk_size = train_size % self.chunk_size
        steps_last_chunk = math.ceil(last_chunk_size / self.config.batch_size)

        self.chunks_per_epoch = math.ceil(train_size / self.chunk_size)
        steps_per_epoch = (
            train_size // self.chunk_size
        ) * steps_per_chunk + steps_last_chunk

        # Get epoch and chunk index from global training step
        self.global_step = 0
        if checkpoint is not None:
            self.global_step = checkpoint.global_step

        self.epoch = self.global_step // steps_per_epoch
        self.chunk_index = (self.global_step % steps_per_epoch) // steps_per_chunk

        logger.info(
            f"Starting training from epoch {self.epoch+1}/{self.config.max_epochs}, "
            f"chunk {self.chunk_index+1}/{self.chunks_per_epoch} | "
            f"Global step: {self.global_step:,}"
        )

        # Initialize scheduler step - needed if resuming from checkpoint
        self.scheduler_step = self.epoch * self.chunks_per_epoch + self.chunk_index
        self.scheduler.step(self.scheduler_step)

        while self.epoch < self.config.max_epochs:
            if is_ddp():
                if hasattr(self.val_loader, "sampler"):
                    self.val_loader.sampler.set_epoch(self.epoch)

            early_stop = self._train_epoch()
            if early_stop:
                break
            self.epoch += 1

        logger.info(f"Finished training after {self.epoch + 1} epochs")

    def print_nan_parameters(self):
        error_list = ""
        error_list += "=== NaN sweep over model parameters ===\n"
        found = False
        for name, p in self.model.named_parameters():
            if torch.isnan(p).any():
                found = True
                error_list += f"NaNs in parameter: {name}\n"
        if not found:
            error_list += "No NaNs found in parameters.\n"
        return error_list

    def run(self):
        """
        Run complete training pipeline:

        1. Setup and preprocess data
        2. Build the model
        3. Setup optimizer and scheduler
        4. Run the training loop
        5. Run evaluation on validation and test sets
        """
        try:
            # Training pipeline
            logger.info("Starting training pipeline")
            logger.info("Setting up data loaders")
            self.setup_loaders()
            logger.info("Preparing metadata")
            self.prepare_metadata(self.checkpoint)
            logger.info("Building model")
            self.build_model(self.checkpoint)
            logger.info("Setting up optimizer")
            self.setup_optimizer(self.checkpoint)
            logger.info("Starting training")
            self.train(self.checkpoint)

            # Final evaluation (after training)
            ## Run and log the evaluation on train, val and test data
            logger.info("Final model evaluation:")
            model = ModelBuilder.build_from_checkpoint(self.best_checkpoint_filename)
            model = model.to(self.device)

            if is_ddp():
                self.model = DDP(module=model, device_ids=[self.local_rank])
            else:
                self.model = model

            ## On train data
            train_sampler = (
                DistributedSampler(self.train_data, shuffle=False) if is_ddp() else None
            )
            train_loader = make_loader(
                dataset=self.train_data,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=train_sampler,
            )
            (
                (train_recon_loss,
                train_acc,
                train_kl_loss,
                train_size_loss,
                train_acc_with_sizes,
                train_edit_distance,
                train_edit_distance_acc),
                train_accuracy_by_size,
                train_edit_distance_by_size
            ) = self.evaluator.forward(
                model=self.model,
                loader=train_loader,
            )
            if self.is_main_process:
                logger.info(
                    f"\tTrain Reconstruction Loss: {train_recon_loss:.6f} | KL Loss: {train_kl_loss:.6f} | Accuracy: {train_acc:.6f} | Edit Distance: {train_edit_distance:.6f}"
                )

            ## On val data
            (
                val_recon_loss,
                val_acc,
                val_kl_loss,
                val_size_loss,
                val_acc_with_sizes,
                val_edit_distance,
                val_edit_distance_acc,
            ), val_accuracy_by_size, val_edit_distance_by_size = (
                self.evaluator.forward(
                    model=self.model,
                    loader=self.val_loader,
                )
            )
            if self.is_main_process:
                logger.info(
                    f"\tValid Reconstruction Loss: {val_recon_loss:.6f} | KL Loss: {val_kl_loss:.6f} | Accuracy: {val_acc:.6f} | Edit Distance: {val_edit_distance:.6f}"
                )

            ## On test data
            (
                test_recon_loss,
                test_acc,
                test_kl_loss,
                test_size_loss,
                test_acc_with_sizes,
                test_edit_distance,
                test_edit_distance_acc,
            ), test_accuracy_by_size, test_edit_distance_by_size = self.evaluator.forward(
                model=self.model,
                loader=self.test_loader,
            )
            if self.is_main_process:
                logger.info(
                    f"\tTest Reconstruction Loss: {test_recon_loss:.6f} | KL Loss: {test_kl_loss:.6f} | Accuracy: {test_acc:.6f} | Edit Distance: {test_edit_distance:.6f}"
                )

            # Log to tracker
            results = {
                "train_reconstruction_loss": train_recon_loss,
                "train_kl_loss": train_kl_loss,
                "train_accuracy": train_acc,
                "train_size_loss": train_size_loss,
                "train_accuracy_with_sizes": train_acc_with_sizes,
                "train_edit_distance": train_edit_distance,
                "train_edit_distance_accuracy": train_edit_distance_acc,
                "val_reconstruction_loss": val_recon_loss,
                "val_kl_loss": val_kl_loss,
                "val_accuracy": val_acc,
                "val_size_loss": val_size_loss,
                "val_accuracy_with_sizes": val_acc_with_sizes,
                "val_edit_distance": val_edit_distance,
                "val_edit_distance_accuracy": val_edit_distance_acc,
                "test_reconstruction_loss": test_recon_loss,
                "test_kl_loss": test_kl_loss,
                "test_accuracy": test_acc,
                "test_size_loss": test_size_loss,
                "test_accuracy_with_sizes": test_acc_with_sizes,
                "test_edit_distance": test_edit_distance,
                "test_edit_distance_accuracy": test_edit_distance_acc,
            }
            if self.is_main_process:
                self.tracker.log_summary(metrics=results)
        except Exception as e:
            logger.error(self.print_nan_parameters())
            results = None
            logger.error(f"[{self.rank}] Training failed with error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if self.is_main_process:
                logger.complete()
                logger.remove(self.log_id)

            if self.tracker is not None:
                self.tracker.finish()

            if is_ddp():
                dist.destroy_process_group()
        return results