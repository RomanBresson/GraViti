# %%
import os
import argparse

from src.train import Trainer
from src.utils.training_config import TrainingConfig
from src.data.constants import AVAILABLE_DATASETS


# %%
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="VGAE")

    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="QM9NoHydro",
        choices=AVAILABLE_DATASETS,
        help="Training dataset name",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory where dataset will be stored",
    )
    parser.add_argument(
        "--models_root",
        type=str,
        default="saved_models",
        help="Root directory where models will be saved",
    )
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="Force re-processing the dataset (needed when pre_transform is changed)",
    )
    parser.add_argument(
        "--splits",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=[0.8, 0.1, 0.1],
        help="Dataset split: train val test. Must sum to 1.0. Example: --splits 0.8 0.1 0.1",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Input batch size for training",
    )
    parser.add_argument(
        "--with_aromatic",
        type=int,
        default=1,
        help="Keep aromatic bonds as edge class (if 0 Kekulize bonds - replace with alternating single and double bonds)",
    )

    # Model architecture arguments
    parser.add_argument(
        "--latent_size",
        type=int,
        default=256,
        help="Latent space size",
    )
    parser.add_argument(
        "--encoder_hidden_size",
        type=int,
        default=256,
        help="Governs the hidden size of the transformers",
    )
    parser.add_argument(
        "--encoder_output_size",
        type=int,
        default=128,
        help="Governs the output size of the transformers",
    )
    parser.add_argument(
        "--encoder_heads",
        type=int,
        default=4,
        help="Attention heads of the transformers",
    )
    parser.add_argument(
        "--encoder_layers",
        type=int,
        default=3,
        help="Number of transformer layers for encoder",
    )
    parser.add_argument(
        "--decoder_layers",
        type=int,
        default=3,
        help="Number of transformer layers for decoder",
    )
    parser.add_argument(
        "--decoder_sigma",
        type=float,
        default=0.01,
        help="Noise for the decoder's sequence transformer",
    )
    parser.add_argument(
        "--reg_weight",
        type=float,
        default=0.0,
        help="Regression weight (0 = no regressor)",
    )
    parser.add_argument(
        "--variational",
        type=int,
        default=1,
        help="1 if VAE, 0 if AE",
    )
    parser.add_argument(
        "--use_atom_attr",
        type=int,
        default=1,
        help="Use atoms properties",
    )

    # Training arguments
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (maximum)",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.1,
        help="Max learning rate decay factor by cycle",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-8,
        help="Weight decay for regularization",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=1e-4,
        help="Maximal value of beta",
    )
    parser.add_argument(
        "--beta_cycle",
        type=int,
        required=False,
        help="Number of scheduler cycles for beta annealing. Defaults to 2 LR cycles (see `--cycle_length`)",
    )
    parser.add_argument(
        "--cycle_length",
        type=int,
        default=50,
        help="Period of LR-scheduler cycles (in sub-epochs/chunks)",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout for transformers",
    )
    parser.add_argument(
        "--ratio_negative_edges",
        type=float,
        default=4.0,
        help="Ratio of negative edges",
    )
    parser.add_argument(
        "--use_focal",
        type=int,
        default=0,
        help="Use Focal Loss (0=Cross-Entropy, 1=Focal)",
    )
    parser.add_argument(
        "--weight_edge_loss",
        type=float,
        default=1.0,
        help="Weight of the reconstruction loss for edge",
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        required=False,
        help="Maximum number of epochs (full passes over the dataset). Defaults to 10 cycles (see `--cycle_length`)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        required=False,
        help="Number of warmup epochs (sub-epochs). Defaults to 1/50 of a cycle (see `--cycle_length`)",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        required=False,
        help="Number of sub-epochs (chunks) to wait without improvement before early stopping. Defaults to 3 cycles (see `--cycle_length`)",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        required=False,
        help="Sub-epoch chunk size, for updating schedulers within a large epoch.",
    )

    # Experiment tracking and logging arguments
    parser.add_argument(
        "--log_interval",
        type=int,
        default=5,
        help="Interval for evaluation and logging (default=5 epochs)",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="graph-vae",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity name",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name for tracking",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="+",
        default=[],
        help="Tags for the experiment",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Group for current experiments",
    )
    parser.add_argument(
        "--wandb_notes",
        type=str,
        default=None,
        help="Notes for the experiment",
    )
    parser.add_argument(
        "--track_gradients",
        action="store_true",
        help="Track gradients in wandb",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    # Extra arguments
    parser.add_argument(
        "--exists_ok",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Behavior if a model with the given configuration already exists: 0=Raise Error, 1=Overwrite, 2=Continue Training",
    )

    parser.add_argument(
        "--use_classes_weights",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use weights for classes in loss computation, 1=Use weights, 0=Don't use weights",
    )

    parser.add_argument(
        "--predict_hydrogens_formal_charges",
        type=int,
        default=1,
        choices=[0, 1],
        help="Predict hydrogen counts and formal charges, 1=Predict, 0=Don't predict",
    )

    parser.add_argument(
        "--reinject_size",
        type=int,
        default=1,
        choices=[0, 1],
        help="Reinject graph size in decoder, 1=Reinject, 0=Don't reinject",
    )

    parser.add_argument(
        "--use_grale_loss",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use GRALE loss for edges, 1=Use GRALE, 0=Don't use GRALE",
    )

    return parser.parse_args()


# %%
if __name__ == "__main__":
    args = parse_args()

    # Convert args to dict and handle boolean conversion
    kwargs = vars(args)
    kwargs["with_aromatic"] = bool(kwargs.get("with_aromatic"))
    kwargs["variational"] = bool(kwargs.get("variational"))
    kwargs["use_atom_attr"] = bool(kwargs.get("use_atom_attr"))
    kwargs["use_focal"] = bool(kwargs.get("use_focal"))
    kwargs["use_classes_weights"] = bool(kwargs.get("use_classes_weights"))
    kwargs["predict_hydrogens_formal_charges"] = bool(kwargs.get("predict_hydrogens_formal_charges"))
    kwargs["reinject_size"] = bool(kwargs.get("reinject_size"))
    kwargs["use_grale_loss"] = bool(kwargs.get("use_grale_loss"))

    # Handle wandb disable
    if kwargs.pop("disable_wandb", False):
        os.environ["WANDB_MODE"] = "online"

    # Get dataset loading params
    data_root = kwargs.pop("data_root", "data")
    models_root = kwargs.pop("models_root", "saved_models")
    force_reload = kwargs.pop("force_reload", False)
    splits = kwargs.pop("splits", [0.8, 0.1, 0.1])

    # Start Training
    config = TrainingConfig(**kwargs)
    trainer = Trainer(config, data_root, splits, force_reload, models_root=models_root)
    trainer.run()
