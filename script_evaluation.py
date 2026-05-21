#%%
import torch
from rdkit import RDLogger
import argparse
RDLogger.DisableLog('rdApp.error')
from datetime import datetime
import matplotlib.pyplot as plt

#%%
from src.model.builder import ModelBuilder
from src.model.checkpoint import ModelCheckpoint
from src.data.treat_data import load_data, make_loader
from src.utils.evaluations import (
    run_generation_trials,
    get_smiles_loader,
    get_first_embeddings,
    interpolation_wrapper,
    logp_optim_wrapper_cma,
)
from src.model.objectives_class import LossStack, Evaluator
from src.utils.coloring import coloring_graph_to_string
from src.utils.bayesian_net import bayesian_net_graph_to_string
import os
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_or_create_smiles_txt(cache_path, dataset, train_ds, train_loader):
    """
    Load SMILES from a .txt file if it exists.
    Otherwise compute them, save them, and return the list.
    """

    # 1. Load if file exists
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    # 2. Otherwise compute
    if dataset.startswith("Coloring"):
        smiles_list = [
            coloring_graph_to_string(graph.x, graph.node_mask, dataset)
            for graph in train_ds
        ]
    elif dataset.startswith("BayesianNet"):
        smiles_list = [
            bayesian_net_graph_to_string(
                graph.x,
                graph.edge_attr_adj,
                graph.node_mask,
                dataset,
            )
            for graph in train_ds
        ]
    else:
        smiles_list = get_smiles_loader(train_loader)

    # 3. Save to txt
    with open(cache_path, "w") as f:
        for smi in smiles_list:
            f.write(smi + "\n")
    return smiles_list


def summarize_metrics(metrics):
    summary = {}
    for key in ["validity", "uniqueness", "novelty", "count"]:
        values = np.array([m[key] for m in metrics])
        summary[key] = {
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
        }
    return summary


def write_metric_block(f, title, summary):
    f.write(f"\n--- {title} ---\n")
    for k in ["validity", "uniqueness", "novelty", "count"]:
        mean = summary[k]["mean"]
        std = summary[k]["std"]
        f.write(f"{k.capitalize():<12}: {mean:.4f} ± {std:.4f}\n")

def main():
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Evaluate trained molecular model.")
    parser.add_argument("--model_name", type=str, required=True, help="Model checkpoint name inside saved_models/")
    parser.add_argument("--models_root", type=str, required=True, help="Root directory for models")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for data")
    args = parser.parse_args()

    model_name = args.model_name
    with_aromatic = "arom1" in model_name
    dataset = model_name.split('_')[1]
    model_path = f"{args.models_root}/{model_name}"

    test_generation = model_name.startswith('vae') and (float(model_name.split("_b")[-1].split("_")[0]) > 0) and (dataset != "PubChem32")

    print(f"Evaluating model '{model_name}' on {dataset}")
    # %%
    # load data
    splits = [0.8, 0.1, 0.1]
    max_train_size = 9
    if dataset == 'PubChem16':
        splits = [0.9736, 0.003, 0.0234]
        max_train_size = 16
    elif dataset == 'PubChem32':
        splits = [0.9978, 0.0011, 0.0011]
        max_train_size = 32
    train_ds, val_ds, test_ds = load_data(dataset, splits=splits, data_root=args.data_root, with_aromatic=with_aromatic)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # %%
    # load model
    checkpoint = ModelCheckpoint.load(model_path)
    model = ModelBuilder.build_from_checkpoint(model_path)
    model = model.to(device)

    # %%
    # Evaluate
    loss_stack = LossStack(
        node_class_weights=checkpoint.metadata_state_dict['node_class_weights'].to(device),
        edge_class_weights=checkpoint.metadata_state_dict['edge_class_weights'].to(device),
        use_focal=checkpoint.config_state_dict['use_focal'],
        ratio_negative_edges=checkpoint.config_state_dict['ratio_negative_edges'],
        use_grale_loss=checkpoint.config_state_dict['use_grale_loss']
    )
    evaluator = Evaluator(loss_stack=loss_stack, check_with_permutations=False)
    test_loader = make_loader(test_ds, batch_size=64, shuffle=False)
    val_loader = make_loader(val_ds, batch_size=64, shuffle=False)
    train_loader = make_loader(train_ds, batch_size=64, shuffle=False)

    print(f"# batches: {len(train_loader)}")
    #%%
    #print("train")
    #(_, reconstruction_accuracy, _, _, reconstruction_accuracy_with_size, _, _), reconstruction_by_size, edit_distance_by_size = evaluator.forward(
    #    model=model, loader=train_loader
    #)
    #reconstruction_by_size = {size: acc.item() for size, acc in enumerate(reconstruction_by_size) if size > 0}
    #edit_distance_by_size = {size: dist.item() for size, dist in enumerate(edit_distance_by_size) if size > 0}
    #print(f"Reconstruction accuracy: {reconstruction_accuracy}")
    #print(f"Reconstruction accuracy with size: {reconstruction_accuracy_with_size}")
    #print(f"Edit distance: {edit_distance}")
    #print(f"Edit distance accuracy (<1e-5): {edit_distance_acc}")
    #print(f"Reconstruction accuracy by size: {reconstruction_by_size}")
    #print(f"Edit distance by size: {edit_distance_by_size}")
    first_embeddings = get_first_embeddings(num_batches=20, loader=test_loader, model=model)
    interpolation_wrapper(first_embeddings, model, model_name, dataset, num_pairs=20, steps=100, interp_mode='lin')
    interpolation_wrapper(first_embeddings, model, model_name, dataset, num_pairs=20, steps=100, interp_mode='rot')
    if not dataset.startswith("Coloring"):
        logp_optim_wrapper_cma(first_embeddings, model, model_name, dataset, with_aromatic=with_aromatic)

    print("val")
    (_, reconstruction_accuracy, _, _, reconstruction_accuracy_with_size, edit_distance, edit_distance_acc), reconstruction_by_size, edit_distance_by_size = evaluator.forward(
        model=model, loader=val_loader
    )
    reconstruction_by_size = {size: acc.item() for size, acc in enumerate(reconstruction_by_size) if size > 0}
    edit_distance_by_size = {size: dist.item() for size, dist in enumerate(edit_distance_by_size) if size > 0}
    print(f"Reconstruction accuracy: {reconstruction_accuracy}")
    print(f"Reconstruction accuracy with size: {reconstruction_accuracy_with_size}")
    print(f"Edit distance: {edit_distance}")
    print(f"Edit distance accuracy (<1e-5): {edit_distance_acc}")
    print(f"Reconstruction accuracy by size: {reconstruction_by_size}")
    print(f"Edit distance by size: {edit_distance_by_size}")

    print("test")
    (_, reconstruction_accuracy, _, _, reconstruction_accuracy_with_size, edit_distance, edit_distance_acc), reconstruction_by_size, edit_distance_by_size = evaluator.forward(
        model=model, loader=test_loader
    )
    reconstruction_by_size = {size: acc.item() for size, acc in enumerate(reconstruction_by_size) if size > 0}
    edit_distance_by_size = {size: dist.item() for size, dist in enumerate(edit_distance_by_size) if size > 0}
    print(f"Reconstruction accuracy: {reconstruction_accuracy}")
    print(f"Reconstruction accuracy with size: {reconstruction_accuracy_with_size}")
    print(f"Edit distance: {edit_distance}")
    print(f"Edit distance accuracy (<1e-5): {edit_distance_acc}")
    print(f"Reconstruction accuracy by size: {reconstruction_by_size}")
    print(f"Edit distance by size: {edit_distance_by_size}")

    #%%
    size_results = {}
    if test_generation:
        train_molecules = load_or_create_smiles_txt(os.path.join("data", f'{dataset}_{str(splits)}_smiles.txt'), dataset, train_ds, train_loader)
        print(f"\n=== Generation with unconstrained size ===")

        metrics_unconstrained = run_generation_trials(
            model=model,
            training_smiles=train_molecules,
            dataset=dataset,
            num_graphs=1024,
            num_trials=10,
        )
        summary_unconstrained = summarize_metrics(metrics_unconstrained)

        print("=== Generation (10 trials, unconstrained) ===")
        for k, v in summary_unconstrained.items():
            print(f"{k:10s}: {v['mean']:.4f} ± {v['std']:.4f}")

        # TODO: Check `size` constraint
        size_results = {}

        for size in range(1,max_train_size*2):
            print(f"\n=== Generation with forced size = {size} ===")

            metrics = run_generation_trials(
                model=model,
                training_smiles=train_molecules,
                dataset=dataset,
                num_graphs=1024,
                num_trials=10,
                forced_size=size
            )
            size_results[size] = summarize_metrics(metrics)
    print('Generation ran OK')

    # %%
    #embeddings, sizes, PLogPs, MWs = get_all_embeddings(model, test_loader)

    # ---------------------------------------------------------
    # If dataset is PubChem16, also evaluate on PubChem32
    # ---------------------------------------------------------
    extra_results = None

    if dataset=="PubChem16":
        print("\n=== Additional Evaluation on PubChem32 ===")

        # Load PubChem32
        train32, val32, test32 = load_data("PubChem32", splits=[0.99, 0.005, 0.005])
        test_loader32 = make_loader(test32, batch_size=64, shuffle=False)

        # Evaluate reconstruction only
        (_, rec_acc32, _, _, rec_acc_size32, edit_distance32, edit_distance_acc32), rec_by_size32, edit_distance_by_size32 = evaluator.forward(
            model=model, loader=test_loader32
        )

        rec_by_size32 = {size: acc.item() for size, acc in enumerate(rec_by_size32) if size > 0}
        edit_distance_by_size32 = {size: dist.item() for size, dist in enumerate(edit_distance_by_size32) if size > 0}

        print(f"PubChem32 Reconstruction accuracy: {rec_acc32}")
        print(f"PubChem32 Reconstruction accuracy with size: {rec_acc_size32}")
        print(f"PubChem32 Edit distance: {edit_distance32}")
        print(f"PubChem32 Edit distance accuracy (<1e-5): {edit_distance_acc32}")
        print(f"PubChem32 Reconstruction accuracy by size: {rec_by_size32}")
        print(f"PubChem32 Edit distance by size: {edit_distance_by_size32}")
        # Store results for writing to file
        extra_results = {
            "rec_acc32": rec_acc32,
            "rec_acc_size32": rec_acc_size32,
            "edit_distance32": edit_distance32,
            "edit_distance_acc32": edit_distance_acc32,
            "rec_by_size32": rec_by_size32,
            "edit_distance_by_size32": edit_distance_by_size32,
        }

    # %%
    # Output directory for this model

    output_dir = os.path.join("results", model_name)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "results.txt")

    with open(output_file, "w") as f:
        f.write("=== Model Evaluation Report ===\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Model: {model_name}\n\n")

        f.write("--- Reconstruction (Primary Dataset) ---\n")
        f.write(f"Reconstruction accuracy: {reconstruction_accuracy:.4f}\n")
        f.write(f"Reconstruction accuracy (with size): {reconstruction_accuracy_with_size:.4f}\n")
        f.write(f"Edit distance: {edit_distance:.4f}\n")
        f.write(f"Edit distance accuracy (<1e-5): {edit_distance_acc:.4f}\n")
        f.write("Reconstruction accuracy by size:\n")
        for size, acc in reconstruction_by_size.items():
            f.write(f"  Size {size}: {acc:.4f}\n")
        f.write("Edit distance by size:\n")
        for size, dist in edit_distance_by_size.items():
            f.write(f"  Size {size}: {dist:.4f}\n")

        if test_generation:
            f.write("\n=== Generation (10 independent trials) ===\n")
            f.write("Protocol: 10 trials with different random seeds, "
                    "1024 molecules per trial.\n")

            write_metric_block(
                f,
                "Unconstrained generation",
                summary_unconstrained
            )
            f.write("\n--- Per-size generation (mean ± std over 10 trials) ---\n")

            for size in sorted(size_results.keys()):
                s = size_results[size]
                f.write(
                    f"Size {size:>2}: "
                    f"Validity={s['validity']['mean']:.4f} ± {s['validity']['std']:.4f}, "
                    f"Uniqueness={s['uniqueness']['mean']:.4f} ± {s['uniqueness']['std']:.4f}, "
                    f"Novelty={s['novelty']['mean']:.4f} ± {s['novelty']['std']:.4f}, "
                    f"Count={s['count']['mean']:.1f}\n"
                )
    # Add PubChem32 results if available

        if extra_results is not None:
            f.write("\n\n=== Additional Evaluation on PubChem32 ===\n")

            f.write("\n--- Reconstruction (PubChem32) ---\n")
            f.write(f"Reconstruction accuracy: {extra_results['rec_acc32']:.4f}\n")
            f.write(f"Reconstruction accuracy (with size): {extra_results['rec_acc_size32']:.4f}\n")
            f.write(f"Edit distance: {extra_results['edit_distance32']:.4f}\n")
            f.write(f"Edit distance accuracy (<1e-5): {extra_results['edit_distance_acc32']:.4f}\n")
            f.write("Reconstruction accuracy by size:\n")
            for size, acc in extra_results["rec_by_size32"].items():
                f.write(f"  Size {size}: {acc:.4f}\n")
            f.write("Edit distance by size:\n")
            for size, dist in extra_results["edit_distance_by_size32"].items():
                f.write(f"  Size {size}: {dist:.4f}\n")


    print(f"Results written to {output_file}")
    # Save embeddings figure
    #if PLogPs is not None and MWs is not None:
    #    colors = [sizes.detach().cpu(), PLogPs.detach().cpu(), MWs.detach().cpu()]
    #    color_labels = ['sizes', 'PLogP', 'MW']
    #else:
    #    colors = [sizes.detach().cpu()]
    #    color_labels = ['sizes']
    #figs = plot_embeddings(embeddings.detach().cpu(), colors, color_labels)
    #for fig, col in zip(figs, color_labels):
    #    fig_path = os.path.join(output_dir, col + "_embeddings.png")
    #    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    #    plt.close(fig)

    #print(f"Embedding figure saved to {fig_path}")


if __name__ == "__main__":
    main()
