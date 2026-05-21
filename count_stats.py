import argparse
from pathlib import Path
import json

from src.data.stats import process_dataset_stats
from src.data.constants import AVAILABLE_DATASETS

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Dataset Statistics")
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        choices=AVAILABLE_DATASETS,
        help="Dataset name",
    )
    parser.add_argument(
        "--with_aromatic",
        action="store_true",
        default=False,
        help="Use aromatic bonds (no kekulization)",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=-1,
        help="Maximum number of atoms per molecule",
    )
    parser.add_argument(
        "--force_reload",
        action="store_true",
        default=False,
        help="Force re-processing of the dataset",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory where dataset will be stored",
    )
    args = vars(parser.parse_args())
    dataset = args["dataset"]
    with_aromatic = args["with_aromatic"]
    max_size = args["max_size"]
    force_reload = args["force_reload"]
    data_root = args["data_root"]

    # Stats path
    stats_dir = Path("data") / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Calculate and log stats
    print(f"Computing stats for {dataset}")
    print(f"\twith_aromatic={with_aromatic}, max_size={max_size}")
    stats = process_dataset_stats(
        name=dataset,
        with_aromatic=with_aromatic,
        max_size=max_size,
        force_reload=force_reload,
        data_root=data_root,
    )

    print(f"{dataset} (max_size={max_size}) Stats")
    print(
        f"Atom count: {stats['mu_atom_total']:.3f} ± {stats['std_atom_total']:.3f} atoms/molecule"
    )
    print(
        f"Bond count: {stats['mu_bond_total']:.3f} ± {stats['std_bond_total']:.2f} bonds/molecule"
    )

    print(
        "Mean number of atom count per graph for each node class:\n",
        stats["mu_atom_count"],
    )
    print(
        "Standard deviation of atom count per graph for each node class:\n",
        stats["std_atom_count"],
    )

    print(
        "Mean number of bond count per graph for each edge type:\n",
        stats["mu_bond_count"],
    )
    print(
        "Standard deviation of bond count per graph for each edge type:\n",
        stats["std_bond_count"],
    )

    if stats.get("mu_mw") is not None:
        print(f"Molecular weight: {stats['mu_mw']:.3f} ± {stats['std_mw']:.3f}")
    if stats.get("mu_plogp") is not None:
        print(f"Penalized logP: {stats['mu_plogp']:.3f} ± {stats['std_plogp']:.3f}")

    if stats.get("mu_atom_attr") is not None:
        print(
            "Mean value for each atom attribute:\n",
            stats["mu_atom_attr"],
        )
        print(
            "Standard deviation of each atom attribute:\n",
            stats["std_atom_attr"],
        )

    if stats.get("max_hydrogens") is not None:
        print(f"Maximum hydrogens per atom: {stats['max_hydrogens']}")
    if stats.get("min_formal_charge") is not None:
        print(
            f"Formal Charge range: [{stats['min_formal_charge']}, {stats['max_formal_charge']}]"
        )

    # Save to JSON
    filename = f"stats_{dataset}"
    if not dataset.startswith("Coloring"):
        filename += f"_{'aromatic' if with_aromatic else 'kekule'}"
    filepath = stats_dir / f"{filename}.json"
    with open(filepath, "w") as f:
        json.dump(stats, f, indent=4)
