"""
Statistics for datasets. Minimal processing pre-tranforms for each dataset, that
can be used in order to calculate basic stats for each dataset with `calculate_stats`.

- mu_atom_count: Mean number of atom count per graph for each node class
- std_atom_count: Standard deviation of atom count per graph for each node class
- mu_bond_count: Mean number of bond count per graph for each edge type
- std_bond_count: Standard deviation of bond count per graph for each edge type

- mu_atom_total: Mean number of total atoms per graph
- std_atom_total: Standard deviation of total atoms per graph
- mu_bond_total: Mean number of total bonds per graph
- std_bond_total: Standard deviation of total bonds per graph

- mu_mw: Mean molecular weight per graph
- std_mw: Standard deviation of molecular weight per graph
- mu_plogp: Mean penalized logP per graph
- std_plogp: Standard deviation of penalized logP per graph

- mu_atom_attr: Mean atom attributes (per atom, per attribute dimension)
- std_atom_attr: Standard deviation of atom attributes
"""

import os
from joblib import Parallel, delayed
from typing import List, Dict, Tuple

import torch

from torch_geometric.data import Dataset
from torch_geometric.datasets import QM9, ZINC

import numpy as np
from tqdm import tqdm

from .pubchem import PubChem
from .coloring import Coloring
from .bayesian_net import BayesianNetwork
from .processing import PreTransformQM9, PreTransformZINC, PreTransformPubChem
from .constants import DatasetName, get_valid_atoms, get_valid_bonds, get_dataset_stats
from .utils import filter_dataset_with_size, get_cpu_count


def calculate_stats(
    dataset: Dataset,
    name: DatasetName,
    with_aromatic: bool = True,
    num_workers: int = max(os.cpu_count() - 2, 1),
) -> Dict[str, List[float] | float]:
    """
    Calculate dataset statistics by streaming the dataset
    in parallel, using stable Welford accumulation.

    Args:
        dataset (torch_geometric.data.Dataset):
            Torch Geometric Dataset whose stats will be calculated.
        name (DatasetName):
            Any available dataset between the defined dataset names (`data.constants.DatasetName`)
        with_aromatic (bool, default=True):
            Whether the given Dataset contains aromatic bonds.
        num_workers (int, optional):
            Number of workers used for parallel processing of the dataset
            (default :obj:`os.cpu_count()-2`)

    Returns:
        Dict[str,List[float]|float]:
            The calculated statistics counts for `mu_atom_total`, `std_atom_total`,
            `mu_bond_total`, `std_bond_total`, `mu_atom_count`, `std_atom_count`,
            `mu_bond_count`, `std_bond_count`, `mu_mw`, `std_mw`, `mu_plogp`, `std_plogp`,
            `mu_atom_attr`, `std_atom_attr`.
    """
    total_mols = len(dataset)
    num_atom_classes = len(get_valid_atoms(name))
    num_bond_classes = len(get_valid_bonds(with_aromatic, name))
    attr_stats = get_dataset_stats(name, False).get("mu_atom_attr")
    num_atom_attr = len(attr_stats) if attr_stats is not None else 0

    # Initialize global accumulators
    count, attr_count = 0, 0

    mu_atom_total, m2_atom_total = 0.0, 0.0
    mu_bond_total, m2_bond_total = 0.0, 0.0

    mu_atom_count = np.zeros(num_atom_classes, dtype=np.float64)
    m2_atom_count = np.zeros(num_atom_classes, dtype=np.float64)

    mu_bond_count = np.zeros(num_bond_classes, dtype=np.float64)
    m2_bond_count = np.zeros(num_bond_classes, dtype=np.float64)

    mu_mw, m2_mw = 0.0, 0.0
    mu_plogp, m2_plogp = 0.0, 0.0

    if num_atom_attr > 0:
        mu_atom_attr = np.zeros(num_atom_attr, np.float64)
        m2_atom_attr = np.zeros(num_atom_attr, np.float64)

    max_hs = 0
    min_fc, max_fc = 0, 0

    # Chunk the dataset
    chunk_size = 1000
    chunks = [
        list(range(i, min(i + chunk_size, total_mols)))
        for i in range(0, total_mols, chunk_size)
    ]
    sc_size = max(1, num_workers * 4)

    # Process dataset
    for sc_start in tqdm(range(0, len(chunks), sc_size), desc="Processing chunks"):
        sc_end = min(sc_start + sc_size, len(chunks))
        current_chunks = chunks[sc_start:sc_end]

        # Process chunks in parallel
        batch_results = Parallel(n_jobs=num_workers)(
            delayed(_process_molecule_batch)(
                dataset, chunk, num_atom_classes, num_bond_classes
            )
            for chunk in current_chunks
        )

        # Update global Welford stats per molecule
        for batch_stats in batch_results:
            for i in range(len(batch_stats["total_atoms"])):
                # Update atom total statistics
                _, mu_atom_total, m2_atom_total = _update_welford(
                    count, mu_atom_total, m2_atom_total, batch_stats["total_atoms"][i]
                )

                # Update bond total statistics
                _, mu_bond_total, m2_bond_total = _update_welford(
                    count, mu_bond_total, m2_bond_total, batch_stats["total_bonds"][i]
                )

                # Update atom count statistics
                _, mu_atom_count, m2_atom_count = _update_welford(
                    count, mu_atom_count, m2_atom_count, batch_stats["atom_counts"][i]
                )

                # Update bond count statistics
                _, mu_bond_count, m2_bond_count = _update_welford(
                    count, mu_bond_count, m2_bond_count, batch_stats["bond_counts"][i]
                )

                # Update molecular weight statistics
                if batch_stats["mw"] is not None:
                    _, mu_mw, m2_mw = _update_welford(
                        count, mu_mw, m2_mw, batch_stats["mw"][i]
                    )
                else:
                    mu_mw = None
                    m2_mw = None

                # Update penalized logP statistics
                if batch_stats["plogp"] is not None:
                    _, mu_plogp, m2_plogp = _update_welford(
                        count, mu_plogp, m2_plogp, batch_stats["plogp"][i]
                    )
                else:
                    mu_plogp = None
                    m2_plogp = None

                count += 1

                # Update per-atom attribute statistics
                if batch_stats["atom_attrs"] is not None:
                    atom_attrs = batch_stats["atom_attrs"][i]
                    if atom_attrs is not None:
                        for attr_vec in atom_attrs:
                            attr_count, mu_atom_attr, m2_atom_attr = _update_welford(
                                attr_count, mu_atom_attr, m2_atom_attr, attr_vec
                            )
                    else:
                        attr_count = 2
                        mu_atom_attr = None
                        m2_atom_attr = None
                else:
                    attr_count = 2
                    mu_atom_attr = None
                    m2_atom_attr = None

                # Update hydrogens statistics
                if batch_stats["max_hydrogens"] is not None:
                    max_hs = max(max_hs, batch_stats["max_hydrogens"])
                else:
                    max_hs = None

                # Update formal charge statistics
                if batch_stats["min_formal_charge"] is not None:
                    min_fc = min(min_fc, batch_stats["min_formal_charge"])
                else:
                    min_fc = None
                if batch_stats["max_formal_charge"] is not None:
                    max_fc = max(max_fc, batch_stats["max_formal_charge"])
                else:
                    max_fc = None

    # Compute final statistics
    try:
        mu_atom_total, _, var_atom_total = _finalize_welford(
            count, mu_atom_total, m2_atom_total
        )
        std_atom_total = var_atom_total**0.5

        mu_bond_total, _, var_bond_total = _finalize_welford(
            count, mu_bond_total, m2_bond_total
        )
        std_bond_total = var_bond_total**0.5

        mu_atom_count, _, var_atom_count = _finalize_welford(
            count, mu_atom_count, m2_atom_count
        )
        std_atom_count = var_atom_count**0.5

        mu_bond_count, _, var_bond_count = _finalize_welford(
            count, mu_bond_count, m2_bond_count
        )
        std_bond_count = var_bond_count**0.5

        if mu_mw is not None:
            mu_mw, _, var_mw = _finalize_welford(count, mu_mw, m2_mw)
            std_mw = var_mw**0.5
        std_mw = var_mw**0.5 if mu_mw is not None else None

        if mu_plogp is not None:
            mu_plogp, _, var_plogp = _finalize_welford(count, mu_plogp, m2_plogp)
        std_plogp = var_plogp**0.5 if mu_plogp is not None else None

        if mu_atom_attr is not None:
            mu_atom_attr, _, var_atom_attr = _finalize_welford(
                attr_count, mu_atom_attr, m2_atom_attr
            )
            std_atom_attr = var_atom_attr**0.5
        std_atom_attr = var_atom_attr**0.5 if mu_atom_attr is not None else None

    except ValueError as e:
        print(f"Exception: _finalize_welford failed: {e}")

        std_atom_total = 1.0
        std_bond_total = 1.0
        std_atom_count = np.ones_like(mu_atom_count)
        std_bond_count = np.ones_like(mu_bond_count)
        std_mw = 1.0
        std_plogp = 1.0
        std_atom_attr = np.ones_like(mu_atom_attr)

    return {
        "mu_atom_total": float(mu_atom_total),
        "std_atom_total": float(std_atom_total),
        "mu_bond_total": float(mu_bond_total),
        "std_bond_total": float(std_bond_total),
        "mu_atom_count": mu_atom_count.tolist(),
        "std_atom_count": std_atom_count.tolist(),
        "mu_bond_count": mu_bond_count.tolist(),
        "std_bond_count": std_bond_count.tolist(),
        "mu_mw": float(mu_mw) if mu_mw is not None else None,
        "std_mw": float(std_mw) if std_mw is not None else None,
        "mu_plogp": float(mu_plogp) if mu_plogp is not None else None,
        "std_plogp": float(std_plogp) if std_plogp is not None else None,
        "mu_atom_attr": mu_atom_attr.tolist() if mu_atom_attr is not None else None,
        "std_atom_attr": std_atom_attr.tolist() if std_atom_attr is not None else None,
        "max_hydrogens": max_hs if max_hs is not None else None,
        "min_formal_charge": min_fc if min_fc is not None else None,
        "max_formal_charge": max_fc if max_fc is not None else None,
    }


def process_dataset_stats(
    name: DatasetName,
    with_aromatic: bool = True,
    max_size: int = -1,
    force_reload: bool = False,
    data_root: str = "data",
) -> Dict[str, List[float] | float]:
    available_workers = max(get_cpu_count() - 1, 1)

    # QM9 dataset
    if name.startswith("QM9"):
        data_path = f"{data_root}/{name}"

        with_hydro = name == "QM9WithHydro"
        effective_max_size = max_size
        if max_size <= 0:
            effective_max_size = 29 if with_hydro else 9

        dataset = QM9(
            data_path,
            pre_transform=PreTransformQM9(with_hydro, with_aromatic),
            force_reload=force_reload,
            pre_filter=filter_dataset_with_size(name, effective_max_size),
        )

    # ZINC dataset
    elif name.startswith("ZINC"):
        data_path = f"{data_root}/ZINC"

        subset = name == "ZINC12k"
        effective_max_size = max_size
        if max_size <= 0:
            effective_max_size = 38

        dataset = ZINC(
            data_path,
            subset=subset,
            split="train",
            pre_transform=PreTransformZINC(with_aromatic, subset),
            pre_filter=filter_dataset_with_size(name, effective_max_size),
            force_reload=force_reload,
        )

    # PubChem dataset
    elif name.startswith("PubChem"):
        data_path = f"{data_root}/PubChem"

        effective_max_size = max_size
        # ~20M molecules of <=16 atoms in PubChem16
        if name in ["PubChem16S", "PubChem16"]:
            max_mols = int(2e7)
            if max_size <= 0:
                effective_max_size = 16

        # ~100M molecules of <=32 atoms in PubChem32
        elif name in ["PubChem32S", "PubChem32"]:
            max_mols = int(1e8)
            if max_size <= 0:
                effective_max_size = 32

        # ~120M molecules of <=64 atoms in PubChem64
        elif name in ["PubChem64S", "PubChem64"]:
            max_mols = int(1.2e8)
            if max_size <= 0:
                effective_max_size = 64
        variant = str(effective_max_size)

        # Subset is set if either "PubChem16S", "PubChem32S" or "PubChem64S"
        # is selected. In this case use maximum 500K molecules
        subset = name.endswith("S")
        if subset:
            max_mols = int(5e5)
            variant += "_500k"

        variant += f"_{'aromatic' if with_aromatic else 'kekule'}"

        dataset = PubChem(
            data_path,
            pre_transform=PreTransformPubChem(with_aromatic),
            pre_filter=filter_dataset_with_size(name, effective_max_size),
            force_reload=force_reload,
            subset=subset,
            max_mols=max_mols,
            variant=variant,
            num_workers=available_workers,
        )

    # Coloring dataset
    elif name.startswith("Coloring"):
        data_path = f"{data_root}/Coloring"

        # if name ends with 'Canonical' enforce canonical reordering
        is_canonical = name.endswith("Canonical")
        base_name = name
        if is_canonical:
            base_name = name[: -len("Canonical")]
        variant = base_name.split("Coloring")[-1].lower()

        # NOTE: when Canonical reordering is requested, always use
        # 1-WL for 3 iterations
        canonical_ordering = "wl" if is_canonical else None
        wl_iterations = 3

        dataset = Coloring(
            data_path,
            variant=variant,
            split="train",
            force_reload=force_reload,
            with_images=False,
            canonical_ordering=canonical_ordering,
            wl_iterations=wl_iterations,
            seed=42,
            num_workers=available_workers,
        )

    # Bayesian network dataset
    elif name.startswith("BayesianNet"):
        data_path = f"{data_root}/BayesianNet"
        variant = name.split("BayesianNet")[-1].lower()
        dataset = BayesianNetwork(
            data_path,
            variant=variant,
            force_reload=force_reload,
            directed=False,
        )

    # Error - Unknown dataset
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    return calculate_stats(dataset, name, with_aromatic, available_workers)


# Helpers for stats calculation


def _process_molecule_batch(
    dataset: Dataset,
    indices: List[int],
    num_atom_classes: int,
    num_bond_classes: int,
) -> Dict[str, List[float] | List[np.ndarray]]:
    """
    Extract raw counts for each molecule in a subset (batch)
    of given dataset.

    Args:
        dataset (torch_geometric.data.Dataset):
            Torch Geometric Dataset whose stats will be calculated.
        indices (List[int]):
            List of indices within the `dataset`, defining the subset
            to calculate stats for.
        num_atom_classes (int):
            Number of available atom classes in given dataset, necessary
            for counting atom counts per class.
        num_bond_classes (int):
            Number of available bond types in given dataset, necessary
            for counting bond counts per type.

    Returns:
        Dict[str,List[float]|List[np.ndarray]]:
            The calculated raw counts for `total_atoms`, `total_bonds`,
            `atom_counts`, `bond_counts`, `mw`, `plogp`, `atom_attrs`.
    """
    batch_stats = {
        "total_atoms": [],
        "total_bonds": [],
        "atom_counts": [],
        "bond_counts": [],
        "mw": [],
        "plogp": [],
        "atom_attrs": [],
        "max_hydrogens": 0,
        "min_formal_charge": 0,
        "max_formal_charge": 0,
    }

    for idx in indices:
        mol = dataset[idx]

        # Total counts per molecule
        batch_stats["total_atoms"].append(float(mol.num_nodes))
        batch_stats["total_bonds"].append(float(mol.num_edges))

        # Atom count per class
        atom_count = (
            torch.bincount(mol.x.view(-1).cpu(), minlength=num_atom_classes)
            .numpy()
            .astype(np.float64)
        )
        batch_stats["atom_counts"].append(atom_count)

        # Bond count per class. Most graph datasets store undirected edges in
        # so each bond appears twice due to symmetry. 
        # Assume that directed graphs also have a `directed_edge_index` attribute.
        edge_multiplier = 0.5
        if (
            hasattr(mol, "directed_edge_index")
            and mol.edge_index.size(1) == mol.num_edges
        ):
            edge_multiplier = 1.0
        bond_count = (
            torch.bincount(mol.edge_attr.cpu(), minlength=num_bond_classes)
            .numpy()
            .astype(np.float64)
            * edge_multiplier
        )
        batch_stats["bond_counts"].append(bond_count)

        # Molecular weight and penalized logP
        if hasattr(mol, "MW"):
            batch_stats["mw"].append(float(mol.MW))
        else:
            batch_stats["mw"] = None

        if hasattr(mol, "PLogP"):
            batch_stats["plogp"].append(float(mol.PLogP))
        else:
            batch_stats["plogp"] = None

        # Atom attributes: shape (num_nodes, num_attributes)
        if hasattr(mol, "atom_attr"):
            atom_attrs = mol.atom_attr.cpu().numpy().astype(np.float64)
            batch_stats["atom_attrs"].append(atom_attrs)
        else:
            batch_stats["atom_attrs"] = None

        # Minimum/maximum values tracking for hydrogens
        if hasattr(mol, "num_hydrogens"):
            max_hs = mol.num_hydrogens.max().item()
            batch_stats["max_hydrogens"] = max(batch_stats["max_hydrogens"], max_hs)
        else:
            batch_stats["max_hydrogens"] = None

        # Minimum/maximum values tracking for formal charge
        if hasattr(mol, "formal_charge"):
            min_fc = mol.formal_charge.min().item()
            max_fc = mol.formal_charge.max().item()
            batch_stats["min_formal_charge"] = min(
                batch_stats["min_formal_charge"], min_fc
            )
            batch_stats["max_formal_charge"] = max(
                batch_stats["max_formal_charge"], max_fc
            )
        else:
            batch_stats["min_formal_charge"] = None
            batch_stats["max_formal_charge"] = None

    return batch_stats


def _update_welford(
    count: int,
    mean: float | np.ndarray,
    m2: float | np.ndarray,
    new_value: float | np.ndarray,
) -> Tuple[int, float | np.ndarray, float | np.ndarray]:
    """
    Update step for
    [Welford's online algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm).

    Can work either with scalar or vector values.

    Args:
        count (int): Number of samples seen so far.
        mean (float|np.ndarray): Accumulated mean value of the dataset so far.
        m2 (float|np.ndarray): Aggregated squared distance from mean so far.
        new_value (float|np.ndarray): The new sample to update statistics for.
    Returns:
        Tuple[int,float|np.ndarray,float|np.ndarray]:
            The updated values for (count, mean, m2), after taking into
            account the provided `new_value`.
    """
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    m2 += delta * delta2
    return count, mean, m2


def _finalize_welford(
    count: int,
    mean: float | np.ndarray,
    m2: float | np.ndarray,
) -> Tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """
    Retrieve the mean, population variance and sample variance using
    [Welford's online algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm).

    Can work either with scalar or vector values.

    Args:
        count (int): Number of samples seen.
        mean (float|np.ndarray): Accumulated mean value of the dataset.
        m2 (float|np.ndarray): Aggregated squared distance from mean.
    Returns:
        Tuple[float|np.ndarray,float|np.ndarray,float|np.ndarray]:
            The final values for (mean, population_variance, sample_variance) from calculated
            aggregates.
    """
    if count < 2:
        raise ValueError("Count must be at least 2")

    population_variance = m2 / count
    sample_variance = m2 / (count - 1)

    return mean, population_variance, sample_variance
