import torch
from torch.utils.data import random_split, Sampler

from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9, ZINC

from typing import Optional, Tuple, List, Dict, Iterable, Any

from src.data.constants import DatasetName
from src.data.pubchem import PubChem
from src.data.coloring import Coloring, COLORING_CONFIGS
from src.data.bayesian_net import BayesianNetwork
from src.data.processing import (
    PreTransformQM9,
    PreTransformZINC,
    PreTransformPubChem,
    TransformGraph,
)
from src.data.utils import filter_dataset_with_size, get_cpu_count


def load_data(
    dataset: DatasetName,
    splits: Optional[List[float]] = [0.8, 0.1, 0.1],
    force_reload: Optional[bool] = False,
    max_size: int = -1,
    with_aromatic: bool = True,
    data_root: str = "data",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the train, validation and test splits of a supported dataset,
    with corresponding preprocessing steps.

    Args:
        dataset (str):
            Any available dataset between the defined dataset names (`data.constants.DatasetName`)
        splits (list[float], optional, default=[0.8, 0.1, 0.1]):
            train/validation/test split ratios - only applicable to `QM9`,
            `PubChem`, and `BayesianNet` datasets (`ZINC` and `Coloring`
            datasets are pre-splitted)
        force_reload (bool, optional, default=False):
            Whether to re-process the dataset.
        max_size (int, default=-1):
            Max size of graphs to be filtered (-1 means use the default per dataset)
            (QM9WithHydro: 28, QM9NoHydro: 9, ZINC: 38, PubChem16: 16,
             PubChem32: 32, PubChem64: 64, ColoringSmall: 10, ColoringMedium: 15, ColoringBig: 32)
        with_aromatic (bool, default=True):
            Optionally keep aromatic bonds. If `False`, drop aromatic bonds with
            Kekulization (replace with alternating single-double bonds).
        data_root (str, default="data"):
            Root directory where datasets are be stored

    Returns:
        tuple[torch.data.Dataset,torch.data.Dataset,torch.data.Dataset]:
            a tuple of (train, validation, test) `Dataset`s

    NOTES:
        * For QM9/PubChem/BayesianNet we split the whole dataset according to `splits` ratios.
        * For ZINC/Coloring we rely on built-in split handling.
    """

    def _load_QM9(name: str):
        with_hydro = name == "QM9WithHydro"

        effective_max_size = max_size
        if max_size <= 0:
            effective_max_size = 29 if with_hydro else 9

        data = QM9(
            f"{data_root}/{name}",
            pre_transform=PreTransformQM9(with_hydro, with_aromatic),
            pre_filter=filter_dataset_with_size(name, effective_max_size),
            transform=TransformGraph(name, with_aromatic, effective_max_size),
            force_reload=force_reload,
        )
        generator = torch.Generator().manual_seed(0)
        return random_split(data, splits, generator=generator)

    def _load_ZINC_split(name: str, split: str):
        subset = name == "ZINC12k"

        effective_max_size = max_size
        if max_size <= 0:
            effective_max_size = 38

        return ZINC(
            f"{data_root}/ZINC",
            subset=subset,
            split=split,
            pre_transform=PreTransformZINC(with_aromatic, subset),
            pre_filter=filter_dataset_with_size(name, effective_max_size),
            transform=TransformGraph(name, with_aromatic, effective_max_size),
            force_reload=force_reload and (split == "train"),
        )

    def _load_ZINC(name: str):
        return tuple(
            [_load_ZINC_split(name, split) for split in ["train", "val", "test"]]
        )

    def _load_PubChem(name: str):
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
        else:
            max_mols = int(1.2e8)
            if max_size <= 0:
                effective_max_size = 64
        variant = str(effective_max_size)

        # Subset is set if either "PubChem16S", "PubChem32S" or "PubChem64S" is selected
        subset = name.endswith("S")
        max_mols = None
        if subset:
            max_mols = int(5e5)
            variant += "_500k"
        variant += f"_{'aromatic' if with_aromatic else 'kekule'}"

        data = PubChem(
            f"{data_root}/PubChem",
            pre_transform=PreTransformPubChem(with_aromatic),
            pre_filter=filter_dataset_with_size(name, effective_max_size),
            transform=TransformGraph(name, with_aromatic, effective_max_size),
            force_reload=force_reload,
            subset=subset,
            max_mols=max_mols,
            variant=variant,
            num_workers=max(get_cpu_count() - 2, 1),
        )
        generator = torch.Generator().manual_seed(0)
        return random_split(data, splits, generator=generator)

    def _load_Coloring(name: str):
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

        effective_max_size = max_size
        if max_size <= 0:
            effective_max_size = COLORING_CONFIGS.get(variant).max_nodes

        base_path = f"{data_root}/Coloring"
        train_data = Coloring(
            base_path,
            variant=variant,
            split="train",
            transform=TransformGraph(base_name, max_size=effective_max_size),
            force_reload=force_reload,
            with_images=False,
            canonical_ordering=canonical_ordering,
            wl_iterations=wl_iterations,
            seed=42,
            num_workers=max(get_cpu_count() - 2, 1),
        )
        val_data = Coloring(
            base_path,
            variant=variant,
            split="val",
            transform=TransformGraph(base_name, max_size=effective_max_size),
            force_reload=False,
            with_images=False,
            canonical_ordering=canonical_ordering,
            wl_iterations=wl_iterations,
            seed=42,
            num_workers=max(get_cpu_count() - 2, 1),
        )
        test_data = Coloring(
            base_path,
            variant=variant,
            split="test",
            transform=TransformGraph(base_name, max_size=effective_max_size),
            force_reload=False,
            with_images=False,
            canonical_ordering=canonical_ordering,
            wl_iterations=wl_iterations,
            seed=42,
            num_workers=max(get_cpu_count() - 2, 1),
        )
        return train_data, val_data, test_data

    def _load_BayesianNet(name: str):
        base_path = f"{data_root}/BayesianNet"
        variant = name.split("BayesianNet")[-1].lower()

        effective_max_size = max_size
        if max_size <= 0:
            effective_max_size = 8

        data = BayesianNetwork(
            base_path,
            variant=variant,
            transform=TransformGraph(name, max_size=effective_max_size),
            force_reload=force_reload,
            directed=False,
        )
        generator = torch.Generator().manual_seed(0)
        return random_split(data, splits, generator=generator)

    # Prepare splits / datasets
    if dataset.startswith("QM9"):
        train_dataset, val_dataset, test_dataset = _load_QM9(name=dataset)
    elif dataset.startswith("ZINC"):
        train_dataset, val_dataset, test_dataset = _load_ZINC(name=dataset)
    elif dataset.startswith("PubChem"):
        train_dataset, val_dataset, test_dataset = _load_PubChem(name=dataset)
    elif dataset.startswith("Coloring"):
        train_dataset, val_dataset, test_dataset = _load_Coloring(name=dataset)
    elif dataset.startswith("BayesianNet"):
        train_dataset, val_dataset, test_dataset = _load_BayesianNet(name=dataset)
    else:
        raise ValueError(f"Unknown Dataset {dataset}")

    return train_dataset, val_dataset, test_dataset


def treat_batch(batch: Batch, device: torch.device | str) -> Dict[str, Any]:
    """
    Processes a single batch and move tensors to the specified device.

    Args:
        batch (torch_geometric.data.Batch): A PyTorch Geometric batch containing molecular graph data.
        device (torch.device | str): Target device.
    """
    batch = batch.to(device, non_blocking=True)

    num_nodes = batch.edge_attr_adj.size(1)
    max_num_nodes = int(batch.node_mask.sum(-1).max().item())
    node_mask = batch.node_mask.reshape(-1, num_nodes)[:, :max_num_nodes]
    bs = node_mask.size(0)
    graph_sizes = node_mask.sum(-1)
    E = batch.classes_edges.reshape(bs, num_nodes, num_nodes)[
        :, :max_num_nodes, :max_num_nodes
    ]
    if E.max() == 4:
        aromatic_mask = (E == 3).any(-1)
    else:
        aromatic_mask = torch.zeros_like(node_mask, dtype=torch.bool)

    return {
        "X": batch.classes_nodes.reshape(bs, num_nodes)[:, :max_num_nodes],
        "E": E,
        "y": batch.global_features.reshape(bs, -1),
        "PE": {
            "pe_lap": batch.pe_lap.reshape(bs, num_nodes, -1)[:, :max_num_nodes],
            "pe_rw": batch.pe_rw.reshape(bs, num_nodes, -1)[:, :max_num_nodes],
        },
        "targets": (
            batch.y.reshape(bs, -1) if getattr(batch, "y", None) is not None else None
        ),
        "reg_target": (
            batch.reg_target.reshape(bs, -1)
            if getattr(batch, "reg_target", None) is not None
            else None
        ),
        "hydrogens": (
            batch.num_hydrogens.reshape(bs, num_nodes)[:, :max_num_nodes]
            if getattr(batch, "num_hydrogens", None) is not None
            else None
        ),
        "formal_charges": (
            batch.formal_charge.reshape(bs, num_nodes)[:, :max_num_nodes]
            if getattr(batch, "formal_charge", None) is not None
            else None
        ),
        "chem_properties": (
            torch.stack((batch.MW, batch.PLogP), dim=0).T
            if getattr(batch, "MW", None) is not None
            and getattr(batch, "PLogP", None) is not None
            else None
        ),
        "edge_index": batch.edge_index,
        "batch": batch.batch,
        "atom_attr": (
            batch.atom_attr.reshape(bs, num_nodes, -1)[:, :max_num_nodes]
            if getattr(batch, "atom_attr", None) is not None
            else None
        ),
        "node_mask": node_mask,
        "graph_sizes": graph_sizes,
        "aromatic_mask": aromatic_mask,
    }


def treat_data_all(
    loader: torch.utils.data.DataLoader, device: torch.device | str
) -> List[Dict[str, Any]]:
    """
    Process all batches from a DataLoader into a list of processed dictionaries.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader yielding PyTorch Geometric Batch objects.
        device (torch.device | str): Device to move tensors to ("cpu" or "cuda").

    Returns:
        list[dict[str, Any]]: List of dictionaries containing processed batch data
                              (same format as `treat_batch`).
    """
    data = []

    for batch in loader:
        data.append(treat_batch(batch, device))

    return data


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    workers: int = None,
    sampler: Sampler | Iterable = None,
) -> DataLoader:
    """
    Create a torch_geometric.loader.DataLoader for the given dataset and parameters,

    Args:
        dataset (Dataset): The dataset from which to load data
        batch_size (int): How many samples to load per batch
        shuffle (bool, default=False): If set to `True`, the data will be reshuffled at every epoch.
        workers (int, default=None): Forced number of workers to be used in DataLoader. If None (default),
            the number of workers will be determined based on the available workers on the system.
        sampler (Sampler|Iterable, optional): defines the strategy to draw samples from the dataset. Can
            be any ``Iterable`` with ``__len__`` implemented. If specified, :attr:`shuffle` must not be specified.

    Returns:
        torch_geometric.loader.DataLoader:
            DataLoader for the given dataset and parameters
    """
    available_workers = get_cpu_count()
    if workers is not None:
        loader_workers = min(available_workers, workers)
    else:
        loader_workers = min(available_workers // 2, 8)
        loader_workers = max(1, loader_workers)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=loader_workers,
        pin_memory=True,
        sampler=sampler,
    )

# Dirty functions to treat QM9 as PubChem

def make_stats_qm9_pc():
    from src.data.constants import get_dataset_stats
    ret = {}
    for data_name in ['QM9NoHydro', 'PubChem16']:
        stats = get_dataset_stats(data_name, with_aromatic=True)
        mu_atom_count = torch.tensor(stats["mu_atom_count"])
        std_atom_count = torch.tensor(stats["std_atom_count"])
        mu_bond_count = torch.tensor(stats["mu_bond_count"])
        std_bond_count = torch.tensor(stats["std_bond_count"])
        mu_atom_total = torch.tensor([stats["mu_atom_total"]])
        std_atom_total = torch.tensor([stats["std_atom_total"]])
        if stats.get("mu_atom_attr") is not None:
            attr_mu = torch.tensor(stats["mu_atom_attr"])
            attr_std = torch.tensor(stats["std_atom_attr"])
        else:
            attr_mu, attr_std = None, None
        ## Define `mu` and `std` for global features
        global_mus = torch.cat([mu_atom_count, mu_bond_count, mu_atom_total])
        global_stds = torch.cat([std_atom_count, std_bond_count, std_atom_total])
        ret[data_name] = (global_mus, global_stds, attr_mu, attr_std)
    return ret

def treat_qm9_as_pc(batch, device="cpu"):
    ret = make_stats_qm9_pc()
    bat = treat_batch(batch, device=device)
    new_y = torch.zeros((bat['y'].shape[0], 36), device=device)
    update_y = (bat['y'] * ret['QM9NoHydro'][1].unsqueeze(0).to(device)) + ret['QM9NoHydro'][0].unsqueeze(0).to(device)
    new_y[:, :4] = update_y[:,:4]
    new_y[:, -5:] = update_y[:,4:]
    new_y = (new_y - ret['PubChem16'][0].unsqueeze(0).to(device)) / ret['PubChem16'][1].unsqueeze(0).to(device)
    bat['y'] = new_y
    bat['atom_attr'] = (bat['atom_attr'] * ret['QM9NoHydro'][3].unsqueeze(0).to(device)) + ret['QM9NoHydro'][2].unsqueeze(0).to(device)
    bat['atom_attr'] = (bat['atom_attr'] - ret['PubChem16'][2].unsqueeze(0).to(device)) / ret['PubChem16'][3].unsqueeze(0).to(device)
    bat['reg_target'] = batch.reg_target.to(device)
    return bat