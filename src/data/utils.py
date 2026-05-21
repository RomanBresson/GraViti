"""
Various utility functions for data processing and filtering.
"""

import os
from functools import partial
from typing import List, Callable, Optional, Any

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from rdkit import Chem

from .constants import DatasetName, get_valid_atoms, get_valid_bonds
from .features import build_edge_adj_attr
from ..utils.chem import make_molecule_from_xe, get_largest, is_valid_smiles

"""
Data normalization and pre-treatment.
"""


def normalize_molecule(
    dataset: DatasetName,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int,
    num_hydrogens: torch.Tensor = None,
    formal_charge: torch.Tensor = None,
    with_hydro: bool = False,
    with_aromatic: bool = True,
) -> Chem.Mol | None:
    """
    Construct and normalize an RDKit `Chem.Mol` object from graph-based molecular data.

    This function rebuilds an RDKit molecule from non-padded node and edge representations
    (as used in PyTorch Geometric molecular datasets), applies optional hydrogen
    addition/removal, and controls whether aromatic bonds are preserved or converted
    to explicit single/double bonds.

    The procedure includes:
    1. Building a `torch_geometric.data.Data` graph and computing the dense
       adjacency tensor (`edge_attr_adj`).
    2. Reconstructing the RDKit molecule using node and edge tensors.
    3. Selecting the largest connected component (to handle fragmented graphs).
    4. Adding or removing explicit hydrogens, based on `with_hydro` (implicit hydrogens are always removed).
    5. Preserving or kekulizing aromatic structures, based on `with_aromatic`.

    Args:
        dataset (DatasetName):
            Name of the source dataset - used to determine atom/bond type mappings.
        x (torch.Tensor):
            Node (atom) feature tensor.
        edge_index (torch.Tensor):
            Edge index tensor of shape `[2, num_edges]`, defining molecular bonds.
        edge_attr (torch.Tensor):
            Edge (bond) feature tensor of shape `[num_edges, num_bond_features]`.
        num_hydrogens (torch.Tensor, optional, default=None):
            Number of hydrogens per atom.
        formal_charge (torch.Tensor, optional, default=None):
            Formal charge per atom.
        with_hydro (bool, optional, default=False):
            If `True`, explicit hydrogens are retained or added (`Chem.AddHs`).
            If `False`, hydrogens are removed (`Chem.RemoveHs`).
        with_aromatic (bool, optional, default=True):
            If `True`, aromaticity perception is applied (`Chem.SanitizeMol`).
            If `False`, aromatic bonds are kekulized into alternating single/double bonds
            (`Chem.Kekulize`).

    Returns:
        rdkit.Chem.Mol|None:
            A normalized RDKit molecule object with hydrogens and aromaticity handled
            according to the specified flags. If the molecule cannot be reconstructed,
            `None` will be returned.
    """
    node_mask = torch.ones((x.shape[0],), dtype=torch.bool)

    # One-hot encode node and edge features - as expected by `build_edge_adj_attr`
    ## One-hot encode atom types
    if x.dim() == 1 or x.shape[-1] == 1:
        num_node_classes = len(get_valid_atoms(dataset))
        x = F.one_hot(x.flatten(), num_node_classes)

    ## One-hot encode bond types
    if edge_attr.dim() == 1:
        num_edge_types = len(get_valid_bonds())
        edge_attr = F.one_hot(edge_attr.long(), num_edge_types)

    # Assume a molecule has aromatic bonds if the one-hot representation of `edge_attr` has 4 bond types
    has_aromatic = edge_attr.shape[1] == 4

    # Create a new molecule
    mol = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    mol = build_edge_adj_attr(mol)

    # Build `rdkit.Chem.Mol` from the one-hot nodes and adjacency matrix
    molecule = make_molecule_from_xe(
        x=mol.x.argmax(-1),
        e=mol.edge_attr_adj.argmax(-1),
        node_mask=node_mask,
        num_hydrogens=num_hydrogens,
        formal_charge=formal_charge,
        dataset_name=dataset,
        with_aromatic=has_aromatic,
    )
    molecule = get_largest(molecule)

    # Drop implicit (and optionally explicit) hydrogen atoms
    molecule = Chem.RemoveHs(molecule, implicitOnly=with_hydro)
    if with_hydro:
        molecule = Chem.AddHs(molecule, explicitOnly=True)
        Chem.SanitizeMol(molecule)

    # Drop aromatic bonds if `with_aromatic` flag is not set
    if not with_aromatic:
        Chem.Kekulize(molecule, clearAromaticFlags=False)

    return molecule


def data_from_molecule(
    molecule: Chem.Mol, atom_classes: List[int], bond_types: List[Chem.BondType]
) -> Data | None:
    """
    Convert an RDKit `Chem.Mol` object into a PyTorch Geometric `Data` graph.

    This function encodes a molecule into a graph representation compatible
    with molecular graph datasets. It builds node (atom) and edge (bond) tensors
    using dataset-specific atom and bond vocabularies.

    The resulting `Data` object contains:
      - `x`: node tensor with atom-type indices.
      - `edge_index`: edge connectivity tensor.
      - `edge_attr`: bond-type indices.
      - `num_hydrogens`: number of hydrogens per atom
      - `formal_charge`: formal charge per atom
      - metadata such as `num_nodes`, `num_edges`, and `smiles`.

    Args:
        molecule (rdkit.Chem.Mol):
            RDKit molecule to convert into graph form.
        atom_classes (List[int]):
            List with valid atom types (atomic numbers).
        bond_types (List[rdkit.Chem.BondType]):
            List with valid bond types.

    Returns:
        Data|None:
            PyTorch Geometric `Data` object with the following attributes:
                - `x`: LongTensor of shape `[num_atoms, 1]`, containing atom-type indices.
                - `edge_index`: LongTensor of shape `[2, num_edges]`, defining bond connections.
                - `edge_attr`: LongTensor of shape `[num_edges * 2, 1]`, containing bond-type indices.
                - `num_hydrogens`: LongTensor of shape `[num_atoms, 1]`, containing hydrogen counts per atom.
                - `formal_charge`: LongTensor of shape `[num_atoms, 1]`, containing formal charge per atom.
                - `num_nodes`: Total number of atoms in the molecule.
                - `num_edges`: Total number of unique bonds.
                - `smiles`: Canonical SMILES string of the molecule.
            or `None` if any atom or bond type is not in the valid vocabulary.
    """
    # Node (atom) features
    node_labels, node_hs, node_fc = [], [], []
    for atom in molecule.GetAtoms():
        label = safe_index(atom_classes, atom.GetAtomicNum())
        if label is None:
            return None
        node_labels.append(label)
        node_hs.append(atom.GetTotalNumHs())
        node_fc.append(atom.GetFormalCharge())

    x = torch.tensor(node_labels, dtype=torch.long).view(-1, 1)
    num_hydrogens = torch.tensor(node_hs, dtype=torch.int).view(-1, 1)
    formal_charge = torch.tensor(node_fc, dtype=torch.int).view(-1, 1)

    # Edge (bond) features
    rows, cols, edge_labels = [], [], []
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        btype = bond.GetBondType()
        label = safe_index(bond_types, btype)
        if label is None:
            return None
        rows += [i, j]
        cols += [j, i]
        edge_labels += [label, label]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(edge_labels, dtype=torch.long).view(-1)

    # Create `Data` object for molecule
    smiles = Chem.MolToSmiles(molecule)
    mol = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_hydrogens=num_hydrogens,
        formal_charge=formal_charge,
        num_nodes=molecule.GetNumAtoms(),
        num_edges=molecule.GetNumBonds(),
        smiles=smiles,
    )
    return mol


"""
Data filtering functions, for all available datasets.

QM9 - ZINC - PubChem
"""


def filter_with_size_qm9(
    data: Data, with_hydro: bool = True, max_size: int = -1
) -> bool:
    """
    Pre-filter function for QM9 Dataset.

    Check that number of atoms in the molecule is less than the specified
    limit, and the molecule's SMILES representation is valid.

    Args:
        data (torch_geometric.data.Data):
            The molecule (`Data` object) to be examined.
        with_hydro (bool, default=True):
            If `False`, don't consider hydrogen molecules for limiting
            graph's maximum size.
        max_size (int, default=-1):
            Maximum size of nodes (atoms) per graph (molecule).

    Returns:
        bool: `True` iff the given `Data` object is valid according to restrictions.
    """
    num_nodes = data.num_nodes if with_hydro else data.x[:, 1:5].sum()
    return num_nodes <= max_size and is_valid_smiles(data.smiles)


def filter_with_size_zinc(data: Data, max_size: int = -1) -> bool:
    """
    Pre-filter function for ZINC Dataset.

    Check that number of atoms in the molecule is less than the specified
    limit.

    Args:
        data (torch_geometric.data.Data):
            The molecule (`Data` object) to be examined.
        max_size (int, default=-1):
            Maximum size of nodes (atoms) per graph (molecule).

    Returns:
        bool: `True` iff the given `Data` object is valid according to restrictions.
    """
    return data.num_nodes <= max_size


def filter_with_size_pubchem(data: Data, max_size: int = -1) -> bool:
    """
    Pre-filter function for PubChem Dataset.

    Check that number of atoms in the molecule is less than the specified
    limit, and the molecule's SMILES representation is valid.

    Args:
        data (torch_geometric.data.Data):
            The molecule (`Data` object) to be examined.
        max_size (int, default=-1):
            Maximum size of nodes (atoms) per graph (molecule).

    Returns:
        bool: `True` iff the given `Data` object is valid according to restrictions.
    """
    num_nodes = data.num_nodes
    return num_nodes <= max_size and is_valid_smiles(data.smiles)


def filter_dataset_with_size(
    dataset: DatasetName,
    max_size: int = -1,
) -> Callable[[Data], bool]:
    """
    Helper function for creating a `Callable` of the functions for filtering
    the available datasets:
    - `filter_with_size_qm9`
    - `filter_with_size_zinc`
    - `filter_with_size_pubchem`

    Torch Geometric's `Dataset` expects a `pre_filter` function that only accepts
    a `Data` argument.

    Args:
        dataset (Literal):
            Any available dataset between: `QM9NoHydro`, `QM9WithHydro`, `ZINC250k`, `ZINC12k`, `PubChem`
        max_size (int, default=-1):
            Maximum size of nodes (atoms) per graph (molecule).

    Returns:
        Callable[[Data],bool]:
            The filtering function corresponding to the selected dataset, ready
            to be used as a `pre_filter` function on a Torch Geometric's `Dataset`.
    """
    # QM9 dataset
    if dataset.startswith("QM9"):
        with_hydro = dataset == "QM9WithHydro"
        return partial(filter_with_size_qm9, with_hydro=with_hydro, max_size=max_size)

    # ZINC dataset
    if dataset.startswith("ZINC"):
        return partial(filter_with_size_zinc, max_size=max_size)

    # PubChem dataset
    if dataset.startswith("PubChem"):
        return partial(filter_with_size_pubchem, max_size=max_size)

    raise ValueError(f"Unknown dataset {dataset}")


"""
Various utility functions.
"""


def safe_index(lst: List[Any], item: Any) -> Optional[int]:
    """
    Safely get the index of an item in a list.

    Args:
        lst (list): The list to search.
        item (Any): The item to find.

    Returns:
        Optional[int]: The index of the item if found, otherwise `None`.
    """
    try:
        return lst.index(item)
    except:
        return None


def get_cpu_count() -> int:
    """
    Get number of available CPUs that can be allocated.

    Checks for SLURM set environment variables, and falls
    back to `os.cpu_count()`.
    """
    # Get local world size (for single node/multiple processes)
    lws = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    lws = max(lws, 1)

    cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK")
    if cpus_per_task is not None:
        return int(cpus_per_task) // lws

    ntasks = os.getenv("SLURM_NTASKS")
    job_cpus = os.getenv("SLURM_JOB_CPUS_PER_NODE")
    if ntasks and job_cpus:
        return int(job_cpus.split("(")[0]) // lws

    cpus_on_node = os.getenv("SLURM_CPUS_ON_NODE")
    if cpus_on_node:
        return int(cpus_on_node) // lws

    os_cpus = os.cpu_count()
    if os_cpus:
        return os_cpus // lws

    return 1
