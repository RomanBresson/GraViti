from functools import lru_cache
from typing import List, Union

import torch

import networkx as nx

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer

from mendeleev import element

from ..data.constants import DatasetName, get_valid_atoms, get_valid_bonds, treat_aromatic_valid_table

PERIODIC_TABLE = Chem.GetPeriodicTable()


def molecular_weight(mol: Chem.Mol) -> float:
    """
    Compute the exact molecular weight of an RDKit Mol.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule.

    Returns:
        float: Exact molecular weight.
    """
    return Descriptors.ExactMolWt(mol)


def logP(mol: Chem.Mol) -> float:
    """
    Compute the partition coefficient (logP) of an RDKit Mol.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule.

    Returns:
        float: logP value.
    """
    return Descriptors.MolLogP(mol)


def get_largest(molecule: Chem.Mol) -> Chem.Mol | None:
    """
    Given an RDKit Mol that may be disconnected, return the largest connected fragment.

    Args:
        molecule (rdkit.Chem.Mol): Input RDKit molecule.

    Returns:
        rdkit.Chem.Mol|None:
            Largest fragment molecule, or None if sanitization fails.
    """
    try:
        mol_frags = Chem.rdmolops.GetMolFrags(
            molecule, asMols=True, sanitizeFrags=False
        )
        largest_mol = max(
            mol_frags, default=molecule, key=lambda m: m.GetNumHeavyAtoms()
        )
        return largest_mol
    except:
        return None


def is_valid(molecule: Chem.Mol) -> bool:
    """
    Check if a `Chem.Mol` object corresponds to a valid RDKit molecule.

    Args:
        molecule (rdkit.Chem.Mol): Molecule to be validated

    Returns:
        bool:
            True if RDKit can parse and basic descriptors compute successfully.
    """
    try:
        get_largest(molecule)
        logP(molecule)
        molecular_weight(molecule)
        return True
    except:
        return False


def is_valid_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string corresponds to a valid RDKit molecule.

    Args:
        smiles (str): Molecule's SMILES representation.

    Returns:
        bool:
            True if RDKit can parse and basic descriptors compute successfully.
    """
    mol = Chem.MolFromSmiles(smiles)
    return is_valid(mol)


def compute_penalized_logP(mol: Chem.Mol) -> float:
    """
    Compute penalized logP as in 'Lift Your Molecules' paper:
        logP - synthetic accessibility (SA) - large-cycle penalty.

    Args:
        mol (rdkit.Chem.Mol): Input RDKit molecule.

    Returns:
        float: Penalized logP score.
    """
    get_largest(mol)
    current_log_P_value = Descriptors.MolLogP(mol)
    current_SA_score = -sascorer.calculateScore(mol)
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))

    # Detect cycles
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])

    # Penalize long cycles (>6 atoms)
    cycle_length = max(cycle_length - 6, 0)
    current_cycle_score = -cycle_length

    score = current_log_P_value + current_SA_score + current_cycle_score
    return score


def make_molecule_from_xe(
    x: torch.Tensor,
    e: torch.Tensor,
    node_mask: torch.Tensor,
    num_hydrogens: torch.Tensor,
    formal_charge: torch.Tensor,
    dataset_name: DatasetName,
    with_aromatic: bool = True,
) -> Chem.Mol:
    """
    Construct an RDKit `Mol` from node and edge features tensors. Optionally, the number of hydrogens
    per atom can be provided, in order to be explicitly set during reconstruction.

    Args:
        x (torch.Tensor):
            Node feature tensor.
        e (torch.Tensor):
            Edge feature tensor.
        node_mask (torch.Tensor):
            A boolean tensor of shape (num_nodes) where True indicates a real node and False indicates a non-node.
        num_hydrogens (torch.Tensor):
            Number of hydrogens per atom. If `None`, a default value of zeros will be used (automatically infer with `rdkit`).
        formal_charge (torch.Tensor):
            Formal charge per atom. If `None`, a default value of zeros will be used (try to infer using heuristics).
        dataset_name (DatasetName):
            Any available dataset between the defined dataset names (`data.constants.DatasetName`).
            Used to define valid atom and bond vocabularies.
        with_aromatic (bool, optional, default=True):
            Whether the original molecule contains aromatic bonds - needed to find the `index_no_bond`.

    Returns:
        rdkit.Chem.Mol: Constructed RDKit molecule.
    """
    assert x.dim()==1
    assert e.dim()==2
    # Atom classes (if one-hot encoded, extract actual index)
    atom_dict = {i: t for i, t in enumerate(get_valid_atoms(dataset_name))}
    atom_classes = x.clone()
    bond_dict = {i: t for i, t in enumerate(get_valid_bonds(with_aromatic))}
    bonds = e.clone()
    index_no_bond = len(bond_dict)
    # Indices of non-padding atoms
    real_atoms = node_mask.bool().cpu().clone()
    atom_classes = atom_classes[real_atoms]
    bonds = bonds[real_atoms][:, real_atoms]
    size = len(atom_classes)

    # Number of hydrogens per atom (if not defined, default to 0s)
    no_implicit_hs = num_hydrogens is not None
    if no_implicit_hs:
        atom_hydrogens = num_hydrogens.to(torch.int)
        atom_hydrogens = atom_hydrogens[real_atoms]

    # Formal charge per atom (if not defined, default to 0s)
    disable_charge_heuristic = formal_charge is not None
    if disable_charge_heuristic:
        atom_charges = formal_charge.to(torch.int).clone()
        atom_charges = atom_charges[real_atoms]

    # Molecule creation
    molecule = Chem.RWMol()

    ## Add atoms
    for class_idx in atom_classes:
        atom = Chem.Atom(atom_dict[class_idx.item()])
        molecule.AddAtom(atom)

    ## Add bonds
    for i in range(size):
        for j in range(i + 1, size):
            if bonds[i, j] != index_no_bond:
                bond = bonds[i, j]
                bond_type = bond_dict[bond.item()]
                molecule.AddBond(i, j, bond_type)

    ## Fix number of Hydrogens and Formal Charge on atom iff
    ## it's part of an aromatic cycle (considered ambiguous)
    molecule.UpdatePropertyCache(strict=False)
    for idx, atom in enumerate(molecule.GetAtoms()):
        is_aromatic = any(b.GetBondType() == Chem.BondType.AROMATIC for b in atom.GetBonds())

        # Set hydrogens explicitly
        if is_aromatic and no_implicit_hs:
            # Take into account hydrogens that are already attached
            # to the atom through bonds
            neighbors = atom.GetNeighbors()
            attached_hs = sum([n.GetSymbol() == "H" for n in neighbors])
            hs_set = atom_hydrogens[idx].item() - attached_hs
            atom.SetNumExplicitHs(max(hs_set, 0))
            atom.SetNoImplicit(True)
            atom.UpdatePropertyCache(strict=False)

        # Force formal charge
        if is_aromatic and disable_charge_heuristic:
            atom.SetFormalCharge(atom_charges[idx].item())
            atom.UpdatePropertyCache(strict=False)
        # Heuristic formal charge assignment based on atom valence
        else:
            fix_formal_charge(atom)

    molecule = molecule.GetMol()

    # Sanitize and drop implicit hydrogens
    try:
        Chem.SanitizeMol(molecule)
        molecule = get_largest(molecule)
        molecule = Chem.RemoveHs(molecule, implicitOnly=True)
        return molecule
    except:
        return None


def make_molecules(
    X: torch.Tensor,
    E: torch.Tensor,
    node_mask: torch.Tensor,
    num_hydrogens: torch.Tensor = None,
    formal_charges: torch.Tensor = None,
    dataset_name: DatasetName = "PubChem32",
    with_aromatic: bool = True,
) -> List[Chem.Mol]:
    """
    Construct an RDKit `Mol` from node features `x` and edge features `e`, for a list
    of (x, e, mask) represented molecules. Optionally, the number of hydrogens
    per atom can be provided, in order to be explicitly set during reconstruction.

    Args:
        x (torch.Tensor):
            Node feature matrix (num_nodes, num_atom_classes).
        e (torch.Tensor):
            Edge feature tensor (num_nodes, num_nodes, num_bond_types).
        node_mask (torch.Tensor):
            A boolean tensor of shape (num_nodes) where True indicates a real node and False indicates a non-node.
        num_hydrogens (torch.Tensor, optional):
            Number of hydrogens per atom (per graph/molecule). Defaults to zeros for all atoms.
        formal_charges (torch.Tensor, optional):
            Formal charge per atom (per graph/molecule). Defaults to zeros for all atoms.
        dataset_name (DatasetName):
            Any available dataset between the defined dataset names (`data.constants.DatasetName`).
            Used to define valid atom and bond vocabularies.
        with_aromatic (bool, optional, default=True):
            Whether the original molecule contains aromatic bonds - needed to find the `index_no_bond`.

    Returns:
        List[rdkit.Chem.Mol]: Constructed RDKit molecules.
    """
    if num_hydrogens is not None and formal_charges is not None:
        return [
            make_molecule_from_xe(x, e, n, hs, fc, dataset_name, with_aromatic)
            for x, e, n, hs, fc in zip(X, E, node_mask, num_hydrogens, formal_charges)
        ]
    else:
        return [
            make_molecule_from_xe(x, e, n, None, None, dataset_name, with_aromatic)
            for x, e, n in zip(X, E, node_mask)
        ]

def make_molecules_from_batch(
    batch,
    dataset_name: DatasetName = "PubChem32",
    with_aromatic: bool = True
) -> List[Chem.Mol]:

    num_hydrogens = batch.get("hydrogens").clone()

    formal_charges = batch.get("formal_charges").clone()
    if formal_charges is not None:
        fc_min = treat_aromatic_valid_table(dataset_name)[1]
        formal_charges += fc_min

    return make_molecules(
        X=batch['X'].clone(),
        E=batch['E'].clone(),
        node_mask=batch['node_mask'].clone(),
        num_hydrogens=num_hydrogens,
        formal_charges=formal_charges,
        dataset_name=dataset_name,
        with_aromatic=with_aromatic,
    )

def make_molecules_from_outputs(
    batch, 
    dataset_name: DatasetName = "PubChem32",
    with_aromatic: bool = True
) -> List[Chem.Mol]:
    h = batch.get("predicted_hydrogens")
    num_hydrogens = h.argmax(-1) if h is not None else None

    fc_min = treat_aromatic_valid_table(dataset_name)[1]
    fc = batch.get("predicted_formal_charges")
    formal_charges = (fc.argmax(-1) + fc_min) if fc is not None else None

    return make_molecules(
        X=batch['X'].argmax(-1),
        E=batch['E'].argmax(-1),
        node_mask=batch['used_node_filter'],
        num_hydrogens=num_hydrogens,
        formal_charges=formal_charges,
        dataset_name=dataset_name,
        with_aromatic=with_aromatic,
    )

def atom_weight(atom: Union[int, str, Chem.Atom]) -> float:
    """
    Get the mass for a given atom.

    Args:
        atom (int|str|rdkit.Chem.Atom): Atom representation as `int` (atomic number),
            `str` (symbol) or `rdkit.Chem.Atom` object.

    Returns:
        float:
            Mass (weight) of the given atom.
    """
    if not isinstance(atom, Chem.Atom):
        atom = Chem.Atom(atom)
    return atom.GetMass()


def valence_electrons(atom: Union[int, str, Chem.Atom]) -> float:
    """
    Get the number of valence electrons for a given atom.

    Args:
        atom (int|str|rdkit.Chem.Atom): Atom representation as `int` (atomic number),
            `str` (symbol) or `rdkit.Chem.Atom` object.

    Returns:
        float:
            Number of valence electrons of the given atom.
    """
    if isinstance(atom, Chem.Atom):
        atom = atom.GetAtomicNum()
    return PERIODIC_TABLE.GetNOuterElecs(atom)


@lru_cache(maxsize=None)
def electronegativity(atom: Union[int, str, Chem.Atom]) -> float:
    """
    Get the Pauling electronegativity for a given atom.

    Results are cached after first lookup for faster access,
    using `lru_cache`.

    Args:
        atom (int|str|rdkit.Chem.Atom): Atom representation as `int` (atomic number),
            `str` (symbol) or `rdkit.Chem.Atom` object.

    Returns:
        float:
            Electronegativity of the given atom (or 0 in case of error).
    """
    if isinstance(atom, Chem.Atom):
        atom = atom.GetSymbol()
    elif isinstance(atom, int):
        atom = Chem.Atom(atom).GetSymbol()

    en = element(atom).electronegativity_pauling()
    return 0.0 if en is None else en


def fix_formal_charge(atom: Chem.Atom):
    atom.UpdatePropertyCache(strict=False)
    val = atom.GetValence(which=Chem.ValenceType.EXPLICIT)

    if (atom.GetSymbol() == "N") and (val == 4):
        atom.SetFormalCharge(1)

    elif (atom.GetSymbol() == "O") and (val == 3):
        atom.SetFormalCharge(1)
    elif (atom.GetSymbol() == "O") and (val == 1):
        atom.SetFormalCharge(-1)

    elif (atom.GetSymbol() in ["B", "Al"]) and (val == 4):
        atom.SetFormalCharge(-1)
    elif (atom.GetSymbol() in ["B", "Al"]) and (val == 5):
        atom.SetFormalCharge(-2)
    elif (atom.GetSymbol() in ["B", "Al"]) and (val == 6):
        atom.SetFormalCharge(-3)

    elif (atom.GetSymbol() == "P") and (val == 4):
        atom.SetFormalCharge(1)
    elif (atom.GetSymbol() == "P") and (val == 6):
        atom.SetFormalCharge(-1)

    elif (atom.GetSymbol() == "As") and (val == 6):
        atom.SetFormalCharge(-1)

    elif (atom.GetSymbol() == "Si") and (val == 6):
        atom.SetFormalCharge(-2)
    elif (atom.GetSymbol() == "Si") and (val == 5):
        atom.SetFormalCharge(-1)

    elif (atom.GetSymbol() == "Sn") and (val == 6):
        atom.SetFormalCharge(-2)
    elif (atom.GetSymbol() == "Sn") and (val == 5):
        atom.SetFormalCharge(-1)

    elif (atom.GetSymbol() == "I") and (val == 6):
        atom.SetFormalCharge(1)

    elif (atom.GetSymbol() in ["Cl", "Br"]) and (val > 1):
        atom.SetFormalCharge(int(val - 1))

    elif (atom.GetSymbol() in ["Ba", "K"]) and (val == 0):
        atom.SetFormalCharge(valence_electrons(atom.GetAtomicNum()))
