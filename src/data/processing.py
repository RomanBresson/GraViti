"""
Data processing PyTorch Geometric transforms, for all available datasets.

QM9 - ZINC - PubChem
"""

from typing import Optional

import torch
import torch.nn.functional as F

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

from .constants import (
    DatasetName,
    get_dataset_stats,
    get_valid_atoms,
    get_valid_bonds,
    treat_aromatic_valid_table,
)
from .features import add_pe, build_edge_adj_attr, pad_to_shape, make_global_features
from .utils import normalize_molecule, data_from_molecule
from ..utils.chem import (
    molecular_weight,
    atom_weight,
    valence_electrons,
    electronegativity,
    compute_penalized_logP,
)


class ProcessGraphBase(BaseTransform):
    """
    PyTorch Geometric transform for processing molecules.

    Minimal processing pre-tranform, that normalizes each `mol`
    of the dataset by sanitizing the represented molecule, removing
    implicit (and optionally explicit - see `with_hydro`) hydrogens,
    and keeping only the largest connected fragment.

    Finally returns a `torch_geometric.data.Data` object with the following attributes:
        - x (`torch.Tensor`): Molecule's atoms, represented with zero-based indices
            of the dataset's valid atom classes (see `data.constants.get_valid_atoms`).
        - edge_index (`torch.Tensor`): Molecule's bonds, represented with two rows
            where `row[0,i]=k, row[1,i]=l` means that there is a bond (edge) between
            atom `x[k]` and `x[l]`.
            (NOTE: `row[0,i+1]=l, row[1,i+1]=k` to represent symmetry)
        - edge_attr (`torch.Tensor`): Molecule's bond types, represented with zero-
            based indices of the dataset's valid bond types (see `data.constants.get_valid_bonds`).
            `edge_attr[i]=k` means that the i-th bond in `edge_index` is of type `k`.
        - num_hydrogens (`torch.Tensor`): Hydrogen counts per atom.
        - formal_charge (`torch.Tensor`): Formal charge per atom.
        - num_nodes (`int`): Number of atoms in the molecule.
        - num_edges (`int`): Number of bonds in the molecule.
        - smiles (`str`): SMILES representation of the molecule.
        - atom_attr (`torch.Tensor`): Per-atom calculated attributes (atomic number, atomic weight,
        valence electrons, electronegativity, hydrogens, focal charge)

    Args:
        dataset (DatasetName):
            Any available dataset between the defined dataset names (`data.constants.DatasetName`)
        with_hydro (bool, default=False):
            Optionally keep explicit hydrogen atoms.
        with_aromatic (bool, default=True):
            Optionally keep aromatic bonds. If `False`, drop aromatic bonds with
            Kekulization (replace with alternating single-double bonds).
    """

    def __init__(
        self,
        dataset: DatasetName,
        with_hydro: bool = False,
        with_aromatic: bool = True,
    ):
        super().__init__()

        self.dataset = dataset
        self.with_hydro = with_hydro
        self.with_aromatic = with_aromatic

        self.valid_atom_classes = get_valid_atoms(dataset)
        self.valid_bond_types = get_valid_bonds(with_aromatic, dataset)

        # Prepare atom attributes for each of the available atom types.
        # Creates the `atom_type_attrs` property with basic attributes:
        # for all available atom types:
        #   Atomic Number, Atomic Mass, Valence Electrons, Electronegativity.
        #     shape = (num_node_types, 4)
        atom_type_attrs = [
            [
                atom_type,
                atom_weight(atom_type),
                valence_electrons(atom_type),
                electronegativity(atom_type),
            ]
            for atom_type in self.valid_atom_classes
        ]
        self.atom_type_attrs = torch.tensor(atom_type_attrs, dtype=torch.float32)

    def forward(self, mol: Data) -> Optional[Data]:
        molecule, new_mol = self._prepare_molecule(mol)

        # Calculate penalized logP and molecular weight attributes
        try:
            new_mol.PLogP = float(compute_penalized_logP(molecule))
            new_mol.MW = float(molecular_weight(molecule))
        except:
            raise Exception(
                "Failed to calculate penalized logP or weight for the molecule"
            )

        return new_mol

    def _prepare_molecule(self, mol: Data):
        # Normalize molecule
        molecule = normalize_molecule(
            dataset=self.dataset,
            x=mol.x,
            edge_index=mol.edge_index,
            edge_attr=mol.edge_attr,
            num_nodes=mol.num_nodes,
            num_hydrogens=getattr(mol, "num_hydrogens", None),
            formal_charge=getattr(mol, "formal_charge", None),
            with_hydro=self.with_hydro,
            with_aromatic=self.with_aromatic,
        )
        if molecule is None:
            raise Exception(f"Failed to 'normalize_molecule'")

        # Turn into graph `Data`
        new_mol = data_from_molecule(
            molecule=molecule,
            atom_classes=self.valid_atom_classes,
            bond_types=self.valid_bond_types,
        )
        if new_mol is None:
            raise Exception(
                "Failed to create mol `Data` object with 'data_from_molecule'"
            )

        new_mol.atom_attr = self._calculate_attributes(new_mol)

        return molecule, new_mol

    def _calculate_attributes(self, mol: Data):
        # Get attributes for each atom type of given molecule
        atom_attr = self.atom_type_attrs[mol.x.flatten()]
        atom_attr = torch.cat((atom_attr, mol.num_hydrogens, mol.formal_charge), dim=1)
        return atom_attr


class PreTransformQM9(ProcessGraphBase):
    """
    PyTorch Geometric transform for processing molecules of QM9 dataset,
    extending `ProcessGraphBase` class.

    Args:
        with_hydro (bool, default=False):
            Optionally keep explicit hydrogen atoms.
            If set to True, "QM9WithHydro" will be used instead of "QM9NoHydro".
        with_aromatic (bool, default=True):
            Optionally keep aromatic bonds. If `False`, drop aromatic bonds with
            Kekulization (replace with alternating single-double bonds).
    """

    dataset = "QM9NoHydro"

    def __init__(self, with_hydro: bool = False, with_aromatic: bool = True):
        if with_hydro:
            self.dataset = "QM9WithHydro"

        super().__init__(
            dataset=self.dataset,
            with_hydro=with_hydro,
            with_aromatic=with_aromatic,
        )

    def _prepare_molecule(self, mol: Data):
        # Handle hydrogens in the original dataset
        dataset = self.dataset
        self.dataset = "QM9WithHydro"

        # Only keep the one-hot nodes representation (the rest are features)
        mol.num_hydrogens = mol.x[:, -1].unsqueeze(-1).clone()
        mol.x = mol.x[:, :5].clone()
        # TODO: Get formal charge for QM9

        molecule, new_mol = super()._prepare_molecule(mol)
        new_mol.y = mol.y
        # Restore dataset
        self.dataset = dataset
        return molecule, new_mol


class PreTransformZINC(ProcessGraphBase):
    """
    PyTorch Geometric transform for processing molecules of ZINC dataset,
    extending `ProcessGraphBase` class.

    Args:
        with_aromatic (bool, default=True):
            Optionally keep aromatic bonds. If `False`, drop aromatic bonds with
            Kekulization (replace with alternating single-double bonds).
        subset (bool, default=False):
            Use subset of ZINC (12k molecules instead of 250k).
    """

    dataset = "ZINC250k"

    def __init__(
        self,
        with_aromatic: bool = True,
        subset: bool = False,
    ):
        if subset:
            self.dataset = "ZINC12k"

        super().__init__(
            dataset=self.dataset,
            with_hydro=False,
            with_aromatic=with_aromatic,
        )

    def _prepare_molecule(self, mol: Data):
        # Standardize atom ordering and edge attributes
        zinc_mapping = torch.tensor(
            [
                [0, 0, 0],  # 0: C
                [2, 0, 0],  # 1: O
                [1, 0, 0],  # 2: N
                [3, 0, 0],  # 3: F
                [0, 0, 1],  # 4: C H1
                [5, 0, 0],  # 5: S
                [6, 0, 0],  # 6: Cl
                [2, -1, 0],  # 7: O -
                [1, +1, 1],  # 8: N H1 +
                [7, 0, 0],  # 9: Br
                [1, +1, 3],  # 10: N H3 +
                [1, +1, 2],  # 11: N H2 +
                [1, +1, 0],  # 12: N +
                [1, -1, 0],  # 13: N -
                [5, -1, 0],  # 14: S -
                [8, 0, 0],  # 15: I
                [4, 0, 0],  # 16: P
                [2, +1, 1],  # 17: O H1 +
                [1, -1, 1],  # 18: N H1 -
                [2, +1, 0],  # 19: O +
                [5, +1, 0],  # 20: S +
                [4, 0, 1],  # 21: P H1
                [4, 0, 2],  # 22: P H2
                [0, -1, 2],  # 23: C H2 -
                [4, +1, 0],  # 24: P +
                [5, +1, 1],  # 25: S H1 +
                [0, -1, 1],  # 26: C H1 -
                [4, +1, 1],  # 27: P H1 +
            ],
            device=mol.x.device,
        )
        x = mol.x.view(-1).clone()

        mol.x = zinc_mapping[x][:, 0].unsqueeze(-1)
        mol.formal_charge = zinc_mapping[x][:, 1].unsqueeze(-1)
        mol.num_hydrogens = zinc_mapping[x][:, 2].unsqueeze(-1)
        mol.edge_attr = mol.edge_attr - 1

        return super()._prepare_molecule(mol)


class PreTransformPubChem(ProcessGraphBase):
    """
    PyTorch Geometric transform for processing molecules of PubChem dataset,
    extending `ProcessGraphBase` class.

    Args:
        with_aromatic (bool, default=True):
            Optionally keep aromatic bonds. If `False`, drop aromatic bonds with
            Kekulization (replace with alternating single-double bonds).
    """

    dataset = "PubChem16"

    def __init__(self, with_aromatic: bool = True):
        super().__init__(
            dataset=self.dataset,
            with_hydro=False,
            with_aromatic=with_aromatic,
        )


class TransformGraph(BaseTransform):
    """
    PyTorch Geometric transform for processing molecules.

    Performs the following transforms on a molecule `Data` object
    that is the result of `ProcessGraphBase`:
        - One-hot encoding node (`x`) and bond (`edge_attr`) classes.
        - Normalizing atom attributes (`atom_attr`).
        - Creating positional encodings (`pe_rw`, `pe_lap`, `pe` - see `add_pe`).
        - Building the edge adjacency matrix (`edge_attr_adj` - see `build_edge_adj_attr`).
        - Padding node features, positional encodings, and adjacency matrix up to `max_size` nodes.
        - Creating the `classes_nodes`, `classes_edges`, `num_edge_types`, `num_nodes_types` attributes.
        - Creating node- and edge-level features (`global_features` - see `make_global_features`).
    """

    default_max_size = 32

    def __init__(
        self,
        dataset: DatasetName,
        with_aromatic: bool = True,
        max_size: int = -1,
    ):
        self.dataset = dataset
        self.with_aromatic = with_aromatic

        self.num_node_classes = len(get_valid_atoms(dataset))
        self.num_bond_classes = len(get_valid_bonds(with_aromatic, dataset))
        _, self.min_formal_charge, _, _ = treat_aromatic_valid_table(
            dataset, with_aromatic
        )

        if max_size < 0:
            self.max_size = self.default_max_size
        else:
            self.max_size = max_size

        # Read dataset stats
        stats = get_dataset_stats(self.dataset, self.with_aromatic)
        mu_atom_count = torch.tensor(stats["mu_atom_count"])
        std_atom_count = torch.tensor(stats["std_atom_count"])
        mu_bond_count = torch.tensor(stats["mu_bond_count"])
        std_bond_count = torch.tensor(stats["std_bond_count"])
        mu_atom_total = torch.tensor([stats["mu_atom_total"]])
        std_atom_total = torch.tensor([stats["std_atom_total"]])
        if stats.get("mu_atom_attr") is not None:
            self.attr_mu = torch.tensor(stats["mu_atom_attr"])
            self.attr_std = torch.tensor(stats["std_atom_attr"])
        else:
            self.attr_mu, self.attr_std = None, None

        ## Define `mu` and `std` for global features
        self.global_mus = torch.cat([mu_atom_count, mu_bond_count, mu_atom_total])
        self.global_stds = torch.cat([std_atom_count, std_bond_count, std_atom_total])

    def forward(self, mol: Data) -> Optional[Data]:
        # Move data to same device as original mol
        self._data_to_device(mol, mol.x.device)
        return self._create_features(mol)

    def _create_features(self, mol: Data):
        # One-hot encode atom types (and pad to max_size)
        new_x = torch.zeros(self.max_size, self.num_node_classes)
        new_x[: mol.num_nodes] = F.one_hot(mol.x.flatten(), self.num_node_classes)

        # Normalize and pad mol attributes
        if self.attr_mu is not None and getattr(mol, "atom_attr", None) is not None:
            new_atom_attr = torch.zeros(self.max_size, mol.atom_attr.shape[-1])
            new_atom_attr[: mol.num_nodes] = mol.atom_attr
            new_atom_attr = (new_atom_attr - self.attr_mu) / self.attr_std
        else:
            new_atom_attr = None

        # One-hot encode edge types
        edge_classes = mol.edge_attr.long()
        edge_one_hot = F.one_hot(edge_classes, self.num_bond_classes).float()

        # Pad number of hydrogens
        if getattr(mol, "num_hydrogens", None) is not None:
            new_num_hydrogens = torch.zeros(self.max_size, dtype=torch.int)
            new_num_hydrogens[: mol.num_nodes] = mol.num_hydrogens.flatten()
        else:
            new_num_hydrogens = None

        # Pad and normalize formal charge (0-based)
        if getattr(mol, "formal_charge", None) is not None:
            new_formal_charge = torch.zeros(self.max_size, dtype=torch.int)
            new_formal_charge[: mol.num_nodes] = mol.formal_charge.flatten()
            new_formal_charge[: mol.num_nodes] -= self.min_formal_charge
        else:
            new_formal_charge = None

        # Create the transformed molecule
        new_mol = Data(
            x=new_x,
            edge_index=mol.edge_index,
            edge_attr=edge_one_hot,
            num_hydrogens=new_num_hydrogens,
            formal_charge=new_formal_charge,
            atom_attr=new_atom_attr,
            MW=getattr(mol, "MW", None),
            PLogP=getattr(mol, "PLogP", None),
            smiles=getattr(mol, "smiles", None),
            reg_target=getattr(mol, 'y', None)
        )

        new_mol = add_pe(new_mol, num_pe=8)
        new_mol = build_edge_adj_attr(new_mol)
        new_mol = pad_to_shape(new_mol, self.max_size)
        new_mol = make_global_features(new_mol, self.global_mus, self.global_stds)

        return new_mol

    def _data_to_device(self, mol: Data, device: torch.device | str):
        if getattr(mol, "attr_mu", None) is not None:
            self.attr_mu = self.attr_mu.to(device)
            self.attr_std = self.attr_std.to(device)
        if getattr(mol, "global_mus", None) is not None:
            self.global_mus = self.global_mus.to(device)
            self.global_stds = self.global_stds.to(device)
