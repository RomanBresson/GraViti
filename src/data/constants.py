"""
Various constants for datasets.
- Available datasets' names
- Valid bond types and atom classes per dataset
- Calculated statistics per dataset
- Getters for all mentioned values (`get_valid_atoms`, `get_valid_bonds`, `get_dataset_stats`)
"""

from typing import List, Literal, Optional, TypeAlias, Tuple, Dict, get_args, Set
from rdkit import Chem

"""
Define available datasets
"""


BOND_ORDER = {
    Chem.BondType.SINGLE: 1.0,
    Chem.BondType.DOUBLE: 2.0,
    Chem.BondType.TRIPLE: 3.0,
    Chem.BondType.AROMATIC: 1.5,
}

# --------------------------
# Ambiguous (H, charge) pairs to predict for aromatic heteroatoms
# under a representation without degree/valence/hydrogens.
AROMATIC_VALID_TABLE_PUBCHEM = {
    13: [  # Al
        (0, +1),
    ],
    5: [   # B
        (0, 0),
        (0, -1),
        (1, -1),
    ],
    6: [   # C
        (0, 0),
        (0, +1),
        (0, -1),
        (1, 0),
        (1, +1),
        (1, -1),
    ],
    7: [   # N
        (0, 0),
        (0, +1),
        (0, -1),
        (1, 0),
        (1, +1),
    ],
    8: [   # O
        (0, 0),
        (0, +1),
        (1, +1),
    ],
    15: [  # P
        (0, 0),
        (0, +1),
        (0, -1),
        (1, 0),
        (1, +1),
    ],
    16: [  # S
        (0, 0),
        (0, +1),
        (1, +1),
    ],
    34: [  # Se
        (0, 0),
        (0, +1),
        (1, +1),
    ],
    14: [  # Si
        (0, 0),
        (0, -1),
        (0, +2),
        (1, 0),
        (1, -1),
    ],
    52: [  # Te
        (0, 0),
        (0, +1),
    ],
}

AROMATIC_VALID_TABLE_QM9 = {k:AROMATIC_VALID_TABLE_PUBCHEM[k] for k in [6,7,8]}
AROMATIC_VALID_TABLE_ZINC = {k:AROMATIC_VALID_TABLE_PUBCHEM[k] for k in [6,7,8,15,16]}

DatasetName: TypeAlias = Literal[
    "QM9NoHydro",
    "QM9WithHydro",
    "ZINC250k",
    "ZINC12k",
    "PubChem16S",  # Subset of PubChem with 500K molecules (small) of up to 16 atoms
    "PubChem16",  # Subset of PubChem with all molecules of up to 16 atoms
    "PubChem32S",  # Subset of PubChem with 500K molecules (small) of up to 32 atoms
    "PubChem32",  # Subset of PubChem with all molecules of up to 32 atoms
    "PubChem64S",  # Subset of PubChem with 500K molecules (small) of up to 64 atoms
    "PubChem64",  # Subset of PubChem with all molecules of up to 64 atoms
    "ColoringSmall",
    "ColoringSmallCanonical",
    "ColoringMedium",
    "ColoringMediumCanonical",
    "ColoringBig",
    "ColoringBigCanonical",
    "BayesianNetAsia",
]

AROMATIC_VALID_TABLES = {
    "PubChem16S": AROMATIC_VALID_TABLE_PUBCHEM,
    "PubChem16": AROMATIC_VALID_TABLE_PUBCHEM,
    "PubChem32S": AROMATIC_VALID_TABLE_PUBCHEM,
    "PubChem32": AROMATIC_VALID_TABLE_PUBCHEM,
    "PubChem64S": AROMATIC_VALID_TABLE_PUBCHEM,
    "PubChem64": AROMATIC_VALID_TABLE_PUBCHEM,
    "QM9WithHydro": AROMATIC_VALID_TABLE_QM9,
    "QM9NoHydro": AROMATIC_VALID_TABLE_QM9,
    "ZINC250k": AROMATIC_VALID_TABLE_ZINC,
    "ZINC12k": AROMATIC_VALID_TABLE_ZINC
}

# Elements for which we consider aromatic inference. Others → skip (no ambiguity handling).
AVAILABLE_DATASETS = list(get_args(DatasetName))

"""
Getters for defined constants - Helpers for easier access.
"""


def treat_aromatic_valid_table(dataset: DatasetName, with_aromatic: bool = True) -> Dict[int, List[Tuple[int, int]]]:
    """
    Return the aromatic validity table for the given dataset.

    Args:
        dataset: Any available dataset between the defined dataset names (`data.constants.DatasetName`).
        with_aromatic: Whether there is an aromatic variant of the dataset. If `False`, the returned table will be empty (no aromatic rules).

    Returns:
        A dictionary mapping atomic numbers to lists of valid (H, FC) tuples in aromatic environments.
        If the dataset does not include aromatic rules, returns an empty dictionary.
    """

    AROMATIC_VALID_TABLE = AROMATIC_VALID_TABLES.get(dataset, None)

    # If dataset has no aromatic rules → return empty stats
    if (not AROMATIC_VALID_TABLE) or (not with_aromatic):
        return 0, 0, 0, {}

    # Extract hydrogen and FC sets per atom
    h_dict: Dict[int, Set[int]] = {
        Z: {h for (h, _) in pairs}
        for Z, pairs in AROMATIC_VALID_TABLE.items()
    }
    fc_dict: Dict[int, Set[int]] = {
        Z: {fc for (_, fc) in pairs}
        for Z, pairs in AROMATIC_VALID_TABLE.items()
    }

    max_h = max([max(h_set) for h_set in h_dict.values()]) if h_dict else 0
    max_fc = max([max(fc_set) for fc_set in fc_dict.values()]) if fc_dict else 0
    min_fc = min([min(fc_set) for fc_set in fc_dict.values()]) if fc_dict else 0

    # Determine which atoms require classifiers
    must_use_classifier = {}
    for Z in AROMATIC_VALID_TABLE.keys():
        need_H  = len(h_dict[Z])  > 1
        need_FC = len(fc_dict[Z]) > 1
        must_use_classifier[Z] = (need_H, need_FC)

    return max_h, min_fc, max_fc, must_use_classifier


def get_valid_atoms(
    dataset: DatasetName,
) -> List[int]:
    """
    Get the list of valid atomic numbers for given dataset.

    Args:
        dataset (DatasetName):
            Any available dataset between the defined dataset names (`data.constants.DatasetName`)

    Returns:
        List[int]:
            List of atomic numbers used in the given dataset.
    """
    if dataset == "QM9WithHydro":
        return VALID_ATOM_NUMS_QM9_WITH_HYDRO
    if dataset == "QM9NoHydro":
        return VALID_ATOM_NUMS_QM9_NO_HYDRO
    if dataset.startswith("ZINC"):
        return VALID_ATOM_NUMS_ZINC
    if dataset.startswith("PubChem"):
        return VALID_ATOM_NUMS_PUBCHEM
    if dataset.startswith("Coloring"):
        return VALID_NODES_COLORING
    if dataset.startswith("BayesianNet"):
        return VALID_NODES_BAYESIAN_NET_ASIA

    raise ValueError(f"Unknown dataset {dataset}")


def get_valid_bonds(
    with_aromatic: bool = True, dataset: Optional[DatasetName] = None
) -> List[Chem.BondType]:
    """
    Get the list of valid bond types.

    Args:
        with_aromatic (bool, default=True):
            Include aromatic bonds (set to `False` if kekulization will be used)
        dataset (DatasetName|None, optional):
            If provided for dataset-specific handling (e.g. Coloring),
            returns the appropriate edge classes for that dataset.

    Returns:
        List[rdkit.Chem.BondType]:
            List of bond types used for the representation of each molecule.
    """
    if dataset is not None and dataset.startswith("Coloring"):
        return VALID_EDGE_TYPES_COLORING
    if dataset is not None and dataset.startswith("BayesianNet"):
        return VALID_EDGE_TYPES_BAYESIAN_NET
    if not with_aromatic:
        return VALID_BOND_TYPES[:-1]
    return VALID_BOND_TYPES


def get_dataset_stats(
    dataset: DatasetName,
    with_aromatic: bool,
) -> dict:
    """
    Get atom (node) and bond (edge) statistics for selected dataset.

    Args:
        dataset (DatasetName):
            Any available dataset between the defined dataset names (`data.constants.DatasetName`)
        with_aromatic (bool):
            Get stats calculated including aromatic bonds. If `False` get stats for kekulized variant.

    Returns:
        dict[str,float|List[float]]:
            Calculated stats for selected dataset:
            - mu and std of atom count per graph (`mu_atom_total`, `std_atom_total`)
            - mu and std of bond count per graph (`mu_bond_total`, `std_bond_total`)
            - mu and std of atom class frequency per graph (`mu_atom_count`, `std_atom_count`)
            - mu and std of bond type frequency per graph (`mu_bond_count`, `std_bond_count`)
            - mu and std of molecular weights per graph (`mu_mw`, `std_mw`)
            - mu and std of penalized logP per graph (`mu_plogp`, `std_plogp`)
            - mu and std of atom attributes per graph (`mu_atom_attr`, `std_atom_attr`)
    """
    if dataset == "QM9WithHydro":
        stats = QM9_WITH_HYDRO_STATS
    elif dataset == "QM9NoHydro":
        stats = QM9_NO_HYDRO_STATS
    elif dataset.startswith("ZINC"):
        stats = ZINC_STATS
    elif dataset.startswith("PubChem"):
        stats = PUBCHEM_STATS
    elif dataset.startswith("Coloring"):
        return COLORING_STATS
    elif dataset.startswith("BayesianNet"):
        return BAYESIAN_NET_ASIA_STATS
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    return {
        **{
            k: stats[k]
            for k in [
                "mu_atom_total",
                "std_atom_total",
                "mu_bond_total",
                "std_bond_total",
                "mu_atom_count",
                "std_atom_count",
                "mu_mw",
                "std_mw",
                "mu_plogp",
                "std_plogp",
                "mu_atom_attr",
                "std_atom_attr",
            ]
        },
        **{
            k: stats["aromatic" if with_aromatic else "kekule"][k]
            for k in ["mu_bond_count", "std_bond_count"]
        },
    }


"""
Dataset-specific structure constants
"""

VALID_BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]

VALID_EDGE_TYPES_COLORING = [0]  # only 1 edge type
VALID_EDGE_TYPES_BAYESIAN_NET = [0]  # only 1 directed edge type

VALID_ATOM_NUMS_QM9_WITH_HYDRO = [
    1,  # H (Hydrogen)
    6,  # C (Carbon)
    7,  # N (Nitrogen)
    8,  # O (Oxygen)
    9,  # F (Fluorine)
]

VALID_ATOM_NUMS_QM9_NO_HYDRO = [
    6,  # C (Carbon)
    7,  # N (Nitrogen)
    8,  # O (Oxygen)
    9,  # F (Fluorine)
]

VALID_ATOM_NUMS_ZINC = [
    6,  # C (Carbon)
    7,  # N (Nitrogen)
    8,  # O (Oxygen)
    9,  # F (Fluorine)
    15,  # P (Phosphorus)
    16,  # S (Sulfur)
    17,  # Cl (Chlorine)
    35,  # Br (Bromine)
    53,  # I (Iodine)
]

# This list of atomic numbers was obtained from the official GRALE repository: https://github.com/KrzakalaPaul/GRALE
VALID_ATOM_NUMS_PUBCHEM = [
    6,  # C (Carbon)
    7,  # N (Nitrogen)
    8,  # O (Oxygen)
    9,  # F (Fluorine)
    15,  # P (Phosphorus)
    16,  # S (Sulfur)
    17,  # Cl (Chlorine)
    35,  # Br (Bromine)
    53,  # I (Iodine)
    14,  # Si (Silicon)
    11,  # Na (Sodium)
    33,  # As (Arsenic)
    80,  # Hg (Mercury)
    50,  # Sn (Tin)
    5,  # B (Boron)
    20,  # Ca (Calcium)
    19,  # K (Potassium)
    30,  # Zn (Zinc)
    26,  # Fe (Iron)
    34,  # Se (Selenium)
    13,  # Al (Aluminum)
    29,  # Cu (Copper)
    12,  # Mg (Magnesium)
    82,  # Pb (Lead)
    24,  # Cr (Chromium)
    27,  # Co (Cobalt)
    28,  # Ni (Nickel)
    56,  # Ba (Barium)
    78,  # Pt (Platinum)
    25,  # Mn (Manganese)
    52,  # Te (Tellurium)
]

VALID_NODES_COLORING = [
    "blue",
    "green",
    "red",
    "yellow",
]

VALID_NODES_BAYESIAN_NET_ASIA = list(range(8))


"""
Dataset-specific calculated stats
"""

QM9_WITH_HYDRO_STATS = {
    "mu_atom_total": 18.05018137847664,
    "std_atom_total": 2.943793629839058,
    "mu_bond_total": 18.659241000837067,
    "std_bond_total": 3.144939032477548,
    "mu_atom_count": [
        9.25529408117062,
        6.3585170371748,
        1.0153939168449373,
        1.3978699655846005,
        0.023106377701299047,
    ],
    "std_atom_count": [
        2.816840763802325,
        1.232051965854609,
        1.0744702258049077,
        0.8860971730722759,
        0.21716283911599848,
    ],
    "aromatic": {
        "mu_bond_count": [
            16.9042104610421,
            0.6021067807645866,
            0.2812606579232917,
            0.8716631011068909,
        ],
        "std_bond_count": [
            4.685357989766857,
            0.718980011377896,
            0.5281759389345242,
            2.028014809435352,
        ],
    },
    "kekule": {
        "mu_bond_count": [
            17.429742969646412,
            0.9482373732675929,
            0.2812606579232917,
        ],
        "std_bond_count": [
            3.942443678885886,
            0.9895874837895698,
            0.5281759389345242,
        ],
    },
    "mu_mw": 122.87102485216771,
    "std_mw": 7.6386315693303395,
    "mu_plogp": -7.044836911982709,
    "std_plogp": 1.21694395844188,
    "mu_atom_attr": [
        3.651213772550152,
        6.799244762577535,
        2.676720817283475,
        2.4688560320175115,
        0.012350720791517755,
        0.0004732274368851155,
    ],
    "std_atom_attr": [
        2.771208002791685,
        6.035969516067202,
        1.800351209565348,
        0.3656437138077588,
        0.11054255757037877,
        0.028725131083507962,
    ],
}

QM9_NO_HYDRO_STATS = {
    "mu_atom_total": 8.794887297305749,
    "std_atom_total": 0.5107028040202518,
    "mu_bond_total": 9.40394691966617,
    "std_bond_total": 1.1637077001089604,
    "mu_atom_count": [
        6.3585170371748,
        1.0153939168449373,
        1.3978699655846005,
        0.023106377701299047,
    ],
    "std_atom_count": [
        1.232051965854609,
        1.0744702258049077,
        0.8860971730722759,
        0.21716283911599848,
    ],
    "aromatic": {
        "mu_bond_count": [
            7.648916379871683,
            0.6021067807645866,
            0.2812606579232917,
            0.8716631011068909,
        ],
        "std_bond_count": [
            2.588139817708458,
            0.718980011377896,
            0.5281759389345242,
            2.028014809435352,
        ],
    },
    "kekule": {
        "mu_bond_count": [
            8.174448888475522,
            0.9482373732675929,
            0.2812606579232917,
        ],
        "std_bond_count": [
            1.807147586903011,
            0.9895874837895698,
            0.5281759389345242,
        ],
    },
    "mu_mw": 122.87102485216771,
    "std_mw": 7.6386315693303395,
    "mu_plogp": -4.008468739699812,
    "std_plogp": 1.526427944514304,
    "mu_atom_attr": [
        6.441216908181261,
        12.893657506224864,
        4.441216908181261,
        2.7517864569765984,
        1.077697361115274,
        0.0009712280305927947,
    ],
    "std_atom_attr": [
        0.761702679479848,
        1.5287132487997812,
        0.761702679479848,
        0.3439074580567665,
        0.9750720294553518,
        0.041145821206337176,
    ],
}

ZINC_STATS = {
    "mu_atom_total": 23.15485134834176,
    "std_atom_total": 4.5073678096723855,
    "mu_bond_total": 24.906559217493243,
    "std_bond_total": 5.293177799205774,
    "mu_atom_count": [
        17.06111967128926,
        2.8260268804741573,
        2.310220852593755,
        0.3180431887496501,
        0.0005272463649544822,
        0.41227029557613276,
        0.17193685770256895,
        0.051092899900458975,
        0.003613455690851813,
    ],
    "std_atom_count": [
        3.718496061857261,
        1.3885599536759883,
        1.2980263998635844,
        0.7805198496144126,
        0.023734631974674767,
        0.6103043592263611,
        0.44678638563779866,
        0.22659687122527408,
        0.060830999073486935,
    ],
    "aromatic": {
        "mu_bond_count": [
            12.864820395343775,
            1.587220638968081,
            0.06240597061056122,
            10.392112212571421,
        ],
        "std_bond_count": [
            4.141540721092316,
            1.114953318100454,
            0.2555965635659942,
            5.281270918542057,
        ],
    },
    "kekule": {
        "mu_bond_count": [
            18.50655194512992,
            6.337601301753277,
            0.06240597061056122,
        ],
        "std_bond_count": [
            3.8238563867303292,
            2.645868408322221,
            0.2555965635659942,
        ],
    },
    "mu_mw": 331.433631079598,
    "std_mw": 61.96550939512845,
    "mu_plogp": -0.18614674546220472,
    "std_plogp": 1.5971927417010483,
    "mu_atom_attr": [
        6.6940609564925015,
        13.447809822281062,
        4.427797850233696,
        2.7242207706036212,
        0.8755689177087809,
        -0.00726573624517628,
    ],
    "std_atom_attr": [
        2.245956575561803,
        4.915374670226969,
        0.7880887987253579,
        0.33048988138296176,
        0.9495196151568385,
        0.10705152465044059,
    ],
}

PUBCHEM_STATS = {
    "mu_atom_total": 21.7470622529769,
    "std_atom_total": 5.515460097062807,
    "mu_bond_total": 23.15218947162415,
    "std_bond_total": 6.355047219760842,
    "mu_atom_count": [
        16.010094265535024,
        2.4466767753219285,
        2.2162923361556954,
        0.3809789862485704,
        0.011821354999141684,
        0.34326264270127144,
        0.20746905995705808,
        0.09227599963090784,
        0.015012347499168499,
        0.013109145758605185,
        1.708391322233025e-06,
        0.0002479319721267013,
        7.325904835339548e-05,
        0.0005761986920986099,
        0.0075764195722999,
        1.4393533187314803e-06,
        1.1030558143545977e-06,
        2.542409133086969e-06,
        5.52873097195307e-06,
        0.0010699775918245316,
        0.00025358177020012455,
        1.6007761208328294e-06,
        1.4931609194316253e-06,
        4.535980739030247e-05,
        1.229503676000601e-05,
        3.4302345446405383e-06,
        5.421115770549659e-06,
        1.9101698248586807e-06,
        5.622894273176807e-06,
        2.77109143606318e-06,
        0.00017974429013914905,
    ],
    "std_atom_count": [
        4.640332295265356,
        1.5903941615286925,
        1.597049804861007,
        0.9802659164544859,
        0.13184433871102982,
        0.5892403152089013,
        0.5228744529606099,
        0.32621275210619677,
        0.138836911312687,
        0.14960564349739275,
        0.0014439673253302107,
        0.018104551729124296,
        0.009413594035061666,
        0.02562587973614253,
        0.11136053853909073,
        0.0027714676675277554,
        0.0012164324158945154,
        0.0016361301008981813,
        0.0024466436380346486,
        0.03917317378616142,
        0.018294431848105693,
        0.0012863052442151578,
        0.0015603819195718117,
        0.007329722539754686,
        0.0036345034023491274,
        0.0019511217374325479,
        0.0023966501754587944,
        0.003397311710375842,
        0.002416214629870277,
        0.0017280955413657483,
        0.015467711386065803,
    ],
    "aromatic": {
        "mu_bond_count": [
            12.143747274305134,
            1.4344675213291347,
            0.071554584986005,
            9.502420091003854,
        ],
        "std_bond_count": [
            4.648475156761185,
            1.2141143526734757,
            0.2824066621516779,
            5.884128049922766,
        ],
    },
    "kekule": {
        "mu_bond_count": [
            17.273935104702762,
            5.806889013310936,
            0.07155358865760567,
        ],
        "std_bond_count": [
            4.631847518268939,
            2.867659166866684,
            0.2824066107136622,
        ],
    },
    "mu_mw": 316.9555243065116,
    "std_mw": 78.88649625517935,
    "mu_plogp": -0.4051175326431947,
    "std_plogp": 2.2652783922898108,
    "mu_atom_attr": [
        6.800310482194923,
        13.700714973233763,
        4.444184802853124,
        2.7281847877117524,
        0.8861220634946353,
        -0.012867426618887722,
    ],
    "std_atom_attr": [
        2.8452437015257885,
        6.447978995006755,
        0.8216546123180213,
        0.3413798210967808,
        0.9680717571536731,
        0.1328262656825521,
    ],
}

COLORING_STATS = {
    "mu_atom_total": 12.509296666666511,
    "std_atom_total": 4.6145557264382475,
    "mu_bond_total": 24.93021999999993,
    "std_bond_total": 11.429519654444366,
    "mu_atom_count": [
        3.1280833333333296,
        3.125646666666672,
        3.127230000000017,
        3.1283366666666623
    ],
    "std_atom_count": [
        1.3692300424053316,
        1.3718597554000167,
        1.369993473235193,
        1.3700119061882665
    ],
    "mu_bond_count": [24.93021999999993],
    "std_bond_count": [11.429519654444366],
}

BAYESIAN_NET_ASIA_STATS = {
    "mu_atom_total": 8.0,
    "std_atom_total": 0.0,
    "mu_bond_total": 8.00433999999984,
    "std_bond_total": 2.389234125530295,
    "mu_atom_count": [1.0] * 8,
    "std_atom_count": [0.0] * 8,
    "mu_bond_count": [8.00433999999984],
    "std_bond_count": [2.389234125530295],
    "mu_atom_attr": [1.000542500000028, 1.0005425000000228, 1.0547418749999802],
    "std_atom_attr": [1.0693130600239686, 1.0684675564602317, 1.1245147808965832],
}
