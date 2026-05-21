import torch

from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
from torch_geometric.data import Data


def add_pe(mol: Data, num_pe: int) -> Data:
    """
    Adds Laplacian Eigenvector and Random Walk Positional Encodings (PEs)
    to the molecule graph.

    Args:
        mol (torch_geometric.data.Data): Graph data object.
        num_pe (int): Desired PE dimensionality.

    Returns:
        torch_geometric.data.Data:
            Graph with `mol.pe_lap`, `mol.pe_rw`, and `mol.pe` fields
            added (shape [num_nodes, num_pe]).
    """
    num_nodes = mol.num_nodes
    gotten_pe = min(num_pe, num_nodes - 1)

    # Handle graphs with no edges: just pad with zeros
    if mol.edge_index.shape[1] == 0:
        mol.pe_lap = torch.zeros(mol.num_nodes, num_pe)
        mol.pe_rw = torch.zeros(mol.num_nodes, 8)

    else:
        # Add positional encodings - Laplacian Eigenvector and Random Walk
        PE_transform_lap = AddLaplacianEigenvectorPE(gotten_pe, attr_name="pe_lap")
        PE_transform_rw = AddRandomWalkPE(8, attr_name="pe_rw")

        mol = PE_transform_lap(mol)
        mol = PE_transform_rw(mol)

        # Pad with zeros if fewer eigenvectors than requested
        mol.pe_lap = torch.cat(
            [mol.pe_lap, torch.zeros(num_nodes, num_pe - gotten_pe)], dim=1
        )

    mol.pe = torch.cat([mol.pe_lap, mol.pe_rw], dim=1)
    return mol


def build_edge_adj_attr(mol: Data) -> Data:
    """
    Builds a dense adjacency tensor encoding edge attributes.

    Creates tensor of shape [num_nodes, num_nodes, num_edge_features+1].
    The last channel encodes presence/absence of an edge.

    Args:
        mol (torch_geometric.data.Data): Graph data object.

    Returns:
        torch_geometric.data.Data:
            Graph with `mol.edge_attr_adj` field added (shape [num_nodes, num_nodes, num_edge_features+1]).
    """
    edge_index = mol.edge_index
    edge_attr = mol.edge_attr

    # Initially mark all pairs as "no-edge"
    edge_attr_adj = torch.zeros(mol.num_nodes, mol.num_nodes, edge_attr.shape[1] + 1)
    edge_attr_adj[:, :, -1] = 1

    # Fill in edge attributes
    edge_attr_adj[edge_index[0], edge_index[1], :-1] = edge_attr.float()
    edge_attr_adj[edge_index[0], edge_index[1], -1] = 0

    mol.edge_attr_adj = edge_attr_adj
    return mol


def pad_to_shape(mol: Data, max_size: int) -> Data:
    """
    Pads a molecule graph to a fixed number of nodes.

    - Adds 'dummy' nodes with last feature = 1.0.
    - Pads positional encodings and adjacency accordingly.

    Args:
        mol (torch_geometric.data.Data): Graph data object.
        max_size (int): Maximum number of nodes in graph - pads up to this size.

    Returns:
        torch_geometric.data.Data:
            Graph with `mol.x`, `mol.pe`, and `mol.edge_attr_adj` fields padded
            to fill size of `max_size`.
    """
    num_nodes = mol.x.shape[0]
    num_nodes_types = mol.x.sum(0)
    num_edge_types = mol.edge_attr_adj.sum((0, 1))
    false_nodes_needed = max_size - num_nodes

    # Pad node features (mark dummy nodes with 1)
    new_x = torch.zeros(max_size, mol.x.shape[1] + 1)
    new_x[:num_nodes, :-1] = mol.x
    new_x[:, -1] = 1 - new_x.sum(1)  # set padding nodes to be "not a real atom"
    mol.x = new_x
    mol.node_mask = new_x[:, -1] != 1

    # Pad positional encodings
    mol.pe = torch.cat(
        (mol.pe, torch.zeros(size=(false_nodes_needed, mol.pe.shape[1]))), dim=0
    )
    mol.pe_lap = torch.cat(
        (mol.pe_lap, torch.zeros(size=(false_nodes_needed, mol.pe_lap.shape[1]))), dim=0
    )
    mol.pe_rw = torch.cat(
        (mol.pe_rw, torch.zeros(size=(false_nodes_needed, mol.pe_rw.shape[1]))), dim=0
    )

    # Pad adjacency matrix
    new_edge_attr_adj = torch.zeros(max_size, max_size, mol.edge_attr_adj.shape[2])
    new_edge_attr_adj[:, :, -1] = 1
    new_edge_attr_adj[:num_nodes, :num_nodes] = mol.edge_attr_adj
    mol.edge_attr_adj = new_edge_attr_adj

    mol.classes_nodes = mol.x.argmax(-1)
    mol.classes_edges = mol.edge_attr_adj.argmax(-1)
    mol.num_edge_types = num_edge_types
    mol.num_nodes_types = num_nodes_types

    return mol


def make_global_features(
    mol: Data,
    mus: torch.Tensor = torch.tensor(0.0),
    stds: torch.Tensor = torch.tensor(1.0),
) -> Data:
    """
    Compute and normalize global features for nodes and edges.

    - Node-level features: summed over all nodes (excluding padded or false nodes).
    - Edge-level features: summed over the adjacency tensor (excluding false edges).
    - Includes an additional scalar feature for total node count.

    The resulting vector is standardized using the provided mean (`mus`)
    and standard deviation (`stds`).
    Args:
        mol (torch_geometric.data.Data): Graph data object.
        mus (torch.Tensor, optional): Mean tensor for feature normalization.
            Defaults to 0.0 (no centering).
        stds (torch.Tensor, optional): Standard deviation tensor for normalization.
            Zero values are replaced by 1.0 to avoid division by zero.
            Defaults to 1.0 (no scaling).

    Returns:
        torch_geometric.data.Data:
            Graph with `mol.global_features` field added (for nodes and edges features).
    """
    base_x = torch.sum(mol.x[:, :-1], dim=0)  # exclude false_nodes
    nodes_count = torch.sum(base_x)
    base_e = torch.sum(mol.edge_attr_adj[:, :, :-1], dim=[0, 1])  # exclude false_edges
    global_features = torch.cat([base_x, base_e, torch.tensor([nodes_count])])
    stds = torch.where(stds != 0, stds, 1.0)
    mol.global_features = (global_features - mus) / stds
    return mol
