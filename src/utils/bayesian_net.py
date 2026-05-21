import torch
import torch.nn.functional as F
from torch_geometric.data import Data

import networkx as nx
import matplotlib.pyplot as plt

import io
import math
from typing import List, Optional, Tuple
from PIL import Image

from ..data.features import build_edge_adj_attr
from ..data.constants import DatasetName, get_valid_atoms, get_valid_bonds


def is_valid(
    x: torch.Tensor,
    e: torch.Tensor,
    node_mask: torch.Tensor = None,
    dataset_name: DatasetName = "BayesianNetAsia",
    directed: bool = False,
    require_all_node_types: bool = True,
) -> bool:
    """
    Check whether a decoded graph is a valid Bayesian-network graph.

    For a BN dataset, validity means:
    - real nodes use valid variable ids
    - every variable id appears exactly once (by default)
    - edge labels are valid and self-loops are absent
    - the interpreted directed graph is acyclic

    Args:
        x (torch.Tensor):
            Node feature tensor.
        e (torch.Tensor):
            Edge feature tensor (adjacency matrix). The value of `e[i,j]` represents the edge for
            nodes `i->j`. If `e[i,j]=number of possible edge types for this dataset`, it represents the absence
            of an edge between nodes i and j. Otherwise, it represents the edge type.
        node_mask (torch.Tensor, optional, default=None):
            A boolean tensor of shape (num_nodes) where True indicates a real node and False indicates a non-node.
            If `None` (default), all nodes in `x` are considered as real nodes.
        dataset_name (DatasetName, optional, default=BayesianNetAsia):
            Any available dataset between the defined BN dataset names (`data.constants.DatasetName`).
            Used to define valid node and edge vocabularies.
        directed (bool, optional, default=False):
            If `directed=False` input edges are treated as an undirected skeleton and
            oriented from smaller variable id to larger variable id.
            If `directed=True`, edge directions are preserved and backward edges are
            rejected when variable ids are unique.
        require_all_node_types (bool, optional, default=True):
            If `require_all_node_types=True`, all node types are expected to be found in
            `x` (exactly once).

    Returns:
        bool: `True` if the given graph fulfills all constraints, otherwise `False`.
    """
    labels, edges, mask = _prepare_x_e_mask(x, e, node_mask, dataset_name)
    labels = labels[mask]
    edges = edges[mask][:, mask]

    num_node_types = len(get_valid_atoms(dataset_name))
    num_edge_types = len(get_valid_bonds(with_aromatic=None, dataset=dataset_name))
    index_no_edge = num_edge_types

    if labels.numel() == 0:
        return False

    if not ((labels >= 0) & (labels < num_node_types)).all().item():
        return False

    if not ((edges >= 0) & (edges <= index_no_edge)).all().item():
        return False

    if (edges.diagonal() != index_no_edge).any().item():
        return False

    unique_labels = labels.unique(sorted=True)
    if unique_labels.numel() != labels.numel():
        return False

    if require_all_node_types:
        expected = torch.arange(num_node_types, device=labels.device)
        if labels.numel() != num_node_types or not torch.equal(unique_labels, expected):
            return False

    adjacency = _directed_adjacency_from_labels(
        labels=labels,
        edges=edges,
        no_edge_index=index_no_edge,
        directed=directed,
    )

    if directed:
        label_order = labels[:, None] < labels[None, :]
        if (adjacency & ~label_order).any().item():
            return False

    graph = nx.DiGraph()
    graph.add_nodes_from(range(labels.numel()))
    src, dst = torch.where(adjacency)
    graph.add_edges_from((int(i), int(j)) for i, j in zip(src, dst))
    return nx.is_directed_acyclic_graph(graph)


def bayesian_net_graph_to_string(
    x: torch.Tensor,
    e: torch.Tensor,
    node_mask: torch.Tensor = None,
    dataset_name: DatasetName = "BayesianNetAsia",
    directed: bool = False,
) -> str:
    """
    Convert a Bayesian-network graph into a string representation, returning
    the upper-triangular parent-indicator bits ordered by variable id,
    e.g. `||1|00|101|...`.

    Invalid or partially decoded graphs still receive a deterministic fallback
    id containing the node labels and dense edge classes.

    Args:
        x (torch.Tensor): Class indices `(N,)` or logits/one-hot represenation `(N,C)`.
        e (torch.Tensor): Edge feature tensor (adjacency matrix) `(N,N)` or logits/one-hot
            represenation `(N,N,C)`.
        node_mask (torch.Tensor, optional): Boolean mask indicating positions of real nodes.
        dataset_name (DatasetName, optional, default=BayesianNetAsia): Any available dataset
            between the defined BN dataset names (`data.constants.DatasetName`). Used to define
            valid node and edge vocabularies.
        directed (bool, optional, default=False):
            If `directed=False` input edges are treated as an undirected skeleton and
            oriented from smaller variable id to larger variable id.
            If `directed=True`, edge directions are preserved and backward edges are
            rejected when variable ids are unique.

    Returns:
        str: string representation of the BN graph
    """
    labels, edges, mask = _prepare_x_e_mask(x, e, node_mask, dataset_name)
    labels = labels[mask]
    edges = edges[mask][:, mask]

    num_node_types = len(get_valid_atoms(dataset_name))
    num_edge_types = len(get_valid_bonds(with_aromatic=None, dataset=dataset_name))
    index_no_edge = num_edge_types
    valid = is_valid(
        x=x,
        e=e,
        node_mask=node_mask,
        dataset_name=dataset_name,
        directed=directed,
        require_all_node_types=True,
    )

    if not valid:
        label_part = ",".join(str(int(v)) for v in labels.detach().cpu().tolist())
        edge_part = "".join(str(int(v)) for v in edges.detach().cpu().flatten().tolist())
        return f"{dataset_name}:invalid:x={label_part}:e={edge_part}"

    order = torch.argsort(labels)
    labels = labels[order]
    edges = edges[order][:, order]
    adjacency = _directed_adjacency_from_labels(
        labels=labels,
        edges=edges,
        no_edge_index=index_no_edge,
        directed=directed,
    )

    parts = []
    for child in range(num_node_types):
        parent_bits = [
            "1" if adjacency[parent, child].item() else "0" for parent in range(child)
        ]
        parts.append("".join(parent_bits))

    return "|".join(parts)


def make_nx_graph_from_data(
    graph: Data,
    dataset_name: DatasetName = "BayesianNetAsia",
    directed: bool = False,
) -> nx.DiGraph:
    """
    Construct a NetworkX `DiGraph` from a Bayesian-network `Data` object.

    Raw dataset objects may contain `directed_edge_index`. In that case, the
    original DAG edges are used directly.

    Args:
        graph (torch_geometric.data.Data):
            `BayesianNet` graph from processed dataset.
        dataset_name (DatasetName, optional, default=BayesianNetAsia):
            Any available dataset between the defined BN dataset names (`data.constants.DatasetName`).
            Used to define valid node/edge vocabularies.

    Returns:
        networkx.DiGraph: Constructed NetworkX Directed Graph, with `variable` and `label` attributes
        added to each node, and graph attributes `id` (see `bayesian_net_graph_to_string`) and `valid`
        (see `is_valid`).
    """
    if hasattr(graph, "directed_edge_index") and not hasattr(graph, "edge_attr_adj"):
        labels = _node_labels_from_x(graph.x)
        node_mask = getattr(graph, "node_mask", None)
        if node_mask is None:
            node_mask = torch.ones(
                labels.shape[0], dtype=torch.bool, device=labels.device
            )
        else:
            node_mask = node_mask.to(device=labels.device).squeeze().bool()

        G = nx.DiGraph()
        for idx, label in enumerate(labels[node_mask].detach().cpu().tolist()):
            G.add_node(idx, variable=int(label), label=str(int(label)))

        old_to_new = {}
        for new_idx, old_idx in enumerate(
            torch.where(node_mask)[0].detach().cpu().tolist()
        ):
            old_to_new[int(old_idx)] = int(new_idx)

        edge_index = graph.directed_edge_index.detach().cpu()
        for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            if int(src) in old_to_new and int(dst) in old_to_new:
                G.add_edge(old_to_new[int(src)], old_to_new[int(dst)])

        G.graph["id"] = _nx_graph_to_parent_string(G, dataset_name)
        num_node_types = len(get_valid_atoms(dataset_name))
        variables = sorted(int(attrs["variable"]) for _, attrs in G.nodes(data=True))
        has_all_variables = variables == list(range(num_node_types))
        forward_edges = all(
            int(G.nodes[src]["variable"]) < int(G.nodes[dst]["variable"])
            for src, dst in G.edges()
        )
        G.graph["valid"] = (
            has_all_variables and forward_edges and nx.is_directed_acyclic_graph(G)
        )
        return G

    node_mask = getattr(graph, "node_mask", None)
    x = graph.x

    # Build the adjacency matrix if it's not already defined
    if not hasattr(graph, "edge_attr_adj"):
        # One-hot encode node and edge features - as expected by `build_edge_adj_attr`
        ## One-hot encode atom types
        if x.dim() == 1 or x.shape[-1] == 1:
            num_node_classes = len(get_valid_atoms(dataset_name))
            x = F.one_hot(x.flatten(), num_node_classes)
        graph.x = x

        ## One-hot encode bond types
        edge_attr = graph.edge_attr
        if edge_attr.dim() == 1:
            num_edge_types = len(get_valid_bonds(None, dataset_name))
            edge_attr = F.one_hot(edge_attr.long(), num_edge_types)
        graph.edge_attr = edge_attr

        graph = build_edge_adj_attr(graph)

    return make_nx_graph_from_xe(
        x=graph.x,
        e=graph.edge_attr_adj,
        node_mask=node_mask,
        dataset_name=dataset_name,
        directed=directed,
    )


def make_nx_graph_from_xe(
    x: torch.Tensor,
    e: torch.Tensor,
    node_mask: torch.Tensor = None,
    dataset_name: DatasetName = "BayesianNetAsia",
    directed: bool = False,
) -> nx.DiGraph:
    """
    Construct a NetworkX `DiGraph` from node and edge features tensors.

    Args:
        x (torch.Tensor):
            Node feature tensor.
        e (torch.Tensor):
            Edge feature tensor (adjacency matrix). The value of `e[i,j]` represents the edge for
            nodes `i->j`. If `e[i,j]=number of possible edge types for this dataset`, it represents the absence
            of an edge between nodes i and j. Otherwise, it represents the edge type.
        node_mask (torch.Tensor, optional, default=None):
            A boolean tensor of shape (num_nodes) where True indicates a real node and False indicates a non-node.
            If `None` (default), all nodes in `x` are considered as real nodes.
        dataset_name (DatasetName, optional, default=BayesianNetAsia):
            Any available dataset between the defined BN dataset names (`data.constants.DatasetName`).
            Used to define valid node and edge vocabularies.
        directed (bool, optional, default=False):
            If `directed=False` input edges are treated as an undirected skeleton and
            oriented from smaller variable id to larger variable id.
            If `directed=True`, edge directions are preserved and backward edges are
            rejected when variable ids are unique.

    Returns:
        networkx.DiGraph: Constructed NetworkX Directed Graph, with `variable` and `label` attributes
        added to each node, and graph attributes `id` (see `bayesian_net_graph_to_string`) and `valid`
        (see `is_valid`).
    """
    labels, edges, mask = _prepare_x_e_mask(x, e, node_mask, dataset_name)
    labels = labels[mask]
    edges = edges[mask][:, mask]

    num_edge_types = len(get_valid_bonds(with_aromatic=None, dataset=dataset_name))
    index_no_edge = num_edge_types
    adjacency = _directed_adjacency_from_labels(
        labels=labels,
        edges=edges,
        no_edge_index=index_no_edge,
        directed=directed,
    )

    # Graph creation
    G = nx.DiGraph()

    ## Add nodes
    for idx, label in enumerate(labels.detach().cpu().tolist()):
        label = int(label)
        G.add_node(idx, variable=label, label=str(label))

    ## Add edges
    src, dst = torch.where(adjacency)
    G.add_edges_from((int(i), int(j)) for i, j in zip(src, dst))

    # Graph properties
    G.graph["id"] = bayesian_net_graph_to_string(
        x, e, node_mask, dataset_name, directed
    )
    G.graph["valid"] = is_valid(x, e, node_mask, dataset_name, directed)

    return G


def make_nx_graphs(
    X: torch.Tensor,
    E: torch.Tensor,
    node_mask: torch.Tensor,
    dataset_name: DatasetName = "BayesianNetAsia",
    directed: bool = False,
) -> List[nx.DiGraph]:
    """
    Construct NetworkX `DiGraph`s from node features `x` and edge features `e`, for a list
    of (x, e, mask) represented BN graphs.

    Args:
        x (torch.Tensor):
            Node feature matrix (num_graphs, num_nodes, num_classes).
        e (torch.Tensor):
            Edge feature tensor (num_graphs, num_nodes, num_nodes, 1).
        node_mask (torch.Tensor):
            A boolean tensor of shape (num_nodes) where True indicates a real node and False indicates a non-node.
        dataset_name (DatasetName, optional, default=BayesianNetAsia):
            Any available dataset between the defined BN dataset names (`data.constants.DatasetName`).
            Used to define valid node and edge vocabularies.
        directed (bool, optional, default=False):
            If `directed=False` input edges are treated as an undirected skeleton and
            oriented from smaller variable id to larger variable id.
            If `directed=True`, edge directions are preserved and backward edges are
            rejected when variable ids are unique.

    Returns:
       List[networkx.DiGraph]: Constructed NetworkX Directed Graphs, with `variable` and `label` attributes
        added to each node, and graph attributes `id` (see `bayesian_net_graph_to_string`) and `valid`
        (see `is_valid`).
    """
    return [
        make_nx_graph_from_xe(x, e, n, dataset_name, directed)
        for x, e, n in zip(X, E, node_mask)
    ]


def make_nx_graphs_from_batch(
    batch,
    dataset_name: DatasetName = "BayesianNetAsia",
    directed: bool = False,
) -> List[nx.DiGraph]:
    return make_nx_graphs(
        X=batch["X"],
        E=batch["E"],
        node_mask=batch["node_mask"],
        dataset_name=dataset_name,
        directed=directed,
    )


def make_nx_graphs_from_outputs(
    batch,
    dataset_name: DatasetName = "BayesianNetAsia",
    directed: bool = False,
) -> List[nx.DiGraph]:
    return make_nx_graphs(
        X=batch["X"].argmax(-1),
        E=batch["E"].argmax(-1),
        node_mask=batch["used_node_filter"],
        dataset_name=dataset_name,
        directed=directed,
    )


def plot_bayesian_net(
    G: nx.DiGraph,
    title: Optional[str] = None,
    with_labels: bool = True,
):
    """
    Plot a Bayesian-network `DiGraph`, constructed either with
    `make_nx_graph_from_data` or `make_nx_graph_from_xe`.

    Args:
        G (networkx.DiGraph): Graph object.
        title (str|None): Optional title for the displayed plot.
        with_labels (bool, default=True): Show/hide node labels from the displayed plot.
    """
    pos = _bayesian_net_layout(G)
    labels = {node: G.nodes[node].get("label", str(node)) for node in G.nodes()}
    node_colors = [
        "#d7e8ff" if G.graph.get("valid", False) else "#ffd8d8" for _ in G.nodes()
    ]

    plt.figure(figsize=(7, 4))
    nx.draw_networkx(
        G,
        pos=pos,
        labels=labels if with_labels else None,
        with_labels=with_labels,
        node_color=node_colors,
        node_size=650,
        edgecolors="black",
        linewidths=1.1,
        arrows=True,
        arrowsize=16,
        font_size=10,
    )
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()


def plot_nx_graphs_grid(
    graphs: List[nx.DiGraph],
    legends: Optional[List[str]] = None,
    graphs_per_row: int = 5,
    sub_img_size: Tuple[int, int] = (250, 250),
    with_labels: bool = True,
) -> Image.Image:
    """
    Render a list of NetworkX directed graphs into a single grid image (PIL),
    similar to RDKit's `MolsToGridImage`.
    """
    if len(graphs) == 0:
        raise ValueError("No graphs to render.")

    cols = min(graphs_per_row, len(graphs))
    rows = math.ceil(len(graphs) / cols)
    dpi = 100
    fig_w = (cols * sub_img_size[0]) / dpi
    fig_h = (rows * sub_img_size[1]) / dpi
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi)

    try:
        axes = axes.flatten().tolist()
    except Exception:
        axes = [axes]

    for idx, G in enumerate(graphs):
        ax = axes[idx]
        pos = _bayesian_net_layout(G)
        labels = {node: G.nodes[node].get("label", str(node)) for node in G.nodes()}
        node_colors = [
            "#d7e8ff" if G.graph.get("valid", False) else "#ffd8d8" for _ in G.nodes()
        ]
        nx.draw_networkx(
            G,
            pos=pos,
            labels=labels if with_labels else None,
            with_labels=with_labels,
            node_color=node_colors,
            node_size=620,
            edgecolors="black",
            linewidths=1.0,
            arrows=True,
            arrowsize=14,
            font_size=9,
            ax=ax,
        )
        if legends is not None and idx < len(legends):
            ax.set_title(legends[idx], fontsize=10)
        else:
            state = "valid" if G.graph.get("valid", False) else "invalid"
            ax.set_title(f"{state}: {G.graph.get('id', '')}", fontsize=8)
        ax.axis("off")

    for idx in range(len(graphs), len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image = image.copy()
    image.info.clear()
    buffer.close()
    return image


"""
Helpers
"""


def _prepare_x_e_mask(
    x: torch.Tensor,
    e: torch.Tensor,
    node_mask: torch.Tensor,
    dataset_name: DatasetName,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = _node_labels_from_x(x)
    edges = _edge_classes_from_e(e)
    num_nodes = labels.shape[0]

    if edges.shape[0] != num_nodes or edges.shape[1] != num_nodes:
        raise ValueError(
            f"Expected e to have shape (N,N,...) with N={num_nodes}. "
            f"Got {tuple(edges.shape)}"
        )

    if node_mask is None:
        mask = torch.ones(num_nodes, dtype=torch.bool, device=labels.device)
    else:
        mask = node_mask.to(device=labels.device).squeeze().bool()
        if mask.dim() != 1 or mask.shape[0] != num_nodes:
            raise ValueError(
                f"Expected node_mask to have shape (N,). Got {tuple(mask.shape)}"
            )

    return labels.to(torch.long), edges.to(torch.long), mask


def _node_labels_from_x(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x
    if x.dim() == 2:
        if x.shape[-1] == 1:
            return x.squeeze(-1)
        return x.argmax(dim=-1)
    raise ValueError(f"Expected x to have shape (N,) or (N,C). Got {tuple(x.shape)}")


def _edge_classes_from_e(e: torch.Tensor) -> torch.Tensor:
    if e.dim() == 2:
        return e
    if e.dim() == 3:
        if e.shape[-1] == 1:
            return e.squeeze(-1)
        return e.argmax(dim=-1)
    raise ValueError(f"Expected e to have shape (N,N) or (N,N,C). Got {tuple(e.shape)}")


def _directed_adjacency_from_labels(
    labels: torch.Tensor,
    edges: torch.Tensor,
    no_edge_index: int,
    directed: bool,
) -> torch.Tensor:
    edge_present = edges != no_edge_index
    edge_present = edge_present & ~torch.eye(
        edge_present.shape[0], dtype=torch.bool, device=edge_present.device
    )

    if directed:
        return edge_present

    skeleton = edge_present | edge_present.transpose(0, 1)
    label_order = labels[:, None] < labels[None, :]
    return skeleton & label_order


def _bayesian_net_layout(G: nx.DiGraph) -> dict:
    layers = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if not preds:
            layers[node] = 0
        else:
            layers[node] = max(layers[p] for p in preds) + 1

    pos = {}
    layer_nodes = {}
    for node, layer in layers.items():
        layer_nodes.setdefault(layer, []).append(node)

    for layer, nodes in layer_nodes.items():
        for i, node in enumerate(nodes):
            pos[node] = (i, -layer)  # x spreads, y is depth

    return pos


def _nx_graph_to_parent_string(G: nx.DiGraph, dataset_name: DatasetName) -> str:
    num_node_types = len(get_valid_atoms(dataset_name))
    var_to_node = {
        int(attrs.get("variable", node)): node for node, attrs in G.nodes(data=True)
    }
    if set(var_to_node.keys()) != set(range(num_node_types)):
        return f"{dataset_name}:invalid"

    parts = []
    for child in range(num_node_types):
        child_node = var_to_node[child]
        bits = []
        for parent in range(child):
            parent_node = var_to_node[parent]
            bits.append("1" if G.has_edge(parent_node, child_node) else "0")
        parts.append("".join(bits))
    return f"{dataset_name}:" + "|".join(parts)
