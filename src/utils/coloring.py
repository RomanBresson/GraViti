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
    dataset_name: DatasetName = "ColoringBig",
) -> bool:
    """
    Check if given graph is a valid coloring graph, meaning that no node is adjacent
    to a node with the same color.

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
        dataset_name (DatasetName, optional, default=ColoringBig):
            Any available dataset between the defined dataset names (`data.constants.DatasetName`).
            Used to define valid node and edge vocabularies.

    Returns:
        bool: `True` if the given graph fulfills coloring constraints, otherwise `False`.
    """
    if x.dim() not in (1, 2):
        raise ValueError(
            f"Expected x to have shape (N,) or (N,C). Got {tuple(x.shape)}"
        )
    num_nodes = x.shape[0]

    if node_mask is None:
        node_mask = torch.ones(num_nodes, dtype=torch.bool, device=x.device)
    else:
        node_mask = node_mask.to(device=x.device).squeeze()
        if node_mask.dim() != 1 or node_mask.shape[0] != num_nodes:
            raise ValueError(
                f"Expected node_mask to have shape (N,). Got {tuple(node_mask.shape)}"
            )
        node_mask = node_mask.bool()

    # Colors: allow either indices (N,) or one-hot/logits (N,C).
    if x.dim() == 1:
        colors = x
    else:
        if x.shape[-1] == 1:
            colors = x.squeeze(-1)
        else:
            colors = x.argmax(dim=-1)
    colors = colors.to(torch.long)

    # Edges: allow either indices (N,N) or one-hot/logits (N,N,C).
    if e.dim() == 2:
        edges = e
    elif e.dim() == 3:
        if e.shape[-1] == 1:
            edges = e.squeeze(-1)
        else:
            edges = e.argmax(dim=-1)
    else:
        raise ValueError(
            f"Expected e to have shape (N,N) or (N,N,C). Got {tuple(e.shape)}"
        )
    edges = edges.to(torch.long)

    if edges.shape[0] != num_nodes or edges.shape[1] != num_nodes:
        raise ValueError(
            f"Expected e to have shape (N,N,...) with N={num_nodes}. Got {tuple(e.shape)}"
        )

    num_colors = len(get_valid_atoms(dataset_name))
    num_edge_types = len(get_valid_bonds(with_aromatic=None, dataset=dataset_name))
    index_no_edge = num_edge_types

    # Basic vocabulary sanity for real nodes only (helps catch invalid decodes early).
    real = node_mask
    if real.any():
        if not ((colors[real] >= 0) & (colors[real] < num_colors)).all().item():
            return False
        er = edges[real][:, real]
        if not ((er >= 0) & (er <= index_no_edge)).all().item():
            return False

    # Subgraph of real nodes.
    colors = colors[real]
    edges = edges[real][:, real]
    n = colors.numel()
    if n <= 1:
        return True

    # Self-loops are invalid for the Coloring constraint
    if (edges.diagonal() != index_no_edge).any().item():
        return False

    # Treat adjacency as undirected: if either direction predicts an edge, nodes are adjacent.
    adj = (edges != index_no_edge) | (edges.transpose(0, 1) != index_no_edge)
    adj = adj & ~torch.eye(n, dtype=torch.bool, device=adj.device)

    same_color = colors[:, None].eq(colors[None, :])
    violates = (adj & same_color).any().item()
    return not bool(violates)


def coloring_graph_to_string(
    x: torch.Tensor,
    node_mask: torch.Tensor = None,
    dataset_name: DatasetName = "ColoringBig",
) -> str:
    """
    Convert a Coloring graph's node colors into a compact string (e.g. "BGRY...").

    Args:
        x (torch.Tensor): Class indices `(N,)` or logits/one-hot represenation `(N,C)`.
        node_mask (torch.Tensor, optional): Boolean mask indicating positions of real nodes.
        dataset_name (DatasetName, optional, default=ColoringBig): Name of dataset for
            getting the color vocabulary.

    Returns:
        str: string representation of the color graph, each letter corresponds to the
            capital letter of node's color string (e.g. blue -> B)
    """
    if x.dim() not in (1, 2):
        raise ValueError(
            f"Expected x to have shape (N,) or (N,C). Got {tuple(x.shape)}"
        )

    vocab = get_valid_atoms(dataset_name)

    # Remove one-hot/logits encoding if present.
    if x.dim() == 2:
        if x.shape[-1] == 1:
            labels = x.squeeze(-1)
        else:
            labels = x.argmax(dim=-1)
    else:
        labels = x

    labels = labels.to(torch.long).detach()

    if node_mask is not None:
        node_mask = node_mask.to(device=labels.device).squeeze().bool()
        labels = labels[node_mask]

    chars = ""
    for v in labels.tolist():
        v_int = int(v)
        color = str(vocab[v_int])
        color = color[0].upper() if color else "?"
        chars += color

    return chars


def make_nx_graph_from_data(graph: Data, dataset_name: DatasetName) -> nx.Graph:
    """
    Construct a NetworkX `Graph` from a coloring `Data` graph, containing the
    `edge_index` and `edge_attr`, or `edge_attr_adj` attributes.

    Args:
        graph (torch_geometric.data.Data):
            `Coloring` graph from processed dataset.
        dataset_name (DatasetName, optional, default=ColoringBig):
            Any available dataset between the defined Coloring dataset names (`data.constants.DatasetName`).
            Used to define valid node/edge vocabularies.

    Returns:
        networkx.Graph: Constructed NetworkX Graph, with `color` attribute added to each node.
    """
    node_mask = getattr(graph, "node_mask", torch.ones_like(graph.x))
    x = graph.x

    # Build the adjacency matrix if it's not already defined
    if not hasattr(graph, "edge_attr_adj"):
        edge_attr = graph.edge_attr

        # One-hot encode node and edge features - as expected by `build_edge_adj_attr`
        ## One-hot encode atom types
        if x.dim() == 1 or x.shape[-1] == 1:
            num_node_classes = len(get_valid_atoms(dataset_name))
            x = F.one_hot(x.flatten(), num_node_classes)
        graph.x = x

        ## One-hot encode bond types
        if graph.edge_attr.dim() == 1:
            num_edge_types = len(get_valid_bonds(False, dataset_name))
            edge_attr = F.one_hot(edge_attr.long(), num_edge_types)
        graph.edge_attr = edge_attr

        graph = build_edge_adj_attr(graph)

    return make_nx_graph_from_xe(
        x=graph.x,
        e=graph.edge_attr_adj,
        node_mask=node_mask,
        dataset_name=dataset_name,
    )


def make_nx_graph_from_xe(
    x: torch.Tensor,
    e: torch.Tensor,
    node_mask: torch.Tensor = None,
    dataset_name: DatasetName = "ColoringBig",
) -> nx.Graph:
    """
    Construct a NetworkX `Graph` from node and edge features tensors.

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
        dataset_name (DatasetName, optional, default=ColoringBig):
            Any available dataset between the defined dataset names (`data.constants.DatasetName`).
            Used to define valid node and edge vocabularies.

    Returns:
        networkx.Graph: Constructed NetworkX Graph, with `color` attribute added to each node, and graph attributes
            `id` (see `coloring_graph_to_string`) and `valid` (see `is_valid`).
    """
    if node_mask is None:
        node_mask = torch.ones_like(x)

    # Color classes (if one-hot encoded, extract actual index)
    color_dict = {i: t for i, t in enumerate(get_valid_atoms(dataset_name))}
    color_classes = x
    if x.dim() == 2:
        color_classes = torch.argmax(x, dim=-1).cpu().detach()

    # Edge types (if one-hot encoded, extract actual index)
    edge_dict = {i: t for i, t in enumerate(get_valid_bonds(None, dataset_name))}
    edges = e
    if e.dim() == 3:
        edges = torch.argmax(e, dim=-1)
    index_no_edge = len(edge_dict)

    # Indices of non-padding nodes
    real_colors = node_mask.bool().cpu()
    color_classes = color_classes[real_colors]
    edges = edges[real_colors][:, real_colors]
    size = len(color_classes)

    # Graph creation
    G = nx.Graph()

    ## Add nodes
    for i, class_idx in enumerate(color_classes):
        color = color_dict[class_idx.item()]
        G.add_node(i, color=color)

    ## Add edges
    for i in range(size):
        for j in range(i + 1, size):
            if edges[i, j] != index_no_edge:
                G.add_edge(u_of_edge=i, v_of_edge=j)

    # Graph properties
    G.graph["id"] = coloring_graph_to_string(x, node_mask, dataset_name)
    G.graph["valid"] = is_valid(x, e, node_mask, dataset_name)

    return G


def make_nx_graphs(
    X: torch.Tensor,
    E: torch.Tensor,
    node_mask: torch.Tensor,
    dataset_name: DatasetName = "ColoringBig",
) -> List[nx.Graph]:
    """
    Construct a NetworkX `Graph` from node features `x` and edge features `e`, for a list
    of (x, e, mask) represented coloring graphs.

    Args:
        x (torch.Tensor):
            Node feature matrix (num_nodes, num_color_classes).
        e (torch.Tensor):
            Edge feature tensor (num_nodes, num_nodes, num_edge_types). (num_edge_types=1 for coloring)
        node_mask (torch.Tensor):
            A boolean tensor of shape (num_nodes) where True indicates a real node and False indicates a non-node.
        dataset_name (DatasetName, optional, default=ColoringBig):
            Any available dataset between the defined dataset names (`data.constants.DatasetName`).
            Used to define valid node and edge vocabularies.

    Returns:
        List[networkx.Graph]: Constructed NetworkX Graphs, with `color` attribute added to each node, and graph attributes
            `id` (see `coloring_graph_to_string`) and `valid` (see `is_valid`).
    """

    return [
        make_nx_graph_from_xe(x, e, n, dataset_name) for x, e, n in zip(X, E, node_mask)
    ]


def make_nx_graphs_from_batch(
    batch, dataset_name: DatasetName = "ColoringBig"
) -> List[nx.Graph]:
    return make_nx_graphs(
        X=batch["X"],
        E=batch["E"],
        node_mask=batch["node_mask"],
        dataset_name=dataset_name,
    )


def make_nx_graphs_from_outputs(
    batch, dataset_name: DatasetName = "ColoringBig"
) -> List[nx.Graph]:
    return make_nx_graphs(
        X=batch["X"].argmax(-1),
        E=batch["E"].argmax(-1),
        node_mask=batch["used_node_filter"],
        dataset_name=dataset_name,
    )


def plot_color_graph(
    G: nx.Graph,
    title: Optional[str] = None,
    with_labels: Optional[bool] = True,
):
    """
    Plot a Graph from `Coloring` dataset, constructed either with
    `make_nx_graph_from_data` or `make_nx_graph_from_xe` in order to
    contain the `color` attribute on each node.

    Args:
        G (networkx.Graph): Graph object with `color` attribute on each node.
        title (str|None): Optional title for the displayed plot.
        with_labels (bool, default=True): Show/hide node labels from the displayed plot.
    """
    node_colors = [G.nodes[i]["color"] for i in G.nodes()]

    plt.figure(figsize=(6, 6))
    nx.draw_networkx(
        G,
        pos=nx.spring_layout(G, seed=0),
        with_labels=with_labels,
        node_color=node_colors,
        node_size=650,
        edgecolors="black",
        linewidths=1.2,
        font_size=10,
    )

    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()


def plot_nx_graphs_grid(
    graphs: List[nx.Graph],
    legends: Optional[List[str]] = None,
    graphs_per_row: int = 5,
    sub_img_size: Tuple[int, int] = (250, 250),
    with_labels: bool = True,
) -> Image.Image:
    """
    Render a list of NetworkX graphs into a single grid image (PIL),
    similar to RDKit's `MolsToGridImage`.

    Args:
        graphs (List[nx.Graph]): Graphs to render.
        legends (List[str]|None): Optional per-graph labels shown as subplot titles.
        graphs_per_row (int): Number of graphs per row.
        sub_img_size (tuple[int, int]): Approximate width/height in pixels per cell.
        with_labels (bool): Whether to draw node labels.

    Returns:
        PIL.Image.Image: Rendered grid image.
    """
    if len(graphs) == 0:
        raise ValueError("No graphs to render.")

    cols = min(graphs_per_row, len(graphs))
    rows = math.ceil(len(graphs) / cols)
    dpi = 100
    fig_w = (cols * sub_img_size[0]) / dpi
    fig_h = (rows * sub_img_size[1]) / dpi
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi)

    # Normalize `axes` to a flat list
    if not isinstance(axes, (list, tuple)):
        try:
            axes = axes.flatten().tolist()
        except Exception:
            axes = [axes]
    else:
        axes = list(axes)

    for idx, G in enumerate(graphs):
        ax = axes[idx]
        node_colors = [G.nodes[i]["color"] for i in G.nodes()]
        nx.draw_networkx(
            G,
            pos=nx.spring_layout(G, seed=0),
            with_labels=with_labels,
            node_color=node_colors,
            node_size=650,
            edgecolors="black",
            linewidths=1.2,
            font_size=10,
            ax=ax,
        )
        if legends is not None and idx < len(legends):
            ax.set_title(legends[idx], fontsize=10)
        else:
            title = f"{G.graph.get('id', '')} ({'valid' if G.graph.get('valid', False) else 'invalid'})"
            ax.set_title(title, fontsize=10)
        ax.axis("off")

    for idx in range(len(graphs), len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    img = Image.open(buffer).convert("RGB")
    img = img.copy()
    img.info.clear()
    buffer.close()
    return img
