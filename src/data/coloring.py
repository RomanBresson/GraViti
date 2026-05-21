# This code is based on the official GRALE repository: https://github.com/KrzakalaPaul/GRALE

import os
import os.path as osp

from joblib import Parallel, delayed
from typing import Callable, Dict, List, Optional, Tuple

from dataclasses import dataclass

import torch
from torch_geometric.data import Data, InMemoryDataset
import networkx as nx
from scipy.ndimage import gaussian_filter
from scipy.sparse import csgraph
import numpy as np
from tqdm import tqdm

from .constants import get_valid_atoms, get_valid_bonds


@dataclass(frozen=True)
class ColoringConfig:
    train_size: int
    valid_size: int
    test_size: int
    min_nodes: int
    max_nodes: int
    pixels: int

    @property
    def split_sizes(self) -> Dict[str, int]:
        return {
            "train": self.train_size,
            "val": self.valid_size,
            "test": self.test_size,
        }


COLORING_CONFIGS = {
    "small": ColoringConfig(
        train_size=100_000,
        valid_size=2_000,
        test_size=2_000,
        min_nodes=5,
        max_nodes=10,
        pixels=32,
    ),
    "medium": ColoringConfig(
        train_size=300_000,
        valid_size=10_000,
        test_size=10_000,
        min_nodes=5,
        max_nodes=20,
        pixels=64,
    ),
    "big": ColoringConfig(
        train_size=600_000,
        valid_size=10_000,
        test_size=10_000,
        min_nodes=6,
        max_nodes=32,
        pixels=64,
    ),
}

COLORING_IMAGE_PALETTE = np.array(
    [
        [1 / 4, 1 / 4, 3 / 4],
        [1 / 4, 3 / 4, 1 / 4],
        [3 / 4, 1 / 4, 1 / 4],
        [1.0, 4 / 5, 2 / 5],
    ],
    dtype=np.float32,
)

CANONICAL_ORDERINGS = {"geom", "wl"}


class Coloring(InMemoryDataset):
    """
    Synthetic graph-coloring dataset generated from the GRALE sampling
    procedure and configuration presets (`small`, `medium`, `big`).

    Args:
        root (str): Root directory where the dataset should be saved.
        variant (str): Any of the available :obj:`Coloring` variants:
            - `small` graphs of 5-10 nodes (split 100k-2k-2k).
            - `medium` graphs of 5-15 nodes (split 300k-10k-10k).
            - `big` graphs of 6-32 nodes (split 600k-10k-10k).
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            a :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            boolean value, indicating whether the data object should be
            included in the final dataset. (default: :obj:`None`)
        log (bool, optional): Whether to print any console output while
            downloading and processing the dataset. (default: :obj:`True`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        with_images (bool, optional): Include image to the generated dataset
            apart from the graph representation.
            (default: :obj:`False`)
        canonical_ordering (str, optional): Canonical ordering strategy for nodes.
            Available options:
            - `None`: keep sampled order
            - `geom`: geometry-driven order by node position (y, x)
            - `wl`: Weisfeiler-Lehman structural ordering
            (default: :obj:None)
        wl_iterations (int, optional): Number of WL refinement iterations when using `wl` ordering.
            (default: :obj:`3`)
        seed (int, optional): Base random seed for deterministic synthetic
            data generation. If :obj:`None`, generation is non-deterministic.
            (default: :obj:`None`)
        num_workers (int, optional): Number of workers used for parallel
            processing of the dataset
            (default :obj:`os.cpu_count()-2`)
    """

    _valid_variants = ["small", "medium", "big"]
    _valid_splits = ["train", "val", "test"]

    def __init__(
        self,
        root: str,
        variant: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
        with_images: bool = False,
        canonical_ordering: Optional[str] = None,
        wl_iterations: int = 3,
        seed: Optional[int] = None,
        num_workers: int = max(os.cpu_count() - 2, 1),
    ) -> None:
        variant = variant.lower()
        assert variant in self._valid_variants
        self.variant = variant

        self.name = f"Coloring{variant.capitalize()}"
        self.valid_node_classes = get_valid_atoms(self.name)
        self.valid_edge_types = get_valid_bonds(with_aromatic=None, dataset=self.name)

        split = split.lower()
        assert split in self._valid_splits
        self.split = split

        self.num_workers = num_workers
        self.seed = seed
        self.with_images = with_images
        if (
            canonical_ordering is not None
            and canonical_ordering not in CANONICAL_ORDERINGS
        ):
            raise ValueError(
                f"Unknown canonical ordering '{canonical_ordering}'. "
                f"Expected one of: {sorted(CANONICAL_ORDERINGS)}"
            )
        self.canonical_ordering = canonical_ordering
        self.wl_iterations = max(int(wl_iterations), 1)

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            log=log,
            force_reload=force_reload,
        )

        self.load(self._split_path(self.split))

    @property
    def raw_file_names(self) -> List[str]:
        # Synthetic dataset: no raw files required.
        return []

    @property
    def processed_dir(self) -> str:
        filename = f"processed_{self.variant}"
        if self.with_images:
            filename += "_img"
        if self.canonical_ordering is not None:
            filename += f"_{self.canonical_ordering}"
            if self.canonical_ordering == "wl":
                filename += f"_it{self.wl_iterations}"

        return osp.join(self.root, filename)

    @property
    def processed_file_names(self) -> List[str]:
        return ["train.pt", "val.pt", "test.pt"]

    def _split_path(self, split: str) -> str:
        split_to_idx = {"train": 0, "val": 1, "test": 2}
        return self.processed_paths[split_to_idx[split]]

    def download(self) -> None:
        # Synthetic dataset: nothing to download.
        return

    def process(self) -> None:
        config = COLORING_CONFIGS.get(self.variant)
        assert config is not None, f"{self.variant} configuration not defined"
        split_sizes = config.split_sizes
        total_size = sum(split_sizes.values())
        sample_seeds: List[Optional[int]]
        if self.seed is None:
            sample_seeds = [None] * total_size
        else:
            rng = np.random.default_rng(self.seed)
            sample_seeds = [
                int(s) for s in rng.integers(0, np.iinfo(np.int64).max, size=total_size)
            ]

        # Generate samples
        all_samples = Parallel(n_jobs=self.num_workers)(
            delayed(_sample_graph_worker)(
                num_colors=len(self.valid_node_classes),
                min_nodes=config.min_nodes,
                max_nodes=config.max_nodes,
                pixels=config.pixels,
                seed=sample_seeds[i],
                canonical_ordering=self.canonical_ordering,
                wl_iterations=self.wl_iterations,
            )
            for i in tqdm(
                range(total_size), desc="Generating samples", disable=not self.log
            )
        )

        # Per-split processing
        offset = 0
        written = {}
        for split in self._valid_splits:
            # Get split's data
            size = split_sizes[split]
            samples = all_samples[offset : offset + size]
            offset += size
            data_list = []

            # Process data
            for graph in tqdm(
                samples, desc=f"Processing {split} dataset", disable=not self.log
            ):
                x = torch.as_tensor(graph["node_colors"], dtype=torch.long)
                num_nodes = int(x.shape[0])

                # Build directed edge index for undirected graph adjacency.
                adjacency = torch.as_tensor(graph["adjacency_matrix"], dtype=torch.long)
                src, dst = torch.where(adjacency > 0)
                edge_index = torch.stack([src, dst], dim=0)
                edge_attr = torch.zeros(edge_index.shape[1], dtype=torch.long)

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=num_nodes,
                    num_edges=int(edge_index.shape[1] // 2),
                )

                if self.with_images:
                    image = torch.as_tensor(graph["image"], dtype=torch.uint8)
                    image = image.permute(2, 0, 1).contiguous()  # (C, H, W)
                    data.image = image

                # Apply pre-filter if available
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                # Apply pre-transform if available
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            written[split] = len(data_list)
            self.save(data_list, self._split_path(split))

        # Log dataset stats
        if self.log:
            print("=" * 60)
            print(f"Coloring dataset variant '{self.variant}' generated:")
            print(
                f"Canonical ordering={self.canonical_ordering}"
                f"{f' (WL iterations={self.wl_iterations})' if self.canonical_ordering =='wl' else ''}"
            )
            print(
                f"Train={written['train']:,} | "
                f"Val={written['val']:,} | "
                f"Test={written['test']:,}"
            )
            print(f"Dataset saved to: {self.processed_dir}")
            print("=" * 60)


def _canonical_node_permutation(
    node_colors: np.ndarray,
    adjacency_matrix: np.ndarray,
    node_positions: np.ndarray,
    ordering: Optional[str] = None,
    wl_iterations: int = 3,
) -> Optional[np.ndarray]:
    """
    Build a deterministic permutation of node indices for one sampled graph.

    The goal is to reduce arbitrary node-index variance across isomorphic graphs,
    so reconstruction metrics based on absolute node positions become meaningful.
    Depending on `ordering`, we use:
    - `geom`: only geometry (`y`, `x`) and light tie-breakers
    - `wl`: structural 1-WL labels, graph-distance and geometry tie-breakers

    Args:
        node_colors (np.ndarray): Color label per node (shape `[N]`).
        adjacency_matrix (np.ndarray): Binary adjacency matrix (shape `[N, N]`).
        node_positions (np.ndarray): Node coordinates `(x, y)` (shape `[N, 2]`).
        ordering (str, optional): Canonical ordering strategy for nodes.
            Available options:
            - `None`: keep sampled order
            - `geom`: geometry-driven order by node position (y, x)
            - `wl`: Weisfeiler-Lehman structural ordering
            (default: :obj:None)
        wl_iterations (int, optional): Number of WL refinement iterations when using `wl` ordering.
            (default: :obj:`3`)

    Returns:
        np.ndarray|None:
            Permutation `perm` such that `new_nodes = old_nodes[perm]`.
            Returns `None` when no reordering is requested.
    """
    if ordering is None:
        return None

    n_nodes = int(node_colors.shape[0])
    if n_nodes <= 1:
        return np.arange(n_nodes, dtype=np.int32)

    if ordering not in CANONICAL_ORDERINGS:
        raise ValueError(
            f"Unknown canonical ordering '{ordering}'. "
            f"Expected one of: {sorted(CANONICAL_ORDERINGS)}"
        )

    deg = adjacency_matrix.sum(axis=1).astype(np.int32, copy=False)
    x = node_positions[:, 0]
    y = node_positions[:, 1]

    # Geometry-driven order top-left to bottom-right, then degree, then index
    if ordering == "geom":
        return np.asarray(
            sorted(range(n_nodes), key=lambda i: (y[i], x[i], deg[i], i)),
            dtype=np.int32,
        )

    # WL labels summarize local structure; they are the main sorting key.
    wl_labels = _wl_refinement_labels(
        adjacency_matrix=adjacency_matrix,
        node_labels=node_colors.astype(np.int64, copy=False),
        iterations=wl_iterations,
    )

    # Smallest node by WL label (break ties by degree and then top-left to bottom-right ordering)
    root = min(
        range(n_nodes),
        key=lambda i: (
            int(wl_labels[i]),
            int(deg[i]),
            float(y[i]),
            float(x[i]),
            int(i),
        ),
    )
    d_root = _bfs_distances(adjacency_matrix, root)

    # Use a second anchor to further break symmetries that 1-WL may leave tied.
    anchor = max(
        range(n_nodes),
        key=lambda i: (
            int(d_root[i]),
            int(deg[i]),
            -int(wl_labels[i]),
            int(i),
        ),
    )
    d_anchor = _bfs_distances(adjacency_matrix, anchor)

    # Generate permutation of initial nodes
    key_fn = lambda i: (
        int(wl_labels[i]),
        int(d_root[i]),
        int(d_anchor[i]),
        int(deg[i]),
        float(y[i]),
        float(x[i]),
        int(i),
    )
    return np.asarray(sorted(range(n_nodes), key=key_fn), dtype=np.int32)


def _wl_refinement_labels(
    adjacency_matrix: np.ndarray,
    node_labels: np.ndarray,
    iterations: int = 3,
) -> np.ndarray:
    """
    Run deterministic 1-WL refinement starting from given node labels.

    Each iteration builds a node signature:
        (current_label, sorted(neighbor_labels))
    and compresses signatures into compact integer IDs.
    This captures progressively larger neighborhoods.

    Note:
        The refinement may stabilize before `iterations`; in that case we
        terminate early (no further label changes are possible).

    Args:
        adjacency_matrix (np.ndarray): Binary adjacency matrix (shape `[N, N]`).
        node_labels (np.ndarray): Initial labels (shape `[N]`).
        iterations (int, optional): Maximum number of WL iterations.

    Returns:
        np.ndarray: Final WL labels (shape `[N]`).
    """
    labels = node_labels.astype(np.int64, copy=True)
    iterations = max(int(iterations), 1)
    neighbors = [np.where(adjacency_matrix[i] > 0)[0] for i in range(labels.shape[0])]

    for _ in range(iterations):
        signatures: List[Tuple[int, Tuple[int, ...]]] = []
        for i in range(labels.shape[0]):
            # Sort neighbor labels to make signatures invariant to node ordering.
            neigh_labels = tuple(sorted(int(labels[j]) for j in neighbors[i]))
            signatures.append((int(labels[i]), neigh_labels))

        # Deterministic compression of signatures into dense IDs.
        vocab = {sig: idx for idx, sig in enumerate(sorted(set(signatures)))}
        new_labels = np.asarray([vocab[sig] for sig in signatures], dtype=np.int64)

        # Early exit when refinement has converged.
        if np.array_equal(new_labels, labels):
            break

        labels = new_labels

    return labels


def _bfs_distances(adjacency_matrix: np.ndarray, source: int) -> np.ndarray:
    """
    Compute unweighted shortest-path distances from one source via BFS.

    Distances are used as global tie-breakers in canonical ordering.
    Unreachable nodes (disconnected components) get sentinel distance `N + 1`.

    Args:
        adjacency_matrix (np.ndarray): Binary adjacency matrix (shape `[N, N]`).
        source (int): Source node index.

    Returns:
        np.ndarray: Distance vector (shape `[N]`).
    """
    n_nodes = int(adjacency_matrix.shape[0])
    unreachable = n_nodes + 1
    dist = np.full(n_nodes, unreachable, dtype=np.int32)
    dist[source] = 0

    queue = [source]
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        neigh = np.where(adjacency_matrix[node] > 0)[0]
        for nxt in neigh:
            if dist[nxt] == unreachable:
                dist[nxt] = dist[node] + 1
                queue.append(int(nxt))

    return dist


def _sample_graph_worker(
    num_colors: int,
    min_nodes: int = 2,
    max_nodes: int = 10,
    pixels: int = 32,
    sigma_filter: float = 0.4,
    sigma_noise: float = 0.02,
    node_weight_threshold: Optional[float] = None,
    edge_weight_threshold: Optional[float] = None,
    precompute_shortest_paths: bool = False,
    seed: Optional[int] = None,
    canonical_ordering: Optional[str] = None,
    wl_iterations: int = 3,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Sample a synthetic colored graph via a Voronoi-like pixel partition.

    This code is based on the coloring sample from the official GRALE 
    repository: https://github.com/KrzakalaPaul/GRALE

    Args:
        num_colors (int): Number of available node colors.
        min_nodes (int, optional): Minimum number of sampled nodes.
        max_nodes (int, optional): Maximum number of sampled nodes.
        pixels (int, optional): Pixel resolution used to build regions.
        sigma_filter (float, optional): Gaussian blur sigma for image synthesis.
        sigma_noise (float, optional): Gaussian noise std for image synthesis.
        node_weight_threshold (float | None, optional): Reject samples where any
            node weight falls below this value.
        edge_weight_threshold (float | None, optional): Threshold used to convert
            edge weights to a binary adjacency matrix.
        precompute_shortest_paths (bool, optional): Whether to compute all-pairs
            shortest-path distances.
        seed (int | None, optional): Seed for deterministic sampling.
        canonical_ordering (str, optional): Node ordering strategy (`geom`, `wl`).
        wl_iterations (int, optional): Number of WL refinement steps for `wl`.

    Returns:
        Dict[str, Optional[np.ndarray]]: Graph dictionary with
        `node_colors`, `adjacency_matrix`, optionally `SP_matrix`, and `image`.
    """
    rng = np.random.default_rng(seed)
    node_weight_normalization = float(pixels * pixels)
    edge_weight_normalization = float(pixels)
    pixel_axis = np.linspace(0.0, 1.0, pixels)

    while True:
        # Sample graph size and Voronoi centroids.
        n_nodes = int(rng.integers(min_nodes, max_nodes + 1))
        centroids = rng.uniform(0.0, 1.0, (2, n_nodes))

        # Assign each pixel to the nearest centroid (L1 distance).
        dists = np.abs(pixel_axis[:, None, None] - centroids[None, 0, :]) + np.abs(
            pixel_axis[None, :, None] - centroids[1, None, :]
        )
        closest = np.argmin(dists, axis=2).astype(np.int32, copy=False)

        # Count pixels per node and resample if any node is empty.
        counts = np.bincount(closest.ravel(), minlength=n_nodes).astype(np.float64)
        if np.any(counts == 0):
            continue

        node_weights = counts / node_weight_normalization
        if node_weight_threshold is not None and np.any(
            node_weights < node_weight_threshold
        ):
            continue

        # Build undirected weighted edges from horizontal/vertical boundaries.
        horizontal = np.column_stack((closest[:, :-1].ravel(), closest[:, 1:].ravel()))
        vertical = np.column_stack((closest[:-1, :].ravel(), closest[1:, :].ravel()))
        edges = np.vstack((horizontal, vertical))

        # Keep only boundaries crossing different regions.
        edges = edges[edges[:, 0] != edges[:, 1]]
        if edges.size > 0:
            # Canonicalize undirected edges and aggregate multiplicities.
            edges = np.sort(edges, axis=1)
            unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
            edge_weights = edge_counts.astype(np.float64) / edge_weight_normalization
        else:
            unique_edges = np.empty((0, 2), dtype=np.int32)
            edge_weights = np.empty((0,), dtype=np.float64)

        if edge_weight_threshold is not None and np.any(edge_weights > 1.0):
            continue

        graph = nx.Graph()
        graph.add_nodes_from(
            (node, {"pos": centroids[:, node], "weight": node_weights[node]})
            for node in range(n_nodes)
        )
        if unique_edges.shape[0] > 0:
            graph.add_weighted_edges_from(
                (
                    int(u),
                    int(v),
                    float(weight),
                )
                for (u, v), weight in zip(unique_edges, edge_weights)
            )

        coloring = _color_graph(graph, num_colors, rng=rng)
        node_colors = np.fromiter(
            (coloring[node] for node in range(n_nodes)), dtype=np.uint8, count=n_nodes
        )

        adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
        if unique_edges.shape[0] > 0:
            edge_mask = (
                edge_weights > edge_weight_threshold
                if edge_weight_threshold is not None
                else np.ones_like(edge_weights, dtype=bool)
            )
            kept_edges = unique_edges[edge_mask]
            if kept_edges.shape[0] > 0:
                adjacency_matrix[kept_edges[:, 0], kept_edges[:, 1]] = 1
                adjacency_matrix[kept_edges[:, 1], kept_edges[:, 0]] = 1

        node_positions = centroids.T.astype(np.float32, copy=False)  # (n_nodes, 2)
        permutation = _canonical_node_permutation(
            node_colors=node_colors,
            adjacency_matrix=adjacency_matrix,
            node_positions=node_positions,
            ordering=canonical_ordering,
            wl_iterations=wl_iterations,
        )

        if permutation is not None:
            inverse_perm = np.empty(n_nodes, dtype=np.int32)
            inverse_perm[permutation] = np.arange(n_nodes, dtype=np.int32)

            node_colors = node_colors[permutation]
            adjacency_matrix = adjacency_matrix[np.ix_(permutation, permutation)]
            node_positions = node_positions[permutation]
            closest = inverse_perm[closest]

        if num_colors > COLORING_IMAGE_PALETTE.shape[0]:
            raise ValueError(
                f"Palette supports at most {COLORING_IMAGE_PALETTE.shape[0]} colors; "
                f"got {num_colors}"
            )
        node_rgb = COLORING_IMAGE_PALETTE[node_colors]
        image = np.take(node_rgb, closest, axis=0)
        image = gaussian_filter(image, sigma_filter, mode="nearest")
        image = image + rng.normal(0.0, sigma_noise, size=image.shape)
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255).astype(np.uint8)

        if precompute_shortest_paths:
            adjacency_sparse = csgraph.csgraph_from_dense(adjacency_matrix)
            sp_matrix = csgraph.shortest_path(
                adjacency_sparse, directed=False, unweighted=True
            ).astype(np.uint8)
        else:
            sp_matrix = None

        sample = {
            "node_colors": node_colors,
            "adjacency_matrix": adjacency_matrix,
            "SP_matrix": sp_matrix,
            "image": image,
        }
        return sample


def _color_graph(
    graph: nx.Graph, num_colors: int, rng: np.random.Generator
) -> Dict[int, int]:
    """
    Color a graph with randomized greedy retries until a valid coloring is found.

    This code is based on the graph coloring from the official GRALE 
    repository: https://github.com/KrzakalaPaul/GRALE

    Args:
        graph (nx.Graph): Undirected graph whose nodes are integer ids.
        num_colors (int): Number of available colors.
        rng (np.random.Generator): Random number generator for deterministic
            color sampling.

    Returns:
        Dict[int, int]: Mapping node id -> color id.
    """
    assert num_colors > 0, "num_colors must be positive"

    nodes = list(graph.nodes)
    neighborhoods = {node: tuple(graph.neighbors(node)) for node in nodes}

    while True:
        coloring = {}
        for node in nodes:
            used_colors = set()
            for neighbor in neighborhoods[node]:
                neighbor_color = coloring.get(neighbor)
                if neighbor_color is not None:
                    used_colors.add(neighbor_color)
            valid_colors = [c for c in range(num_colors) if c not in used_colors]

            if not valid_colors:
                break

            color_idx = int(rng.integers(0, len(valid_colors)))
            coloring[node] = valid_colors[color_idx]
        else:
            return coloring
