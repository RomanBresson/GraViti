# This code is based on the 'asia_200k' dataset introduced in
# the official D-VAE repository: https://github.com/muhanzhang/D-VAE

import ast
import os
import os.path as osp
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from .constants import get_valid_atoms


@dataclass(frozen=True)
class BayesianNetworkConfig:
    raw_file_name: str
    url: str
    num_node_types: int
    line_to_data: Callable[[str, bool, Optional[int]], Data]


def _asia_dvae_line_to_data_dispatcher(
    line: str,
    directed: bool,
    num_node_types: Optional[int],
) -> Data:
    return _asia_dvae_line_to_data(
        line=line,
        directed=directed,
        num_node_types=num_node_types,
    )


BAYESIAN_NETWORK_CONFIGS: Dict[str, BayesianNetworkConfig] = {
    "asia": BayesianNetworkConfig(
        raw_file_name="asia_200k.txt",
        url="https://raw.githubusercontent.com/muhanzhang/D-VAE/master/data/asia_200k.txt",
        num_node_types=len(get_valid_atoms("BayesianNetAsia")),
        line_to_data=_asia_dvae_line_to_data_dispatcher,
    ),
}


class BayesianNetwork(InMemoryDataset):
    """
    Generic Bayesian-network graph dataset.

    The class is variant-driven, meaning various sources can be supported by being
    registered in ``BAYESIAN_NETWORK_CONFIGS`` with their raw filename, URL,
    number of node types, and a row processor that converts one raw record into
    a :class:`~torch_geometric.data.Data` object.

    Implemented variants:
    - ``"asia"``: based on D-VAE's ``asia_200k.txt`` format. The resulting graph keeps
                    ``x`` as the variable/node type and uses ``y`` for graph-level
                    targets: ``[BIC, n_nodes, n_edges, n_roots, n_sinks, avg_depth,
                    max_depth]``. Simple structural attributes are exposed in
                    ``atom_attr``: ``[in_degree, out_degree, depth]``.

    Args:
        root (str): Root directory where the dataset should be saved.
        variant (str, optional): Bayesian network variant. Currently ``"asia"``.
            (default: ``"asia"``)
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
        directed (bool, optional): Keep DAG edges directed. This is kept as a
            low-level dataset option for future directed models or manual
            experiments; the named pipeline dataset ``BayesianNetAsia`` uses
            the undirected skeleton representation. If ``False``, edges are
            symmetrized when the selected row processor supports that.
            (default: :obj:`False`)
        max_graphs (int, optional): Limit the number of raw rows processed,
            useful for debugging.
            (default: :obj:`None`)
    """

    _valid_variants = tuple(BAYESIAN_NETWORK_CONFIGS.keys())

    def __init__(
        self,
        root: str,
        variant: str = "asia",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
        directed: bool = False,
        max_graphs: Optional[int] = None,
    ) -> None:
        variant = variant.lower()
        assert (
            variant in self._valid_variants
        ), f"Unknown BayesianNetwork variant '{variant}'. Expected one of: {list(self._valid_variants)}."
        self.variant = variant

        self.config = BAYESIAN_NETWORK_CONFIGS[variant]
        self.name = f"BayesianNet{variant.capitalize()}"

        self.directed = directed
        self.max_graphs = max_graphs

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            log=log,
            force_reload=force_reload,
        )

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [self.config.raw_file_name]

    @property
    def processed_dir(self) -> str:
        direction = "directed" if self.directed else "undirected"
        parts = ["processed", self.variant, direction]
        if self.max_graphs is not None:
            parts.append(f"n{self.max_graphs}")
        return osp.join(self.root, "_".join(parts))

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self) -> None:
        raw_path = self.raw_paths[0]
        if osp.exists(raw_path):
            return

        os.makedirs(self.raw_dir, exist_ok=True)
        urllib.request.urlretrieve(self.config.url, raw_path)

        if self.log:
            print("Download complete.")

    def process(self) -> None:
        with open(self.raw_paths[0], "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        if self.max_graphs is not None:
            lines = lines[: self.max_graphs]

        data_list = []
        for line in tqdm(
            lines,
            desc=f"Processing BayesianNet {self.variant}",
            disable=not self.log,
        ):
            data = self.config.line_to_data(
                line=line,
                directed=self.directed,
                num_node_types=self.config.num_node_types,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

        if self.log:
            print("=" * 60)
            print(
                f"BayesianNetwork variant '{self.variant}' processed "
                f"({'directed' if self.directed else 'undirected'} edges):"
            )
            print(f"Graphs={len(data_list):,}")
            print(f"Dataset saved to: {self.processed_dir}")
            print("=" * 60)


def _asia_dvae_line_to_data(
    line: str,
    directed: bool = False,
    num_node_types: Optional[int] = None,
) -> Data:
    """
    Convert one D-VAE Asia raw text line into a PyG graph.

    The D-VAE Asia format stores one record as ``node_specs, score``. Each
    ``node_specs[i]`` is ``[node_type, parent_0, ..., parent_{i-1}]`` in
    topological order.
    """
    node_specs, score = _parse_asia_dvae_line(line)
    node_types, adjacency, atom_attr, structural_targets = _asia_dvae_specs_to_arrays(
        node_specs=node_specs,
        num_node_types=num_node_types,
    )
    if score is not None:
        y = np.concatenate(
            [np.asarray([float(score)], dtype=np.float32), structural_targets],
            axis=0,
        )
    else:
        y = structural_targets

    edge_adjacency = adjacency if directed else np.maximum(adjacency, adjacency.T)
    src, dst = np.nonzero(edge_adjacency)
    edge_index = torch.as_tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr = torch.zeros(edge_index.shape[1], dtype=torch.long)

    data = Data(
        x=torch.as_tensor(node_types, dtype=torch.long).view(-1, 1),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.as_tensor(y, dtype=torch.float32).view(1, -1),
        atom_attr=torch.as_tensor(atom_attr, dtype=torch.float32),
        num_nodes=int(node_types.shape[0]),
        num_edges=int(adjacency.sum()),
    )
    if directed:
        data.directed_edge_index = torch.as_tensor(
            np.stack(np.nonzero(adjacency), axis=0), dtype=torch.long
        )
    if score is not None:
        data.raw_score = torch.tensor([float(score)], dtype=torch.float32)
    return data


def _parse_asia_dvae_line(line: str) -> Tuple[List[List[int]], Optional[float]]:
    """
    Parse a raw D-VAE Asia Bayesian-network line.

    Supports both ``(<node_specs>, score)`` and ``<node_specs>`` records.
    """
    record = ast.literal_eval(line.strip())
    score = None

    if isinstance(record, tuple) and len(record) == 2:
        node_specs, score = record
    elif (
        isinstance(record, list)
        and len(record) == 2
        and isinstance(record[0], list)
        and record[0]
        and isinstance(record[0][0], list)
    ):
        node_specs, score = record
    else:
        node_specs = record

    if not isinstance(node_specs, list) or not all(
        isinstance(spec, list) for spec in node_specs
    ):
        raise ValueError(f"Could not parse D-VAE Asia node specs from: {line}")

    return [[int(value) for value in spec] for spec in node_specs], score


def _asia_dvae_specs_to_arrays(
    node_specs: Sequence[Sequence[int]],
    num_node_types: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert D-VAE Asia node specs into node labels, adjacency, attributes and target.
    """
    n_nodes = len(node_specs)
    node_types = np.asarray([spec[0] for spec in node_specs], dtype=np.int64)

    if num_node_types is not None:
        invalid = node_types[(node_types < 0) | (node_types >= num_node_types)]
        if invalid.size > 0:
            raise ValueError(
                f"Found node types outside [0, {num_node_types}): "
                f"{sorted(set(invalid.tolist()))}"
            )

    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    for child_idx, spec in enumerate(node_specs):
        parents = spec[1:]
        if len(parents) != child_idx:
            raise ValueError(
                "Bayesian network rows must be in topological order: "
                f"row {child_idx} has {len(parents)} parent indicators"
            )
        for parent_idx, has_edge in enumerate(parents):
            if has_edge:
                adjacency[parent_idx, child_idx] = 1

    in_degree = adjacency.sum(axis=0)
    out_degree = adjacency.sum(axis=1)
    depth = _dag_depths(adjacency)
    atom_attr = np.stack([in_degree, out_degree, depth], axis=1).astype(np.float32)

    n_edges = float(adjacency.sum())
    n_roots = float((in_degree == 0).sum())
    n_sinks = float((out_degree == 0).sum())
    avg_depth = float(depth.mean()) if n_nodes > 0 else 0.0
    max_depth = float(depth.max()) if n_nodes > 0 else 0.0
    target = np.asarray(
        [n_nodes, n_edges, n_roots, n_sinks, avg_depth, max_depth],
        dtype=np.float32,
    )

    return node_types, adjacency, atom_attr, target


def _dag_depths(adjacency: np.ndarray) -> np.ndarray:
    n_nodes = int(adjacency.shape[0])
    depth = np.zeros(n_nodes, dtype=np.int64)
    for node_idx in range(n_nodes):
        parents = np.where(adjacency[:, node_idx] == 1)[0]
        if parents.size > 0:
            depth[node_idx] = int(depth[parents].max()) + 1
    return depth
