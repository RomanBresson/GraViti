import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import global_add_pool

from typing import Tuple

from .transformer_utils import (
    Film,
    res_and_post_treat,
    post_treat_y,
    AttentionBlock,
    AttentionBlockEdges,
    make_mlp,
)


class GraphTransformerLayer(nn.Module):
    """
    Simple node-only Graph Transformer layer.

    Applies multi-head self-attention on node features followed by the
    residual + feed-forward post-treatment block.
    """

    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
    ):
        """
        Initialize a GraphTransformerlayer.

        Args:
            in_dim (int): node feature dimension (input and output).
            num_heads (int): number of attention heads.
            activation (nn.Module, default=nn.ReLU): Activation class to use in res_and_post_treat.
            dropout (float, default=0.0): dropout probability for res_and_post_treat.
        """
        super().__init__()

        self.in_dim = in_dim
        self.att_mod = AttentionBlock(in_dim, in_dim, num_heads)
        self.post_treat = res_and_post_treat(in_dim, activation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        num_nodes = x.shape[1]

        x_att = self.att_mod(x)
        x = self.post_treat(x, x_att)

        return x.reshape(batch_size, num_nodes, self.in_dim)


class GraphTransformerLayerEdge(nn.Module):
    """
    Transformer layer that jointly updates node and edge embeddings.

    * Uses AttentionBlockEdges to compute node updates and an edge-aware
    attention-conditioned edge update.
    * Applies a residual + FFN block separately to nodes and edges.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        num_heads: int,
        activation: nn.Module,
        dropout: float,
    ):
        """
        Initialize a GraphTransformerLayerEdge.

        Args:
            node_in_dim (int): node feature dimension.
            edge_in_dim (int): edge feature dimension.
            num_heads (int): number of attention heads.
            activation (torch.nn.Module): activation class for post blocks.
            dropout (float): dropout probability for post blocks.
        """
        super().__init__()
        self.node_in_dim = node_in_dim
        self.att_mod = AttentionBlockEdges(node_in_dim, edge_in_dim, num_heads)

        self.post_node = res_and_post_treat(node_in_dim, activation, dropout)
        self.post_edge = res_and_post_treat(edge_in_dim, activation, dropout)

    def forward(
        self, x: torch.Tensor, e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_x, new_e = self.att_mod(x, e)

        new_x = self.post_node(x, new_x)
        new_e = self.post_edge(e, new_e)

        return new_x, new_e


class GraphTransformerEdge(nn.Module):
    """
    Graph-level Transformer that uses node+edge transformer layers and
    produces a graph-level representation.

    Key components:
    - Embedding LUT for node and edge features.
    - MLP to embed positional encodings.
    - A stack of GraphTransformerLayerEdge layers.
    - A final MLP to produce output from pooled node features.

    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        trans_dim_nodes: int,
        trans_dim_edges: int,
        num_heads: int,
        trans_layers: int,
        pe_dim: int,
        out_dim: int,
        activation: nn.Module,
        dropout: float,
    ):
        """
        Initialize a GraphTransformerEdge.

        Args:
            node_features (int): node feature dimension.
            edge_features (int): edge feature dimension.
            trans_dim_nodes (int): transformer side - node feature dimension.
            trans_dim_edges (int): transformer side - edge feature dimension.
            num_heads (int): number of attention heads.
            trans_layers (int): number of transformer layers (using `GraphTransformerLayerEdge`).
            pe_dim (int): positional encodings feature dimension.
            out_dim (int): output feature dimension.
            activation (torch.nn.Module): Activation class to use in res_and_post_treat.
            dropout (float): dropout probability for res_and_post_treat.
            norm (torch.nn.Module (Optional)): Normalization layer class (e.g., nn.BatchNorm1d).
        """
        super().__init__()

        self.classes_node = node_features
        self.classes_edge = edge_features

        self.embed_node = nn.Embedding(node_features, trans_dim_nodes)
        self.embed_edge = nn.Embedding(edge_features, trans_dim_edges)

        # MLP Network for Positional Encodings
        self.mlp_pe = make_mlp(
            in_dim=pe_dim,
            hidden_dim=trans_dim_nodes,
            out_dim=trans_dim_nodes,
            nb_layers=2,
            dropout=0.0,
            activation=activation,
            norm=None,
        )

        # Transformer layers that update node & edge representations
        self.node_rep_update = nn.ModuleList(
            [
                GraphTransformerLayerEdge(
                    trans_dim_nodes, trans_dim_edges, num_heads, activation, dropout
                )
                for _ in range(trans_layers)
            ]
        )

        # Output projection
        self.mlp_out = make_mlp(
            in_dim=trans_dim_nodes,
            hidden_dim=trans_dim_nodes,
            out_dim=out_dim,
            nb_layers=2,
            dropout=dropout,
            activation=activation,
            norm=None,
        )
        self.dropout = dropout

    def embed_nodes_and_edges(
        self, x: torch.Tensor, e: torch.Tensor, pe: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embed nodes and edges, add positional encodings, and run transformer updates.

        Args:
            x (torch.Tensor): raw node features
            e (torch.Tensor): raw edge features
            pe (torch.Tensor): positional encodings

        Returns:
            Tuple[torch.Tensor,torch.Tensor]:
                x (batch, n, trans_dim_nodes), e (batch, n, n, trans_dim_edges)
        """
        x = self.embed_node(x)
        e = self.embed_edge(e)
        pe = self.mlp_pe(pe)
        x = x + pe

        for gtr in self.node_rep_update:
            x, e = gtr(x, e)
        return x, e

    def forward(
        self, xpe: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward expects a tuple (x, pe, e) as described in argument `xpe` below.

        Args:
            xpe (Tuple[torch.Tensor,torch.Tensor,torch.Tensor]):
                * x: (batch, n, node_features)
                * pe: (batch, n, pe_dim)
                * e: (batch, n, n, edge_features)

        Returns:
            torch.Tensor: graph-level tensor of shape (batch, out_dim)
        """
        x, pe, e = xpe
        x, e = self.embed_nodes_and_edges(x, e, pe)

        x = torch.sum(x, dim=1)
        x = self.mlp_out(x)
        return x


class GraphTransformerLayerXEY(nn.Module):
    """
    Transformer layer that updates nodes, edges and the global features y.

    * Uses AttentionBlockEdges for node+edge interaction.
    * Uses FiLM to condition node/edge updates on the global features y.
    * Updates y using post_treat_y which aggregates node/edge statistics.

    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        y_in_dim: int,
        num_heads: int,
        activation: nn.Module,
        dropout: float,
    ):
        """
        Initialize a Graph Transformer Layer (for nodes, edges and global features)
        Args:
            node_in_dim (int): node feature dimension.
            edge_in_dim (int): edge feature dimension.
            y_in_dim (int): global feature dimension.
            num_heads (int): number of attention heads.
            activation (torch.nn.Module): activation class for post blocks.
            dropout (float): dropout probability for post blocks.
        """
        super().__init__()
        self.node_in_dim = node_in_dim
        self.att_mod = AttentionBlockEdges(node_in_dim, edge_in_dim, num_heads)

        self.post_node = res_and_post_treat(node_in_dim, activation, dropout)
        self.post_edge = res_and_post_treat(edge_in_dim, activation, dropout)
        self.post_y = post_treat_y(node_in_dim, edge_in_dim, y_in_dim)

        self.FilmYE = Film(y_in_dim, edge_in_dim)
        self.FilmYX = Film(y_in_dim, node_in_dim)

    def forward(
        self, x: torch.Tensor, e: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_x, new_e = self.att_mod(x, e)

        # Condition updates on global y via FiLM
        new_x = self.FilmYX(y, new_x)
        new_e = self.FilmYE(y, new_e)

        new_x = self.post_node(x, new_x)
        new_e = self.post_edge(e, new_e)
        new_y = self.post_y(x, e, y)

        return new_x, new_e, new_y


class GraphTransformerEy(GraphTransformerEdge):
    """
    Extension of GraphTransformerEdge that additionally consumes and updates
    a global graph-level feature vector `y`.

    The main changes compared to the base class:
    - embeds y via an MLP
    - uses GraphTransformerLayerXEY blocks which condition on y and update y
    - output MLP consumes both pooled node and pooled edge representations

    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        y_features: int,
        trans_dim_nodes: int,
        trans_dim_edges: int,
        trans_dim_y: int,
        num_heads: int,
        trans_layers: int,
        pe_dim: int,
        out_dim: int,
        activation: nn.Module = nn.SiLU,
        dropout: float = 0.5,
    ):
        """
        Initialize a Graph TransformerEy.

        Args:
            node_features (int): node feature dimension.
            edge_features (int): edge feature dimension.
            y_features (int): graph feature dimension.
            trans_dim_nodes (int): transformer side - node feature dimension.
            trans_dim_edges (int): transformer side - edge feature dimension.
            trans_dim_y (int): transformer side - graph feature dimension.
            num_heads (int): number of attention heads.
            trans_layers (int): number of transformer layers (using `GraphTransformerLayerXEY`).
            pe_dim (int): positional encodings feature dimension.
            out_dim (int): output feature dimension.
            activation (nn.Module, default=nn.SiLU): Activation class to use in res_and_post_treat.
            dropout (float, default=0.5): dropout probability for res_and_post_treat.
            norm (nn.Module (Optional)): Normalization layer class (e.g., nn.BatchNorm1d).
        """
        super().__init__(
            node_features=node_features,
            edge_features=edge_features,
            trans_dim_nodes=trans_dim_nodes,
            trans_dim_edges=trans_dim_edges,
            num_heads=num_heads,
            trans_layers=trans_layers,
            pe_dim=pe_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout
        )

        # Embed y
        self.mlp_in_y = make_mlp(
            in_dim=y_features,
            hidden_dim=trans_dim_y,
            out_dim=trans_dim_y,
            nb_layers=2,
            activation=activation,
            dropout=dropout,
            norm=None,
        )

        # Use `GraphTransformerLayerXEY` transformer (override/replace `GraphTransformerLayerEdge`)
        self.node_rep_update = nn.ModuleList(
            [
                GraphTransformerLayerXEY(
                    node_in_dim=trans_dim_nodes,
                    edge_in_dim=trans_dim_edges,
                    y_in_dim=trans_dim_y,
                    num_heads=num_heads,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(trans_layers)
            ]
        )

        # output now consumes both node- and edge-pooled representations
        self.mlp_out = make_mlp(
            in_dim=trans_dim_nodes + trans_dim_edges,
            hidden_dim=trans_dim_nodes,
            out_dim=out_dim,
            nb_layers=2,
            activation=activation,
            dropout=0,
            norm=None,
        )

    def embed_nodes_and_edges(
        self, x: torch.Tensor, pe: torch.Tensor, e: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed nodes, edges and global features, add positional encodings, and run 
        transformer updates.

        Args:
            x: raw node features, shape (batch, n, node_features)
            pe: positional encodings, shape (batch, n, trans_dim_nodes) or broadcastable
            e: raw edge features, shape (batch, n, n, edge_features)
            y: raw graph-level features, shape (batch, n, y_features)

        Returns:
            tuple[torch.Tensor,torch.Tensor,torch.Tensor]: x, e, y (all in their transformed dimensions)
        """
        x = self.embed_node(x)
        e = self.embed_edge(e)
        y = self.mlp_in_y(y)
        pe = self.mlp_pe(pe)

        x = x + pe
        for gtr in self.node_rep_update:
            x, e, y = gtr(x, e, y)
        return x, e, y

    def forward(self, data: dict):
        """
        Expects a dictionary with keys "X", "PE", "E", "y" as explained
        in the `Args` section.

        Args:
            data (dict): a dictionary with the following properties:
                * X: node features (batch, n, node_features_onehot or encoded)
                * PE.pe_lap: positional encodings (batch, n, pe_dim)
                * E: edge features (batch, n, n, edge_features_onehot or encoded)
                * y: global features (batch, y_features)


        Returns:
            torch.tensor: graph-level tensor of shape (batch, out_dim)
        """
        xt, pe, et, y = data["X"], data["PE"]["pe_lap"], data["E"], data["y"]
        x, e, y = self.embed_nodes_and_edges(xt, pe, et, y)

        # Detect real nodes/edges assuming last index in the one-hot/vector is padding
        real_nodes_indices = torch.where(xt != self.classes_node - 1)
        real_edges_indices = torch.where(et != self.classes_edge - 1)

        # Gather real (non-padded) nodes and edges
        real_nodes = x[real_nodes_indices[0], real_nodes_indices[1]]
        real_edges = e[real_edges_indices[0], real_edges_indices[1], real_edges_indices[2]]

        # Pool per-graph using the batch indices from the first element of the 'where' result
        x = global_add_pool(real_nodes, real_nodes_indices[0])
        e = global_add_pool(real_edges, real_edges_indices[0])

        # Layer-norm and concatenate
        x = torch.nn.functional.layer_norm(x, (x.shape[1],))
        e = torch.nn.functional.layer_norm(e, (e.shape[1],))

        x = torch.cat((x, e), 1)
        x = self.mlp_out(x)
        return x


class GATLayer(nn.Module):
    """
    Thin wrapper around PyG's `GATv2Conv`.
    """

    def __init__(
        self, in_feats: int, out_feats: int, num_heads: int, edge_features: int
    ):
        """
        Initialize a GAT layer.

        Args:
            in_feats (int): input feature dim
            out_feats (int): output feature dim per head
            num_heads (int): number of attention heads
            edge_features (int): edge feature dim (edge_dim for GATv2Conv)
        """
        super().__init__()

        self.attention_layer = GATv2Conv(
            in_channels=in_feats,
            out_channels=out_feats,
            num_heads=num_heads,
            edge_dim=edge_features,
        )

    def forward(self, x, edge_index, edge_attr):
        return self.attention_layer(x, edge_index, edge_attr)


class GATModel(nn.Module):
    """
    Simple stacked GAT model using embeddings for categorical node/edge inputs.

    Consists of a sequence of GAT layers.
    """

    def __init__(
        self,
        num_layers: int,
        node_features: int,
        edge_features: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int,
    ):
        """
        Initialize a GAT Model.

        Args:
            num_layers (int): Number of GAT layers in sequence before output layer (minimum 1)
            node_features (int): node feature dimension.
            edge_features (int): edge feature dimension.
            hidden_dim (int): output feature dim per head
            out_dim (int): output feature dimension.
            num_heads (int): number of attention heads (per layer)
        """
        super().__init__()

        self.layers = nn.ModuleList()

        self.emb_node = nn.Embedding(node_features, hidden_dim)
        self.emb_edges = nn.Embedding(edge_features, hidden_dim)

        # Input layer
        self.layers.append(
            GATLayer(
                in_feats=hidden_dim,
                out_feats=hidden_dim,
                num_heads=num_heads,
                edge_features=hidden_dim,
            )
        )

        # Add additional GAT layers
        for _ in range(num_layers - 1):
            self.layers.append(
                GATLayer(
                    in_feats=hidden_dim * num_heads,
                    out_feats=hidden_dim,
                    num_heads=num_heads,
                    edge_features=hidden_dim,
                )
            )

        # Output layer (can also aggregate differently)
        self.output_layer = GATLayer(
            hidden_dim * num_heads, out_dim // num_heads, num_heads, hidden_dim
        )

    def forward(self, data):
        x, edge_index, edge_attr = data["x"], data["edge_index"], data["edge_attr"]
        x = torch.argmax(x, -1).long()

        edge_attr = torch.argmax(edge_attr, -1).long()
        x = self.emb_node(x)

        edge_attr = self.emb_edges(edge_attr)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.gelu(x)

        x = self.output_layer(x, edge_index, edge_attr)
        x = global_add_pool(x, data["batch"])
        return x
