import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple, Optional, Any
from .catflow_models.transformer import GraphTransformer

class GraphTransformerReadout(nn.Module):
    def __init__(self, node_dim, edge_dim, graph_dim, hidden_dim, dim_feedforward, output_dim, n_heads, dropout):
        super().__init__()

        # Projection layers to match hidden_dim
        self.node_proj_in = nn.Linear(node_dim, hidden_dim)
        self.edge_proj_in = nn.Linear(edge_dim, hidden_dim)
        self.graph_proj_in = nn.Linear(graph_dim, hidden_dim)

        self.norm_nodes_in = nn.LayerNorm(hidden_dim)
        self.norm_edges_in = nn.LayerNorm(hidden_dim)
        self.norm_nodes_out = nn.LayerNorm(hidden_dim)
        self.norm_edges_out = nn.LayerNorm(hidden_dim)
        self.norm_graph = nn.LayerNorm(hidden_dim)

        # Transformer decoders
        decoder_layer_nodes = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout, dim_feedforward=dim_feedforward, activation='gelu')
        decoder_layer_edges = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout, dim_feedforward=dim_feedforward, activation='gelu')
        self.node_decoder = nn.TransformerDecoder(decoder_layer_nodes, num_layers=2)
        self.edge_decoder = nn.TransformerDecoder(decoder_layer_edges, num_layers=2)

        # Output MLP
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, X_out, E_out, Y_out, node_filter):
        # Apply node filter
        edge_mask = node_filter.unsqueeze(2) & node_filter.unsqueeze(1)

        # Project to hidden_dim
        node_features = self.norm_nodes_in(self.node_proj_in(X_out))
        edge_features = self.norm_edges_in(self.edge_proj_in(E_out))
        graph_features = self.norm_graph(self.graph_proj_in(Y_out))

        # Prepare queries
        batch_size = node_features.size(0)
        node_query = graph_features.unsqueeze(0)
        edge_query = graph_features.unsqueeze(0)

        # Transformer decoding
        padding_mask_nodes = ~node_filter.bool()
        node_rep = self.node_decoder(tgt=node_query, memory=node_features.transpose(0, 1), memory_key_padding_mask=padding_mask_nodes).squeeze(0)
        node_rep = self.norm_nodes_out(node_rep)

        flattened_edge_features = edge_features.view(batch_size, -1, edge_features.size(-1))
        padding_mask_edges = ~edge_mask.bool()
        padding_mask_edges = padding_mask_edges.view(batch_size, -1)
        edge_rep = self.edge_decoder(edge_query, flattened_edge_features.transpose(0, 1), memory_key_padding_mask=padding_mask_edges).squeeze(0)
        edge_rep = self.norm_edges_out(edge_rep)

        # Combine with graph-level feature
        graph_rep = torch.cat([node_rep, edge_rep], dim=-1)
        return self.out_layer(graph_rep)

class CatFlowWrapper(nn.Module):
    """
    Wrapper around CatFlow's GraphTransformer.

    - Embeds node, edge, and global graph features.
    - Applies positional encodings (Laplacian & random walk).
    - Processes graph through a GraphTransformer.
    - Aggregates node, edge, and global features into a latent vector.
    """

    def __init__(
        self,
        n_layers: int,
        input_dims: Dict[str, int],
        hidden_mlp_dims: Dict[str, int],
        hidden_dims: Dict[str, int],
        X_classes: int,
        E_classes: int,
        y_size: int,
        latent_dim: int,
        use_atom_attr = True,
        atom_attr_dim = 0,
        sigma_symmetry_breaking_noise: Optional[float] = 0.,
        dropout=0.
    ):
        """
        Initialize a CatFlowWrapper.

        Args:
            n_layers (int): Number of layers in the GraphTransformer.
            input_dims (dict): Input dimensions for each modality
                * "X": Node input dim
                * "E": Edge input dim
                * "y": Graph input dim
                * "PE": Positional encoding dim
                    * "pe_lap", "pe_rw": Input dims for Laplacian and RW encodings
            hidden_mlp_dims (dict): Hidden dimensions for MLPs in the transformer.
            hidden_dims (dict): Hidden dimensions (e.g., number of heads).
            X_classes (int): Number of node classes.
            E_classes (int): Number of edge classes.
            y_size (int): Dimension of graph-level input features.
            latent_dim (int): Latent representation dimension.
        """
        super().__init__()
        assert input_dims['PE'] % 2 == 0, "input_dims['PE'] must be even"
        self.use_atom_attr = use_atom_attr
        self.edge_classes = E_classes
        self.node_classes = X_classes
        self.latent_dim = latent_dim

        dim_node_features = input_dims['X'] - input_dims['PE']
        x_emb_dim = dim_node_features
        if self.use_atom_attr:
            assert dim_node_features % 2 == 0
            x_emb_dim = dim_node_features//2
            self.atom_attr_in = nn.Linear(atom_attr_dim, x_emb_dim)

        self.x_emb = nn.Embedding(X_classes+1, x_emb_dim, padding_idx=X_classes) #account for padding class
        self.e_emb = nn.Embedding(E_classes, input_dims['E'])
        self.y_emb = nn.Linear(y_size, input_dims['y'])
        self.sigma_symmetry_breaking_noise = sigma_symmetry_breaking_noise

        # Positional encoding
        self.pe_lap_layer = nn.Linear(input_dims['pe_lap'], input_dims['PE'] // 2)
        self.pe_rw_layer = nn.Linear(input_dims['pe_rw'], input_dims['PE'] // 2)
        self.pe_fusion_layer = nn.Linear(input_dims['PE'], input_dims['PE'])
        self.pe_lap_norm = nn.LayerNorm(input_dims['PE'] // 2)
        self.pe_rw_norm = nn.LayerNorm(input_dims['PE'] // 2)

        self.transformer = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=input_dims,
            act_fn_in=nn.GELU(),
            act_fn_out=nn.GELU(),
            dropout=dropout
        ) # should have same input and output sizes for recurrent connexions

        # Normalization
        self.attr_norm = nn.LayerNorm(x_emb_dim)
        self.pe_norm = nn.LayerNorm(input_dims['PE'])
        self.input_x_norm = nn.LayerNorm(x_emb_dim)
        self.input_e_norm = nn.LayerNorm(input_dims['E'])
        self.input_y_norm = nn.LayerNorm(input_dims['y'])
        self.norm_x_cated  = nn.LayerNorm(dim_node_features)
        self.norm_x_final = nn.LayerNorm(input_dims['X'])
        self.node_dim_for_matcher = input_dims['X']

        self.readout_transformer = GraphTransformerReadout(
            node_dim = input_dims['X'],
            edge_dim = input_dims['E'],
            graph_dim = input_dims['y'],
            hidden_dim = hidden_dims['dx'],
            dim_feedforward = hidden_dims['dim_ffX'],
            n_heads = hidden_dims['n_head'],
            output_dim = latent_dim,
            dropout = dropout
        )

    def readout(self, X_out, E_out, Y_out, node_filter):
        graph_rep = self.readout_transformer(X_out, E_out, Y_out, node_filter)
        return graph_rep

    def prepare_nodes(self, X_in, P, atom_attr):
        # Embeddings
        X = self.x_emb(X_in)
        X = self.input_x_norm(X)
        X = F.gelu(X)
        X += torch.randn_like(X) * self.sigma_symmetry_breaking_noise

        # Positional encodings (Laplacian + RandomWalk fusion)
        pe_lap = self.pe_lap_layer(P['pe_lap'])+self.pe_lap_layer(-P['pe_lap'])
        pe_lap = self.pe_lap_norm(pe_lap)

        pe_rw = self.pe_rw_layer(P['pe_rw'])
        pe_rw = self.pe_rw_norm(pe_rw)

        PE = torch.cat([pe_lap, pe_rw], dim=-1)
        PE = F.gelu(PE)
        PE = self.pe_fusion_layer(PE)
        PE = self.pe_norm(PE)
        PE = F.gelu(PE)

        if self.use_atom_attr:
            if atom_attr is None:
                raise ValueError('atom_attr required when use_atom_attr=True')
            attr_embed = self.atom_attr_in(atom_attr)
            attr_embed = self.attr_norm(attr_embed)
            attr_embed = F.gelu(attr_embed)
            X = torch.cat([X, attr_embed], dim=-1)
            X = self.norm_x_cated(X)

        X = torch.cat([X, PE], dim=-1)
        X = self.norm_x_final(X)
        return X

    def embed_nodes_and_edges(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed nodes, edges and global features, add positional encodings, and run
        transformer updates.

        Args:
            inputs (dict): a dictionary with the following properties:
                * X: node features (batch, n)
                * PE: positional encodings (batch, n, pe_dim) (pe_lap and pe_rw)
                * E: edge features (batch, n, n)
                * y: global features (batch, y_features)

        Returns:
            tuple[torch.Tensor,torch.Tensor,torch.Tensor]: X, E, Y (all in their transformed dimensions)
        """
        X_in, E_in, Y_in, P, atom_attr = inputs['X'], inputs['E'], inputs['y'], inputs['PE'], inputs['atom_attr']
        node_filter = inputs['node_mask'] if 'node_mask' in inputs else (X_in != self.node_classes)

        X = self.prepare_nodes(X_in, P, atom_attr) # b x n x input_dims['X']

        E = self.e_emb(E_in)
        E = self.input_e_norm(E)
        E = F.gelu(E)
        E += torch.randn_like(E) * self.sigma_symmetry_breaking_noise # b x n x n x input_dims['E']

        Y = self.y_emb(Y_in)
        Y = self.input_y_norm(Y)
        Y = F.gelu(Y) # b x n x input_dims['y']

        transformed = self.transformer(X, E, Y, node_mask=node_filter) # input_dim -> output_dims
        X_out, E_out, Y_out = transformed.X, transformed.E, transformed.y
        return X_out, E_out, Y_out, node_filter

    def forward(self, inputs):
        X_out, E_out, Y_out, node_filter = self.embed_nodes_and_edges(inputs)
        graph_rep = self.readout(X_out.clone(), E_out, Y_out, node_filter)
        return graph_rep, X_out
