import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Dict, Tuple, Optional

from .catflow_models.transformer import GraphTransformer

class SinPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding implementation.

    Adds fixed sinusoidal embeddings to input sequences so that the model
    can exploit order information without learnable position embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize SinPositionalEncoding:

        Args:
            d_model (int): Embedding dimension.
            max_len (int, default=5000): Maximum sequence length.
        """
        super().__init__()
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        self.pe_scale = nn.Parameter(torch.ones(d_model))
        # Compute the sinusoidal frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]*self.pe_scale
        return x

class LatentToGraph(nn.Module):
    def __init__(self,
        n_layers: int,
        latent_dim: int,
        num_nodes: int,
        output_dims: Dict[str, int],
        transformer_dim: int,
        transformer_dim_ff: int,
        n_heads: int,
        dropout: float=0.1,
        sigma: float=0.01,
        predicted_properties: int=0,
        reinject_size: bool=True,
        MAX_SIZE_MEMORY: int=100 #to avoid memory overflow
    ):
        super().__init__()
        self.transformer_dim = transformer_dim # must be the same as hidden dim of encoder readout
        self.num_nodes_default = num_nodes
        self.sigma_noise = sigma
        self.norm_pre = nn.LayerNorm(transformer_dim*2)
        self.norm_in = nn.LayerNorm(transformer_dim*2)
        # Map latent embeddings into input space for Transformer
        self.latent2seq = nn.Linear(latent_dim, transformer_dim*2)
        self.positional_encoder = SinPositionalEncoding(transformer_dim*2) #doubled to be split later
        self.MAX_SIZE_MEMORY = MAX_SIZE_MEMORY
        # Transformer encoder without dropout
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim*2,
            nhead=n_heads,
            dim_feedforward=transformer_dim_ff,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer_sequence = nn.TransformerEncoder(
            transformer_layer, num_layers=n_layers, norm=nn.LayerNorm(2*transformer_dim)
        )
        self.nodes_proj_out = nn.Sequential(
                nn.Linear(transformer_dim, output_dims['X']),
                nn.GELU()
        )
        self.edges_proj_out = nn.Sequential(
                nn.Linear(transformer_dim*2, output_dims['E']),
                nn.GELU()
        )
        self.graph_proj_out = nn.Sequential(
                nn.Linear(transformer_dim*2, output_dims['y']),
                nn.GELU()
        )
        self.graph_size_predictor = nn.Sequential(
            nn.Linear(8, transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, 1)
        )
        self.graph_size_reencoder = None
        if reinject_size:
            self.graph_size_reencoder = nn.Sequential(
                nn.Linear(1, transformer_dim),
                nn.GELU(),
                nn.Linear(transformer_dim, 8)
            )
        if predicted_properties>0:
            self.property_predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, predicted_properties))
        else:
            self.property_predictor = None

    def forward(self, latent_embeddings: torch.Tensor, override_size_padding: int=None, force_size_graph: torch.Tensor=None):
        size_embedding = latent_embeddings[:, :8] # take part of the latent embedding to predict the size
        predicted_sizes_log = self.graph_size_predictor(size_embedding).squeeze(-1) # b, predict the log size to ensure positivity and better numerical stability

        if force_size_graph is not None:
            assert force_size_graph.shape[0] == latent_embeddings.shape[0], "force_size_graph must have the same batch size as latent_embeddings"
            assert force_size_graph.ndim == 1, "force_size_graph must be a 1D tensor of shape (batch_size,)"
            reinjected_log_sizes = force_size_graph.float().clamp_min(1e-6).log()
            graph_sizes_crisp = force_size_graph.int()
        else:
            reinjected_log_sizes = predicted_sizes_log
            graph_sizes_crisp = reinjected_log_sizes.exp().round().long().clamp_min(1).clamp_max(self.MAX_SIZE_MEMORY)

        if self.training:
            reinjected_log_sizes = reinjected_log_sizes + 0.02 * torch.randn_like(reinjected_log_sizes)

        updated_latent = latent_embeddings
        if self.graph_size_reencoder is not None:
            log_size_reencoded = self.graph_size_reencoder(reinjected_log_sizes.unsqueeze(-1))
            updated_latent = torch.cat([log_size_reencoded, latent_embeddings[:, 8:]], dim=-1)

        if override_size_padding is not None:
            num_nodes_padded = override_size_padding
        else:
            num_nodes_padded = max(self.num_nodes_default, (graph_sizes_crisp.max().item()))

        batch_size = latent_embeddings.size(0)
        node_filter = torch.arange(num_nodes_padded, device=latent_embeddings.device).expand(batch_size, num_nodes_padded) < graph_sizes_crisp.unsqueeze(1)
        node_filter = node_filter.to(latent_embeddings.device)

        # Expand latent embeddings to sequences
        expanded_embeddings = self.latent2seq(updated_latent)  # b x hidden_dims['X']
        expanded_embeddings = self.norm_pre(expanded_embeddings)
        y = expanded_embeddings
        # add activation here?
        # Expand to identical sequences b x n x h (all identical graphwise) and add noise + PE
        sequence = expanded_embeddings.unsqueeze(1).expand(-1, num_nodes_padded, -1)
        sequence = sequence + self.sigma_noise * torch.randn_like(sequence)
        sequence = self.positional_encoder(sequence)
        sequence = self.norm_in(sequence)*node_filter.float().unsqueeze(-1)

        X = self.transformer_sequence(sequence, src_key_padding_mask=~node_filter)  # b x n x h

        X_x = X[:, :, :self.transformer_dim]
        X_e = X[:, :, self.transformer_dim:]

        seq1 = X_e.unsqueeze(2).repeat(1, 1, num_nodes_padded, 1)
        seq2 = X_e.unsqueeze(1).repeat(1, num_nodes_padded, 1, 1)
        edge_mask = node_filter.unsqueeze(1) * node_filter.unsqueeze(2)
        E = torch.cat([seq1, seq2], -1)
        E = self.edges_proj_out(E)*edge_mask.unsqueeze(-1)
        E = (E + E.transpose(1, 2)) /2
        E = E * edge_mask.unsqueeze(-1)
        X = self.nodes_proj_out(X_x)
        y = self.graph_proj_out(y)
        # predicted_log_size is predicted, node_filter is actually used
        return X, E, y, predicted_sizes_log, node_filter

class GraphTransformerDecoder(nn.Module):
    """
    Graph-based Transformer Decoder that reconstructs node and edge features
    from latent embeddings using a sequence Transformer + Graph Transformer.
    """

    def __init__(
        self,
        n_layers: int,
        input_dims: Dict[str, int],
        hidden_mlp_dims: Dict[str, int],
        hidden_dims: Dict[str, int],
        X_classes: int,
        E_classes: int,
        latent_dim: int,
        num_nodes_default: int,
        dropout: float = 0.1,
        sigma: float = 0.01,
        max_hydrogen_values: int = 0,
        bounds_formal_charge_values: Tuple[int, int] = (0, 0),
        reinject_size: bool = True,
        predict_node_mask: bool = False
    ):
        """
        Args:
            n_layers (int): Number of layers in the GraphTransformer.
            input_dims (dict): Input dimensions for nodes ("X") and edges ("E").
            hidden_mlp_dims (dict): Hidden dimensions for MLP layers.
            hidden_dims (dict): Hidden dimensions, must include "n_head".
            X_classes (int): Number of node classes.
            E_classes (int): Number of edge classes.
            latent_dim (int): Dimension of latent embeddings.
            num_nodes_default (int): Default number of nodes per graph (including padding).
            sigma (float, default=0.01): Noise scale for latent expansion.
        """
        super().__init__()

        self.dim_embedding_nodes = input_dims['X']

        self.num_nodes_default = num_nodes_default

        self.latent_to_graph = LatentToGraph(
            n_layers=2,
            latent_dim=latent_dim,
            num_nodes=num_nodes_default,
            output_dims=input_dims,
            transformer_dim=hidden_dims['dx'],
            transformer_dim_ff=hidden_dims['dim_ffX'],
            n_heads=hidden_dims['n_head'],
            dropout=dropout,
            sigma=sigma,
            reinject_size=reinject_size,
        )

        self.activation = nn.GELU()

        self.transformer_graph = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=input_dims,
            act_fn_in=nn.GELU(),
            act_fn_out=nn.GELU(),
            dropout=dropout
        )

        # Classification layers
        self.node_atom_classifier = nn.Sequential(
            nn.Linear(input_dims["X"], input_dims["X"]*4),
            nn.GELU(),
            nn.Linear(input_dims["X"]*4, X_classes))

        self.node_hydrogen_predictor = None
        self.node_formal_charge_predictor = None
        self.soft_node_mask_predictor = None
        predict_hydrogens = max_hydrogen_values > 0
        predict_formal_charge = (bounds_formal_charge_values[1] > bounds_formal_charge_values[0])
        # Classification layers
        if predict_hydrogens:
            self.node_hydrogen_predictor = nn.Sequential(
                nn.Linear(input_dims["X"], input_dims["X"]*4),
                nn.GELU(),
                nn.Linear(input_dims["X"]*4, max_hydrogen_values+1))

        self.possible_formal_charge_values = None
        if predict_formal_charge:
            self.possible_formal_charge_values = list(range(bounds_formal_charge_values[0], bounds_formal_charge_values[1]+1))
            num_formal_charge_values = bounds_formal_charge_values[1] - bounds_formal_charge_values[0] + 1
            self.node_formal_charge_predictor = nn.Sequential(
                nn.Linear(input_dims["X"], input_dims["X"]*4),
                nn.GELU(),
                nn.Linear(input_dims["X"]*4, num_formal_charge_values))

        if predict_node_mask:
            self.soft_node_mask_predictor = nn.Sequential(
                nn.Linear(input_dims["X"], input_dims["X"]*4),
                nn.GELU(),
                nn.Linear(input_dims["X"]*4, 1) # logits are returned for compatibility with downstream losses
            )

        self.edge_classifier = nn.Sequential(
            nn.Linear(input_dims["E"], input_dims["E"]*4),
            nn.GELU(),
            nn.Linear(input_dims["E"]*4, E_classes))

    def embed_XEy(self, latent_embeddings: torch.Tensor, override_size_padding: int=None, force_size_graph: torch.Tensor=None):
        """
            first step of decoding, get last representation before classifiers.
        """
        query = latent_embeddings
        X, E, y, predicted_sizes_log, used_node_filter = self.latent_to_graph(
            latent_embeddings=query,
            override_size_padding=override_size_padding,
            force_size_graph=force_size_graph
        )

        transformed_graph = self.transformer_graph(X, E, y, used_node_filter)
        X, E, y = transformed_graph.X, transformed_graph.E, transformed_graph.y
        return X, E, y, predicted_sizes_log, used_node_filter

    def predict_output(self, X, E, used_node_filter):
        activated_X = self.activation(X)
        X = self.node_atom_classifier(activated_X)
        predicted_hydrogens = None
        predicted_formal_charge = None
        predicted_node_mask = None
        if self.node_hydrogen_predictor is not None:
            predicted_hydrogens = self.node_hydrogen_predictor(activated_X)
        if self.node_formal_charge_predictor is not None:
            predicted_formal_charge = self.node_formal_charge_predictor(activated_X)
        if self.soft_node_mask_predictor is not None:
            predicted_node_mask = self.soft_node_mask_predictor(activated_X).squeeze(-1)
        E = (E + E.transpose(1, 2)) /2
        E = self.edge_classifier(self.activation(E))
        X = X * used_node_filter.unsqueeze(-1)
        E = E * used_node_filter.unsqueeze(1).unsqueeze(-1) * used_node_filter.unsqueeze(2).unsqueeze(-1)
        return X, E, predicted_hydrogens, predicted_formal_charge, predicted_node_mask

    def forward(
        self,
        latent_embeddings: torch.Tensor,
        override_size_padding: Optional[int] = None,
        force_size_graph: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GraphTransformerDecoder.

        Args:
            latent_embeddings (torch.Tensor): Latent embeddings of shape (batch_size, latent_dim).
            override_size_padding (int, optional): If provided, overrides default num_nodes.
            node_filter (torch.Tensor, optional): Boolean mask of shape (batch_size, num_nodes)
                indicating valid nodes (True) vs padding (False).
        Returns:
            Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
            * Node predictions: (batch_size, num_nodes, X_classes)
            * Edge predictions: (batch_size, num_nodes, num_nodes, E_classes)
            * Global feature predictions
            * Predicted graph sizes
            * The node mask actually used inside the decoder
        """
        X, E, y, predicted_sizes_log, used_node_filter = self.embed_XEy(latent_embeddings, override_size_padding, force_size_graph)
        X_for_matcher = X.clone()
        X, E, predicted_hydrogens, predicted_formal_charges, predicted_node_mask = self.predict_output(X, E, used_node_filter)

        outputs = {
            'X': X,
            'E': E,
            'y': y,
            'predicted_sizes_log': predicted_sizes_log,
            'used_node_filter': used_node_filter,
            'predicted_hydrogens': predicted_hydrogens,
            'predicted_formal_charges': predicted_formal_charges,
            'predicted_node_mask': predicted_node_mask,
            'node_embeddings_decoder': X_for_matcher
        }
        return outputs