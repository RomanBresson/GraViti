import torch
import torch.nn as nn
import torch.nn.functional as F


class Film(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) block.

    Applies a learned affine transformation to one tensor (`M2`) conditioned
    on another tensor (`M1`).

    Formula (simplified):
        FiLM(M1, M2) = W1(M1) + (W2(M1) * M2) + M2
    """

    def __init__(self, in_dim1, in_dim2):
        """
        Initialize a FiLM layer.

        Args:
            in_dim1 (int): Dimension of the conditioning input (M1).
            in_dim2 (int): Dimension of the features to be modulated (M2).
        """
        super().__init__()
        self.W1 = nn.Linear(in_dim1, in_dim2, bias=False)
        self.W2 = nn.Linear(in_dim1, in_dim2, bias=False)

    def forward(self, M1: torch.Tensor, M2: torch.Tensor) -> torch.Tensor:
        w1m1 = self.W1(M1)
        w2m1 = self.W2
        # Handle broadcasting depending on tensor rank
        if M1.dim() == 2:

            if M2.dim() == 3:
                w1m1 = w1m1.unsqueeze(-2)
                w2m1 = w2m1.unsqueeze(-2)

            elif M2.dim() == 4:
                w1m1 = w1m1.view(w1m1.shape[0], 1, 1, w1m1.shape[1])
                w2m1 = w2m1.view(w2m1.shape[0], 1, 1, w2m1.shape[1])
        else:
            w1m1 = w1m1.squeeze(-1)
            w2m1 = w2m1.squeeze(-1)

        # Apply modulation
        return w1m1 + (w2m1 * M2) + M2


class res_and_post_treat(nn.Module):
    """
    Residual connection + Feedforward + Normalization block.
    Mirrors the "Post-Attention" block in Transformers.
    """

    def __init__(self, dim: int, activation: nn.Module = nn.ReLU, dropout: float = 0.3):
        """
        Initialize the residual and post-treatment layer.

        Args:
            dim (int): Feature dimension.
            activation (nn.Module, default=nn.ReLU): Activation function class (not instance).
            dropout (float, default=0.3): Dropout probability.
        """
        super().__init__()
        self.dim = dim
        self.O = nn.Linear(dim, dim)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim),
            activation(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=dim, out_features=dim),
        )

        # Normalization Layers
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

    def forward(
        self, old_tensor: torch.Tensor, new_tensor: torch.Tensor
    ) -> torch.Tensor:
        # Residual + projection
        new_tensor = self.O(new_tensor) + old_tensor

        # Flatten across nodes
        new_tensor = new_tensor.view(-1, self.dim)

        # Norm + FFN
        new_tensor = self.norm_1(new_tensor)
        new_tensor_ffn = self.ffn(new_tensor)
        new_tensor = self.norm_2(new_tensor + new_tensor_ffn)

        return new_tensor.view(old_tensor.shape)


class post_treat_y(nn.Module):
    """
    Updates global graph-level representation Y using aggregated node and edge stats.

    Aggregates nodes and edges via mean, min, max, and std, then projects them
    into the Y dimension and combines with current Y.
    """

    def __init__(self, node_dim: int, edge_dim: int, y_dim: int):
        """
        Initialize the post-treatment layer for global features.

        Args:
            node_dim (int): Node feature dimension.
            edge_dim (int): Edge feature dimension.
            y_dim (int): Global feature dimension.
        """
        super().__init__()

        self.lin_y = nn.Linear(y_dim, y_dim)
        self.lin_n = nn.Linear(4 * node_dim, y_dim)
        self.lin_e = nn.Linear(4 * edge_dim, y_dim)
        self.lin_out = nn.Linear(y_dim, y_dim)

    def forward(
        self, X: torch.Tensor, E: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        # Node stats
        mean_node = X.mean(dim=1)
        min_node = X.min(dim=1)[0]
        max_node = X.max(dim=1)[0]
        std_node = X.std(dim=1)

        # Edge stats
        mean_edge = E.mean(dim=(1, 2))
        min_edge = E.min(dim=2)[0].min(dim=1)[0]
        max_edge = E.max(dim=2)[0].max(dim=1)[0]
        std_edge = E.std(dim=(1, 2))

        # Projections
        x_rep = torch.hstack((mean_node, min_node, max_node, std_node))
        x_rep = self.lin_n(x_rep)
        e_rep = torch.hstack((mean_edge, min_edge, max_edge, std_edge))
        e_rep = self.lin_e(e_rep)
        y_rep = self.lin_y(Y)

        agg = x_rep + e_rep + y_rep
        out = self.lin_out(agg)
        return agg  # TODO: Check this - maybe return `out` ???


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention over node features (Transformer-style).
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        """
        Initialize the attention block.

        Args:
            in_dim (int): Input node feature dimension.
            out_dim (int): Output node feature dimension.
            num_heads (int): Number of attention heads.
        """
        super().__init__()

        self.num_heads = num_heads
        self.out_dim = out_dim
        self.dim_by_head = out_dim // num_heads

        self.Q = nn.Linear(in_dim, out_dim)
        self.K = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)

    def qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.Q(x).reshape(x.shape[0], x.shape[1], self.num_heads, self.dim_by_head)
        k = self.K(x).reshape(x.shape[0], x.shape[1], self.num_heads, self.dim_by_head)
        v = self.V(x).reshape(x.shape[0], x.shape[1], self.num_heads, self.dim_by_head)
        return q, k, v

    def mha(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch:
        # Shape: (batch, heads, nodes, features)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attention_weights = torch.matmul(q, k.transpose(-2, -1))
        attention_weights = attention_weights / (self.dim_by_head**0.5)
        attention_weights = F.softmax(attention_weights, dim=-1)

        out = torch.matmul(attention_weights, v)
        return out.permute(0, 2, 1, 3).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape b x n x d
        q, k, v = self.qkv(x)
        out = self.mha(q, k, v)
        out = out.view(x.shape[0], x.shape[1], self.out_dim)
        return out


class AttentionBlockEdges(AttentionBlock):
    """
    Edge-aware multi-head attention.

    Extends node attention by conditioning attention weights on edge embeddings.
    """

    def __init__(self, node_dim: int, edge_dim: int, num_heads: int):
        """
        Initialize the edge-aware attention block.

        Args:
            node_dim (int): Node feature dimension.
            edge_dim (int): Edge feature dimension.
            num_heads (int): Number of attention heads.
        """
        super().__init__(node_dim, node_dim, num_heads)

        self.edge_dim = edge_dim
        self.node_dim = node_dim

        self.E = nn.Linear(edge_dim, edge_dim)

        self.FilmEAtt = Film(edge_dim // num_heads, 1)

    def forward(
        self, x: torch.Tensor, e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x has shape b x n x d
        q, k, v = self.qkv(x)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Project edge features and reshape for heads
        e = self.E(e)
        e = e.reshape(
            e.shape[0],
            e.shape[1],
            e.shape[1],
            self.num_heads,
            self.edge_dim // self.num_heads,
        )
        e = e.permute(0, 3, 1, 2, 4)

        pre_attention_weights = torch.matmul(q, k.transpose(-2, -1))
        pre_attention_weights = pre_attention_weights / (self.dim_by_head**0.5)
        pre_attention_weights = self.FilmEAtt(e, pre_attention_weights)
        attention_weights = F.softmax(pre_attention_weights.clamp(-5, 5), dim=-1)

        # Node update
        out = torch.matmul(attention_weights, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(x.shape[0], x.shape[1], self.node_dim)

        # Edge update
        new_e = e * pre_attention_weights.unsqueeze(-1)
        new_e = new_e.permute(0, 2, 3, 1, 4).contiguous()
        new_e = new_e.view(
            new_e.shape[0], new_e.shape[1], new_e.shape[2], self.edge_dim
        )

        return out, new_e


def make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    nb_layers: int,
    activation: nn.Module,
    dropout: float,
    norm: nn.Module | None,
) -> nn.Module:
    """
    Utility function to construct a feed-forward MLP.

    Args:
        in_dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        out_dim (int): Output dimension.
        nb_layers (int): Number of layers.
        activation (torch.nn.Module): Activation function class.
        dropout (float): Dropout probability.
        norm (nn.Module (Optional)): Normalization layer class (e.g., nn.BatchNorm1d).

    Returns:
        nn.Module: The MLP.
    """
    lays = []
    if nb_layers > 1:
        # Create (nb_layers - 1) hidden layers
        for l in range(nb_layers - 1):
            dim_in = in_dim if l == 0 else hidden_dim
            lays.append(nn.Linear(dim_in, hidden_dim))
            lays.append(activation())

            if norm is not None:
                lays.append(norm(hidden_dim))

            if dropout > 0.0:
                lays.append(nn.Dropout(dropout))

        lays.append(nn.Linear(hidden_dim, out_dim))

    else:
        lays.append(nn.Linear(in_dim, out_dim))
        lays.append(activation())

    return nn.Sequential(*lays)
