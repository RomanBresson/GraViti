import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Any
from src.model.grale_losses.grale_functions import SinkhornMatcher

def init_weights_gelu(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class AutoEncoder(nn.Module):
    """
    Base `AutoEncoder` class for graph representation learning.

    Encodes input batch into latent embeddings, decodes the embeddings into graph
    predictions, and optionally computes a regression prediction from the embeddings.

    Args:
        encoder (torch.nn.Module): Encoder network mapping input data to latent representation.
        decoder (torch.nn.Module): Decoder network reconstructing graph from latent space.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, use_matcher: bool=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = self.encoder.latent_dim

        self.beta = 0.0
        self.has_regressor = False
        self.regressor = None

        self.mu = None
        self.log_sigma = None

        self.matcher = None
        if use_matcher:
            self.matcher = SinkhornMatcher(
                node_model_dim=encoder.node_dim_for_matcher,
                matcher_dim=32,
                n_nodes_max=decoder.num_nodes_default,
                max_iter_sinkhorn=50,
                normalize_cost_matrix=True,
                epsilon=0.0001,
                fixed_n_iters_sinkhorn=True
            )

        self.apply(init_weights_gelu)

    def forward(
        self, data: Dict[str, Any], hard_matching: bool=False, override_size_padding: Optional[int]=None
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
    ]:
        """
        Forward pass.

        Args:
            data (Dict[str,Any]): Input data batch as dictionary.
        Returns:
            Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,Optional[torch.Tensor]]:
                (X, E, y, predicted_sizes, used_node_filter, predicted_hydrogens, predicted_formal_charge)
        """
        x_graph, node_embeddings_encoder = self.encode(data)
        outputs = self.decode(
            x_graph,
            force_size_graph=data.get("graph_sizes"),
            override_size_padding=override_size_padding
        )

        predicted_perm_matrix = None
        if self.matcher is not None:
            node_embeddings_decoder = outputs.get('node_embeddings_decoder')
            node_mask = ~data.get('node_mask').bool()
            predicted_perm_matrix, log_solver = self.matcher.forward(
                node_embeddings_inputs = node_embeddings_encoder, 
                node_masks_inputs = node_mask,
                node_embeddings_outputs = node_embeddings_decoder,
                hard = hard_matching)

        outputs['latent_embeddings'] = x_graph
        outputs['latent_mu'] = self.mu
        outputs['latent_log_sigma'] = self.log_sigma
        outputs['predicted_perm_matrix'] = predicted_perm_matrix
        return outputs

    def encode(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Encodes input data into latent representation.

        Args:
            data (Dict[str,Any]): Input data batch as dictionary.
        Returns:
            torch.Tensor: Latent representation.
        """
        x_g, node_embeddings_encoder = self.encoder(data)
        return x_g, node_embeddings_encoder

    def decode(
        self, x_g: torch.Tensor, force_size_graph: Optional[Any] = None, override_size_padding: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decodes latent representation into graph predictions.

        Args:
            x_g (torch.Tensor): Latent representation.
            enforced_graph_sizes (torch.Tensor, optional): Mask to filter nodes during decoding.
        Returns:
            Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
            * Node predictions: (batch_size, num_nodes, X_classes)
            * Edge predictions: (batch_size, num_nodes, num_nodes, E_classes)
            * Global feature predictions
            * Predicted graph sizes
            * The node mask actually used inside the decoder
        """
        outputs = self.decoder(
            latent_embeddings=x_g,
            force_size_graph=force_size_graph,
            override_size_padding=override_size_padding
        )
        return outputs

    def loss_kl(self) -> torch.Tensor:
        """
        KL divergence term (zero for vanilla AutoEncoder).
        Overridden in VariationalAutoEncoder.

        Returns:
            torch.Tensor: Always zero.
        """
        return torch.tensor(0.0, device=next(self.parameters()).device)


class VariationalAutoEncoder(AutoEncoder):
    """
    `Variational AutoEncoder` (VAE) for graph representation learning.

    Extends the `AutoEncoder` class, introducing stochastic latent variables (mu, sigma)
    and KL divergence regularization.

    Args:
        encoder (torch.nn.Module): Encoder producing mean and log-variance.
        decoder (torch.nn.Module): Decoder reconstructing graph structure.
        beta (float): Weight for KL divergence in loss.
        regressor (torch.nn.Module, optional): Optional downstream regressor on latent space.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        beta: float,
        regressor: nn.Module = None,
        use_matcher: bool = False
    ):
        super().__init__(encoder, decoder, use_matcher=use_matcher)
        self.beta = beta

        self.has_regressor = regressor is not None
        self.regressor = regressor
        self.encoder.latent_dim = self.encoder.latent_dim//2
        self.latent_dim = self.encoder.latent_dim

    def reparameterize(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        """
        Reparameterization trick: samples z = mu + eps * sigma.

        Args:
            mu (torch.Tensor): Latent mean.
            log_sigma (torch.Tensor): Latent log standard deviation.

        Returns:
            torch.Tensor: Sampled latent variable.
        """
        if self.training:
            eps = torch.randn_like(log_sigma)
            std = torch.exp(log_sigma)
            return mu + eps * std
        else:
            return mu

    def encode(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Encodes input data into latent distribution parameters (mu, log_sigma),
        then samples latent variable via reparameterization trick.

        Args:
            data (Dict[str,Any]): Input data batch as dictionary.
        Returns:
            torch.Tensor: Sampled latent representation.
        """
        x_g, node_embeddings_encoder = self.encoder(data)

        # Split output into mean and log_sigma
        mu_g = x_g[:, : x_g.shape[1] // 2]
        log_sigma_g = x_g[:, x_g.shape[1] // 2 :]

        # Clamp log_sigma to prevent numerical instability
        log_sigma_g = log_sigma_g.clamp(max=10)

        self.mu = mu_g
        self.log_sigma = log_sigma_g
        x_g = self.reparameterize(self.mu, self.log_sigma)
        return x_g, node_embeddings_encoder

    def loss_kl(self):
        """
        Computes KL divergence between approximate posterior q(z|x) and prior p(z).

        KL(q(z|x) | p(z)) = - 1/2 sum[1 - mu^2 - std^2 + 2log(std)]

        Returns:
            torch.Tensor: KL divergence term (averaged over batch).
        """

        kl = torch.sum(
            1 - self.mu.pow(2) - torch.exp(self.log_sigma).pow(2) + 2 * self.log_sigma,
            dim=1,
        )
        kl = -0.5 * torch.mean(kl)

        return kl
