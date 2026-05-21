from pathlib import Path
from typing import Tuple, Union

from .components import (
    AutoEncoder,
    VariationalAutoEncoder,
    CatFlowWrapper,
    GraphTransformerDecoder,
    make_mlp,
)

from .metadata import ModelMetadata
from .checkpoint import ModelCheckpoint
from ..utils.training_config import TrainingConfig


class ModelBuilder:
    @classmethod
    def build_from_config(
        cls, config: TrainingConfig, metadata: ModelMetadata
    ) -> AutoEncoder | VariationalAutoEncoder:
        """
        Build model from given configuration for training or inference.

        Args:
            config (TrainingConfig): Training parameters.
            metadata (ModelMetadata): Data characteristics and preprocessing info.

        Returns:
            AutoEncoder|VariationalAutoEncoder: The constructed autoencoder model.
        """
        # Create Encoder and Decoder
        encoder = cls._create_encoder(config, metadata)
        decoder = cls._create_decoder(config, metadata)

        # Regressor is not used
        regressor = None
        """
        has_reg = config.reg_weight != 0
        if has_reg:
            nb_targets = target_mu.shape[0]
            regressor = make_mlp(in_dim=config.latent_size, hidden_dim=1024, out_dim=nb_targets, nb_layers=2, activation=nn.SiLU, dropout=config.dropout, norm=None).to(device)
        """

        # Create autoencoder
        if config.variational:
            model = VariationalAutoEncoder(
                encoder=encoder,
                decoder=decoder,
                beta=config.beta,
                regressor=regressor,
                use_matcher=config.use_grale_loss
            )
        else:
            model = AutoEncoder(encoder, decoder, use_matcher=config.use_grale_loss)

        return model

    @classmethod
    def build_from_checkpoint(
        cls, checkpoint: Union[ModelCheckpoint, Path, str]
    ) -> AutoEncoder | VariationalAutoEncoder:
        """
        Build model from given checkpoint.

        Args:
            checkpoint (ModelCheckpoint|Path|str):
                A checkpoint instance or path to load a checkpoint from.

        Returns:
            AutoEncoder|VariationalAutoEncoder: The constructed autoencoder model.
        """
        if not isinstance(checkpoint, ModelCheckpoint):
            checkpoint = ModelCheckpoint.load(Path(checkpoint))

        config = TrainingConfig.from_state_dict(checkpoint.config_state_dict)
        metadata = ModelMetadata.from_state_dict(checkpoint.metadata_state_dict)
        model = cls.build_from_config(config, metadata)

        model.load_state_dict(checkpoint.model_state_dict)
        return model

    @classmethod
    def _create_encoder(
        cls, config: TrainingConfig, metadata: ModelMetadata
    ) -> CatFlowWrapper:
        """
        Create the CatFlow-based encoder based on given configuration

        Args:
            config (TrainingConfig): Training parameters.
            metadata (ModelMetadata): Data characteristics and preprocessing info.

        Returns:
            CatFlowWrapper: The constructed encoder model.
        """
        out_size_encoder = config.latent_size * (2 if config.variational else 1)

        embedding_dims, hidden_dims, hidden_mlp_dims = cls._get_transformer_dims(
            config, metadata
        )

        return CatFlowWrapper(
            n_layers=config.encoder_layers,
            input_dims=embedding_dims,
            hidden_dims=hidden_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            X_classes=metadata.num_node_features,
            E_classes=metadata.num_edge_features,
            y_size=metadata.y_size,
            latent_dim=out_size_encoder,
            use_atom_attr=config.use_atom_attr,
            atom_attr_dim=metadata.atom_attr_dim,
            sigma_symmetry_breaking_noise=0.01,
            dropout=config.dropout,
        )

    @classmethod
    def _create_decoder(
        cls, config: TrainingConfig, metadata: ModelMetadata
    ) -> GraphTransformerDecoder:
        """
        Create a decoder based on given configuration.

        Args:
            config (TrainingConfig): Training parameters.
            metadata (ModelMetadata): Data characteristics and preprocessing info.

        Returns:
            GraphTransformerDecoder: The constructed decoder model.
        """
        embedding_dims, hidden_dims, hidden_mlp_dims = cls._get_transformer_dims(
            config, metadata
        )

        return GraphTransformerDecoder(
            n_layers=config.decoder_layers,
            input_dims=embedding_dims,
            hidden_dims=hidden_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            X_classes=metadata.num_node_features,
            E_classes=metadata.num_edge_features,
            latent_dim=config.latent_size,
            num_nodes_default=metadata.max_size,
            sigma=config.decoder_sigma,
            dropout=config.dropout,
            max_hydrogen_values=metadata.max_hydrogens,
            bounds_formal_charge_values=metadata.formal_charge_bounds,
            reinject_size=metadata.reinject_size,
            predict_node_mask=config.use_grale_loss
        )

    @classmethod
    def _get_transformer_dims(
        cls, config: TrainingConfig, metadata: ModelMetadata
    ) -> Tuple[dict, dict, dict]:
        embedding_dims = {
            "X": config.encoder_output_size,
            "E": config.encoder_output_size // 2,
            "y": config.encoder_output_size,
            "pe_lap": metadata.pe_lap_size,
            "pe_rw": metadata.pe_rw_size,
            "PE": 32,
        }

        hidden_dims = {
            "dx": config.encoder_hidden_size,
            "de": config.encoder_hidden_size // 2,
            "dy": config.encoder_hidden_size,
            "n_head": config.encoder_heads,
            "dim_ffX": config.encoder_hidden_size * 2,
            "dim_ffE": config.encoder_hidden_size,
            "dim_ffy": config.encoder_hidden_size * 2,
        }

        hidden_mlp_dims = {
            "X": config.encoder_hidden_size * 2,
            "E": config.encoder_hidden_size,
            "y": config.encoder_hidden_size * 2,
        }

        return embedding_dims, hidden_dims, hidden_mlp_dims
