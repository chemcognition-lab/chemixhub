import sys
import functools
from mixhub.model.aggregation import (
    MeanAggregation,
    MaxAggregation,
    PrincipalNeighborhoodAggregation,
    PNAMixtureSizeScaled,
    AttentionAggregation,
    Set2SetAggregation,
)

from mixhub.model.predictor import (
    PredictiveHead,
    PhysicsPredictiveHead,
    ScaledCosineRegressor,
)
from mixhub.model.utils import ActivationEnum
from mixhub.model.graph import GraphNets
from mixhub.model.linear import FullyConnectedNet
from mixhub.model.mixture import SelfAttentionBlock, DeepSet, MixtureModel
from mixhub.data.graph_utils import NODE_DIM, EDGE_DIM


import torch
from torch import nn


if sys.version_info >= (3, 11):
    import enum
    from enum import StrEnum
else:
    from backports.strenum import StrEnum


# Molecule block
class MoleculeEncoderEnum(StrEnum):
    """Basic str enum for molecule encoders (en-to-end training)."""

    gnn = enum.auto()
    linear = enum.auto()

# Molecular Fractions
class FractionsAggregationEnum(StrEnum):
    """Basic str enum for mole fraction aggregation (en-to-end training)."""
    concat = enum.auto()
    multiply = enum.auto()


# Mixture block
class MixtureEncoderEnum(StrEnum):
    """Basic str enum for mixture encoders (en-to-end training)."""

    deepset = enum.auto()
    self_attn = enum.auto()

# Mixture block
class AggEnum(StrEnum):
    """Basic str enum for molecule aggregators."""

    mean = enum.auto()
    max = enum.auto()
    pna = enum.auto()
    scaled_pna = enum.auto()
    attn = enum.auto()
    set2set = enum.auto()

# Prediction head
class RegressorEnum(StrEnum):
    """Basic str enum for regressors."""
    mlp = enum.auto()
    physics_based = enum.auto()
    scaled_cosine = enum.auto()


def build_mixture_model(config):

    # Molecule block
    mol_encoder_methods = {
        MoleculeEncoderEnum.gnn: GraphNets(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            global_dim=config.mol_encoder.gnn.global_dim,
            hidden_dim=config.mol_encoder.gnn.hidden_dim,
            depth=config.mol_encoder.gnn.depth,
            dropout_rate=config.dropout_rate,
        ),
        MoleculeEncoderEnum.linear: nn.LazyLinear(
            # config.mol_encoder.gnn.global_dim,
            config.mol_encoder.output_dim,
        ),
    }

    # Projection layer
    project_input = nn.LazyLinear(config.attn_aggregation.embed_dim)

    # Mixture block
    mol_aggregation_methods = {
        AggEnum.mean: MeanAggregation(),
        AggEnum.max: MaxAggregation(),
        AggEnum.pna: PrincipalNeighborhoodAggregation(),
        AggEnum.scaled_pna: PNAMixtureSizeScaled(),
        AggEnum.attn: AttentionAggregation(
            embed_dim=config.attn_aggregation.embed_dim,
            num_heads=config.attn_num_heads,
            dropout_rate=config.dropout_rate,
        ),
        AggEnum.set2set: Set2SetAggregation(
            in_channels=config.set2set_aggregation.in_channels,
            processing_steps=config.set2set_aggregation.processing_steps,
        )
    }

    # Mixture block
    mixture_encoder_methods = {
        MixtureEncoderEnum.self_attn: SelfAttentionBlock(  # TODO: Rename
            num_layers=config.mix_encoder.num_layers,
            embed_dim=config.mix_encoder.embed_dim,
            num_heads=config.attn_num_heads,
            add_mlp=config.mix_encoder.self_attn.add_mlp,
            dropout_rate=config.dropout_rate,
            mol_aggregation=mol_aggregation_methods[config.mol_aggregation],
        ),
        MixtureEncoderEnum.deepset: DeepSet(
            embed_dim=config.mix_encoder.embed_dim,
            num_layers=config.mix_encoder.num_layers,
            dropout_rate=config.dropout_rate,
            mol_aggregation=mol_aggregation_methods[config.mol_aggregation],
        ),
    }

    # Prediction head
    regressor_type = {
        RegressorEnum.mlp: PredictiveHead(
            hidden_dim=config.regressor.hidden_dim,
            num_layers=config.regressor.num_layers,
            output_dim=config.regressor.mlp.output_dim,
            dropout_rate=config.dropout_rate,
        ),
        RegressorEnum.physics_based: PhysicsPredictiveHead(
            law=config.regressor.physics_based.law,
            hidden_dim=config.regressor.hidden_dim,
            num_layers=config.regressor.num_layers,
            dropout_rate=config.dropout_rate,
        ),
        RegressorEnum.scaled_cosine: ScaledCosineRegressor(
            output_dim=config.regressor.mlp.output_dim,
            act=ActivationEnum.hardtanh,
        ),
    }

    mixture_model = MixtureModel(
        mol_encoder=mol_encoder_methods[config.mol_encoder.type],
        projection_layer=project_input,
        mix_encoder=mixture_encoder_methods[config.mix_encoder.type],
        regressor=regressor_type[config.regressor.type],
        fraction_aggregation_type=config.fraction_aggregation.type,
        context_aggregation_type=config.context_aggregation.type,
        fraction_film_activation=config.fraction_aggregation.film.activation,
        context_film_activation=config.context_aggregation.film.activation,
        fraction_film_output_dim=config.fraction_aggregation.film.output_dim,
        context_film_output_dim=config.context_aggregation.film.output_dim,
    )

    return mixture_model