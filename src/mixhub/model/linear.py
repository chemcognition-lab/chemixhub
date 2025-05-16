import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from xgboost import XGBRegressor
from mixhub.model.aggregation import MeanAggregation, PrincipalNeighborhoodAggregation
from mixhub.model.utils import compute_key_padding_mask
from typing import Tuple

if sys.version_info >= (3, 11):
    import enum
    from enum import StrEnum
else:
    from backports.strenum import StrEnum


class AggEnum(StrEnum):
    """Basic str enum for molecule aggregators."""

    mean = enum.auto()
    pna = enum.auto()


class SimpleModelEnum(StrEnum):
    """Basic str enum for simple ML model."""

    linear_regression = enum.auto()
    sgd_linear_regression = enum.auto()
    xgboost = enum.auto()


def fit_simple_model(
    x: torch.Tensor,
    y: torch.Tensor,
    mol_aggregation: str,
    model_type: str,
) -> LinearRegression:


    mol_aggregation_methods = {
        AggEnum.mean: MeanAggregation(),
        AggEnum.pna: PrincipalNeighborhoodAggregation(),
    }

    classic_models = {
        SimpleModelEnum.linear_regression: LinearRegression(),
        SimpleModelEnum.sgd_linear_regression: SGDRegressor(max_iter=100),
        SimpleModelEnum.xgboost: XGBRegressor(),
    }

    with torch.no_grad():
        aggregator = mol_aggregation_methods[mol_aggregation]
        key_padding_mask = compute_key_padding_mask(x)
        mix_rep = aggregator(x, key_padding_mask)


    X_np = mix_rep.cpu().numpy()
    y_np = y.cpu().numpy()

    model = classic_models[model_type]
    model.fit(X_np, y_np)

    return model


def predict_simple_model(
    x: torch.Tensor,
    mol_aggregation: str,
    model: Tuple[LinearRegression, SGDRegressor, XGBRegressor],
) -> np.array:

    mol_aggregation_methods = {
        AggEnum.mean: MeanAggregation(),
        AggEnum.pna: PrincipalNeighborhoodAggregation(),
    }

    with torch.no_grad():
        aggregator = mol_aggregation_methods[mol_aggregation]
        key_padding_mask = compute_key_padding_mask(x)
        mix_rep = aggregator(x, key_padding_mask)

    X_np = mix_rep.cpu().numpy()

    return model.predict(X_np)


class FullyConnectedNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout_rate: float = 0.0,
    ):
        super(FullyConnectedNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self.layers = nn.Sequential(nn.LazyLinear(hidden_dim), nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            # self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Dropout(p=dropout_rate))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x, x_ids, device):
        output = self.layers(x)
        return output