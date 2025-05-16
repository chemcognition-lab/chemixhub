import sys
import torch

import math
import numpy as np
import copy
import functools
from pprint import pprint

import numpy as np
import omegaconf
from torch import nn
import torchmetrics.functional as F
from mixhub.data.featurization import UNK_TOKEN
from scipy.stats import pearsonr


if sys.version_info >= (3, 11):
    import enum
    from enum import StrEnum
else:
    from backports.strenum import StrEnum


def compute_key_padding_mask(x, unk_token=UNK_TOKEN):
    return (x == unk_token).all(dim=2)


def print_conf(conf: omegaconf.DictConfig):
    """Pretty prints a configuration object."""
    pprint(omegaconf.OmegaConf.to_container(conf, resolve=True))


# Activation (FiLM and cosine head)
class ActivationEnum(StrEnum):
    """Basic str enum for activation functions."""

    sigmoid = enum.auto()
    relu = enum.auto()
    hardtanh = enum.auto()


ACTIVATION_MAP = {
    ActivationEnum.sigmoid: nn.Sigmoid,
    ActivationEnum.relu: nn.ReLU,
    ActivationEnum.hardtanh: functools.partial(nn.Hardtanh, min_val=0.0, max_val=1.0),

}


class EarlyStopping:
    """Stop training early if a metric stops improving.

    Models often benefit from stoping early after a metric stops improving.
    This implementation assumes the monitored value will be loss-like
    (i.g. val_loss) and will checkpoint when reaching a new best value.
    Checkpointed value can be restored.

    Args:
      model: model to checkpoint.
      patience: number of iterations before flaggin a stop.
      min_delta: minimum value to quanlify as an improvement.
      checkpoint_interval: number of iterations before checkpointing.
      mode: maximise or minimise the monitor value
    """

    def __init__(
        self,
        model: torch.nn.Module,
        patience: int = 100,
        min_delta: float = 0,
        checkpoint_interval: int = 1,
        mode: bool = "maximize",
    ):
        self.patience = patience
        self.min_delta = np.abs(min_delta)
        self.wait = 0
        self.best_step = 0
        self.checkpoint_count = 0
        self.checkpoint_interval = checkpoint_interval
        self.values = []
        self.best_model = copy.deepcopy(model.state_dict())
        if mode == "maximize":
            self.monitor_op = lambda a, b: np.greater_equal(a - min_delta, b)
            self.best_value = -np.inf
        elif mode == "minimize":
            self.monitor_op = lambda a, b: np.less_equal(a + min_delta, b)
            self.best_value = np.inf
        else:
            raise ValueError("Invalid mode for early stopping.")

    def check_criteria(self, monitor_value: float, model: torch.nn.Module) -> bool:
        """Gets learing rate based on value to monitor."""
        self.values.append(monitor_value)
        self.checkpoint_count += 1

        if self.monitor_op(monitor_value, self.best_value):
            self.best_value = monitor_value
            self.best_step = len(self.values) - 1
            self.wait = 0
            if self.checkpoint_count >= self.checkpoint_interval:
                self.checkpoint_count = 0
                self.best_model = copy.deepcopy(model.state_dict())
        else:
            self.wait += 1

        return self.wait >= self.patience

    def restore_best(self):
        print(
            f"Restoring checkpoint at step {self.best_step} with best value at {self.best_value}"
        )
        return self.best_model


TORCH_METRIC_FUNCTIONS = {
    "pearson": F.pearson_corrcoef,
    "rmse": lambda pred, targ: F.mean_squared_error(pred, targ, squared=False),
    # "spearman": F.spearman_corrcoef,
    "kendall": F.kendall_rank_corrcoef,
    # "r2": F.r2_score,
    # "mae": F.mean_absolute_error,
}


def cast_to_torch(x):
    return torch.from_numpy(x) if isinstance(x, np.ndarray) else x


def compute_metrics(y_true, y_pred, metric_functions):
    """Calculate metrics on a set of predictions."""
    y_true = cast_to_torch(y_true.flatten())
    y_pred = cast_to_torch(y_pred.flatten())
    metrics = {}
    for name, func in metric_functions.items():
        metrics[name] = func(y_true, y_pred).detach().cpu().item()
    return metrics


evaluate = functools.partial(compute_metrics, metric_functions=TORCH_METRIC_FUNCTIONS)


def cast_float(x):
    return x if isinstance(x, float) else x.item()


def bootstrap_ci(true_values, predictions, metric_fn, num_samples=500, alpha=0.05):
    """
    Calculates a bootstrap confidence interval for a given metric.

    Args:
        true_values: True values of the target variable.
        predictions: Predicted values.
        metric: A function that takes true_values and predictions as input and returns a scalar metric.
        num_samples: Number of bootstrap samples to generate.
        alpha: Significance level for the confidence interval.

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.
    """

    n = len(true_values)
    values = []
    for _ in range(num_samples):
        indices = np.random.randint(0, n, n)
        bootstrap_true = true_values[indices]
        bootstrap_pred = predictions[indices]
        value = metric_fn(bootstrap_true, bootstrap_pred)

        if torch.isnan(value) and metric_fn ==F.pearson_corrcoef:
            r, p_value = pearsonr(bootstrap_true, bootstrap_pred)
            value = r
            print(bootstrap_true)
            print(bootstrap_pred)
            print(value)


        values.append(cast_float(value))
    lower_bound = np.percentile(values, alpha / 2 * 100)
    upper_bound = np.percentile(values, (1 - alpha / 2) * 100)

    return lower_bound, upper_bound, values