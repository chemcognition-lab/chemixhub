import enum
import sys
from typing import Optional

import torch
import torch.nn as nn
import torchmetrics
import tqdm
from torch.utils.data import DataLoader
from torchtune.training.metric_logging import WandBLogger

from mixhub.model.utils import EarlyStopping


def predict(
    model: nn.Module,
    test_loader: DataLoader,
    device,
):

    # Metrics
    metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.PearsonCorrCoef(),
            torchmetrics.R2Score(),
            torchmetrics.MeanAbsoluteError(),
            torchmetrics.MeanSquaredError(),
            torchmetrics.KendallRankCorrCoef(),
        ]
    )

    metrics = metrics.to(device)

    model.eval()

    all_preds = torch.Tensor().to(device)
    all_labels = torch.Tensor().to(device)
    for i, batch in enumerate(test_loader):

        test_indices = batch["ids"].to(device)
        test_features = batch["features"].to(device)
        test_fractions = batch["fractions"].to(device)
        test_context = batch["context"].to(device)
        test_labels = batch["label"].to(device)

        with torch.no_grad():
            y_pred = model(test_features, test_indices, test_fractions, test_context)

        all_labels = torch.cat((all_labels, test_labels))
        all_preds = torch.cat((all_preds, y_pred.flatten()))
        metrics.update(y_pred.flatten(), test_labels)

    metric_dict = {k: item.cpu().item() for k, item in metrics.compute().items()}

    return metric_dict, all_preds, all_labels