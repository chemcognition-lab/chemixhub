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

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum


class LossEnum(StrEnum):
    """Basic str enum for molecule aggregators."""

    mae = enum.auto()
    mse = enum.auto()


LOSS_MAP = {
    LossEnum.mae: nn.L1Loss,
    LossEnum.mse: nn.MSELoss,
}


def log_metrics(metrics: torchmetrics.MetricCollection, epoch, logger, mode="train"):
    if isinstance(logger, WandBLogger):
        for metric in metrics.keys():
            logger.log(f"{mode}_{metric}", metrics[metric], epoch)
    else:
        logger.log({f"{mode}_{metric}": metrics[metric] for metric in metrics})


def compute_metrics(metrics: torchmetrics.MetricCollection, info):
    metric_dict = {k: item.cpu().item() for k, item in metrics.compute().items()}
    return info | metric_dict


def train_one_epoch(
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    loss_fn,
    metrics: torchmetrics.MetricCollection,
    device: torch.device,
):
    train_loss = 0
    num_batch = 0

    # for i, batch in tqdm.tqdm(enumerate(train_loader)):
    for i, batch in enumerate(train_loader):
        train_indices = batch["ids"].to(device)
        train_features = batch["features"].to(device)
        train_fractions = batch["fractions"].to(device)
        train_context = batch["context"].to(device)
        train_labels = batch["label"].to(device)

        optimizer.zero_grad()

        y_pred = model(train_features, train_indices, train_fractions, train_context)
        loss = loss_fn(y_pred.view(-1), train_labels.view(-1))
        metrics.update(y_pred.flatten(), train_labels)

        loss.backward()
        optimizer.step()

        num_batch += i
        train_loss += loss.detach().cpu().item()

    # avg loss and metric per batch
    overall_train_loss = train_loss / (num_batch + 1)
    overall_train_metrics = compute_metrics(metrics, {"loss": overall_train_loss})

    return overall_train_metrics


def validate_one_epoch(
    val_loader: DataLoader,
    model: nn.Module,
    loss_fn,
    metrics: torchmetrics.MetricCollection,
    device: torch.device,
):
    val_loss = 0
    num_batch = 0

    # for i, batch in tqdm.tqdm(enumerate(val_loader)):
    for i, batch in enumerate(val_loader):
        val_indices = batch["ids"].to(device)
        val_features = batch["features"].to(device)
        val_fractions = batch["fractions"].to(device)
        val_context = batch["context"].to(device)
        val_labels = batch["label"].to(device)

        y_pred = model(val_features, val_indices, val_fractions, val_context)

        loss = loss_fn(y_pred.view(-1), val_labels.view(-1))
        metrics.update(y_pred.flatten(), val_labels)

        num_batch += i
        val_loss += loss.detach().cpu().item()

    # avg loss and metric per batch
    overall_val_loss = val_loss / (num_batch + 1)
    overall_val_metrics = compute_metrics(metrics, {"loss": overall_val_loss})

    return overall_val_metrics


def train(
    root_dir: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_type: str,
    lr_mol_encoder: float,
    lr_other: float,
    device,
    weight_decay: float,
    max_epochs: int,
    patience: Optional[int] = None,
    experiment_name: Optional[str] = None,
    wandb_logger: Optional[WandBLogger] = None,
):
    loss_fn = LOSS_MAP[loss_type]()

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

    # Run a dummy forward pass to initialize Lazy layers
    dummy_batch = next(iter(train_loader))
    dummy_indices = dummy_batch["ids"].to(device)
    dummy_features = dummy_batch["features"].to(device)
    dummy_fractions = dummy_batch["fractions"].to(device)
    dummy_context =dummy_batch["context"].to(device)

    with torch.no_grad():
        _ = model(dummy_features, dummy_indices, dummy_fractions, dummy_context)

    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    mol_encoder_params = list(model.mol_encoder.parameters())
    other_params = [p for name, p in model.named_parameters() if not name.startswith("mol_encoder.")]

    optimizer = torch.optim.Adam([
        {
            'params': mol_encoder_params,
            'lr': lr_mol_encoder,
            'weight_decay': weight_decay,
        },
        {
            'params': other_params,
            'lr': lr_other,
            'weight_decay': weight_decay,
        }
    ])

    es = EarlyStopping(model, patience=patience, mode="minimize")

    # pbar = tqdm.tqdm(range(max_epochs))
    pbar = range(max_epochs)
    for epoch in pbar:
        model.train()

        overall_train_metrics = train_one_epoch(
            train_loader,
            optimizer,
            model,
            loss_fn,
            metrics,
            device,
        )

        metrics.reset()

        # validation + early stopping
        model.eval()
        with torch.no_grad():
            overall_val_metrics = validate_one_epoch(
                val_loader,
                model,
                loss_fn,
                metrics,
                device,
            )

        if wandb_logger:
            log_metrics(
                metrics=overall_train_metrics,
                epoch=epoch,
                logger=wandb_logger,
                mode="train",
            )
            log_metrics(
                metrics=overall_val_metrics,
                epoch=epoch,
                logger=wandb_logger,
                mode="val",
            )

        # pbar.set_description(
        #     f"Train: {overall_train_metrics['loss']:.4f} | Test: {overall_val_metrics['loss']:.4f} | Test pearson: {overall_val_metrics['PearsonCorrCoef']:.4f} | Test MAE: {overall_val_metrics['MeanAbsoluteError']:.4f}"
        # )

        stop = es.check_criteria(overall_val_metrics["MeanAbsoluteError"], model)
        if stop:
            print(f"Early stop reached at {es.best_step} with {es.best_value}")
            break

        metrics.reset()

    # save model weights
    best_model_dict = es.restore_best()
    model.load_state_dict(best_model_dict)  # load the best one trained
    torch.save(model.state_dict(), f"{root_dir}/best_model_dict_{experiment_name}.pt")

    # final eval
    print("Using best model for a final eval")
    model.eval()
    with torch.no_grad():
        overall_val_metrics = validate_one_epoch(
            val_loader,
            model,
            loss_fn,
            metrics,
            device,
        )

    if wandb_logger:
        log_metrics(
            metrics=overall_val_metrics,
            epoch=epoch+1,
            logger=wandb_logger,
            mode="val",
        )

    if wandb_logger and wandb_logger == WandBLogger:
        wandb_logger.close()
