import argparse
import os
import torch
import sys
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchtune.training.metric_logging import WandBLogger
from omegaconf import OmegaConf

from mixhub.data.dataset import MixtureTask
from mixhub.data.data import DATA_CATALOG
from mixhub.data.featurization import FEATURIZATION_TYPE
from mixhub.data.collate import custom_collate
from mixhub.data.splits import SplitLoader

from mixhub.model.train import train
from mixhub.model.model_builder import build_mixture_model


def main(
    config,
    experiment_name,
    wandb_logger=None,
):
    config = copy.deepcopy(config)

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    print(f"Running on: {device}")

    root_dir = config.root_dir
    os.makedirs(root_dir, exist_ok=True)

    featurization = config.dataset.featurization

    if FEATURIZATION_TYPE[featurization] == "graphs" and config.mixture_model.mol_encoder.type != "gnn":
        raise ValueError(f"featurization is:{FEATURIZATION_TYPE[featurization]} but molecule encoder is: {config.mol_encoder.type}")

    if FEATURIZATION_TYPE[featurization] == "tensors" and config.mixture_model.mol_encoder.type == "gnn":
        raise ValueError(f"featurization is:{FEATURIZATION_TYPE[featurization]} but molecule encoder is: {config.mol_encoder.type}")

    # Dataset
    dataset = DATA_CATALOG[config.dataset.name]()
    property = config.dataset.property

    mixture_task = MixtureTask(
        property=property,
        dataset=dataset,
        featurization=featurization,
    )

    # Split Loader
    split_loader = SplitLoader(split_type="kfold")
    train_indices, val_indices, _ = split_loader(
        property=mixture_task.property,
        cache_dir=mixture_task.dataset.data_dir,
        split_num=0,
    )

    # Data Loader
    train_dataset = Subset(mixture_task, train_indices.tolist())
    val_dataset = Subset(mixture_task, val_indices.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=custom_collate,
        num_workers=config.num_workers,
    )

    model = build_mixture_model(config=config.mixture_model)
    model = model.to(device)

    # Save hyper parameters    
    OmegaConf.save(config, f"{root_dir}/hparams_{experiment_name}.yaml")

    # Training
    train(
        root_dir=root_dir,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_type=config.loss_type,
        lr_mol_encoder=config.lr_mol_encoder,
        lr_other=config.lr_other,
        device=device,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        patience=config.patience,
        experiment_name=experiment_name,
        wandb_logger=wandb_logger,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training and evaluation on a random split")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    parser.add_argument("--experiment_name", type=str, default="test_run", help="Name of the experiment")
    parser.add_argument("--wandb_project", type=str, default=None, help="Name of the wandb project (optional)")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    experiment_name = args.experiment_name

    if args.wandb_project is not None:
        wandb_logger = WandBLogger(project=args.wandb_project)
    else:
        wandb_logger = None

    main(
        config=config,
        experiment_name=experiment_name,
        wandb_logger=wandb_logger,
    )
