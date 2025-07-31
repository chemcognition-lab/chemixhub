import argparse
import os
import torch
import sys
import copy
import pandas as pd
import numpy as np

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
from mixhub.model.predict import predict
from mixhub.model.model_builder import build_mixture_model


def main(
    config,
    original_root_dir,
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

    # Zero-shot Dataset
    dataset = DATA_CATALOG[config.dataset.name]()
    property = config.dataset.property


    mixture_task = MixtureTask(
        property=property,
        dataset=dataset,
        featurization=featurization,
    )

    for i in range(5):

        run_name = experiment_name + f"_{i}"

        print(f"Best model on split {i}")
        model = build_mixture_model(config=config.mixture_model)

        # Load model
        checkpoint = torch.load(f"{original_root_dir}/best_model_dict_cv_{i}.pt", weights_only=False)
        model.load_state_dict(checkpoint)

        model = model.to(device)

        print(f"Testing on split {i}")

        # Data Loader (one big batch)
        test_loader = DataLoader(
            mixture_task,
            batch_size=mixture_task.__len__(),
            collate_fn=custom_collate,
            num_workers=config.num_workers,
        )

        metric_dict, y_pred, y_test = predict(
            model=model,
            test_loader=test_loader,
            device=device,
        )

        print(metric_dict)
        test_metrics = pd.DataFrame(metric_dict, index=["metrics"]).transpose()
        test_metrics.to_csv(os.path.join(config.root_dir, f"{run_name}_test_metrics.csv"))
    
        y_pred = y_pred.detach().cpu().numpy().flatten()
        y_test = y_test.detach().cpu().numpy().flatten()
        test_predictions = pd.DataFrame(
            {
                "Predicted_Experimental_Values": y_pred,
                "Ground_Truth": y_test,
                "MAE": np.abs(y_pred - y_test),
            },
            index=range(len(y_pred)),
        )
        test_predictions.to_csv(os.path.join(config.root_dir, f"{run_name}_test_predictions.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run zero shot experiment")
    parser.add_argument("model_checkpoint_dir", type=str, help="Path to the checkpoint directory")
    parser.add_argument("model_type", type=str, help="Specify model type keyword")
    parser.add_argument("ft_target_dataset", type=str, help="Specify desired dataset to infer on")
    parser.add_argument("ft_target_property", type=str, help="Specify desired property to infer on")
    parser.add_argument("--wandb_project", type=str, default=None, help="Name of the wandb project (optional)")

    args = parser.parse_args()

    original_root_dir = os.path.abspath(args.model_checkpoint_dir)

    config_path = os.path.join(original_root_dir, "hparams_cv.yaml")
    config = OmegaConf.load(config_path)

    original_dataset = config.dataset.name
    original_property = config.dataset.property.lower().replace(' ', '_')
    task = f"{original_dataset}_{original_property}_{args.model_type}"

    # Overwrite config root_dir
    config.dataset.name = args.ft_target_dataset
    config.dataset.property = args.ft_target_property
    config.root_dir = f"/project/a/aspuru/rajao/mixture-datasets/results_bighp/bm_{task}/{config.dataset.name}/{config.dataset.property.lower().replace(' ', '_')}"

    experiment_name = "finetune"
 
    if args.wandb_project is not None:
        wandb_logger = WandBLogger(project=args.wandb_project)
    else:
        wandb_logger = None

    main(
        config=config,
        original_root_dir=original_root_dir,
        experiment_name=experiment_name,
        wandb_logger=wandb_logger,
    )
