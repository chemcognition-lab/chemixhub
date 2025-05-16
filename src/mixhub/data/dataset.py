import logging
import os
import pandas as pd
import torch
import math
import torch
from torch.utils.data import Dataset
import torch_geometric as pyg
from torch_geometric.data import Data
from typing import Tuple, List, Dict, Union
from mixhub.data import (
    MixtureDataInfo,
    COLUMN_PROPERTY,
    COLUMN_VALUE,
    COLUMN_UNIT,
)
from mixhub.data.utils import pad_list
from safetensors import safe_open
from safetensors.torch import save_file
from mixhub.data.featurization import FeaturizeMolecules, FEATURIZATION_TYPE
from mixhub.data.utils import UNK_TOKEN


class MixtureTask(Dataset):
    """
    Base for handling a chemical mixture with associated properties.
    """

    def __init__(self, property: str, dataset: MixtureDataInfo, featurization: str = None):

        self.dataset = dataset
        self.data = dataset.data
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Check if the provided property exists in the dataset
        if property not in self.data[COLUMN_PROPERTY].unique():
            raise ValueError(f"Property '{property}' is not in dataset properties: {self.data[COLUMN_PROPERTY].unique()}")
        
        self.property = property
        self.data = self.data[self.data[COLUMN_PROPERTY] == property]
        self.unit = self.data[COLUMN_UNIT].unique()

        self.indices_tensor, self.fraction_tensor, self.context_tensor, self.output_tensor = self.__tensorize__()

        self.featurization = featurization

        if self.featurization:
            self.features = self.__getfeatures__()

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __num_unique_mixtures__(self) -> int:
        """
        Returns the number of unique mixtures in the dataset based on compound IDs.

        Returns:
            int: The number of unique mixtures.
        """
        cmp_ids = self.dataset.metadata["columns"]["id_column"]
        return len(self.data[cmp_ids].drop_duplicates())

    def __max_num_components__(self) -> int:
        """
        Returns the maximum number of components (compounds) in the dataset.

        Returns:
            int: Maximum number of components.
        """
        cmp_ids = self.dataset.metadata["columns"]["id_column"]
        # return self.data[cmp_ids].apply(len).max()

        def row_max_len(row):
            return max(len(row[col]) if isinstance(row[col], list) else 0 for col in cmp_ids)

        return self.data.apply(row_max_len, axis=1).max()

    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Tuple[int], Tuple[float], float]: A tuple consisting of:
                - Tuple of compound IDs
                - Tuple of mole fractions
                - Property value (label)
        """

        sample = {
            "ids": self.indices_tensor[idx],
            "fractions": self.fraction_tensor[idx],
            "context": self.context_tensor[idx],
            "label": self.output_tensor[idx],
        }

        if self.featurization:
            if FEATURIZATION_TYPE[self.featurization] == "graphs":
                sample["features"] = self.features
            elif FEATURIZATION_TYPE[self.featurization] == "tensors":
                sample["features"] = self.features[idx]

        return sample
    
    def __tensorize__(self) -> Dict:
        id_col = self.dataset.metadata["columns"]["id_column"]
        fraction_col = self.dataset.metadata["columns"]["fraction_column"]
        context_cols = self.dataset.metadata["columns"]["context_columns"]
        output_col = self.dataset.metadata["columns"]["output_column"]

        assert len(id_col) == len(fraction_col)

        max_length = self.__max_num_components__()

        output_tensor = torch.tensor(self.data[output_col].tolist())

        indices_tensors = []
        for col in id_col:
            col_tensor = torch.tensor(self.data[col].apply(lambda x: pad_list(x, max_length=max_length)).tolist())
            indices_tensors.append(col_tensor)

        indices_tensor = torch.stack(indices_tensors, dim=-1)

        if len(fraction_col) == 0:
            fraction_tensor = torch.full_like(indices_tensor, fill_value=UNK_TOKEN)
        else:
            fraction_tensors = []
            for i, col in enumerate(fraction_col):
                if col is not None:
                    col_tensor = torch.tensor(self.data[col].apply(lambda x: pad_list(x, max_length=max_length)).tolist())
                else:
                    col_tensor = torch.full_like(indices_tensor[:, :, i], fill_value=UNK_TOKEN)
                fraction_tensors.append(col_tensor)

            fraction_tensor = torch.stack(fraction_tensors, dim=-1)

        if len(context_cols) == 0:
            context_tensor = torch.full_like(output_tensor.unsqueeze(-1), fill_value=UNK_TOKEN)
        else:
            context_tensors = []
            for col in context_cols:
                col_tensor = torch.tensor(self.data[col].tolist()).unsqueeze(1)
                context_tensors.append(col_tensor)

            context_tensor = torch.cat(context_tensors, dim=1)

        return indices_tensor, fraction_tensor, context_tensor, output_tensor


    def __getfeatures__(self) -> Dict:

        cache_dir = os.path.join(self.dataset.data_dir, f"{self.property.lower().replace(' ', '_')}_featurization")
        cache_path = os.path.join(cache_dir, f"{self.featurization}.safetensors")
        if os.path.exists(cache_path):
            with safe_open(cache_path, framework="pt") as f:
                features = f.get_tensor(self.featurization)
        else:
            print("cache not found, featurizing dataset")
            featurizer = FeaturizeMolecules(self.featurization)
            compounds = self.dataset.compounds["smiles"]

            if FEATURIZATION_TYPE[self.featurization] == "graphs":
                features = featurizer.featurize_graphs(compounds=compounds)
            elif FEATURIZATION_TYPE[self.featurization] == "tensors":
                features = featurizer.featurize_tensors(
                    indices_tensor=self.indices_tensor,
                    compounds=compounds,
                )
                os.makedirs(cache_dir, exist_ok=True)
                save_file({self.featurization: features}, cache_path)

        return features
