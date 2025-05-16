import os
import sys
import torch
import pandas as pd
import numpy as np

from typing import Tuple, Optional, List
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from mixhub.data.utils import UNK_TOKEN
from torch.utils.data import random_split
from safetensors import safe_open
from safetensors.torch import save_file
from collections import Counter

def calculate_inner_lengths(row):
    return [len(inner_arr) for inner_arr in row]


def split_indices(indices, valid_percent, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    shuffled = indices[torch.randperm(len(indices))]
    train_size = int((1 - valid_percent) * len(indices))
    return shuffled[:train_size], shuffled[train_size:]


def create_k_molecules_split(
    property: str,
    mixture_indices_tensor: torch.Tensor,
    cache_dir: str,
    k: int,
    val_percent: Optional[float] = 0.1,
    geometric: Optional[bool] = False,
    seed: Optional[int] = None,
) -> None:
    """
    Ablation splits of dataset into mixtures that are <= k and mixtures that have > k molecules (testing).

    Parameters:
    features (np.ndarray): Array of features where each row represents a mixture.
    k (int): Maximum number of molecules in a mixture for it to be included in the training set.
    val_percent (Optional[float]): Percentage of the training data to use for validation. Default is 0.1.

    Returns:
    Tuple[list[int], list[int], list[int]]: A tuple containing lists of indices for the training, validation, and test sets.
    """

    save_path = os.path.join(cache_dir, f'{property.lower().replace(" ", "_")}_splits')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if geometric:
        mixture_lengths = (mixture_indices_tensor != UNK_TOKEN).sum(dim=1)
        mixture_lengths = torch.sqrt(mixture_lengths[:, 0] * mixture_lengths[:, 1]).unsqueeze(-1)
    else:
        mixture_lengths = (mixture_indices_tensor != UNK_TOKEN).sum(dim=1)

    mask = (mixture_lengths <= k).bool().squeeze(1)
    train_indices = torch.arange(mask.shape[0])[mask]
    test_indices = torch.arange(mask.shape[0])[~mask]

    # get indices for training and validation
    train_indices, val_indices = split_indices(
        train_indices, val_percent, seed
    )

    cache_path = os.path.join(save_path, f"num_components_split_{k}.safetensors")

    save_file(
        {"train_indices": train_indices, "val_indices": val_indices, "test_indices": test_indices},
        cache_path,
    )


def create_kfold_split(
    property: str,
    mixture_indices_tensor: torch.Tensor,
    cache_dir: str,
    n_splits: Optional[int] = 5,
    seed: Optional[int] = None,
) -> None:

    save_path = os.path.join(cache_dir, f'{property.lower().replace(" ", "_")}_splits')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset_indices_tensor = torch.arange(mixture_indices_tensor.shape[0])

    train_val_frac = 1.0 - 1.0 / n_splits  # gives 20 for test set
    train_frac = 0.7 / train_val_frac  # get 70/10 of total for train/val sets

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_indices, test_indices) in enumerate(
        kf.split(np.arange(mixture_indices_tensor.shape[0]))
    ):
        # do a split for 70/10/20 train/validation/test
        train_indices, val_indices = train_test_split(
            train_indices, train_size=train_frac, random_state=seed
        )

        train_indices = dataset_indices_tensor[train_indices]
        val_indices = dataset_indices_tensor[val_indices]
        test_indices = dataset_indices_tensor[test_indices]

        cache_path = os.path.join(save_path, f"kfold_split_{i}.safetensors")

        save_file(
            {"train_indices": train_indices, "val_indices": val_indices, "test_indices": test_indices},
            cache_path,
        )


def create_temperature_splits(
    property: str,
    mixture_indices_tensor: torch.Tensor,
    temperature_tensor: torch.Tensor,  # in Kelvin
    cache_dir: str,
    seed: Optional[int] = None,
) -> None:
    """
    Create range-based K-fold splits where each test fold corresponds to a specific temperature range.
    """

    save_path = os.path.join(cache_dir, f'{property.lower().replace(" ", "_")}_splits')
    os.makedirs(save_path, exist_ok=True)

    dataset_indices_tensor = torch.arange(mixture_indices_tensor.shape[0])

    # Define bin edges in Kelvin (produces 5 bins total)
    temp_bin_edges = torch.tensor([298.15, 318.15, 338.15, 358.15], dtype=torch.float32)
    temperature_bins = torch.bucketize(temperature_tensor, temp_bin_edges).squeeze(-1)  # values from 0 to 4
    n_bins = 5

    # Check for empty bins and warn
    bin_counts = torch.bincount(temperature_bins, minlength=n_bins)
    print("bin counts:", bin_counts)
    for bin_idx, count in enumerate(bin_counts):
        if count.item() == 0:
            print(f"Warning: Temperature bin {bin_idx} contains 0 samples and will be skipped.")

    for bin_idx in range(n_bins):
        if bin_counts[bin_idx].item() == 0:
            continue  # Skip empty bin

        # Test set: all samples in the current temperature bin
        test_mask = (temperature_bins == bin_idx)
        test_indices = dataset_indices_tensor[test_mask]

        # Remaining data (not in this bin) is used for train/val
        train_val_mask = ~test_mask
        train_val_indices = dataset_indices_tensor[train_val_mask]
        train_val_temperatures = temperature_tensor[train_val_mask]
        train_val_bins = temperature_bins[train_val_mask]  # for stratified split

        # Stratified train/val split on remaining data
        train_indices, val_indices = train_test_split(
            train_val_indices.numpy(),
            train_size=0.7,
            random_state=seed,
            stratify=train_val_bins.numpy()
        )

        # Convert back to tensors
        train_indices = torch.tensor(train_indices, dtype=torch.long)
        val_indices = torch.tensor(val_indices, dtype=torch.long)

        cache_path = os.path.join(save_path, f"temperature_split_{bin_idx}.safetensors")

        save_file(
            {"train_indices": train_indices, "val_indices": val_indices, "test_indices": test_indices},
            cache_path,
        )


def find_closest_index(value, arr):
    arr = np.array(arr)  # Convert to numpy array for convenience
    index = np.abs(arr - value).argmin()
    return index


def create_molecule_identity_splits(
    property: str,
    mixture_indices_tensor: torch.Tensor,
    cache_dir: str,
    n_splits: Optional[int] = 5,
    valid_percent: Optional[float] = 0.1,
    seed=None,
) -> None:
    """
    Create ablation splits of the dataset based on the exclusion of specific molecules, ensuring that certain molecules do not appear in the training set.

    Parameters:
    mixture_smiles (np.ndarray): Array of molecule mixtures, where each mixture is a list of SMILES strings.
    n_splits (Optional[int]): Number of splits to create. Default is 5.
    valid_percent (Optional[float]): Percentage of the data to use for validation. Default is 0.1.

    Returns:
    Tuple[list[int], list[int], list[int]]: A tuple containing lists of indices for the training, validation, and test sets for each split.
    """

    save_path = os.path.join(cache_dir, f'{property.lower().replace(" ", "_")}_splits')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset_indices_tensor = torch.arange(mixture_indices_tensor.shape[0])

    # Flatten the mixture smiles, since we care only about identity of molecules found in any mixture
    flat = mixture_indices_tensor.flatten()
    filtered = flat[flat != UNK_TOKEN]

    indices_count = Counter(filtered.tolist())
    sorted_indices = sorted(indices_count, key=indices_count.get, reverse=False)

    mixture_list = [[elem for elem in row if elem != UNK_TOKEN] for row in mixture_indices_tensor.squeeze(-1).tolist()]

    # progressively build the training set based on frequency of molecules
    training_increments = [[]]
    training_counts = []
    for i, idx in enumerate(sorted_indices):
        inds = [
            j
            for j, sublist in enumerate(mixture_list)
            if any(item == idx for item in sublist)
        ]
        inds = sorted(list(set(inds + training_increments[i])))
        training_counts.append(len(inds))
        training_increments.append(inds)
    training_increments.pop(0)

    # overlaps in molecular identity will result in unchanged training sets
    # reverse and then get unique, this will pick most inclusive set
    training_counts, uniq_idx = np.unique(training_counts[::-1], return_index=True)
    training_increments = np.array(training_increments[::-1], dtype=object)[
        uniq_idx
    ].tolist()

    # get the increments that are give training sizes closest to equal spacing for ablation test
    split_space = np.linspace(0, len(mixture_list), n_splits + 2)[1:-1]
    closest_indices = [find_closest_index(v, training_counts) for v in split_space]
    training_increments = np.array(training_increments, dtype=object)[
        closest_indices
    ].tolist()
    training_counts = training_counts[closest_indices]

    for i, train_indices in enumerate(training_increments):
        # Split based on exclusion of specific molecules
        test_indices = [j for j in dataset_indices_tensor.tolist() if j not in train_indices]
        train_indices, val_indices = train_test_split(
            train_indices, test_size=valid_percent
        )

        train_indices = dataset_indices_tensor[train_indices]
        val_indices = dataset_indices_tensor[val_indices]
        test_indices = dataset_indices_tensor[test_indices]

        cache_path = os.path.join(save_path, f"leave_one_out_split_{i}.safetensors")

        save_file(
            {"train_indices": train_indices, "val_indices": val_indices, "test_indices": test_indices},
            cache_path,
        )


def create_lso_molecule_identity_splits(
    property: str,
    mixture_indices_tensor: torch.Tensor,
    cache_dir: str,
    n_splits: Optional[int] = 5,
    valid_percent: Optional[float] = 0.1,
    tolerance: Optional[float] = 0.005,
    seed=None,
) -> Tuple[list[int], list[int], list[int]]:
    """
    Create "Leave Some Out" (lso) splits of the dataset based on the exclusion of specific molecules, ensuring that certain molecules do
    not appear in the training set.

    Parameters:
    mixture_smiles (np.ndarray): Array of molecule mixtures, where each mixture is a list of SMILES strings.
    n_splits (Optional[int]): Number of splits to create. Default is 5.
    valid_percent (Optional[float]): Percentage of the data to use for validation. Default is 0.1.
    tolerance (Optional[float]): Tolerance for the size of the test set. Default is 0.005.

    Returns:
    Tuple[list[int], list[int], list[int]]: A tuple containing lists of indices for the training, validation, and test sets for each split.
    """

    save_path = os.path.join(cache_dir, f'{property.lower().replace(" ", "_")}_splits')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset_indices_tensor = torch.arange(mixture_indices_tensor.shape[0])

    flat = mixture_indices_tensor.flatten()
    filtered = flat[flat != UNK_TOKEN]

    indices_count = Counter(filtered.tolist())
    sorted_indices = sorted(indices_count, key=indices_count.get, reverse=False)

    mixture_list = [[elem for elem in row if elem != UNK_TOKEN] for row in mixture_indices_tensor.squeeze(-1).tolist()]

    # Flatten the mixture smiles, since we care only about identity of molecules found in any mixture
    mixture_smiles = np.array(
        [item for item in mixture_list], dtype=object
    )

    # Determine the size of splits
    test_percent = 1.0 / n_splits
    train_percent = 1.0 - test_percent - valid_percent
    N = len(mixture_smiles)

    # Flatten the list of lists and count the frequency of each string
    all_indices = list(range(mixture_smiles.shape[0]))
    all_strings = np.concatenate(mixture_smiles.ravel())

    # incrementally
    splits, smiles_removed = [], []
    for i in range(n_splits):
        while True:
            excluded_mixtures, excluded_smiles = [], []
            sample_strings = all_strings.tolist().copy()
            remaining_mixtures = mixture_smiles.tolist().copy()
            while len(excluded_mixtures) / N < test_percent - tolerance:
                # select a smiles for exclusion
                smi = np.random.choice(sample_strings, size=1)[0]
                excluded_smiles.append(smi)
                excluded_mixtures.extend(
                    [arr for arr in remaining_mixtures if smi in arr]
                )
                sample_strings.remove(smi)  # remove it so it won't be sampled again
                remaining_mixtures = [
                    arr
                    for arr in remaining_mixtures
                    if not any(np.array_equal(arr, excl) for excl in excluded_mixtures)
                ]  # remove arrays that have been added to excluded_mixtures
                assert (
                    len(remaining_mixtures + excluded_mixtures) == N
                ), "List of excluded and remaining not adding up to full dataset."

                if len(excluded_mixtures) / N > test_percent + tolerance:
                    print("Too many excluded sets. Reset.")
                    break

            if (
                len(excluded_mixtures) / N >= test_percent - tolerance
                and len(excluded_mixtures) / N <= test_percent + tolerance
                and excluded_smiles not in smiles_removed
            ):
                # For each array of strings in remaining_mixtures, get the indices as seen inside of mixture_smiles
                remaining_indices = [
                    all_indices[j]
                    for j, sublist in enumerate(mixture_smiles)
                    if any(np.array_equal(sublist, rem) for rem in remaining_mixtures)
                ]
                test_indices = [
                    all_indices[j]
                    for j, sublist in enumerate(mixture_smiles)
                    if any(np.array_equal(sublist, excl) for excl in excluded_mixtures)
                ]

                # perform a split on the remaining indices, which makes up the train and val set
                train_indices, valid_indices = train_test_split(
                    remaining_indices,
                    test_size=valid_percent / (train_percent + valid_percent),
                )
                splits.append((train_indices, valid_indices, test_indices))
                smiles_removed.append(excluded_smiles)
                print(f"Complete split {i}.")
                break
    
    for i in range(n_splits):

        train_indices, val_indices, test_indices = splits[i]
        print(f"Smiles removed from split {i}:", smiles_removed[i])

        train_indices = dataset_indices_tensor[train_indices]
        val_indices = dataset_indices_tensor[val_indices]
        test_indices = dataset_indices_tensor[test_indices]

        cache_path = os.path.join(save_path, f"lso_split_{i}.safetensors")

        save_file(
            {"train_indices": train_indices, "val_indices": val_indices, "test_indices": test_indices},
            cache_path,
        )


SPLIT_MAPPING = {
    "num_components": create_k_molecules_split,
    "kfold": create_kfold_split,
    "temperature": create_temperature_splits,
    "leave_one_out": create_molecule_identity_splits,
    "lso": create_lso_molecule_identity_splits,
}

class SplitLoader(object):

    def __init__(self, split_type: str = "kfold") -> None:

        if split_type not in SPLIT_MAPPING:
            raise ValueError(f"Split type '{split_type}' is not recognized. Choose from {list(SPLIT_MAPPING.keys())}.")

        self.split_type = split_type
    
    def __call__(
        self,
        property: str,
        cache_dir: str,
        split_num: Optional[int] = 0,         
    ):

        save_path = os.path.join(cache_dir, f'{property.lower().replace(" ", "_")}_splits')
        cache_path = os.path.join(save_path, f"{self.split_type}_split_{split_num}.safetensors")

        if os.path.exists(cache_path):
            with safe_open(cache_path, framework="pt") as f:
                train_indices = f.get_tensor("train_indices")
                val_indices = f.get_tensor("val_indices")
                test_indices = f.get_tensor("test_indices")

        return train_indices, val_indices, test_indices
