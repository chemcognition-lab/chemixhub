import os
import numpy as np 
import torch
from safetensors.torch import save_file


if __name__ == "__main__":

    path = os.path.abspath(".")
    split_dir = os.path.join(path, "pommix-splits")
    cache_dir = os.path.join(path, "mixture_similarity_splits")

    split_files = os.listdir(split_dir)

    for f in split_files:

        fpath = os.path.join(split_dir, f)

        cache_path = os.path.join(cache_dir, f.replace("random_cv", "kfold_split_").replace("ablate_components", "num_components_split_").replace("lso_molecules", "lso_split_").replace("npz", "safetensors"))

        pommix_split = np.load(fpath)
        train_indices = torch.from_numpy(pommix_split["training"])
        val_indices = torch.from_numpy(pommix_split["validation"])
        test_indices = torch.from_numpy(pommix_split["testing"])

        save_file(
            {"train_indices": train_indices, "val_indices": val_indices, "test_indices": test_indices},
            cache_path,
        )
