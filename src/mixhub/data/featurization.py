from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from torch_geometric.data import Data
from typing import List, Optional, Dict
import logging
import torch
import numpy as np

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from mixhub.data.data import COLUMN_VALUE
from mixhub.data.utils import indices_to_graphs, UNK_TOKEN

from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer

def parse_status(generator, smiles):
    results = generator.process(smiles)
    try: 
        processed, features = results[0], results[1:]
        if processed is None:
            logging.warning("Descriptastorus cannot process smiles %s", smiles)
        return features
    except TypeError:
        logging.warning("RDKit Error on smiles %s", smiles)
        # if processed is None, the features are are default values for the type


def morgan_fingerprints(
    smiles: List[str],
) -> torch.Tensor:
    """
    Builds molecular representation as a binary Morgan ECFP fingerprints with radius 3 and 2048 bits.

    :param smiles: list of molecular smiles
    :type smiles: list
    :return: tensor of shape [len(smiles), 2048] with ecfp featurized molecules

    """
    generator = MakeGenerator((f"Morgan3",))
    fps = torch.Tensor([parse_status(generator, x) for x in smiles])
    return fps


def rdkit2d_normalized_features(
    smiles: List[str],
) -> torch.Tensor:
    """
    Builds molecular representation as normalized 2D RDKit features.

    :param smiles: list of molecular smiles
    :type smiles: list
    :return: tensor of shape [len(smiles), 200] with featurized molecules

    """
    generator = MakeGenerator((f"rdkit2dhistogramnormalized",))
    fps = torch.Tensor([parse_status(generator, x) for x in smiles])
    return fps


def molt5_embedding(
    smiles: List[str],
):
    """
    Builds molecular representation using MolT5.

    :param smiles: list of molecular smiles
    :type smiles: list
    :return: tensor of shape [len(smiles), 1024]

    """
    model = PretrainedHFTransformer(kind='MolT5', notation='smiles', dtype=np.float32, device="cpu")
    print(model.device)
    embeds = torch.from_numpy(model(smiles))
    return embeds


def pyg_molecular_graphs(
    smiles: List[str], 
) -> List[Data]:
    """
    Convers a list of SMILES strings into PyGeometric molecular graphs.

    :param smiles: list of molecular SMILES
    :type smiles: list
    :return: list of PyGeometric molecular graphs
    """

    from torch_geometric.utils import from_smiles

    return [
        from_smiles(smiles=i) for i in smiles
    ]


def custom_molecular_graphs(
    smiles: List[str],
    init_globals: Optional[bool] = False,
) -> List[Data]:
    """
    Converts a list of SMILES strings into a custom graph tuple
    """

    from .graph_utils import from_smiles

    return [
        from_smiles(smiles=i, init_globals=init_globals) for i in smiles
    ]

FEATURIZATION_MAPPING = {
    "rdkit2d_normalized_features": rdkit2d_normalized_features,
    "morgan_fingerprints": morgan_fingerprints,
    "pyg_molecular_graphs": pyg_molecular_graphs,
    "custom_molecular_graphs": custom_molecular_graphs,
    "molt5_embeddings": molt5_embedding,
}

FEATURIZATION_TYPE = {
    "rdkit2d_normalized_features": "tensors",
    "morgan_fingerprints": "tensors",
    "pyg_molecular_graphs": "graphs",
    "custom_molecular_graphs": "graphs",
    "molt5_embeddings": "tensors",
}


class FeaturizeMolecules(object):
    def __init__(self, featurization: str) -> None:

        if featurization not in FEATURIZATION_MAPPING:
            raise ValueError(f"Featurization type '{featurization}' is not recognized. Choose from {list(FEATURIZATION_MAPPING.keys())}.")

        self.featurization = featurization

    def featurize_tensors(self, indices_tensor: torch.Tensor, compounds: List[str]) -> torch.Tensor:

        featurized_mols = FEATURIZATION_MAPPING[self.featurization](compounds)

        all_features = []
        for ind_tensor in torch.unbind(indices_tensor, dim=-1):
            featurized_tensor = torch.full(
                (ind_tensor.shape[0], ind_tensor.shape[1], featurized_mols.shape[1]),
                UNK_TOKEN,
                dtype=featurized_mols.dtype,
            )
            valid_indices = ind_tensor != UNK_TOKEN
            row_indices = ind_tensor.long()
            featurized_tensor[valid_indices] = featurized_mols[row_indices[valid_indices], :]

            all_features.append(featurized_tensor)
        
        featurized_tensor = torch.stack(all_features, dim=-1)

        return featurized_tensor

    def featurize_graphs(self, compounds: List[str]) -> List[Data]:
        featurized_mols = FEATURIZATION_MAPPING[self.featurization](compounds)
        return featurized_mols
