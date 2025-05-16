from typing import List, Any

import rdkit.Chem as Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import torch
from torch_geometric.data import Data
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator

MAX_ATOMIC_NUM = 53
ATOM_FEATURES = {
    "atomic_num": list(range(MAX_ATOMIC_NUM)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

# hard coded for representation here
EDGE_DIM = 14
NODE_DIM = 85


def onek_encoding_unk(value: Any, choices: List[Any]):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def atom_features(atom: Chem.rdchem.Atom) -> List:
    features = (
        onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES["atomic_num"])
        + onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
        + onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES["chiral_tag"])
        + onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES["num_Hs"])
        + onek_encoding_unk(
            int(atom.GetHybridization()), ATOM_FEATURES["hybridization"]
        )
        + [1 if atom.GetIsAromatic() else 0]
    )
    return list(features)


def bond_features(bond: Chem.rdchem.Bond) -> List:
    if bond is None:
        fbond = [1] + [0] * (EDGE_DIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0),
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def global_features(mol: Chem.rdchem.Mol) -> List:
    generator = MakeGenerator(("rdkit2dhistogramnormalized",))
    feats = generator.process(Chem.MolToSmiles(mol))
    return feats



def from_smiles(smiles: str, init_globals: bool = True) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    smi = Chem.MolToSmiles(mol)         # canonicalized

    a_feats = []
    for a in mol.GetAtoms():
        a_feats.append(atom_features(a))
    
    a_feats = torch.tensor(a_feats, dtype=torch.float)

    b_indices, b_feats = [], []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        feats = bond_features(b)

        b_indices += [[i, j], [j, i]]
        b_feats += [feats, feats]
    
    b_index = torch.tensor(b_indices)
    b_index = b_index.t().to(torch.long).view(2, -1)
    b_feats = torch.tensor(b_feats, dtype=torch.float)

    if b_index.numel() > 0:  # Sort indices.
        perm = (b_index[0] * a_feats.size(0) + b_index[1]).argsort()
        b_index, b_feats = b_index[:, perm], b_feats[perm]
    
    if init_globals:
        g_feats = torch.tensor([global_features(mol)], dtype=torch.float)
    else:
        g_feats = torch.tensor([[0.]], dtype=torch.float)

    return Data(x=a_feats, edge_index=b_index, edge_attr=b_feats, u=g_feats, smiles=smi)


