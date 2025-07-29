import argparse
import os
import glob
import pandas as pd
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import Descriptors


def get_molecule_statistics_per_dataset(root_path, file_extensions=('.csv',)):
    """
    Compute molecule-level statistics per dataset.

    Parameters
    ----------
    root_path : str
        Root directory containing dataset folders.
    file_extensions : tuple
        File extensions to consider as datasets.

    Returns
    -------
    pd.DataFrame
        DataFrame containing molecule statistics per dataset.
    """
    stats_list = []

    descriptor_funcs = {
        'MolWt': Descriptors.MolWt,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'FormalCharge': lambda m: Chem.GetFormalCharge(m),
    }

    for dirpath, _, filenames in os.walk(root_path):
        if os.path.basename(dirpath) == 'raw_data':
            continue

        for fname in filenames:
            if not fname.endswith(file_extensions):
                continue

            fpath = os.path.join(dirpath, fname)
            dataset_name = os.path.normpath(fpath).split(os.sep)[-3]

            try:
                df = pd.read_csv(fpath)

                smiles_cols = [col for col in df.columns if re.search(r'smi|SMILES', col, re.IGNORECASE)]
                if not smiles_cols:
                    continue

                smiles_set = set()
                for col in smiles_cols:
                    valid_smiles = df[col].dropna().astype(str).str.strip()
                    smiles_set.update(valid_smiles[valid_smiles != ''].unique())

                descriptor_records = []
                atom_counts, frag_counts, charges = [], [], []
                for smi in smiles_set:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        atom_counts.append(mol.GetNumAtoms())
                        frag_counts.append(len(Chem.GetMolFrags(mol)))
                        charges.append(Chem.GetFormalCharge(mol))

                        descriptors = {name: func(mol) for name, func in descriptor_funcs.items()}
                        descriptor_records.append(descriptors)

                if atom_counts:
                    descriptor_df = pd.DataFrame(descriptor_records)

                    stats_list.append({
                        'Dataset Name': dataset_name,
                        'Unique molecules': len(atom_counts),
                        'Avg number of atoms per molecule': np.mean(atom_counts),
                        'Max number of atoms per molecule': np.max(atom_counts),
                        'Min number of atoms per molecule': np.min(atom_counts),
                        'Avg number of fragments': np.mean(frag_counts),
                        'Max number of fragments': np.max(frag_counts),
                        'Min number of fragments': np.min(frag_counts),
                        'Fraction charged molecules': np.mean([c != 0 for c in charges]),
                        'Avg rotatable bonds': descriptor_df['NumRotatableBonds'].mean(),
                        'Avg Formal Charge': np.mean(charges),
                    })

            except Exception as e:
                print(f"Skipped {fpath} due to error: {e}")

    return pd.DataFrame(stats_list)


def get_molecule_statistics_per_dataset_numeric(root_path, file_extensions=('.csv',)):
#    """
#     Compute molecular statistics (mean, standard deviation, max, min) for each dataset in a directory.

#     This function iterates through all CSV files in a specified root directory, extracts unique SMILES strings,
#     computes molecular descriptors using RDKit, and aggregates dataset-level statistics.

#     Parameters
#     ----------
#     root_path : str
#         Root directory containing dataset subfolders with CSV files.
#     file_extensions : tuple of str, optional
#         File extensions to consider (default is ('.csv',)).

#     Returns
#     -------
#     pd.DataFrame
#     """



    stats_list = []
    descriptor_funcs = {
        'MolWt': Descriptors.MolWt,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'FormalCharge': lambda m: Chem.GetFormalCharge(m),
    }

    for dirpath, _, filenames in os.walk(root_path):
        if os.path.basename(dirpath) == 'raw_data':
            continue

        for fname in filenames:
            if not fname.endswith(file_extensions):
                continue

            fpath = os.path.join(dirpath, fname)
            dataset_name = os.path.normpath(fpath).split(os.sep)[-3]

            try:
                df = pd.read_csv(fpath)

                smiles_cols = [col for col in df.columns if re.search(r'smi|SMILES', col, re.IGNORECASE)]
                if not smiles_cols:
                    continue

                smiles_set = set()
                for col in smiles_cols:
                    valid_smiles = df[col].dropna().astype(str).str.strip()
                    smiles_set.update(valid_smiles[valid_smiles != ''].unique())

                descriptor_records = []
                atom_counts, frag_counts, charges = [], [], []
                for smi in smiles_set:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        atom_counts.append(mol.GetNumAtoms())
                        frag_counts.append(len(Chem.GetMolFrags(mol)))
                        charges.append(Chem.GetFormalCharge(mol))

                        descriptors = {name: func(mol) for name, func in descriptor_funcs.items()}
                        descriptor_records.append(descriptors)

                if atom_counts:
                    descriptor_df = pd.DataFrame(descriptor_records)

                    stats_list.append({
                        'Dataset Name': dataset_name,
                        'Unique molecules': len(atom_counts),

                        # Atoms per molecule
                        'Atoms/Mol mean': np.mean(atom_counts),
                        'Atoms/Mol std': np.std(atom_counts),
                        'Max atoms per molecule': np.max(atom_counts),
                        'Min atoms per molecule': np.min(atom_counts),

                        # Fragments
                        'Fragments/Mol mean': np.mean(frag_counts),
                        'Fragments/Mol std': np.std(frag_counts),
                        'Max fragments': np.max(frag_counts),

                        # Charge
                        'Formal Charge mean': np.mean(charges),
                        'Formal Charge std': np.std(charges),

                        # Molecular weight
                        'MolWt mean': descriptor_df['MolWt'].mean(),
                        'MolWt std': descriptor_df['MolWt'].std(),

                        # Rotatable bonds
                        'Rotatable Bonds mean': descriptor_df['NumRotatableBonds'].mean(),
                        'Rotatable Bonds std': descriptor_df['NumRotatableBonds'].std(),
                    })

            except Exception as e:
                print(f"Skipped {fpath} due to error: {e}")

    return pd.DataFrame(stats_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract smiles and compute molecular statistics")
    parser.add_argument("root_path", type=str, help="Root path to dataset directories")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)


    print("Computing molecule statistics per dataset...")
    stats_df = get_molecule_statistics_per_dataset_numeric(args.root_path)
    stats_df.to_csv(os.path.join(args.output_dir, "molecule_statistics.csv"), index=False)
    print(f"Saved molecule statistics to {args.output_dir}/molecule_statistics.csv")

