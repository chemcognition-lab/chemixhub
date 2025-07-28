import argparse
import os
import glob
import pandas as pd
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import Descriptors


def extract_all_smiles(root_path, file_pattern='compounds.csv'):
    """
    Extracts all unique SMILES from datasets.

    Parameters
    ----------
    root_path : str
        Root path containing dataset folders.
    file_pattern : str
        Filename pattern to match CSV files containing SMILES.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['smiles', 'dataset'].
    """
    smiles_records = []

    file_paths = glob.glob(os.path.join(root_path, '*', 'processed_data', file_pattern), recursive=True)

    for fpath in file_paths:
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
        try:
            df = pd.read_csv(fpath)

            smiles_cols = [col for col in df.columns if re.search(r'smi|SMILES', col, re.IGNORECASE)]
            if not smiles_cols:
                continue

            for col in smiles_cols:
                smiles_series = df[col].dropna().astype(str).str.strip()
                smiles_series = smiles_series[smiles_series != ''].unique()

                smiles_records.extend([{'smiles': smi, 'dataset': dataset_name} for smi in smiles_series])

        except Exception as e:
            print(f"Skipped {fpath}: {e}")

    smiles_df = pd.DataFrame(smiles_records).drop_duplicates('smiles').reset_index(drop=True)
    return smiles_df



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
        'LogP': Descriptors.MolLogP,
        'TPSA': Descriptors.TPSA,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'FormalCharge': lambda m: Chem.GetFormalCharge(m),
        'NumHDonors': Descriptors.NumHDonors,
        'NumHAcceptors': Descriptors.NumHAcceptors,
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract smiles and compute molecular statistics")
    parser.add_argument("root_path", type=str, help="Root path to dataset directories")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)


    print("Computing molecule statistics per dataset...")
    stats_df = get_molecule_statistics_per_dataset(args.root_path)
    stats_df.to_csv(os.path.join(args.output_dir, "molecule_statistics.csv"), index=False)
    print(f"Saved molecule statistics to {args.output_dir}/molecule_statistics.csv")

