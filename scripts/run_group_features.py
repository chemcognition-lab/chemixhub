import argparse
import os
import pandas as pd
import numpy as np
import re
import ast
from rdkit import Chem
from rdkit.Chem import Descriptors


def compute_molecule_stats_numeric(smiles_list, descriptor_funcs, dataset_label, avg_components=np.nan, std_components=np.nan):
    """
    Compute statistics (mean, std, min, max) for a given list of SMILES.
    Includes average number of components in mixtures (mean and std).

    Parameters
    ----------
    smiles_list : list
        List of SMILES strings.
    descriptor_funcs : dict
        Dictionary of RDKit descriptor functions.
    dataset_label : str
        Name of the dataset.
    avg_components : float, optional
        Average number of components in the mixture.
    std_components : float, optional
        Standard deviation of number of components in the mixture.

    Returns
    -------
    list of dict
        A single-element list containing the computed statistics.
    """
    atom_counts, frag_counts, charges = [], [], []
    descriptor_records = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            atom_counts.append(mol.GetNumAtoms())
            frag_counts.append(len(Chem.GetMolFrags(mol)))
            charges.append(Chem.GetFormalCharge(mol))
            descriptors = {name: func(mol) for name, func in descriptor_funcs.items()}
            descriptor_records.append(descriptors)

    if atom_counts:
        descriptor_df = pd.DataFrame(descriptor_records)
        return [{
            'Dataset Name': dataset_label,
            'Unique molecules': len(atom_counts),

            # Atoms
            'Avg atoms/mol': np.mean(atom_counts),
            'Std atoms/mol': np.std(atom_counts),
            'Max atoms/mol': np.max(atom_counts),
            'Min atoms/mol': np.min(atom_counts),

            # Fragments
            'Avg fragments': np.mean(frag_counts),
            'Std fragments': np.std(frag_counts),
            'Max fragments': np.max(frag_counts),

            # Molecular weight
            'Avg MolWt': descriptor_df['MolWt'].mean(),
            'Std MolWt': descriptor_df['MolWt'].std(),

            # Rotatable bonds
            'Avg Rotatable Bonds': descriptor_df['NumRotatableBonds'].mean(),
            'Std Rotatable Bonds': descriptor_df['NumRotatableBonds'].std(),

            # Formal charge
            'Avg Formal Charge': np.mean(charges),
            'Std Formal Charge': np.std(charges),

            # Components mixture
            'Avg components mixture': avg_components,
            'Std components mixture': std_components
        }]
    else:
        return [{
            'Dataset Name': dataset_label,
            'Unique molecules': 0,
            'Avg atoms/mol': np.nan,
            'Std atoms/mol': np.nan,
            'Max atoms/mol': np.nan,
            'Min atoms/mol': np.nan,
            'Avg fragments': np.nan,
            'Std fragments': np.nan,
            'Max fragments': np.nan,
            'Avg MolWt': np.nan,
            'Std MolWt': np.nan,
            'Avg Rotatable Bonds': np.nan,
            'Std Rotatable Bonds': np.nan,
            'Avg Formal Charge': np.nan,
            'Std Formal Charge': np.nan,
            'Avg components mixture': avg_components,
            'Std components mixture': std_components
        }]


def get_molecule_statistics_with_components(root_path, file_extensions=('.csv',)):
    """
    Compute molecular statistics (mean, std, min, max) for each dataset, 
    including average number of components in mixtures.
    Gives preference to processed*.csv files for component counts.
    
    Parameters
    ----------
    root_path : str
        Path to the root directory containing dataset folders.
    file_extensions : tuple of str, optional
        File extensions to include in the search (default: '.csv').

    Returns
    -------
    pd.DataFrame
        Table containing computed molecular statistics and mixture component
        information for each dataset or dataset subset (for IlThermo).
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

        processed_file = None
        ids_col = None
        avg_components = np.nan
        std_components = np.nan  

        # Look for a processed file to extract mixture info
        for fname in filenames:
            if fname.startswith("processed") and fname.endswith(file_extensions):
                processed_file = os.path.join(dirpath, fname)
                try:
                    df_proc = pd.read_csv(processed_file)
                    ids_cols = [col for col in df_proc.columns if re.search(r'ids', col, re.IGNORECASE)]
                    if ids_cols:
                        ids_col = ids_cols[0]
                        comp_lengths = df_proc[ids_col].dropna().apply(ast.literal_eval).apply(len)
                        avg_components = comp_lengths.mean()
                        std_components = comp_lengths.std()
                except Exception as e:
                    print(f"Error reading {processed_file}: {e}")
                break  

        for fname in filenames:
            if not fname.endswith(file_extensions):
                continue

            fpath = os.path.join(dirpath, fname)
            dataset_name = os.path.normpath(fpath).split(os.sep)[-3]

            try:
                # CASE 1: IlThermo special handling
                if fname == "processed_IlThermoData.csv":
                    compounds_path = os.path.join(
                        root_path, "ionic-liquids", "processed_data", "compounds.csv"
                    )
                    compounds_df = pd.read_csv(compounds_path)
                    df = pd.read_csv(fpath)

                    viscosity_ids = df[df['property'] == 'Viscosity']['cmp_ids'].apply(ast.literal_eval)
                    conductivity_ids = df[df['property'] == 'Electrical conductivity']['cmp_ids'].apply(ast.literal_eval)

                    subsets = {
                        "Viscosity": {
                            "data": compounds_df[compounds_df['compound_id'].isin(
                                x for sublist in viscosity_ids for x in sublist
                            )],
                            "components": viscosity_ids
                        },
                        "Electrical conductivity": {
                            "data": compounds_df[compounds_df['compound_id'].isin(
                                x for sublist in conductivity_ids for x in sublist
                            )],
                            "components": conductivity_ids
                        }
                    }

                    for prop_name, sub_info in subsets.items():
                        smiles_list = sub_info["data"]['smiles'].dropna().astype(str).str.strip().unique()
                        comp_lengths = sub_info["components"].apply(len)
                        avg_comp = comp_lengths.mean() if not comp_lengths.empty else np.nan
                        std_comp = comp_lengths.std() if not comp_lengths.empty else np.nan
                        stats_list.extend(
                            compute_molecule_stats_numeric(smiles_list, descriptor_funcs, "IlThermo_" + prop_name, avg_comp, std_comp)
                        )

                # CASE 2: Other datasets
                elif "ionic-liquids" not in dirpath:
                    df = pd.read_csv(fpath)
                    smiles_cols = [col for col in df.columns if re.search(r'smi|SMILES', col, re.IGNORECASE)]
                    if not smiles_cols:
                        continue

                    smiles_set = set()
                    for col in smiles_cols:
                        valid_smiles = df[col].dropna().astype(str).str.strip()
                        smiles_set.update(valid_smiles[valid_smiles != ''].unique())

                    stats_list.extend(
                        compute_molecule_stats_numeric(smiles_set, descriptor_funcs, dataset_name, avg_components, std_components)
                    )

            except Exception as e:
                print(f"Skipped {fpath} due to error: {e}")

    return pd.DataFrame(stats_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract SMILES and compute molecular statistics (with mixture info)")
    parser.add_argument("root_path", type=str, help="Root path to dataset directories")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Computing molecule statistics per dataset (with components info)...")
    stats_df = get_molecule_statistics_with_components(args.root_path)
    stats_df.to_csv(os.path.join(args.output_dir, "molecule_statistics.csv"), index=False)
    print(f"Saved molecule statistics to {args.output_dir}/molecule_statistics.csv")

