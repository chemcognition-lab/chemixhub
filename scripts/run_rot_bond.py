import os
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt

def calc_rotatable_bonds(smiles):
    """Calculate the number of rotatable bonds for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Descriptors.NumRotatableBonds(mol) if mol else 0
    except Exception:
        return 0

def process_dataset(base_dir):
    """Process datasets and plot rotatable bond distribution."""
    combined_df = []

    for root, dirs, files in os.walk(base_dir):
        if 'processed_data' in root:
            for file in files:
                if file == "compounds.csv":
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path)
                        if 'smiles' in df.columns:
                            df['num_rotatable_bonds'] = df['smiles'].apply(calc_rotatable_bonds)
                            df['num_rotatable_bonds'] = df['num_rotatable_bonds'].fillna(0).astype(int)
                            combined_df.append(df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    if combined_df:
        full_df = pd.concat(combined_df, ignore_index=True)
        more_than_5 = (full_df['num_rotatable_bonds'] > 5).sum()
        less_equal_5 = (full_df['num_rotatable_bonds'] <= 5).sum()

        if (more_than_5 + less_equal_5) > 0:
            plt.figure(figsize=(6, 6))
            plt.pie(
                [less_equal_5, more_than_5],
                labels=['≤5 Rotatable Bonds', '>=5 Rotatable Bonds'],
                autopct='%1.1f%%',
                startangle=90
            )
            plt.title('Overall Rotatable Bond Distribution Across All Datasets')
            plt.show()
        else:
            print("No valid SMILES found to plot pie chart.")
    else:
        print("No compounds.csv files found under 'processed_data' folders.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze rotatable bond distribution from compounds.csv files.")
    parser.add_argument("base_dir", type=str, help="Base directory path to scan for datasets.")
    args = parser.parse_args()

    process_dataset(args.base_dir)
