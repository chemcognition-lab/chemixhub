import pandas as pd
import os
from typing import List, Dict, Tuple
import math
import numpy as np

def create_compound_dataframe(raw_data_path: List[str]) -> pd.DataFrame:

    for path in raw_data_path:

        df = pd.read_csv(path)

        components = ['S1', 'S2', 'S3', 'S4', 'Salt1', 'Salt2']

        smiles_list = df[[f"{cmp} SMILES" for cmp in components]].values.flatten()
        smiles_list = list(set([value for value in smiles_list if pd.notna(value)]))

    compound_df = pd.DataFrame({"smiles": smiles_list})
    compound_df["salt"] = [1 if "." in smi else 0 for smi in compound_df["smiles"]]
    compound_df["polymer"] = [1 if "[Au]" in smi or "[Cu]" in smi else 0 for smi in compound_df["smiles"]]
    compound_df["monomeric_unit"] = [smi if compound_df["polymer"][i] == 1 else None for i, smi in enumerate(compound_df["smiles"])]

    compound_df.index.name = 'compound_id'

    return compound_df


def name_processing(df: pd.DataFrame, name_to_id: Dict) -> pd.DataFrame:

    # df = df.rename(columns={col: col.split(",")[0].lower().replace(" ", "_") for col in df.columns})

    name_columns = [col for col in df.columns if "SMILES" in col]

    for name_col in name_columns:
        df[name_col.replace(" SMILES", "_id")] = df[name_col].map({v: k for k, v in name_to_id.items()})
    
    df = df.drop(columns=name_columns)

    return df


def aggregate_cols_to_list(df: pd.DataFrame) -> pd.DataFrame:

    components = [col.strip("_id") for col in df.columns if "_id" in col]

    for substring in ["id", "mole_fraction", "mw"]:
        selected_columns = [f"{cmp}_{substring}" for cmp in components]
        df[f"cmp_{substring}s"] = df[selected_columns].apply(lambda row: row.dropna().tolist(), axis=1)
        df = df.drop(columns=selected_columns)

    return df


if __name__ == "__main__":

    base_path = os.path.abspath("../raw_data")
    raw_data_path = [os.path.join(base_path, filename) for filename in os.listdir(base_path) if ".py" not in filename]

    compound_df_path = os.path.abspath("./compounds.csv")
    compound_df = create_compound_dataframe(raw_data_path)
    compound_df.to_csv(compound_df_path, index=True)

    name_to_id = compound_df['smiles'].to_dict()

    for path in raw_data_path:

        df = pd.read_csv(path)

        # Remove data source columns
        data_source_cols = [
            'Notes',
            'Compound Notebook Name',
            'Date Copied to Master Database',
            'Data Recorded By',
            'Seshadri Group',
            'Oyaizu Group',
            'DOI',
        ]

        df = df.drop(columns=data_source_cols)

        # Remove polymerization info
        polymerization_info_cols = [col for col in df.columns if "Type" in col or "DOP" in col]

        df = df.drop(columns=polymerization_info_cols)
        
        # Remove inorganic material points
        df = df[df["Inorganic Material 1 (IM1)"].isna()]

        inorganic_cols = [
            'Inorganic Material 1 (IM1)',
            'IM1 Weight %',
            'IM1 Particle Size (nm)',
        ]

        df = df.drop(columns=inorganic_cols)

        # Only keep first SMILES type
        smiles_cols = [col for col in df.columns if "SMILES" in col]
        to_remove_smiles_cols = [col for col in smiles_cols if not col.endswith("SMILES") or "Big" in col]

        df = df.drop(columns=to_remove_smiles_cols)

        # Remove single-compound datapoints
        # smiles_cols = [col for col in df.columns if "SMILES" in col]
        # num_cmp = df[smiles_cols].notna().sum(axis=1)
        # df = df[num_cmp > 1]

        # Remove rows where no Mn/Mw is given
        mw_mn_cols = [col for col in df.columns if "Mw" in col or "Mn" in col]
        df = df[df[mw_mn_cols].isna().all(axis=1) == False]

        # Only keep rows where "Mn or Mw" is given to standardize
        mw_mn_drop = [col for col in mw_mn_cols if "Mn or Mw" not in col]
        df = df.drop(columns=mw_mn_drop)

        # Name processing
        df = name_processing(df=df, name_to_id=name_to_id)

        # what to do with this
        name_cols = ['Solvent 1 (S1)', 'Solvent 2 (S2)', 'Solvent 3 (S3)', 'Solvent 4 (S4)', 'Salt 1', 'Salt 2']
        df = df.drop(columns=name_cols)

        # what to do with this
        endgroup_cols = ['S1 End Group 1', 'S1 End Group 2']
        df = df.drop(columns=endgroup_cols)

        # Remove faulty Ratio rows
        ratio_cols = [col for col in df.columns if "Ratio" in col]
        df = df.drop(columns=ratio_cols)

        # Drop outlier columns
        df = df.drop(columns=["S1 Block Mol Weight", "Heating Rate (K/min)", "Tg (oC)"])

        # Fraction processing (only keep mole fraction)
        id_cols = [col for col in df.columns if "_id" in col]
        weight_terms = ["%", "Molality"]  
      
        def get_non_nan_id(row):
            return [col.strip("_id") for col in row.index if pd.notna(row[col])]

        non_nan_ids = df[id_cols].apply(get_non_nan_id, axis=1)
        non_nan_ids = non_nan_ids.reset_index(drop=True)

        rows_to_keep = []
        for i, (index, row) in enumerate(df.iterrows()):

            matching_columns = [col for col in df.columns if any(sub in col for sub in non_nan_ids[i]) and any(term in col for term in weight_terms)]

            mol_fraction_cols = [col for col in matching_columns if "Mol %" in col]
            # weight_fraction_cols = [col for col in matching_columns if "Weight %" in col]
            # molality_cols = [col for col in matching_columns if "Molality" in col]
            # mw_cols = [col for col in df.columns if any(sub in col for sub in non_nan_ids[i]) and any(term in col for term in ["Mw", "Molar Mass"])]

            # row[mol_fraction_cols].isna().any() == False or 

            if math.isclose(row[mol_fraction_cols].sum(), 1.0, abs_tol=1e-5):
                # Filter out edge cases where molality in original data is > 0 but Mol % is 0
                if 'Salt1 Mol %' in mol_fraction_cols or 'Salt2 Mol %' in mol_fraction_cols:
                    if row["Salt1 Molality (mol salt/kg polymer)"] > 0 and (row["Salt1 Mol %"] == 0 or pd.isna(row["Salt1 Mol %"])):
                        continue
                    if row["Salt2 Molality (mol salt/kg polymer)"] > 0 and (row["Salt2 Mol %"] == 0 or pd.isna(row["Salt2 Mol %"])):
                        continue
                    else:
                        rows_to_keep.append(row)
                else:
                    rows_to_keep.append(row)
        df = pd.DataFrame(rows_to_keep)

        # Remove other fractions/molality
        weight_cols = [col for col in df.columns if "Weight" in col or "Molality" in col]
        df = df.drop(columns=weight_cols)

        # Make property/value/unit cols (predict log conductivity)
        df = df.drop(columns=["Conductivity (S/cm)"])
        df = df.rename(columns={"log Conductivity (S/cm)": "value"})
        df["property"] = "log Conductivity"
        df["unit"] = "S/cm"

        # Convert temperature to Kelvin
        df["Temperature, K"] = df["Temperature (oC)"] + 273.15
        df = df.drop(columns=["Temperature (oC)"])

        # Rename columns
        components = ['S1', 'S2', 'S3', 'S4', 'Salt1', 'Salt2']
        components_to_cmp_id = {component: f"cmp{i+1}" for i, component in enumerate(components)}

        new_cols = {col: col.replace(key, components_to_cmp_id[key]) 
                    for col in df.columns 
                    for key in components_to_cmp_id if col.startswith(key)}

        df = df.rename(columns=new_cols)

        new_cols = {col: col.replace(' Mol %', '_mole_fraction') for col in df.columns if 'Mol %' in col}
        df = df.rename(columns=new_cols)

        new_cols = {col: col.replace(' Molar Mass (g/mol)', '_mw') for col in df.columns if 'Molar Mass (g/mol)' in col}
        df = df.rename(columns=new_cols)

        new_cols = {col: col.replace(' Mn or Mw', '_mn_or_mw') for col in df.columns if 'Mn or Mw' in col}
        df = df.rename(columns=new_cols)

        # Id, MW and Mole fraction cols to single col
        df = aggregate_cols_to_list(df=df)

        # caps ends of polymers with carbons
        compound_df["smiles"] = [smi.replace("[Cu]", "C").replace("[Au]", "C").replace("[Ca]", "C") for smi in compound_df["smiles"]]
        compound_df.to_csv(compound_df_path, index=True)

        csv_name = "processed_" + path.split("/")[-1]
        df.to_csv(f"./{csv_name}", index=False)