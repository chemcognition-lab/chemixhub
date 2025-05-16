import os
import pandas as pd
from typing import List, Dict, Tuple


def create_compound_dataframe(raw_data_path: List[str]) -> pd.DataFrame:

    names_to_smiles = {}

    for path in raw_data_path:

        df = pd.read_csv(path)

        components = {"0": "Drug", "1": "Solvent_1", "2": "Solvent_2"}

        for k,v in components.items():
            cmp_dict = pd.Series(df[f"SMILES_{k}"].values, index=df[f"{v}"]).dropna().to_dict()
            names_to_smiles.update(cmp_dict)

    compound_df = pd.DataFrame(list(names_to_smiles.items()), columns=['name', 'smiles'])
    compound_df["salt"] = [1 if "." in smi else 0 for smi in compound_df["smiles"]]

    compound_df.index.name = 'compound_id'
    return compound_df

def name_processing(df: pd.DataFrame, name_to_id: Dict) -> pd.DataFrame:

    # df = df.rename(columns={col: col.split(",")[0].lower().replace(" ", "_") for col in df.columns})

    fraction_columns = [col for col in df.columns if "comp" in col]
    name_columns = [col for col in df.columns if "SMILES" in col]

    for frac_col in fraction_columns:
        df[frac_col] = df[frac_col] / 100
        df[frac_col] = df[frac_col].replace(0.0, None)

    for name_col in name_columns:
        df[name_col.replace("SMILES_", "cmp") + "_id"] = df[name_col].map({v: k for k, v in name_to_id.items()})
    
    df = df.drop(columns=name_columns)

    df = df.rename(columns={"LogS": "value"})
    df = df.rename(columns={k: k.replace('comp_', 'cmp') + '_mole_fraction' for k in fraction_columns})
    df["property"] = "Log solubility"
    df["unit"] = "g/L"

    return df

def aggregate_cols_to_list_solvent(df: pd.DataFrame) -> pd.DataFrame:

    solvent_components = [col.strip("_id") for col in df.columns if "_id" in col and "0" not in col]


    for substring in ["id", "mole_fraction"]:
        selected_columns = [f"{cmp}_{substring}" for cmp in solvent_components]
        df[f"cmp_{substring}s_solvent"] = df[selected_columns].apply(lambda row: row.dropna().tolist(), axis=1)
        df = df.drop(columns=selected_columns)

    return df

def aggregate_cols_to_list_drug(df: pd.DataFrame) -> pd.DataFrame:

    drug_components = [col.strip("_id") for col in df.columns if "_id" in col and "0" in col]

    for substring in ["id", "mole_fraction"]:
        selected_columns = [f"{cmp}_{substring}" for cmp in drug_components]
        df[f"cmp_{substring}s_drug"] = df[selected_columns].apply(lambda row: row.dropna().tolist(), axis=1)
        df = df.drop(columns=selected_columns)

    return df

if __name__ == "__main__":

    base_path = os.path.abspath("../raw_data")
    raw_data_path = [os.path.join(base_path, filename) for filename in os.listdir(base_path) if ".py" not in filename]

    compound_df_path = os.path.abspath("./compounds.csv")

    if not os.path.exists(compound_df_path):
        compound_df = create_compound_dataframe(raw_data_path)
        compound_df.to_csv(compound_df_path, index=True)
    else:
        compound_df = pd.read_csv(compound_df_path)

    name_to_id = compound_df['smiles'].to_dict()

    for path in raw_data_path:

        df = pd.read_csv(path)

        # Name processing
        df = name_processing(df=df, name_to_id=name_to_id)

        # Remove predicted values
        predict_cols = [col for col in df.columns if "Predict" in col]
        df = df.drop(columns=predict_cols)

        # Remove label columns
        label_cols = [col for col in df.columns if "Label" in col]
        df = df.drop(columns=label_cols)

        # Remove name columns
        name_cols = [col for col in df.columns if "Drug" in col or "Solvent" in col]
        df = df.drop(columns=name_cols)

        df = aggregate_cols_to_list_solvent(df=df)
        df = aggregate_cols_to_list_drug(df=df)

        # df = df.drop(columns=["cmp_mole_fractions_drug"])

        csv_name = "processed_" + path.split("/")[-1]
        df.to_csv(f"./{csv_name}", index=False)