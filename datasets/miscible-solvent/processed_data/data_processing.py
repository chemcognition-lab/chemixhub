import os
import pandas as pd
from typing import List, Dict, Tuple


def create_compound_dataframe(raw_data_path: List[str]) -> pd.DataFrame:

    names_to_smiles = {}

    for path in raw_data_path:

        df = pd.read_csv(path)

        # Split the 'names' column by "|" into 5 separate columns
        split_names = df['label'].str.split('|', expand=True)

        # Rename the columns to name_0 through name_4
        split_names.columns = [f'name_{i}' for i in range(5)]

        # Combine back with the original DataFrame (optional)
        df = pd.concat([df, split_names], axis=1)

        for id in range(5):
            cmp_dict = pd.Series(df[f"SMILES_{id}"].values, index=df[f"name_{id}"]).dropna().to_dict()
            names_to_smiles.update(cmp_dict)


    compound_df = pd.DataFrame(list(names_to_smiles.items()), columns=['name', 'smiles'])
    compound_df["salt"] = [1 if "." in smi else 0 for smi in compound_df["smiles"]]

    compound_df.index.name = 'compound_id'
    return compound_df

def split_property(prop_str):
    parts = prop_str.split('_')
    if 'per' in parts:
        per_idx = parts.index('per')
        unit = '_'.join(parts[per_idx - 1:]).replace("_per_", "/").replace("cubic_meter", "m^3")
        name = '_'.join(parts[:per_idx - 1]).replace("_", " ").capitalize()
    else:
        unit = ''
        name = prop_str
    return pd.Series([name, unit])

def name_processing(df: pd.DataFrame, name_to_id: Dict) -> pd.DataFrame:

    # df = df.rename(columns={col: col.split(",")[0].lower().replace(" ", "_") for col in df.columns})

    name_columns = [col for col in df.columns if "SMILES" in col]

    for name_col in name_columns:
        df[name_col.replace("SMILES_", "cmp") + "_id"] = df[name_col].map({v: k for k, v in name_to_id.items()})
    
    df = df.drop(columns=name_columns)

    df = pd.melt(df, id_vars=[i for i in df.columns if "comp" in i or "cmp" in i], 
                        value_vars=['density_grams_per_cubic_meter','heat_of_vaporization_kcal_per_mol','enthalpy_of_mixing_kJ_per_mol'],
                        var_name='property',
                        value_name='value')


    df[['property', 'unit']] = df['property'].apply(split_property)

    return df

def aggregate_cols_to_list(df: pd.DataFrame) -> pd.DataFrame:

    components = [col.strip("_id") for col in df.columns if "_id" in col]

    for substring in ["id", "mole_fraction"]:
        selected_columns = [f"{cmp}_{substring}" for cmp in components]
        df[f"cmp_{substring}s"] = df[selected_columns].apply(lambda row: row.dropna().tolist(), axis=1)
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

        # Remove single component data points
        # df = df[df["num_components"] != 1]

        # Drop label and simulation properties columns
        df = df.drop(columns=["label", "ID", "num_components"])
        df = df.drop(columns=[col for col in df.columns if col.startswith("Simulation")])


        # Name processing
        df = name_processing(df=df, name_to_id=name_to_id)

        # Mole fraction processing
        df.loc[:, df.columns.str.startswith("comp")] = df.loc[:, df.columns.str.startswith("comp")] / 100
        df.columns = [f'cmp{col[5:]}_mole_fraction' if col.startswith('comp') else col for col in df.columns]

        # Id and Mole fraction cols to single col
        df = aggregate_cols_to_list(df=df)

        csv_name = "processed_" + path.split("/")[-1]
        df.to_csv(f"./{csv_name}", index=False)