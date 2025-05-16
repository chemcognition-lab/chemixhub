import pandas as pd
import os
from typing import Dict

def create_compound_dataframe(path: str) -> pd.DataFrame:

    names_to_smiles = {}

    df = pd.read_csv(path)

    for id in range(42):
        cmp_dict = pd.Series(df[f"smi_{id}"].values, index=df[f"smi_{id}"].values).dropna().to_dict()
        names_to_smiles.update(cmp_dict)


    compound_df = pd.DataFrame(list(names_to_smiles.items()), columns=['name', 'smiles'])
    compound_df = compound_df.drop(columns=["name"])
    compound_df["salt"] = [1 if "." in smi else 0 for smi in compound_df["smiles"]]

    compound_df.index.name = 'compound_id'

    return compound_df

def name_processing(df: pd.DataFrame, name_to_id: Dict) -> pd.DataFrame:

    # df = df.rename(columns={col: col.split(",")[0].lower().replace(" ", "_") for col in df.columns})

    name_columns = [col for col in df.columns if "smi" in col]

    for name_col in name_columns:
        df[name_col.replace("smi_", "cmp") + "_id"] = df[name_col].map({v: k for k, v in name_to_id.items()})
    
    df = df.drop(columns=name_columns)
    return df

def aggregate_cols_to_list(df: pd.DataFrame) -> pd.DataFrame:

    components = [col.strip("_id") for col in df.columns if "_id" in col]

    for substring in ["id"]:
        selected_columns = [f"{cmp}_{substring}" for cmp in components]
        df[f"cmp_{substring}s"] = df[selected_columns].apply(lambda row: row.dropna().tolist(), axis=1)
        df = df.drop(columns=selected_columns)

    return df

if __name__ == "__main__":
    base_path = os.path.abspath("../raw_data")

    smi_mix = os.path.join(base_path, "mixture_smi_definitions_clean.csv")
    mix_label = os.path.join(base_path, "mixtures_combined.csv")

    compound_df_path = os.path.abspath("./compounds.csv")

    if not os.path.exists(compound_df_path):
        compound_df = create_compound_dataframe(smi_mix)
        compound_df.to_csv(compound_df_path, index=True)
    else:
        compound_df = pd.read_csv(compound_df_path)
    
    name_to_id = compound_df['smiles'].to_dict()

    # Name processing
    df_smi_mix = pd.read_csv(smi_mix)
    df = name_processing(df=df_smi_mix, name_to_id=name_to_id)

    # Id and Mole fraction cols to single col
    df = aggregate_cols_to_list(df=df)

    df_mix_label = pd.read_csv(mix_label)
    
    mixture_lookup = df.set_index(['Dataset', 'Mixture Label'])['cmp_ids'].to_dict()

    # Add cmp_ids lists for each mixture
    df_mix_label['cmp_ids_1'] = df_mix_label.apply(lambda row: mixture_lookup.get((row['Dataset'], row['Mixture 1']), []), axis=1)
    df_mix_label['cmp_ids_2'] = df_mix_label.apply(lambda row: mixture_lookup.get((row['Dataset'], row['Mixture 2']), []), axis=1)

    # df_mix_label['cmp_ids'] = df_mix_label.apply(lambda row: [row['cmp_ids_1'], row['cmp_ids_2']], axis=1)

    # Final combined dataframe
    combined_df = df_mix_label[['Dataset', 'Mixture 1', 'Mixture 2', 'Experimental Values', 'cmp_ids_1', 'cmp_ids_2']]
    # combined_df = df_mix_label[['Dataset', 'Mixture 1', 'Mixture 2', 'Experimental Values', 'cmp_ids']]
    combined_df = combined_df.rename(columns={"Experimental Values": "value"})
    combined_df["property"] = "Mixture similarity"
    combined_df["unit"] = None

    csv_name = "processed_OlfactorySimilarity.csv"
    combined_df.to_csv(f"./{csv_name}", index=False)
