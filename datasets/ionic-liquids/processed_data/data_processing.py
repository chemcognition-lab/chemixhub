import pubchempy
from pubchempy import get_compounds
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
import os
import time
import re

def create_compound_dataframe(raw_data_path: List[str]) -> pd.DataFrame:

    names_to_smiles = {}

    for path in raw_data_path:

        df = pd.read_csv(path)

        components = ['cmp1', 'cmp2', 'cmp3']

        for component in components:
            cmp_dict = pd.Series(df[f"{component}_smiles"].values, index=df[f"{component}_name"]).dropna().to_dict()
            names_to_smiles.update(cmp_dict)

    compound_df = pd.DataFrame(list(names_to_smiles.items()), columns=['name', 'smiles'])
    compound_df["salt"] = [1 if "." in smi else 0 for smi in compound_df["smiles"]]

    compound_df.index.name = 'compound_id'
    return compound_df


def name_processing(df: pd.DataFrame, name_to_id: Dict) -> pd.DataFrame:

    # df = df.rename(columns={col: col.split(",")[0].lower().replace(" ", "_") for col in df.columns})

    name_columns = [col for col in df.columns if "smiles" in col]

    for name_col in name_columns:
        df[name_col.replace("_smiles", "_id")] = df[name_col].map({v: k for k, v in name_to_id.items()})
    
    df = df.drop(columns=name_columns)

    return df


def ratio_to_fraction_binary(ratio_to_solvent):
    fraction_substrate = ratio_to_solvent/(ratio_to_solvent + 1)
    fraction_solvent = 1 - ratio_to_solvent/(ratio_to_solvent + 1)
    return fraction_substrate, fraction_solvent


def ratio_to_fraction_ternary(ratio_to_solvent_1, ratio_to_solvent_2):
    fraction_substrate = ratio_to_solvent_1 / (ratio_to_solvent_1 + 1 + (ratio_to_solvent_1 / ratio_to_solvent_2))
    fraction_solvent_1 = (1 / ratio_to_solvent_1) / ((1 / ratio_to_solvent_1) + 1 + (1 / ratio_to_solvent_2))
    fraction_solvent_2 = (1 / ratio_to_solvent_2) / ((1 / ratio_to_solvent_1) + 1 + (1 / ratio_to_solvent_2))
    return fraction_substrate, fraction_solvent_1, fraction_solvent_2


def process_ternary_compounds(df_sub, weights_columns):
    """Process data for ternary compounds (three components)"""
    if len(weights_columns) != 2:
        print("Ternary failed weight_col")
        return
    
    # Check if weights include solvent information
    contains_solvent = [col for col in weights_columns if "Solvent:" in col]
    if contains_solvent:
        compound_name_list = [f"cmp{i}" for i in re.findall(r'cmp(\d)', ' '.join(weights_columns))]

        compound_left = get_remaining_compound(["cmp1", "cmp2", "cmp3"], compound_name_list)
        solvent_components = [f"cmp{i}" for i in re.findall(r'cmp(\d)', ' '.join(contains_solvent))] + [compound_left]

        df_sub = process_binary_compounds(df_sub, contains_solvent, solvent_components)

        column_left = [col for col in weights_columns if col not in contains_solvent]

        if "MolaLity" in column_left[0]:

            # Calculate moles of solvent for 1kg of it
            solvent_mw = 0
            for cmp in solvent_components:
                solvent_mw += df_sub[f'Mole fraction of {cmp} => Liquid']*df_sub[f'{cmp}_mw']

            solvent_moles = 1000 / solvent_mw

            # Calculate moles of solvent's components
            mole_list = []
            for cmp in solvent_components:
                mole_list.append(df_sub[f'Mole fraction of {cmp} => Liquid']*solvent_moles)

            # Get moles of solute
            solute = get_remaining_compound(["cmp1", "cmp2", "cmp3"], solvent_components)
            solute_moles = df_sub[column_left[0]]
            mole_list.append(solute_moles)

            # Total num moles
            total_moles = solvent_moles + solute_moles

            # Get adjusted mole fractions
            fraction_column_name_list = [f'Mole fraction of {cmp} => Liquid' for cmp in solvent_components + [solute]]
            for i, fraction_col in enumerate(fraction_column_name_list):
                df_sub[fraction_col] = mole_list[i] / total_moles

        elif "Weight fraction" in column_left[0]:

            # Calculate molecular weight of solvent                   
            solvent_mw = 0
            for cmp in solvent_components:
                solvent_mw += df_sub[f'Mole fraction of {cmp} => Liquid']*df_sub[f'{cmp}_mw']

            # Weight fraction of solute to mole fraction
            solute = get_remaining_compound(["cmp1", "cmp2", "cmp3"], solvent_components)
            
            solute_fraction = (df_sub[column_left[0]] / df_sub[f'{solute}_mw']) / ((df_sub[column_left[0]] / df_sub[f'{solute}_mw']) + ((1 -df_sub[column_left[0]]) / solvent_mw))

            solvent_fraction = 1 - solute_fraction
            fraction_column_name_list = [f'Mole fraction of {cmp} => Liquid' for cmp in solvent_components]

            for i, fraction_col in enumerate(fraction_column_name_list):
                df_sub[fraction_col] = solvent_fraction * df_sub[fraction_col]

        elif "Mole fraction" in column_left[0]:

            solute_fraction = df_sub[column_left[0]]
            solvent_fraction = 1 - solute_fraction

            fraction_column_name_list = [f'Mole fraction of {cmp} => Liquid' for cmp in solvent_components]
            for i, fraction_col in enumerate(fraction_column_name_list):
                df_sub[fraction_col] = solvent_fraction * df_sub[fraction_col]

        elif "Mole ratio" in column_left[0]:

            # Assume 1 mole of solvent
            solute_moles = df_sub[column_left[0]] 
            total_moles = 1 + solute_moles

            # Get mole fractions for solute
            solute = get_remaining_compound(["cmp1", "cmp2", "cmp3"], solvent_components)
            df_sub[f'Mole fraction of {solute} => Liquid'] = solute_moles / total_moles

            # Get mole fractions for solvent
            fraction_column_name_list = [f'Mole fraction of {cmp} => Liquid' for cmp in solvent_components]

            for i, fraction_col in enumerate(fraction_column_name_list):
                df_sub[fraction_col] = df_sub[fraction_col] / total_moles

        else:
            print(contains_solvent, column_left)
  
    else:

        # Process based on column types
        if "ratio" in weights_columns[0] and "ratio" in weights_columns[1]:
            df_sub = process_ratio(df_sub, weights_columns)
        elif "MolaLity" in weights_columns[0] and "MolaLity" in weights_columns[1]:
            df_sub = process_molality(df_sub, weights_columns)
        elif "Weight fraction" in weights_columns[0] and "Weight fraction" in weights_columns[1]:
            df_sub = process_weight_fraction(df_sub, weights_columns)
        elif "Mole fraction" in weights_columns[0] and "Mole fraction" in weights_columns[1]:
            df_sub = process_mole_fraction(df_sub, weights_columns)
        else:
            # print("heterogenous weights")
            # print(weights_columns)
            print(df_sub[weights_columns + ["property", "cmp1_id"]])
    
    return df_sub


def process_binary_compounds(df_sub, weights_columns, components=["cmp1", "cmp2"]):
    """Process data for binary compounds (two components)"""
    if len(weights_columns) != 1:
        print("Binary failed weight_col")
        return

    column = weights_columns[0]
    contains_solvent = True if "to solvent" in column else False
    
    if "ratio" in column:
        df_sub = process_ratio(df_sub, [column], False, components)
    elif "MolaLity" in column:
        df_sub = process_molality(df_sub, [column], components)
    elif "Weight fraction" in column:
        df_sub = process_weight_fraction(df_sub, [column], components)
    elif "Mole fraction" in column:
        df_sub = process_mole_fraction(df_sub, [column], components)
    
    return df_sub


def get_remaining_compound(compounds_list, compounds_to_remove):
    """Find the remaining compound from a list after removing specified compounds"""
    result = compounds_list.copy()
    for compound in compounds_to_remove:
        if compound in result:
            result.remove(compound)
    return result[0] if result else None


def extract_compound_info(column_name, pattern):
    """Extract compound information using regex pattern"""
    matches = re.search(pattern, column_name)
    return matches.groups() if matches else None


def process_ratio(df_sub, weights_columns, ternary=True, components=["cmp1", "cmp2", "cmp3"]):
    """Process data with ratio columns"""
    pattern = r"(Mole ratio|Volume ratio|Mass ratio|Weight ratio).*?(cmp\d).*?(Liquid)"
    
    # Process each column to extract information
    fraction_column_name_list = []
    compound_name_list = []

    for col in weights_columns:
        property_name, compound_name, phase_name = extract_compound_info(col, pattern)
        property_name = property_name.split(" ratio")[0]
        
        fraction_column_name = f'{property_name} fraction of {compound_name} => {phase_name}'

        fraction_column_name_list.append(fraction_column_name)
        compound_name_list.append(compound_name)
    
    # Find the remaining compound
    compound_left = get_remaining_compound(components, compound_name_list)
    compound_name_list.append(compound_left)

    
    fraction_column_name = f'{property_name} fraction of {compound_left} => {phase_name}'
    fraction_column_name_list.append(fraction_column_name)
    
    # Calculate fractions
    if ternary:
        df_sub[fraction_column_name_list] = (
            df_sub[weights_columns]
            .apply(lambda row: ratio_to_fraction_ternary(row[weights_columns[0]], row[weights_columns[1]]), axis=1)
            .apply(pd.Series)
        )
    else:
        df_sub[fraction_column_name_list] = (
            df_sub[weights_columns]
            .apply(lambda row: ratio_to_fraction_binary(row[weights_columns[0]]), axis=1)
            .apply(pd.Series)
        )

    if property_name == "Mass" or property_name == "Weight":

        # Calculate mole values
        compound_mole_list = []

        for i, fraction_col in enumerate(fraction_column_name_list):
            compound_mole_list.append(df_sub[fraction_col] / df_sub[f"{compound_name_list[i]}_mw"])

        total_moles = sum(compound_mole_list)

        # Calculate fractions for each compound and subtract from the remaining compound
        for i, fraction_col in enumerate(fraction_column_name_list):
            df_sub[fraction_col.replace(property_name, "Mole")] = compound_mole_list[i] / total_moles

    return df_sub


def process_molality(df_sub, weights_columns, components=["cmp1", "cmp2", "cmp3"]):
    """Process data with molality columns"""
    pattern = r"of (.*?), (.*?)\s*=>\s*(.*)"
    
    # Process each column to extract information
    fraction_column_name_list = []
    compound_name_list = []

    for col in weights_columns:
        compound_name, _, phase_name = extract_compound_info(col, pattern)
        
        fraction_column_name = f'Mole fraction of {compound_name} => {phase_name}'

        fraction_column_name_list.append(fraction_column_name)
        compound_name_list.append(compound_name)
    
    # Find the remaining compound
    compound_left = get_remaining_compound(components, compound_name_list)

    fraction_column_name_cmp_left = f'Mole fraction of {compound_left} => {phase_name}'
    fraction_column_name_list.append(fraction_column_name_cmp_left)
    
    # Initialize the remaining compound's fraction to 1
    df_sub[fraction_column_name_cmp_left] = 1
    
    # Calculate fractions for each compound and subtract from the remaining compound
    for i, weights_col in enumerate(weights_columns):
        df_sub[fraction_column_name_list[i]] = df_sub[weights_col] / (df_sub[weights_col] + (1000 / df_sub[f'{compound_left}_mw']))
        df_sub[fraction_column_name_cmp_left] -= df_sub[fraction_column_name_list[i]]

    return df_sub


def process_weight_fraction(df_sub, weights_columns, components=["cmp1", "cmp2", "cmp3"]):
    """Process ternary data with weight fraction columns"""
    pattern = r"of (.*?)\s*=>\s*(.*)"

    # Process each column to extract information
    fraction_column_name_list = []
    compound_name_list = []

    for col in weights_columns:
        compound_name, phase_name = extract_compound_info(col, pattern)
        
        fraction_column_name = f'Mole fraction of {compound_name} => {phase_name}'

        fraction_column_name_list.append(fraction_column_name)
        compound_name_list.append(compound_name)

    # Find the remaining compound
    compound_left = get_remaining_compound(components, compound_name_list)
    
    fraction_column_name_cmp_left = f'Mole fraction of {compound_left} => {phase_name}'
    fraction_column_name_list.append(fraction_column_name_cmp_left)
    
    # Calculate mole values
    compound_mole_list = []

    for i, weights_col in enumerate(weights_columns):
        compound_mole_list.append(df_sub[weights_col] / df_sub[f"{compound_name_list[i]}_mw"])

    remaining_weight_fraction = 1 - sum(df_sub[weights_columns[j]] for j in range(len(weights_columns)))
    compound_mole_list.append(remaining_weight_fraction / df_sub[f'{compound_left}_mw'])

    total_moles = sum(compound_mole_list)

    # Initialize the remaining compound's fraction to 1
    df_sub[fraction_column_name_cmp_left] = 1

    # Calculate fractions for each compound and subtract from the remaining compound
    for i, weights_col in enumerate(weights_columns):
        df_sub[fraction_column_name_list[i]] = compound_mole_list[i] / total_moles
        df_sub[fraction_column_name_cmp_left] -= df_sub[fraction_column_name_list[i]]

    return df_sub


def process_mole_fraction(df_sub, weights_columns, components=["cmp1", "cmp2", "cmp3"]):
    """Process ternary data with mole fraction columns"""

    weights_columns_new = []
    for col in weights_columns:
        if "Solvent: " in col:
            df_sub = df_sub.rename(columns={col: col.split("Solvent: ")[-1]})
            weights_columns_new.append(col.split("Solvent: ")[-1])
        else:
            weights_columns_new.append(col)

    if weights_columns != weights_columns_new:
        weights_columns = weights_columns_new

    pattern = r"of (.*?)\s*=>\s*(.*)"

    # Process each column to extract information
    fraction_column_name_list = []
    compound_name_list = []

    for col in weights_columns:
        compound_name, phase_name = extract_compound_info(col, pattern)
        compound_name_list.append(compound_name)

    # Find the remaining compound
    compound_left = get_remaining_compound(components, compound_name_list)
    
    fraction_column_name_cmp_left = f'Mole fraction of {compound_left} => {phase_name}'
    fraction_column_name_list.append(fraction_column_name_cmp_left)

    df_sub[fraction_column_name_cmp_left] = 1

    # Subtract fractions from the remaining compound
    for i, weights_col in enumerate(weights_columns):
        df_sub[fraction_column_name_cmp_left] -= df_sub[weights_col]

    return df_sub

def aggregate_cols_to_list(df: pd.DataFrame) -> pd.DataFrame:

    components = [col.strip("_id") for col in df.columns if "_id" in col]

    for substring in ["id", "mole_fraction", "mw"]:
        selected_columns = [f"{cmp}_{substring}" for cmp in components]
        df[f"cmp_{substring}s"] = df[selected_columns].apply(lambda row: row.dropna().tolist(), axis=1)
        df = df.drop(columns=selected_columns)

    return df

def fill_na_pressure(df: pd.DataFrame) -> pd.DataFrame:
    df["Pressure_assumption"] = df["Pressure, kPa"].isna().astype(int)
    df["Pressure, kPa"] =  df["Pressure, kPa"].fillna(101.325)
    return df

def fill_na_frequency(df: pd.DataFrame) -> pd.DataFrame:
    df["Frequency_assumption"] = df["Frequency, MHz"].isna().astype(int)
    df["Frequency, MHz"] =  df["Frequency, MHz"].fillna(0)
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

        # Drop rows not conforming to expected property
        df = df.loc[df["property"].isna() == False]

        # Remove MolaRity datapoints
        molarity_or_vol = [col for col in df.columns if "MolaRity" in col or "Volume" in col]
        df = df[df[molarity_or_vol].isna().all(axis=1)]

        # Remove Glass/Gas datapoints
        not_liquid = [col for col in df.columns if "Gas" in col or "Glass" in col or "Crystal of unknown type" in col]
        df = df[df[not_liquid].isna().all(axis=1)]

        # Remove multi-phase ionic liquids
        df = df[df["num_phases"] == 1]

        df = df.dropna(axis="columns", how="all")

        for entry_id in df["entry_ilthermo_id"].unique():

            df_sub = df.loc[df["entry_ilthermo_id"] == entry_id]
            df_sub = df_sub.dropna(axis='columns', how='all')

            weights_columns = [col for col in df_sub.columns if "fraction of" in col or "MolaLity of" in col or "to" in col or "ratio of" in col]

            if "cmp3_id" in df_sub.columns:
                # Ternary compound case
                df_sub = process_ternary_compounds(df_sub, weights_columns)
            else:
                # Binary compound case
                df_sub = process_binary_compounds(df_sub, weights_columns)
            
            df.loc[df["entry_ilthermo_id"] == entry_id] = df_sub
        
        # Remove points with no mole fractions
        cols_mf = [col for col in df.columns if col.startswith("Mole fraction")]

        # Binary mixtures
        df_binary = df.loc[df["cmp3_id"].isna()]
        df_binary = df_binary.dropna(subset=["Mole fraction of cmp2 => Liquid",  "Mole fraction of cmp1 => Liquid"], how="any")
        
        # Ternary mixtures
        df_ternary = df.loc[df["cmp3_id"].isna() == False]
        df_ternary = df_ternary.dropna(subset=cols_mf, how="any")

        df = pd.concat([df_binary, df_ternary])

        # Select relevant columns 
        keywords = ["_mw", "property", "value", "error", "unit", "Temperature", "Pressure", "Frequency"]

        relevant_cols = [
            col for col in df.columns
            if (
                ("_id" in col and "ilthermo" not in col)
                or any(keyword in col for keyword in keywords)
            )
        ]

        relevant_cols += cols_mf

        df = df[relevant_cols]

        # Remove phase info
        df = df.rename(columns={col: col.replace(" => Liquid", "") for col in df.columns})

        # Rename mole fraction
        rename_mf = {col: col.replace('Mole fraction of ', '').replace('cmp', 'cmp') + '_mole_fraction' 
                    for col in df.columns if 'Mole fraction of ' in col}

        # Apply the renaming using df.rename
        df = df.rename(columns=rename_mf)

        # Id, MW and Mole fraction cols to single col
        df = aggregate_cols_to_list(df=df)

        # Add default context values
        df = fill_na_pressure(df=df)
        df = fill_na_frequency(df=df)

        # Apply Log to value column for viscosity and conductivity tasks
        df = df[df["value"] > 0]
        df["value"] = df["value"].apply(np.log)

        # Only select values within a pressure range near 1atm (+/- 2kPa)
        print(df.shape)
        df = df[df["Pressure, kPa"] <= 103.325]
        df = df[df["Pressure, kPa"] >= 99.325]
        print(df.shape)

        csv_name = "processed_" + path.split("/")[-1]
        df.to_csv(f"./{csv_name}", index=False)
