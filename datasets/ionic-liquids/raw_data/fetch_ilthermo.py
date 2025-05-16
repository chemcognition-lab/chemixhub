import ilthermopy
import pandas as pd
from tqdm import tqdm
import re
import html


def extract_entry(entry_id, prop_query):

    entry = ilthermopy.data_structs.GetEntry(entry_id)

    header_dict = entry.header
    entry_df = entry.data
    entry_df = entry_df.rename(columns=header_dict)

    cmp_to_name = {}
    name_to_cmp = {}

    entry_df["num_phases"] = len(entry.phases)

    for i, component in enumerate(entry.components):
        entry_df[f"cmp{i+1}_smiles"] = component.smiles
        entry_df[f"cmp{i+1}_name"] = component.name
        entry_df[f"cmp{i+1}_ilthermo_id"] = component.id
        entry_df[f"cmp{i+1}_mw"] = component.mw

        cmp_to_name[f"cmp{i+1}"] = component.name
        name_to_cmp[component.name] = f"cmp{i+1}"

    if entry.solvent is not None:
        solvents = entry.solvent.split(" + ")
        solvents = [name_to_cmp[name] for name in solvents]
    else:
        solvents = None

    new_columns = {}
    for col in entry_df.columns:
        
        # Renaming
        new_col = col
        for component in entry.components:
            if component.name in col:
                entry_df = entry_df.rename(columns={col: re.sub(re.escape(component.name), name_to_cmp[component.name], col)})

        if new_col != col:
            new_columns[col] = new_col

        if prop_query in col:

            entry_df["property"] = prop_query
            entry_df = entry_df.rename(columns={col: "value"})

            match = re.search(r',\s*(.*?)\s*=>', col)
            if match:
                entry_df["unit"] = html.unescape(match.group(1))

        if "Error" in col:
            entry_df = entry_df.rename(columns={col: "error"})

    entry_df = entry_df.rename(columns=new_columns)

    for col in entry_df.columns:
        for component in entry.components:
            if component.name in col:
                print("did not work", col)

    return entry_df

def fetch_ilthermo_db(prop_query_list, num_components_list):

    final_df = pd.DataFrame()

    for prop_query in prop_query_list:

        for num_components in num_components_list:

            print(f"Extracting {prop_query} data for {num_components}-compounds mixtures")

            mix = ilthermopy.search.Search(n_compounds=num_components, prop=prop_query)

            print(mix["num_phases"].value_counts())

            stacked_df = pd.DataFrame()

            for _, row in tqdm(mix.iterrows(), total=mix.shape[0]):

                entry_df = extract_entry(entry_id=row["id"], prop_query=prop_query)

                entry_df["entry_ilthermo_id"] = row["id"]

                stacked_df = pd.concat([stacked_df, entry_df], ignore_index=True)
            
            final_df = pd.concat([final_df, stacked_df], ignore_index=True)

    final_df.to_csv(f"./IlThermoData.csv", index=False)

if __name__ == "__main__":

    # properties_list = ['Electrical conductivity', 'Density', 'Viscosity',
    #  'Heat capacity at constant pressure', 'Refractive index',
    #  'Equilibrium pressure', 'Heat capacity at vapor saturation pressure',
    #  'Speed of sound', 'Excess volume', 'Excess enthalpy', 'Enthalpy of dilution',
    #  'Composition at phase equilibrium', 'Equilibrium temperature',
    #  'Surface tension liquid-gas', 'Apparent molar volume', 'Apparent enthalpy',
    #  'Binary diffusion coefficient', 'Interfacial tension',
    #  'Osmotic coefficient', 'Activity', 'Partial molar enthalpy ',
    #  'Thermal conductivity', 'Apparent molar heat capacity',
    #  'Partial molar heat capacity ', 'Tracer diffusion coefficient',
    #  'Enthalpy of solution', 'Thermal diffusivity', 'Relative permittivity',
    #  "Henry's Law constant", 'Enthalpy',
    #  'Enthalpy of mixing of a binary solvent with component',
    #  'Partial molar volume ', 'Critical pressure', 'Upper consolute pressure',
    #  'Eutectic composition', 'Eutectic temperature', 'Monotectic temperature',
    #  'Upper consolute temperature', 'Upper consolute composition',
    #  'Critical temperature', 'Isobaric coefficient of volume expansion']

    prop_query_list = ['Electrical conductivity', 'Viscosity']
    num_components = [2, 3]

    fetch_ilthermo_db(prop_query_list, num_components)
