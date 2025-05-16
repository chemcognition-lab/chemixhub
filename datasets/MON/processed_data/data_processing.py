import pandas as pd
import os
import pubchempy as pcp
from tqdm import tqdm
from rdkit import Chem

def smiles_to_name(smiles):
   try:
      compounds = pcp.get_compounds(smiles, namespace='smiles')
      if compounds and compounds[0].iupac_name:
         return compounds[0].iupac_name
      elif compounds and compounds[0].synonyms:
         return compounds[0].synonyms[0]
      return "Unknown"
   except Exception as e:
      print(f"Error retrieving name for SMILES {smiles}: {e}")
      return "Error"

def extract_compounds_df(df: pd.DataFrame):
   all_smiles = pd.unique(df.filter(like="smiles").values.ravel())
   all_smiles = [s for s in all_smiles if pd.notna(s) and str(s).strip() != '']
    
   unique_smiles = pd.Series(all_smiles).drop_duplicates().reset_index(drop=True)
   compound_df = pd.DataFrame({
       'compound_id': range(0, len(unique_smiles)),
       'smiles': unique_smiles
   })

   compound_df['name'] = compound_df['smiles'].apply(smiles_to_name)
   compound_df = compound_df[['compound_id', 'name', 'smiles']]

   compound_df_path = os.path.abspath("./compounds.csv")
   compound_df.to_csv(compound_df_path, index=False)

   return compound_df

def name_processing(df: pd.DataFrame, name_to_id: dict) -> pd.DataFrame:

    # df = df.rename(columns={col: col.split(",")[0].lower().replace(" ", "_") for col in df.columns})

    name_columns = [col for col in df.columns if "smiles" in col]

    for name_col in name_columns:
       df[name_col.replace("_smiles", "_id")] = df[name_col].map({v: k for k, v in name_to_id.items()})
    
    df = df.drop(columns=name_columns)   

    ids_cols = [col for col in df.columns if col.startswith('cmp')]
    df['cmp_ids'] = df[ids_cols].values.tolist()
    df = df.drop(columns=ids_cols)   
    df['cmp_ids'] = df['cmp_ids'].apply(lambda x: [int(v) for v in x if pd.notna(v)])

    
    mole_frac_cols = [col for col in df.columns if col.startswith('Mole fraction of cmp')]
    df[mole_frac_cols] = df[mole_frac_cols].apply(lambda x: x/100)
    df['cmp_mole_fractions'] = df[mole_frac_cols].values.tolist()
    df = df.drop(columns=mole_frac_cols) 
    df['cmp_mole_fractions'] = df['cmp_mole_fractions'].apply(lambda x: [v for v in x if pd.notna(v)])

    return df


def canonicalize_smiles(smiles):
   try:
      mol = Chem.MolFromSmiles(smiles)
      if mol:
         return Chem.MolToSmiles(mol, canonical=True)
   except:
      pass
   #return None



  
    
if __name__ == "__main__":

   base_path = os.path.abspath("../raw_data")
   raw_data_path = [os.path.join(base_path, filename) for filename in os.listdir(base_path) if ".py" not in filename]
    
       
   # compound_df_path = os.path.abspath("./compounds.csv")
   # compound_df = pd.read_csv(compound_df_path)

   # name_to_id = compound_df['smiles'].to_dict()


   for path in raw_data_path:
      df = pd.read_csv(path)
      smiles_cols = [col for col in df.columns if 'smiles' in col]
       
      for col in smiles_cols:
         df[col] = df[col].apply(canonicalize_smiles)


      compound_df = extract_compounds_df(df)
   name_to_id = compound_df['smiles'].to_dict()

   # Name processing
   df = name_processing(df=df, name_to_id=name_to_id)
       
   # Removing duplicates
   df = df.drop_duplicates(
      subset=[
         "value",
         "pred_value",
         "error",
         "Train_Test_Label",
         # "Mole fraction of cmp0"
      ],
      keep="first"
   ).reset_index(drop=True)

   df["property"] = "Motor octane number"
   df["unit"] = None
   df = df.drop(columns=["pred_value", "Label"])

   df.to_csv("./processed_MON.csv", index=False)

