# CheMixHub Datasets 

## Supported datasets overview

CheMixHub consolidates and standardizes data from the following sources:

-   **Miscible Solvents:** Density, enthalpy of mixing, partial molar enthalpy.
    [Source Paper](https://chemrxiv.org/engage/chemrxiv/article-details/677d54c86dde43c908a14a6c)
-   **ILThermo:** Transport properties (ionic conductivity, viscosity) for ionic liquid mixtures.
    [Source Paper](https://ilthermo.boulder.nist.gov/) | _Includes 2 new large-scale tasks._ | To get the dataset, run the following script: ``datasets/ionic-liquids/raw_data/fetch_ilthermo.py``
-   **NIST Viscosity:** Viscosity for organic mixtures from NIST ThermoData Engine (via Zenodo).
    [Source Paper](https://doi.org/10.1016/j.cej.2023.142454)
    | [Link to Dataset](https://zenodo.org/records/8042966) |
    Path to data file in Zenodo: ``nist_dippr_source/NIST_Visc_Data.csv`` | Download and put the file in ``datasets/nist-logV/raw_data`` and to get the processed dataset, run the following script: ``datasets/nist-logV/processed_data/data_processing.py``
-   **Drug Solubility:** Drug solubility in solvent mixtures.
    [Source Paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00911-3)
-   **Solid Polymer Electrolyte Ionic Conductivity (SPEs):** Ionic conductivity for polymer–salt mixtures.
    [Source Paper](https://pubs.acs.org/doi/10.1021/acscentsci.2c01123)
-   **Olfactory Similarity:** Perceptual similarity scores for mixtures.
    [Source Paper](https://arxiv.org/abs/2501.16271)
-   **Motor Octane Number (MON):** Octane numbers for hydrocarbons and fuels.
    [Source Paper](https://www.nature.com/articles/s42004-022-00722-3)

## Dataset repository structure

The dataset repository is structured as follow:

```
dataset_name/
├── raw_data/
└── processed_data/
    ├── dataset_name_splits/
    ├── compounds.csv
    ├── croissant.json
    ├── data_processing.py
    └── processed_dataset_name.csv
```

- `raw_data/` contains the data in its original format. If the dataset is obtained using an API (eg. IlThermo), the script used to obtain it can be included in this folder.
- `processed_data/` contains the data in the CheMixHub format. Namely, 4 files are required:
    1. `compounds.csv` is a CSV file containing two columns `compound_id,smiles` associating smiles to a compound ID. Importantly, the `compound_id` column should contain indices that are up to the `total number of compounds - 1` in your dataset (ie. if you have a mixture dataset of 3 different compounds, values in the `compound_id` column cannot exceed `2`)
    2. `processed_dataset_name.csv` is a CSV file containing the following columns:
        
        (Required)
        - `property`: a string indicating the target property.
        - `value`: a float/int indicating the value of the target property.
        - `cmp_ids`: a string made of a list of int corresponding to the compound IDs associated to the molecules in the mixture for which the property data has been collected.

        (Optional) If useful for prediction, more information can be included:
        - `cmp_mole_fractions`: a string made of a list of float corresponding to the molar fractions associated to the molecules in the mixture for which the property data has been collected. The order of the mole fractions should correspond to the order of the compounds in `cmp_ids`.
        - `T`: a float indicating the temperature at which the target property was measured.
        etc.

        (Optional) Additional columns can be added to preserve sources of information, for example:
        - `unit`: a string indicating the unit of the target property.
        - `Pressure assumption`: a binary indicating whether the standard conditions of pressure were assumed

    3. `croissant.json`: A Croissant file associated with the dataset.
    4. `data_processing.py`: The script converting the data found in `raw_data/` into `compounds.csv` and `processed_dataset_name.csv`
    5. `dataset_name_splits/`: The folder containing the different split types for a defined dataset

## Adding a new dataset

1. Create a dataset folder following the dataset repository structure explained above.
2. Add a dataset class following this format in `chemixhub/src/mixhub/data/data.py`, replacing `dataset_name` by the name of your dataset:

```python
class DatasetName(MixtureDataInfo):
    def __init__(
            self,
            name: str = "Dataset name",
            description: str = """
            Dataset description
            """,
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["cmp_mole_fractions"],
            context_columns: list[str] = [], # Add the name of the columns for which you have mixture context
            output_column: str = "value",
            data_dir: str = os.path.abspath("../datasets/dataset_name/processed_data"),
            compound_csv_name: str = "compounds",
            mixture_csv_name: str = "processed_dataset_name",
    ):
        super().__init__(
            name,
            description,
            id_column,
            fraction_column,
            context_columns,
            output_column,
            data_dir,
            compound_csv_name,
            mixture_csv_name
        )
```

3. Add the class you added to the `DATA_CATALOG` in `chemixhub/src/mixhub/data/data.py`, replacing `dataset_name` by the name of your dataset:

```python
DATA_CATALOG = {
    ...
    "dataset_name": DatasetName,
}
```

4. Create datasets splits, for example to create a simple 5-fold random split:

```
cd ../chemixhub/scripts/
python make_splits.py --dataset_name dataset_name --split_type kfold
```

You're now done with dataset preparation and ready to move on to benchmarking! To do so, please consult the scripts found in `chemixhub/scripts`

## (Just) Loading the data

If you do not want to run the model pipeline of chemixhub but just want use our datasets and dataloader, here is an example of how this is possible with the MON dataset:

```
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from mixhub.data.dataset import MixtureTask
from mixhub.data.data import MONData
from mixhub.data.splits import SplitLoader
from mixhub.data.collate import custom_collate

BATCH_SIZE=1

mixture_task = MixtureTask(
    property="Motor octane number",
    dataset=MONData(),
    featurization="rdkit2d_normalized_features",
)

# Split Loader
split_loader = SplitLoader(split_type="kfold")
train_indices, _, _ = split_loader(
    property=mixture_task.property,
    cache_dir=mixture_task.dataset.data_dir,
    split_num=0,
)

# Data Loader
train_dataset = Subset(mixture_task, train_indices.tolist())

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=custom_collate,
    pin_memory=True,
)

```

