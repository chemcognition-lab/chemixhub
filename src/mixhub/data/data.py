import pandas as pd
import json
import os
import ast

COLUMN_PROPERTY = "property"
COLUMN_VALUE = "value"
COLUMN_UNIT = "unit"


class MixtureDataInfo:
    """
    Base class for Mixture Data Information
    """
    def __init__(
            self,
            name: str,
            description: str,
            id_column: list[str],
            fraction_column: list[str],
            context_columns: list[str],
            output_column: str,
            data_dir: str,
            compound_csv_name: str, 
            mixture_csv_name: str,
    ):
        self.name = name
        self.description = description
        self.metadata = {
            "columns": {
                "id_column": id_column,
                "fraction_column": fraction_column,
                "context_columns": context_columns,
                "output_column": output_column,
            }
        }

        # Load compound data
        compounds_path = os.path.join(data_dir, f"{compound_csv_name}.csv")
        if not os.path.exists(compounds_path):
            raise FileNotFoundError(f"The file {compounds_path} does not exist.")
    
        self.compounds = pd.read_csv(compounds_path, index_col="compound_id")
        
        # Load and filter data
        self.data_dir = data_dir

        data_path = os.path.join(data_dir, f"{mixture_csv_name}.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The file {data_path} does not exist.")
        
        self.data = pd.read_csv(data_path)

        for i in ["id_column", "fraction_column"]:
            cols = self.metadata["columns"][i]
            for col in cols:
                if col is not None:
                    self.data[col] = self.data[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        self.properties = self.data[COLUMN_PROPERTY].unique()

    def export_metadata(self, filepath="dataset_metadata.json"):
        """Exports metadata to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.metadata, f, indent=4)
    
    def summary(self):
        """Prints metadata summary."""
        print(f"Dataset: {self.name}")
        print(f"Description: {self.description}\n")
        print("Column Information:")
        for col, desc in self.metadata["columns"].items():
            print(f"- {col}: {desc}")


class DiffMixData(MixtureDataInfo):
    def __init__(
            self,
            name: str = "DiffMix",
            description: str = """
            The DiffMix dataset is a collection of three tasks centered around thermodynamic and transport
            properties predictions of electrolytes originally gathered by Zhu et al.
            It includes 631 data points of excess molar enthalpy,
            1069 data points of excess molar volume curated from literature 
            and 24,822 data points of ionic conductivity generated using Advanced Electrolyte Model
            for electrolyte solutions.
            """,
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["cmp_mole_fractions"],
            context_columns: list[str] = ["Temperature, K"],
            output_column: str = "value",
            data_dir: str = os.path.abspath("../datasets/diffmix/processed_data"),
            compound_csv_name: str = "compounds",
            mixture_csv_name: str = "processed_DiffMix",
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


class IlThermoData(MixtureDataInfo):
    def __init__(
            self,
            name: str = "IlThermo",
            description: str = """
            ILThermo is a web-based database that provides extensive information on over 50
            chemical and physical properties of pure ILs, as well as their binary and ternary mixtures
            with various solvents. For the scope of this paper, we selected two property prediction
            tasks-ionic conductivty and viscosity- from IlThermo.
            This dataset includes 40,904 data points of ionic conductivity and 75,992 viscosity
            data points curated from the literature.
            """,
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["cmp_mole_fractions"],
            context_columns: list[str] = ["Temperature, K"],
            output_column: str = "value",
            data_dir: str = os.path.abspath("../datasets/ionic-liquids/processed_data"),
            compound_csv_name: str = "compounds",
            mixture_csv_name: str = "processed_IlThermoData",
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


class PolymerElectrolyteData(MixtureDataInfo):
    def __init__(
            self,
            name: str = "PolymerElectrolyte",
            description: str = """
            Bradford et al. compiled this dataset from the literature comprising 11,350
            ionic conductivity measurements across more than 1,700 unique electrolyte formulations.
            Each formulation is uniquely defined by the polymer, salt, salt concentration, polymer
            molecular weight, and any additives present.
            """,
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["cmp_mole_fractions"],
            context_columns: list[str] = ["Temperature, K"],
            output_column: str = "value",
            data_dir: str = os.path.abspath("../datasets/polymer-electrolyte/processed_data"),
            compound_csv_name: str = "compounds",
            mixture_csv_name: str = "processed_PolymerElectrolyteData",
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


class MiscibleSolventData(MixtureDataInfo):
    def __init__(
            self,
            name: str = "MiscibileSolvent",
            description: str = """
            The Miscible Solvents dataset is a set of three tasks-density, heat of vaporization and
            enthalpy of mixing centered around miscible solvent properties, originally generated by
            Chem et al. using molecular dynamics (MD) simulations for 19,238 unique mixtures.
            """,
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["cmp_mole_fractions"],
            context_columns: list[str] = [],
            output_column: str = "value",
            data_dir: str = os.path.abspath("../datasets/miscible-solvent/processed_data"),
            compound_csv_name: str = "compounds",
            mixture_csv_name: str = "processed_MiscibleSolventData",
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

class LogVData(MixtureDataInfo):
    def __init__(
            self,
            name: str = "LogV",
            description: str = """
            A smaller version of the NIST dataset proposed by Bilodeau et al. selected by
            applying two key preprocessing steps:
            (1) removing data entries with SMILES strings containing multiple,
            non-covalently bonded fragments, and 
            (2) excluding entries where either molecule was predicted to
            be a gas or solid in its pure form.
            These steps reduced the dataset to 34,374 data points.
            """,
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["cmp_mole_fractions"],
            context_columns: list[str] = ["T"],
            output_column: str = "value",
            data_dir: str = os.path.abspath("../datasets/logV/processed_data"),
            compound_csv_name: str = "compounds",
            mixture_csv_name: str = "processed_logV",
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


class NISTLogVData(MixtureDataInfo):
    def __init__(
            self,
            name: str = "NIST LogV",
            description: str = """
            NIST Thermodynamics Research Center(TRC) SOURCE data archival system provides
            one of the most comprehensive datasets containing 239,201 dynamic viscosity datapoints
            of binary liquid mixtures.
            """,
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["cmp_mole_fractions"],
            context_columns: list[str] = ["T"],
            output_column: str = "value",
            data_dir: str = os.path.abspath("../datasets/nist-logV/processed_data"),
            compound_csv_name: str = "compounds",
            mixture_csv_name: str = "processed_NISTlogV",
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


class MONData(MixtureDataInfo):
    def __init__(
            self,
            name: str = "MON",
            description: str = """
            Kuzhagaliyeva et al. compiled a database containing 684 data points
            for 352 unique single hydrocarbons and mixtures, reporting experimentally
            measured motor octane numbers (MON) from various literature sources.
            """,
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["cmp_mole_fractions"],
            context_columns: list[str] = [],
            output_column: str = "value",
            data_dir: str = os.path.abspath("../datasets/MON/processed_data"),
            compound_csv_name: str = "compounds",
            mixture_csv_name: str = "processed_MON",
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


class OlfactorySimData(MixtureDataInfo):
    def __init__(
            self,
            name: str = "OlfactorySim",
            description: str = """
            The dataset was originally compiled from previous publications by Tom et al.
            Data for each of these publications was obtained from pyrfume.
            These mixtures are described by 865 pairwise mixture comparisons
            corresponding to labels from two types of experiments:
            1.explicit similarity 2. Triangle discrimination.
            """,
            id_column: list[str] = ["cmp_ids_1", "cmp_ids_2"],
            fraction_column: list[str] = [None, None],
            context_columns: list[str] = [],
            output_column: str = "value",
            data_dir: str = os.path.abspath("../datasets/olfactory-similarity/processed_data"),
            compound_csv_name: str = "compounds",
            mixture_csv_name: str = "processed_OlfactorySimilarity",
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


class DrugSolubilityData(MixtureDataInfo):
    def __init__(
            self,
            name: str = "DrugSolubility",
            description: str = """
            The drug solubility dataset originally curated by Bao et al. from literature
            includes 27,166 data points of drug solubility in mixture of solvents.
            """,
            id_column: list[str] = ["cmp_ids_solvent", "cmp_ids_drug"],
            fraction_column: list[str] = ["cmp_mole_fractions_solvent", "cmp_mole_fractions_drug"],
            context_columns: list[str] = ["Temperature, K"],
            output_column: str = "value",
            data_dir: str = os.path.abspath("../datasets/drug-solubility/processed_data"),
            compound_csv_name: str = "compounds",
            mixture_csv_name: str = "processed_DrugSolubilityData",
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


DATA_CATALOG = {
    "diffmix": DiffMixData,
    "ilthermo": IlThermoData,
    "polymer-electrolyte": PolymerElectrolyteData,
    "miscible-solvent": MiscibleSolventData,
    "logv": LogVData,
    "nist-logv": NISTLogVData,
    "mon": MONData,
    "olfactory": OlfactorySimData,
    "drug-solubility": DrugSolubilityData,
}

PHYSICS_SUPPORTED = {
    "ionic conductivity": ["arrhenius", "vft_cmu", "vft_mit"],
    "Electrical conductivity": ["arrhenius", "vft_cmu", "vft_mit"],
    "Viscosity": ["arrhenius"],
}
