# CheMixHub 🧪📊  
**A Benchmark Suite for Machine Learning on Chemical Mixtures**

Welcome to the official repository for **CheMixHub**, introduced in our paper:  
**"CheMixHub: Datasets and Benchmarks for Chemical Mixture Property Prediction"**

## 🚀 Overview

In this work, we introduce **CheMixHub**, the first comprehensive benchmark suite designed specifically for machine learning tasks involving **chemical mixtures**. The repository provides:

- Curated and standardized mixture datasets
- Robust data splitting strategies for generalization assessment
- Baseline machine learning models to benchmark and compare new approaches

Our goal is to support and accelerate research on learning-based modeling, prediction, and optimization of molecular mixtures in diverse chemical domains.

## 🎯 Key Contributions

- 🔬 **Dataset Curation**: Standardized 11 tasks from 7 existing datasets spanning diverse domains  

  The following datasets are included:

  - **Miscible Solvents**  
    Thermodynamic property dataset with density, enthalpy of mixing, and partial molar enthalpy
    [Source Paper](https://chemrxiv.org/engage/chemrxiv/article-details/677d54c86dde43c908a14a6c)

  - **ILThermo**  
    Transport properties (e.g., ionic conductivity, viscosity) for ionic liquid mixtures from the ILThermo database  
    [Source Paper](https://ilthermo.boulder.nist.gov/)

  - **NIST**  
    Viscosity measurements for organic mixtures extracted from the sourced from NIST ThermoData Engine, published in Zenodo.  
    [Source Paper](https://doi.org/10.1016/j.cej.2023.142454)

  - **Drug Solubility**  
    Solubility values of drugs in various mixtures of solvents.  
    [Source Paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00911-3)

  - **Solid Polymer Electrolyte Ionic Conductivity**  
    Ionic conductivity measurements for polymer–salt mixtures (SPEs)  
    [Source Paper](https://pubs.acs.org/doi/10.1021/acscentsci.2c01123)

  - **Olfactory Similarity**  
    Perceptual similarity scores for mixtures  
    [Source Paper](https://arxiv.org/abs/2501.16271)

  - **Motor Octane Number (MON)**  
    Measured octane numbers for pure hydrocarbons and multi-component fuels  
    [Source Paper](https://www.nature.com/articles/s42004-022-00722-3)
    
- 🧪 **New Tasks**: Added 2 large-scale tasks (116,896 data points) curated from the ILThermo database—larger than any other public mixture dataset.
  
- 🔄 **Generalization Splits**: Created 4 data-splitting strategies:
  - Random split  
  - Unseen chemical components  
  - Varying mixture size/composition  
  - Out-of-distribution (e.g., temperature-based)
  
- 📈 **Baseline Models**: Benchmarked representative ML models to establish initial performance levels


## 📁 Repository Structure

### `datasets/`  
This directory contains the curated datasets used in CheMixHub. Each dataset has its own folder and includes:

- `raw_data/`:  
  Contains the original data files obtained from the source databases or publications.

- `processed_data/`:  
  Includes a `data_processing.py` script used to generate standardized, model-ready data files. This folder contains:
  
  - `processed_data.csv`: A processed mixture-level dataset where each row represents a mixture with its component IDs, mole fractions, temperature (if available), and target property.
  
  - `compounds.csv`: Metadata about the pure components used in the dataset, including their molecular specifications and the IDs that link them to the mixtures in `processed_data.csv`.

  - `{dataset_name}_splits/`:  Contains 5-fold cross-validation data splits for training and validation, used during hyperparameter tuning and model evaluation.

### `scripts/`  
This directory contains utility and execution scripts for training, evaluation, feature precomputation and SMILES canonicalization


## Getting Started

To install and use CheMixHub, follow the steps below:

```bash
git clone https://github.com/<your-username>/chemixhub.git
cd chemixhub
pip install -e .
