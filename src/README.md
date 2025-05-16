### `src/mixhub/data/`  
This folder contains the core logic for handling mixture datasets, including loading, preprocessing, featurization, and dataset splitting.

- `data.py`  
  Entry point for dataset initialization and processing workflows. Coordinates the end-to-end data pipeline including featurization, batching, and metadata handling.

- `dataset.py`  
  Defines the `MixtureTask` dataset class used to wrap chemical mixture data into PyTorch/PyG-compatible format. Handles tensorization of mixtures and labels, including optional featurization.

- `featurization.py`  
  Contains functions for converting SMILES strings into different molecular representations, such as Morgan fingerprints, RDKit 2D descriptors, MolT5 embeddings, and molecular graphs.

- `graph_utils.py`  
  Provides tools for constructing graph representations of molecules from SMILES, including atom, bond, and global feature extraction.

- `splits.py`  
  Implements various dataset splitting strategies for evaluation and generalization testing, including k-fold, component-based, temperature-based, and leave-one-out splits.

- `utils.py`  
  Utility functions such as `pad_list()` for padding variable-length inputs and `indices_to_graphs()` for mapping tensor indices back to molecular graphs.


### `src/mixhub/model/`  
This module implements model architectures, components, and utilities for building and training neural networks for chemical mixture property prediction.

- `aggregation.py`  
  Implements permutation-invariant aggregation strategies such as mean, max, set2set, attention, and PNA. 

- `graph.py`  
  Defines graph-based neural network models including GNN layers and GraphNets architecture for molecular feature extraction using PyTorch Geometric.

- `linear.py`  
  Implements simple baseline models such as linear regression, SGD, and XGBoost, alongside a configurable fully connected neural network (`FullyConnectedNet`).

- `maths_ops.py`  
  Contains masked mathematical operations (mean, variance, min, max) used in aggregation layers to handle variable-length mixtures with padding.

- `mixture.py`  
  Defines the main `MixtureModel` class and its components, including DeepSet and Self-Attention encoders. Handles context and fraction incorporation using FiLM, concatenation, or multiplication.

- `model_builder.py`  
  Central configuration file for instantiating models. 

- `predict.py`  
  Provides evaluation logic for models.

- `predictor.py`  
  Implements different predictive heads, including a physics-informed module that maps model outputs to physical property predictions using equations (e.g., Arrhenius law).

- `train.py`  
  Main training loop.

- `utils.py`  
  Shared model utilities such as activation maps, early stopping, metric evaluation functions, and configuration printing.
