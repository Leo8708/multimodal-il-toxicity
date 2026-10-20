# Multimodal Ionic Liquid Toxicity Prediction

Research code for *A Multimodal Learning Framework for Predicting Ionic Liquid Toxicity Across Cell Lines*.

This repository includes data preprocessing, molecular feature extraction, multimodal FNN/DCNN/XGBoost implementations, random-forest baseline, and grouped cross-validation tools. This is a code-only release; raw datasets, processed data, pretrained weights and manuscript files are not included.

## Repository contents

表格

| File | Purpose |
| --- | --- |
| `prepare_dataset.py` | Data cleaning, SMILES augmentation, fingerprint calculation and structure rendering |
| `config.yaml` | Input/output path configuration for data preprocessing |
| `data_utils.py` | Data loading and ResNet-18 image feature extraction/caching |
| `extract_descriptors.py` | Molecular descriptor calculation |
| `train_dcnn.py` | Original multimodal DCNN with 10-fold CV and hyperparameter search |
| `train_xgboost.py` | Original multimodal XGBoost and random-forest comparison with 5-fold CV |
| `train_fnn.py` | Original multimodal FNN with 5-fold CV |
| `export_*.py` | Model retraining and prediction export scripts |
| `validate_grouped_*.py` | Grouped cross-validation by canonical ionic liquid pairs |
| `tests/` | Basic functionality tests with synthetic inputs |

## Installation

Use Python 3.11 in a virtual environment:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Install PyTorch/torchvision matching your CPU/CUDA environment. A CUDA GPU is recommended for full hyperparameter searches.

## Quick start

### Run tests

Verify basic functionality without study data:

```
python -m pytest -q tests
```

### Prepare input data

1. Obtain the source CSV dataset and place it at `datasets/csv_datasets.zip` (or update paths in `config.yaml`).
2. Run preprocessing:

```
python prepare_dataset.py
python extract_descriptors.py
python data_utils.py
```

### Run training workflows

- Original record-level experiments:
```
python train_dcnn.py
python train_xgboost.py
python train_fnn.py
```
- Grouped validation workflows:
```
python validate_grouped_fnn.py
python validate_grouped_dcnn_xgboost.py
```

## Data and citation

Source dataset: Arakelyan et al., *A comprehensive dataset on cytotoxicity of ionic liquids*, *Scientific Data* 11, 1379 (2024).
DOI: [10.1038/s41597-024-04190-3](https://doi.org/10.1038/s41597-024-04190-3)

Please cite the associated manuscript and source dataset when using this work.

## License

Released under the MIT License. See [LICENSE](LICENSE) for full terms.
