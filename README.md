# Multimodal Ionic Liquid Toxicity Prediction

Foundational research code associated with **A Multimodal Learning Framework for Predicting Ionic Liquid Toxicity Across Cell Lines**.

The package contains data preparation, molecular feature extraction, multimodal FNN, DCNN and XGBoost implementations, a random-forest comparison, and canonical-ionic-liquid-grouped cross-validation tools. It is a code-only release: source datasets, processed data, pretrained weights and manuscript files are not bundled.

**Reproducibility scope:** historical record-level experiments and later grouped-validation implementations are provided as separate workflows. The current grouped scripts are later implementations with different settings from the manuscript's historical grouped analysis. This package does **not** reproduce every manuscript table or figure exactly. See [methods and limitations](docs/METHODS_AND_LIMITATIONS.md).

## Repository contents

| File | Purpose |
| --- | --- |
| `prepare_dataset.py` | Clean the source CSV archive, augment SMILES, compute fingerprints, draw structures and write aligned model inputs. |
| `config.yaml` | Portable input/output paths for data preparation. |
| `data_utils.py` | Load aligned data and compute/cache frozen ImageNet ResNet-18 image features. |
| `extract_descriptors.py` | Compute the 20 molecular descriptors defined by `DESC_NAMES`. |
| `train_dcnn.py` | Historical multimodal DCNN, record-level 10-fold CV and hyperparameter search. |
| `train_xgboost.py` | Historical multimodal XGBoost and random-forest comparison, record-level 5-fold CV and search. |
| `train_fnn.py` | Historical multimodal FNN with fixed hidden widths 256/32 and record-level 5-fold CV. |
| `export_fnn_predictions.py` | Retrain historical FNN and export prediction arrays plus a separate Optuna search history. |
| `export_dcnn_xgboost_predictions.py` | Retrain historical DCNN/XGBoost from saved parameter JSON files and export prediction arrays. |
| `validate_grouped_fnn.py` | Later FNN/grouped validation, nested tuning, ensemble options and data/fold audit. |
| `validate_grouped_dcnn_xgboost.py` | Later DCNN/XGBoost evaluation using the same canonical-IL outer groups. |
| `tests/test_grouped_fnn.py`, `tests/test_grouped_dcnn_xgboost.py` | Data-independent checks selected from the development tests. |
| `data/README.md`, `datasets/README.md` | Input formats, data provenance and missing access information. |
| `environment_snapshot.json` | Local environment recorded during package verification. |
| `FILE_MANIFEST.json` | Package file sizes, SHA-256 hashes and source/copy notes. |

The release contains ten core Python files and two test files. Filenames use lowercase `snake_case`: `prepare_`, `extract_`, `train_`, `validate_` and `export_` identify each script's role, and `data_utils.py` is the shared data module. Source-study input filenames are retained for compatibility.

Release scripts have been renamed, imports and output paths have been updated, and Python comments and docstrings have been removed. They are adapted release files rather than byte-identical copies of the original research scripts. These are the study's implementations and adaptations, not verified copies of external authors' software. The original research directory is unchanged.

## Installation

Use Python 3.11 in an isolated environment. For example, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

The requirements specify minimum versions, including **scikit-learn >= 1.6** for shuffled `GroupKFold`. Tested local package versions are recorded in `environment_snapshot.json`; this is an environment snapshot, not a lockfile or evidence of the environment used for every manuscript experiment. Install a compatible PyTorch/torchvision pair for your CPU or CUDA environment using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

A CPU can execute the code. A CUDA-capable GPU is useful for neural training; full hyperparameter searches are computationally intensive. No full-training runtime benchmark is supplied. The first image-feature extraction downloads the torchvision ResNet-18 ImageNet weights unless they are already cached.

## Check the code without the study data

Run from the repository root:

```powershell
python -m pytest -q tests
python validate_grouped_fnn.py --help
python validate_grouped_dcnn_xgboost.py --help
```

The tests exercise model shapes, preprocessing logic and metric/selection helpers using synthetic inputs. They do not establish reproduction of the paper's results. Dataset-dependent development tests are represented here by the separate `--audit-only` workflow described below.

## Prepare study inputs

1. Obtain the study's source CSV archive and document its public access route and redistribution terms. See [datasets/README.md](datasets/README.md). **A verified download route is not yet supplied with this package.**
2. Place the archive at `datasets/csv_datasets.zip`, or change `data.dataset_zip` in `config.yaml`.
3. Run the following from the repository root:

```powershell
python prepare_dataset.py
python extract_descriptors.py
python data_utils.py
```

This creates `data/processed_data.npz`, `data/used_dataset.xlsx`, `data/rdkit_descriptors.npz`, `data/img_features_resnet18.npz` and `data/molecular_images/`. All arrays must retain the same row ordering. The cached image feature file must be regenerated when the underlying dataset or images change.

The study's processed inputs contain 11,016 model rows from 3,672 source modeling records, with one original and two randomized-SMILES versions per record. These are augmented representations, not 11,016 independent experimental measurements. The grouped loaders enforce 1,223 canonical complete-ion-pair groups and nine endpoint categories (eight named cell lines and pooled `Other`). They are study-specific loaders, not general-purpose dataset interfaces.

The preprocessing script preserves the historical cleaning and filtering choices; review them before using a different dataset. See [data/README.md](data/README.md) for array keys, units and schema.

## Historical record-level workflow

These commands start training immediately and may overwrite earlier generated results:

```powershell
python train_dcnn.py
python train_xgboost.py
python train_fnn.py
```

They write summaries and hyperparameters to `results/dcnn_metrics.json`, `results/xgboost_metrics.json` and `results/fnn_metrics.json`. The second command also writes `results/random_forest_metrics.json`.

Optional historical prediction exports, after the DCNN and XGBoost summaries exist:

```powershell
python export_fnn_predictions.py
python export_dcnn_xgboost_predictions.py
```

Exports are written to `results/predictions/fnn_oof_predictions.npz`, `results/predictions/dcnn_oof_predictions.npz` and `results/predictions/xgboost_oof_predictions.npz`. The first command also runs a 40-trial search and writes `results/predictions/fnn_optimization_history.json`; it is not just a plotting command. These scripts retrain models, so their outputs need not match saved historical runs. In the export files, the legacy key `overall_r2` stores the **mean of fold R2 values**, not pooled prediction R2.

Record-level folds can share canonical structures and augmented variants. Historical FNN training selects an epoch using the held-out fold's R2, and some historical preprocessing is fitted across all rows. These results describe the historical interpolation workflow and should not be interpreted as an independent estimate for unseen structures.

## Later canonical-IL-grouped workflow

With all four processed input files present, first check row alignment, expected dimensions and group isolation:

```powershell
python validate_grouped_fnn.py --audit-only
```

The audit prints `rows=11016`, `canonical_groups=1223` and `audit=PASS` when its checks pass. It also checks outer- and inner-fold group isolation.

Example runs, each with its own analysis settings:

```powershell
python validate_grouped_fnn.py --baseline-only --device cpu
python validate_grouped_fnn.py --full --trials 8 --device cuda
python validate_grouped_fnn.py --ensemble --device cuda
python validate_grouped_dcnn_xgboost.py --device cuda
```

Select `cpu` when CUDA is unavailable. The FNN script writes `results/grouped_fnn_metrics.json` and `results/grouped_fnn_predictions.npz`. The DCNN/XGBoost script writes `results/grouped_dcnn_xgboost_metrics.json` and `results/grouped_dcnn_xgboost_predictions.npz`. Different modes overwrite their corresponding filenames; preserve separate run directories/copies when comparing them.

Outer folds group all observations of the same canonical complete ion pair. Continuous-feature scaling is fitted within training splits; grouped code uses raw descriptor values rather than the historical globally normalized descriptor cache. DCNN/XGBoost defaults include endpoint-specific target normalization and averaging of augmented-triplet predictions; `--no-tta` disables that averaging. The FNN `--ensemble` mode also has distinct preprocessing and selection settings. These modes are **not** interchangeable with the historical grouped runs described in the manuscript.

## Data, citation and license

Source-data reference recorded in the manuscript: Arakelyan et al., *A comprehensive dataset on cytotoxicity of ionic liquids*, *Scientific Data* **11**, 1379 (2024). [https://doi.org/10.1038/s41597-024-04190-3](https://doi.org/10.1038/s41597-024-04190-3). The article DOI identifies the source publication; it is not a verified download link for this package's exact CSV archive.

Please cite the associated manuscript and the source-data publication when using the respective work. No manuscript DOI or software archive DOI has been assigned in this package.

The supplied study code is released under the [MIT License](LICENSE), copyright (c) 2026 Yaowei Li. Third-party datasets, external software and downloaded pretrained weights retain their own terms and are not relicensed by this repository.

After the repository is published, use GitHub Issues for code questions. Code availability wording and version/archive guidance are in [docs/CODE_AVAILABILITY.md](docs/CODE_AVAILABILITY.md).
