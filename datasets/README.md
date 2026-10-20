# Raw datasets

Source: [A comprehensive dataset on cytotoxicity of ionic liquids — figshare](https://doi.org/10.6084/m9.figshare.27979307).

1. Download [cyto-datasets_2025-10-15.zip](https://ndownloader.figshare.com/files/58815787) from figshare.
2. Extract `csv_datasets.zip` from the downloaded archive and place it in `datasets/`.
3. Run `python prepare_dataset.py` from the repository root to generate processed files in `data/`.

The input path is set by `data.dataset_zip` in `config.yaml`.

The archive contains CSV files at its root with these fields:

| Field | Contents |
| --- | --- |
| `Name` | Compound name. |
| `SMILES` | Molecular structure. |
| `Cell line` | Cell-line label. |
| `Method` | Assay method. |
| `Mw,g*mol-1` | Molecular weight in g/mol. |
| `CC50/IC50/EC50, mM` | Toxicity concentration in mM. |

The preprocessing script skips `cell lines.csv`, `methods.csv`, and `media.csv`.
