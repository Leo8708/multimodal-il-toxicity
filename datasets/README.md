# Raw datasets

Place `csv_datasets.zip` in this directory. Its path is set by `data.dataset_zip` in `config.yaml`.

The archive contains CSV files at its root with these fields:

| Field | Contents |
| --- | --- |
| `Name` | Compound name. |
| `SMILES` | Molecular structure. |
| `Cell line` | Cell-line label. |
| `Method` | Assay method. |
| `Mw,g*mol-1` | Molecular weight in g/mol. |
| `CC50/IC50/EC50, mM` | Toxicity concentration in mM. |

The preprocessing script skips `cell lines.csv`, `methods.csv`, and `media.csv`. Run `python prepare_dataset.py` from the repository root to generate processed files in `data/`.
