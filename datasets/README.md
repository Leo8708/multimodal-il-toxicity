# Source input archive

Expected local input: `csv_datasets.zip`, referenced by `data.dataset_zip` in `config.yaml`.

The manuscript cites:

Arakelyan et al. *A comprehensive dataset on cytotoxicity of ionic liquids*. Scientific Data 11, 1379 (2024). <https://doi.org/10.1038/s41597-024-04190-3>.

This citation was recovered from the manuscript. The exact correspondence between the local CSV archive and a public source-data deposit, its version and its redistribution terms remains to be documented. The article DOI is not a verified direct download route for this archive. No data license or accessibility claim is inferred from the article citation.

## Expected format

The archive contains family-specific CSV files directly at the archive root. The preparation script excludes the root entries `cell lines.csv`, `methods.csv` and `media.csv`. It expects compatible source fields including:

- `Name`, `SMILES`, `Cell line` and `Method`;
- a molecular-weight field containing `Mw`, such as `Mw,g*mol-1`;
- a non-statistics toxicity field containing `CC50`, `IC50` or `EC50`, expressed in mM.

For example, one inspected source header is `CC50/IC50/EC50, mM`. The exact source schema and units must be checked before substituting a different archive. An arbitrary CSV with SMILES and targets will not meet the current preparation interface.

## Information to complete before a data-supported release

- Official source repository/deposit URL and assigned identifier, where available.
- Exact dataset version or retrieval date and mapping to `csv_datasets.zip`.
- Applicable data license or other redistribution/access terms.
- Public location of the exact study input files, or a documented route to reconstruct them from accessible source files.
- Dataset citation and processing notes matching the manuscript's Data Availability statement.

The `.gitignore` intentionally excludes raw archives and processed binary data. If study data are deposited separately, update these notes and the manuscript statement with their actual access route.
