# Release-preparation verification

Checked on 2026-10-20 in the local environment recorded in `environment_snapshot.json`.

- The release contains ten core Python files and two test files with descriptive lowercase snake_case names. Three fingerprint-only training scripts and their dedicated data-loader helper have been removed from the public package.
- Python comments and explanatory string statements, including module/function/class docstrings, have been removed. All 12 supplied Python files have zero COMMENT tokens and zero explanatory string statements. Configuration, dependency and ignore files also contain no comment lines.
- All 12 transformed Python files passed normalized AST comparison against the prior release files, allowing only documentation removal, module import renames, artifact-path renames, explicit CLI descriptions and removal of the unused single-modality loader. Training and statistical logic in the retained workflows was preserved; release files are no longer byte-identical to the research originals.
- All 12 supplied Python files passed syntax compilation.
- `python -m pytest -q tests`: **15 passed**. These are selected existing checks using synthetic inputs; no study data are required. One check performs two epochs on a small synthetic dataset.
- Both `validate_grouped_fnn.py --help` and `validate_grouped_dcnn_xgboost.py --help` completed successfully. Their descriptions are explicit runtime strings rather than module docstrings.
- The packaged grouped module's `run_audit` function was run against the separate local study input directory: **audit=PASS**, with 11,016 rows, 1,223 canonical groups, 2,214 fingerprint dimensions, 512 image dimensions, 21 continuous dimensions and nine endpoint categories. Inner/outer group isolation checks passed. These input files are not included in this release.
- Configuration paths are relative, local Markdown links resolve and imported modules are available in the recorded verification environment. Renamed DCNN/XGBoost training JSON outputs match their prediction-export readers.
- A basic absolute-local-path and common credential-pattern scan found no matches in the distributed files; this is not a comprehensive security audit.
- An independent read-only review checked the file selection, copy integrity, dependencies, commands, MIT text and stated reproducibility scope.

These checks do not constitute full model retraining, a clean-environment installation test, numerical reproduction of manuscript results or verification of public dataset/repository access. No GitHub repository was created or uploaded by the package-preparation process. File sizes and SHA-256 checksums are recorded in `FILE_MANIFEST.json`; that manifest excludes itself.
