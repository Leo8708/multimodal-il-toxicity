# Methods and reproducibility scope

## Representation

The study representation contains 2,214 fingerprint bits (166 MACCS and 2,048 ECFP), 512 frozen ImageNet ResNet-18 image features, 20 RDKit molecular descriptors, molecular weight and nine endpoint indicators, totaling 2,756 input features. The DCNN processes fingerprints through a convolutional branch before fusion with image and metadata branches. The FNN and tree workflows use concatenated features. The image backbone is frozen for feature extraction.

The nine endpoint categories comprise eight named cell lines and pooled `Other`. These implementations do not establish generalization to new endpoint categories or external experimental datasets.

## Historical record-level experiments

The historical modeling scripts are `train_dcnn.py`, `train_xgboost.py` and `train_fnn.py`; prediction exports use `export_fnn_predictions.py` and `export_dcnn_xgboost_predictions.py`. These retain the historical experiment logic with release naming and path changes. They are research workflow scripts and execute data loading/training at module import; run them as scripts rather than importing them for discovery.

Their record-level `KFold` splits can place repeated or augmented representations of the same structure in different folds. Historical FNN workflows select their best epoch using held-out fold R2. Some preprocessing is fitted over the complete dataset, including the stored `descriptors_norm`; historical XGBoost also standardizes concatenated features globally. Hyperparameter searches are not uniformly nested within a strictly untouched outer split. These choices limit interpretation as independent generalization estimates.

The implementation uses 20 descriptors, as defined by `DESC_NAMES`; the historical FNN training script has fixed hidden widths of 256/32. Reference-paper R2 constants are comparison values, not validation results for this dataset. Release files use descriptive lowercase `snake_case` filenames, updated imports/output paths, and no Python comments or docstrings; they are not byte-identical copies of the research originals. The study's implementations and adaptations do not establish exact reproduction of external authors' software.

The optional prediction-export scripts retrain models. `export_dcnn_xgboost_predictions.py` reads the saved DCNN/XGBoost parameter summaries; its XGBoost constructor is not identical to the full historical search constructor. No claim of identical predictions between those workflows is made. Legacy `overall_r2` export fields contain mean fold R2; pooled R2 must be computed from all corresponding predictions and targets.

## Later grouped-validation code

`validate_grouped_fnn.py` and `validate_grouped_dcnn_xgboost.py` reconstruct 1,223 canonical complete-ion-pair groups from the row-aligned Excel structures. Grouped outer folds keep the same structure's repeated measurements, endpoint observations and randomized representations together. Split-specific scaling is fitted on training rows.

These scripts include later experimental settings: nested FNN selection/ensembles, optional fingerprint transformations, endpoint-conditioned heads, endpoint-specific target normalization and augmented-triplet prediction averaging. Averaging assumes the original-plus-two-variants row layout created by the historical preparation script. Do not reorder or deduplicate prepared rows before using that option.

The fixed DCNN/XGBoost settings originated in prior multimodal experiments; this is not a claim of a fresh, fully nested hyperparameter search for those models. Group separation does not establish independent provenance for every prior modeling decision.

The original grouped-training implementation used for the manuscript's historical grouped table was not present in the inspected workspace. Current saved grouped outputs reflect different later configurations. The later grouped scripts are supplied for transparency and further evaluation; they must not be presented as an exact executable reconstruction of that historical table.

## What this package verifies

- Existing data-independent tests cover neural outputs, selected training helpers, target transforms, metric definitions, epoch-selection aggregation, calibration and blending logic.
- `--audit-only` checks study input alignment, expected dimensions and group-separated inner/outer partitions when the four study input files are present.
- Syntax checks do not retrain the models.

No full retraining, numerical equivalence to manuscript results, complete figure reconstruction or independent reproduction in a fresh environment is claimed by this release preparation. Exact study data, historical run settings, environment and verified result mappings would be needed to extend those claims. Source datasets and pretrained weights are not included.
