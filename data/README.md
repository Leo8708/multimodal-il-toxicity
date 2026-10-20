# Processed model inputs

This code-only package contains no study data. All files below must come from the same processing run and preserve identical row order. A public access route for the exact study inputs has not yet been supplied.

## Files and array keys

| File | Contents |
| --- | --- |
| `processed_data.npz` | `X_num` (N x 2214 MACCS/ECFP bits); `y` (N log10 toxicity values); `Mw` (N molecular weights, g/mol); `cell_oh` (N x 9 endpoint indicators); `cell_labels`; `cell_categories`; `img_paths`; `train_idx`, `val_idx`, `test_idx`. |
| `used_dataset.xlsx` | Row-aligned structures, endpoints, original mM toxicity values, transformed targets and preparation-stage split labels. Required grouped-loader columns: `SMILES` and `细胞系`. |
| `rdkit_descriptors.npz` | `descriptors` (N x 20 raw values); `descriptors_norm` (historical normalization fitted over all rows); `names` (20 descriptor names). |
| `img_features_resnet18.npz` | `features` (N x 512 frozen ImageNet ResNet-18 global-average-pooling features). |
| `molecular_images/` | Generated 224 x 224 molecular structure images referenced by `img_paths`. Needed when rebuilding the feature cache. |

N is 11,016 for the study-specific grouped loaders: 3,672 source modeling records, each with its original SMILES and two randomized representations. There are 1,223 canonical complete-ion-pair groups and nine endpoint categories. Augmentation does not create new independent measurements.

`y = log10(toxicity_mM * 1000)`, representing log10 of the concentration expressed in micromolar units. The source toxicity column includes CC50, IC50 and EC50 measurements; the legacy variable name `logEC50` does not make them identical biological assays. `Other` is a pooled endpoint category.

## Excel field dictionary

| Field | Meaning |
| --- | --- |
| `IL编号` | Sequential prepared-row identifier. |
| `IL名称` | Name retained from the source archive. |
| `SMILES` | Complete ion-pair representation for that prepared row. |
| `阴离子SMILES` | Historical helper field derived as the last dot-separated fragment; not a validated general ion assignment. |
| `细胞系` | Modeled endpoint category, including pooled `Other`. |
| `测试方法` | Source assay/method label. |
| `毒性值(mM)` | Numeric source concentration after the script's string cleaning. |
| `毒性值(logEC50)` | Base-10 log of the concentration converted from mM to micromolar. |
| `分子量(Mw)` | Molecular weight, g/mol. |
| `数据集划分` | Preparation-stage train/validation/test label; grouped CV builds new splits and does not use this label. |

## Generation and cleaning

From the repository root, with the source archive in place:

```text
python prepare_dataset.py
python extract_descriptors.py
python data_utils.py
```

The historical preparation script merges CSV toxicity/Mw fields, keeps positive numeric toxicity and molecular weight with nonmissing SMILES, strips comparison/annotation characters from toxicity strings, converts mM to micromolar before taking log10, pools endpoints outside the ten most frequent into `Other`, and excludes `HCT-116` and `T-47D`. It generates two randomized SMILES per retained parseable record. These are historical choices requiring review before new applications; removal of inequality symbols does not preserve censoring information.

`extract_descriptors.py` calculates the 20 descriptors listed in `DESC_NAMES`. Grouped CV uses `descriptors` and fits its scalers within each training split. Historical record-level scripts use `descriptors_norm`.

The historical preparation-stage 80/10/10 split is separate from later cross-validation. Grouped loaders reconstruct groups from full canonical SMILES rather than using those stored split indices.

The image feature cache contains features in the prepared-row order without a dataset hash. Delete/regenerate that cache after changing the inputs. The image loader substitutes a white image if an image cannot be opened; verify image completeness when regenerating study inputs.

## Access and rights

MIT applies to the supplied code. Data access conditions and redistribution permissions must be documented separately; no license for the underlying data is asserted here. See [the source-input notes](../datasets/README.md).
