import numpy as np

import train_predictor
from train_predictor import split_training_rows


def test_training_split_excludes_entire_example_group_and_keeps_groups_disjoint():
    groups = np.repeat([f"compound_{i:02d}" for i in range(30)], 3)
    train_idx, val_idx, excluded_idx = split_training_rows(groups, "compound_07", seed=42)

    np.testing.assert_array_equal(excluded_idx, [21, 22, 23])
    assert "compound_07" not in groups[train_idx]
    assert "compound_07" not in groups[val_idx]
    assert set(groups[train_idx]).isdisjoint(groups[val_idx])
    assert len(np.unique(groups[val_idx])) == 3
    np.testing.assert_array_equal(
        np.sort(np.concatenate([train_idx, val_idx, excluded_idx])), np.arange(len(groups))
    )


def test_source_grouping_preserves_wildcard_structures_and_ignores_fragment_order():
    canonicalize = getattr(train_predictor, "canonicalize_source_smiles", None)
    assert callable(canonicalize), "Source grouping requires a permissive RDKit canonicalizer"
    assert canonicalize("[Br-].C[1*]") == canonicalize("C[1*].[Br-]") == "[1*]C.[Br-]"
