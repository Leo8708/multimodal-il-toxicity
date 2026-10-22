import json

import numpy as np
import pytest
import torch

from inference_utils import ToxicityFNN, canonical_smiles, make_features, load_predictor


SMILES = "[Cl-].C1=C[N+](=CN1C)CCCC"


def metadata():
    return {
        "format_version": 1,
        "architecture": {"input_dim": 2756, "hidden1": 512, "hidden2": 32},
        "cell_categories": ["A549", "CaCo-2", "HaCat", "HeLa", "IPC-81", "K562", "MCF-7", "MDA-MB-231", "Other"],
        "descriptor_names": list(range(20)),
    }


def test_rejects_invalid_or_incomplete_ion_pairs():
    for value in ["not-a-smiles", "CCCC", "C[N+]1=CN(C)C=C1", "[Na+].[Cl-].[Cl-]", "[*+].[Cl-]"]:
        with pytest.raises(ValueError):
            canonical_smiles(value)


def test_features_preserve_order_and_fixed_descriptor_scaling():
    mean = np.arange(20, dtype=np.float64)
    scale = np.full(20, 2.0)
    image = np.arange(512, dtype=np.float32)
    features = make_features(SMILES, "HeLa", 174.675, metadata(), mean, scale, image)
    from extract_descriptors import calc_one
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, rdMolDescriptors
    mol = Chem.MolFromSmiles(SMILES)
    expected_fp = np.concatenate([np.asarray(MACCSkeys.GenMACCSKeys(mol))[1:], np.asarray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))])
    assert features.shape == (1, 2756)
    np.testing.assert_array_equal(features[0, :2214], expected_fp)
    np.testing.assert_array_equal(features[0, 2214:2726], image)
    assert features[0, 2726] == pytest.approx(174.675)
    np.testing.assert_array_equal(features[0, 2727:2736], np.eye(9)[3])
    np.testing.assert_allclose(features[0, 2736:], (calc_one(SMILES) - mean) / scale, rtol=1e-6)
    np.testing.assert_array_equal(mean, np.arange(20))


@pytest.mark.parametrize("cell,mw", [("unknown", 174.675), ("HeLa", 0), ("HeLa", float("nan"))])
def test_rejects_unknown_cell_or_invalid_molecular_weight(cell, mw):
    with pytest.raises(ValueError):
        make_features(SMILES, cell, mw, metadata(), np.zeros(20), np.ones(20), np.zeros(512))


def test_loading_saved_model_preserves_prediction(tmp_path):
    torch.manual_seed(42)
    model = ToxicityFNN().eval()
    features = torch.randn(2, 2756)
    expected = model(features).detach()
    torch.save(model.state_dict(), tmp_path / "model.pt")
    np.savez(tmp_path / "preprocessing.npz", descriptor_mean=np.zeros(20), descriptor_scale=np.ones(20))
    (tmp_path / "metadata.json").write_text(json.dumps(metadata()), encoding="utf-8")
    loaded, meta, mean, scale = load_predictor(tmp_path, torch.device("cpu"))
    torch.testing.assert_close(loaded(features), expected)
    assert meta["architecture"]["hidden1"] == 512
    np.testing.assert_array_equal(scale, np.ones(20))
