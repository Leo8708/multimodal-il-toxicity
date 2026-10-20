from pathlib import Path

import numpy as np
import torch

from validate_grouped_dcnn_xgboost import DCNNMultiModal, XGB_PARAMS, _make_xgb
from validate_grouped_fnn import fit_fold_transform, load_grouped_data, transform_features


ROOT = Path(__file__).resolve().parents[1]


def test_dcnn_forward_returns_one_prediction_per_row():
    model = DCNNMultiModal(c1=8, c2=12, h1=16, h2=8, dropout=0.0)
    features = torch.randn(4, 2756)
    prediction = model(features)
    assert prediction.shape == (4,)
    assert torch.isfinite(prediction).all()


def test_xgboost_factory_uses_fixed_multimodal_configuration():
    model = _make_xgb(XGB_PARAMS, torch.device("cpu"))
    params = model.get_params()
    assert params["n_estimators"] == 800
    assert params["max_depth"] == 6
    assert params["tree_method"] == "hist"
