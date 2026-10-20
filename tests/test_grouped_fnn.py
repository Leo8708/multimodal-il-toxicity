from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import r2_score

from validate_grouped_fnn import (
    EndpointConditionedFNN,
    BlockFNN,
    ModernFNN,
    fit_fold_scaler,
    fit_fold_transform,
    fit_neural_model_fixed_epochs,
    fit_target_transform,
    load_grouped_data,
    make_group_folds,
    GroupedData,
    pooled_r2_score,
    fit_linear_calibration,
    apply_linear_calibration,
    aggregate_augmented_predictions,
    fit_blend_weight,
    fit_endpoint_blend_weights,
    mean_fold_r2_score,
    select_endpoint_blend_shrinkage,
    select_refit_epochs,
    transform_features,
)


ROOT = Path(__file__).resolve().parents[1]


def test_endpoint_conditioned_head_returns_one_prediction_per_row():
    model = EndpointConditionedFNN(
        input_dim=16, hidden1=12, hidden2=6, n_endpoints=9, dropout=0.1
    )
    x = torch.randn(11, 16)
    endpoint_id = torch.arange(11) % 9
    pred = model(x, endpoint_id)
    assert pred.shape == (11,)
    assert torch.isfinite(pred).all()


def test_modern_fnn_returns_one_prediction_per_row():
    model = ModernFNN(input_dim=16, hidden1=12, hidden2=6, dropout=0.1)
    x = torch.randn(11, 16)
    pred = model(x)
    assert pred.shape == (11,)
    assert torch.isfinite(pred).all()


def test_block_fnn_returns_one_prediction_per_row():
    model = BlockFNN(hidden1=24, hidden2=12, n_endpoints=9, endpoint_heads=True)
    x = torch.randn(11, 2756)
    endpoint_id = torch.arange(11) % 9
    pred = model(x, endpoint_id)
    assert pred.shape == (11,)
    assert torch.isfinite(pred).all()


def test_fixed_epoch_fit_uses_all_training_rows_without_validation():
    rng = np.random.default_rng(7)
    data = GroupedData(
        x_fp=rng.integers(0, 2, size=(12, 4), dtype=np.int8).astype(np.float32),
        x_img=rng.normal(size=(12, 3)).astype(np.float32),
        x_cont=rng.normal(size=(12, 2)).astype(np.float32),
        endpoint_id=np.arange(12, dtype=np.int64) % 2,
        y=rng.normal(size=12).astype(np.float32),
        groups=np.arange(12, dtype=np.int64),
        endpoint_names=("e0", "e1"),
    )
    transform = fit_fold_transform(data, np.arange(12))
    model, info = fit_neural_model_fixed_epochs(
        "fnn",
        data,
        np.arange(12),
        transform,
        {"hidden1": 8, "hidden2": 4, "learning_rate": 1e-3, "loss": "mse", "batch_size": 4},
        seed=7,
        device=torch.device("cpu"),
        epochs=2,
        group_balance=False,
    )
    assert info["epochs_run"] == 2
    predictions = model(torch.from_numpy(transform_features(data, np.arange(12), transform)))
    assert predictions.shape == (12,)
    assert torch.isfinite(predictions).all()


def test_pooled_r2_is_computed_from_all_inner_rows():
    y_parts = [np.array([0.0, 1.0, 2.0]), np.array([10.0, 11.0])]
    p_parts = [np.array([0.0, 1.0, 1.0]), np.array([9.0, 13.0])]
    pooled = pooled_r2_score(y_parts, p_parts)
    mean_fold = np.mean([r2_score(y, p) for y, p in zip(y_parts, p_parts)])
    assert pooled != pytest.approx(mean_fold)
    assert pooled == pytest.approx(
        1.0 - np.mean(np.concatenate([(y - p) ** 2 for y, p in zip(y_parts, p_parts)]))
        / np.var(np.concatenate(y_parts))
    )


def test_mean_fold_r2_is_arithmetic_and_distinct_from_pooled():
    fold_scores = [0.4, 0.8, 0.9]
    assert mean_fold_r2_score(fold_scores) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        mean_fold_r2_score([])


def test_endpoint_target_transform_uses_training_rows_only():
    y = np.array([0.0, 2.0, 10.0, 14.0], dtype=np.float32)
    endpoint_id = np.array([0, 0, 1, 1], dtype=np.int64)
    transform = fit_target_transform(y, endpoint_id, np.array([0, 1, 2]))
    assert transform.means[0] == pytest.approx(1.0)
    assert transform.means[1] == pytest.approx(10.0)
    assert transform.means[1] != pytest.approx(12.0)
    normalized = transform.transform(y, endpoint_id)
    restored = transform.inverse(normalized, endpoint_id)
    assert np.allclose(restored, y)


def test_refit_epoch_rule_uses_inner_checkpoints_only():
    selection = {"folds": [{"fit": {"best_epoch": 10}}, {"fit": {"best_epoch": 20}}, {"fit": {"best_epoch": 40}}]}
    assert select_refit_epochs(selection, 180, "median") == 20
    assert select_refit_epochs(selection, 180, "mean") == 23
    assert select_refit_epochs({"folds": []}, 180, "median") == 180


def test_linear_calibration_is_fit_from_inner_predictions():
    y = np.array([1.0, 3.0, 5.0, 7.0])
    prediction = np.array([0.5, 1.5, 2.5, 3.5])
    calibration = fit_linear_calibration(y, prediction)
    assert calibration["slope"] == pytest.approx(2.0)
    assert calibration["intercept"] == pytest.approx(0.0)
    assert np.allclose(apply_linear_calibration(prediction, calibration), y)


def test_augmented_triplet_predictions_are_averaged_without_cross_record_mixing():

    prediction = np.arange(1.0, 10.0)
    indices = np.arange(9)
    averaged = aggregate_augmented_predictions(prediction, indices, n_original=3)
    assert np.allclose(
        averaged,
        [10 / 3, 5.0, 20 / 3, 10 / 3, 10 / 3, 5.0, 5.0, 20 / 3, 20 / 3],
    )


def test_blend_weight_is_bounded_and_uses_training_predictions():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    first = np.array([0.2, 0.8, 1.8, 2.8])
    second = np.array([1.0, 1.0, 1.0, 1.0])
    weight = fit_blend_weight(y, first, second)
    assert 0.0 <= weight <= 1.0
    assert weight > 0.5


def test_endpoint_blend_weights_are_fit_separately():
    y = np.array([0.0, 1.0, 10.0, 11.0])
    first = np.array([0.0, 1.0, 8.0, 8.0])
    second = np.array([1.0, 1.0, 10.0, 11.0])
    endpoint_id = np.array([0, 0, 1, 1])
    weights = fit_endpoint_blend_weights(y, first, second, endpoint_id, 2)
    assert weights[0] > weights[1]
    assert all(0.0 <= value <= 1.0 for value in weights)


def test_blend_shrinkage_is_selected_from_inner_fold_predictions():
    y = np.array([0.0, 1.0, 10.0, 11.0, 0.0, 1.0, 10.0, 11.0])
    first = np.array([0.0, 1.0, 8.0, 8.0, 0.0, 1.0, 8.0, 8.0])
    second = np.array([1.0, 1.0, 10.0, 11.0, 1.0, 1.0, 10.0, 11.0])
    endpoint_id = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    fold_id = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    shrinkage, weights, metadata = select_endpoint_blend_shrinkage(
        y, first, second, endpoint_id, fold_id, n_endpoints=2
    )
    assert 0.0 <= shrinkage <= 1.0
    assert np.all((weights >= 0.0) & (weights <= 1.0))
    assert len(metadata["trials"]) >= 3
