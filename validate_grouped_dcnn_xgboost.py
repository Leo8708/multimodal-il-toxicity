from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from validate_grouped_fnn import (
    aggregate_augmented_predictions,
    evaluate_predictions,
    fit_fold_transform,
    fit_target_transform,
    load_grouped_data,
    make_group_folds,
    transform_features,
)


ROOT = Path(__file__).resolve().parent
FP_DIM = 2214
IMG_DIM = 512
CONT_DIM = 21
ENDPOINT_DIM = 9


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Swish(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(value)


class DCNNMultiModal(nn.Module):


    def __init__(
        self,
        c1: int = 64,
        c2: int = 96,
        kernel_size: int = 5,
        h1: int = 192,
        h2: int = 96,
        dropout: float = 0.19594765196206126,
        img_hidden: int = 128,
        meta_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(c1),
            Swish(),
            nn.Dropout(dropout),
            nn.AvgPool1d(2),
            nn.Conv1d(c1, c2, 5, padding=2),
            nn.BatchNorm1d(c2),
            Swish(),
            nn.Dropout(dropout),
            nn.AvgPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.img_head = nn.Sequential(
            nn.Linear(IMG_DIM, img_hidden),
            nn.BatchNorm1d(img_hidden),
            Swish(),
            nn.Dropout(dropout),
        )
        self.meta_head = nn.Sequential(
            nn.Linear(CONT_DIM + ENDPOINT_DIM, meta_hidden),
            Swish(),
        )
        fused_dim = c2 + img_hidden + meta_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, h1),
            nn.BatchNorm1d(h1),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            Swish(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(h2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        fingerprint = features[:, :FP_DIM]
        image = features[:, FP_DIM : FP_DIM + IMG_DIM]
        metadata = features[:, FP_DIM + IMG_DIM :]
        fingerprint_latent = self.conv(fingerprint.unsqueeze(1))
        image_latent = self.img_head(image)
        metadata_latent = self.meta_head(metadata)
        fused = torch.cat(
            [fingerprint_latent, image_latent, metadata_latent], dim=1
        )
        return self.fusion(fused).squeeze(1)


DCNN_PARAMS: dict[str, Any] = {
    "c1": 64,
    "c2": 96,
    "kernel_size": 5,
    "h1": 192,
    "h2": 96,
    "dropout": 0.19594765196206126,
    "learning_rate": 0.0007192388010213356,
    "weight_decay": 0.00010355896477427082,
    "batch_size": 128,
    "betas": [0.99, 0.999],
    "max_epochs": 100,
    "patience": 20,
}

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 800,
    "max_depth": 6,
    "learning_rate": 0.023232346938983735,
    "reg_alpha": 0.001815962037748451,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def _predict_dcnn(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            batch = torch.from_numpy(features[start:stop]).to(device)
            predictions.append(model(batch).detach().cpu().numpy())
    return np.concatenate(predictions).astype(np.float32)


def _dcnn_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(features), torch.from_numpy(targets)
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def _fit_dcnn_early_stopping(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    valid_features: np.ndarray,
    valid_targets: np.ndarray,
    params: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[DCNNMultiModal, dict[str, Any]]:
    set_seed(seed)
    model = DCNNMultiModal(
        c1=int(params["c1"]),
        c2=int(params["c2"]),
        kernel_size=int(params["kernel_size"]),
        h1=int(params["h1"]),
        h2=int(params["h2"]),
        dropout=float(params["dropout"]),
    ).to(device)
    loader = _dcnn_loader(
        train_features,
        train_targets,
        int(params["batch_size"]),
        shuffle=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        betas=tuple(float(x) for x in params["betas"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(int(params["max_epochs"])):
        model.train()
        train_losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        valid_prediction = _predict_dcnn(model, valid_features, device)
        valid_mse = float(np.mean((valid_prediction - valid_targets) ** 2))
        scheduler.step(valid_mse)
        history.append(
            {
                "epoch": epoch + 1,
                "train_mse": float(np.mean(train_losses)),
                "valid_mse": valid_mse,
            }
        )
        if valid_mse < best_loss - 1e-8:
            best_loss = valid_mse
            best_epoch = epoch + 1
            no_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            no_improvement += 1
            if no_improvement >= int(params["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": int(best_epoch),
        "best_valid_mse": float(best_loss),
        "epochs_run": len(history),
        "history": history,
    }


def _fit_dcnn_fixed(
    features: np.ndarray,
    targets: np.ndarray,
    params: dict[str, Any],
    seed: int,
    device: torch.device,
    epochs: int,
) -> tuple[DCNNMultiModal, dict[str, Any]]:
    if int(epochs) < 1:
        raise ValueError("epochs must be positive")
    set_seed(seed)
    model = DCNNMultiModal(
        c1=int(params["c1"]),
        c2=int(params["c2"]),
        kernel_size=int(params["kernel_size"]),
        h1=int(params["h1"]),
        h2=int(params["h2"]),
        dropout=float(params["dropout"]),
    ).to(device)
    loader = _dcnn_loader(
        features,
        targets,
        int(params["batch_size"]),
        shuffle=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        betas=tuple(float(x) for x in params["betas"]),
        weight_decay=float(params["weight_decay"]),
    )
    history: list[float] = []
    for _ in range(int(epochs)):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    return model, {"epochs_run": int(epochs), "train_mse": history}


def run_dcnn(
    data: Any,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    device: torch.device,
    target_mode: str = "endpoint_zscore",
    tta: bool = True,
) -> dict[str, Any]:
    predictions = np.full(data.n_rows, np.nan, dtype=np.float32)
    fold_metrics: list[dict[str, Any]] = []
    configurations: list[dict[str, Any]] = []
    for outer_number, (outer_train, outer_test) in enumerate(folds, start=1):
        fold_seed = seed + outer_number * 100
        split = GroupShuffleSplit(
            n_splits=1, test_size=0.1, random_state=fold_seed
        )
        local_train, local_valid = next(
            split.split(outer_train, groups=data.groups[outer_train])
        )
        fit_idx = outer_train[local_train]
        valid_idx = outer_train[local_valid]
        validation_transform = fit_fold_transform(data, fit_idx, fingerprint_mode="raw")
        validation_target = fit_target_transform(
            data.y, data.endpoint_id, fit_idx, mode=target_mode
        )
        model, selection = _fit_dcnn_early_stopping(
            transform_features(data, fit_idx, validation_transform),
            validation_target.transform(data.y[fit_idx], data.endpoint_id[fit_idx]),
            transform_features(data, valid_idx, validation_transform),
            validation_target.transform(data.y[valid_idx], data.endpoint_id[valid_idx]),
            DCNN_PARAMS,
            fold_seed,
            device,
        )
        selected_epochs = max(1, int(selection["best_epoch"]))
        del model
        full_transform = fit_fold_transform(data, outer_train, fingerprint_mode="raw")
        full_target = fit_target_transform(
            data.y, data.endpoint_id, outer_train, mode=target_mode
        )
        final_model, refit = _fit_dcnn_fixed(
            transform_features(data, outer_train, full_transform),
            full_target.transform(data.y[outer_train], data.endpoint_id[outer_train]),
            DCNN_PARAMS,
            fold_seed + 1,
            device,
            selected_epochs,
        )
        transformed_prediction = _predict_dcnn(
            final_model,
            transform_features(data, outer_test, full_transform),
            device,
        )
        prediction = full_target.inverse(
            transformed_prediction, data.endpoint_id[outer_test]
        )
        if tta:
            prediction = aggregate_augmented_predictions(prediction, outer_test)
        predictions[outer_test] = prediction
        metric = evaluate_predictions(
            data.y[outer_test],
            prediction,
            data.endpoint_id[outer_test],
            data.endpoint_names,
        )
        metric["fold"] = outer_number
        metric["n_groups_test"] = int(np.unique(data.groups[outer_test]).size)
        fold_metrics.append(metric)
        configurations.append(
            {
                "fold": outer_number,
                "params": DCNN_PARAMS,
                "selection": selection,
                "refit": {"selected_epochs": selected_epochs, **refit},
            }
        )
        print(
            f"[DCNN] outer fold {outer_number}: R2={metric['r2']:.4f}, "
            f"epochs={selected_epochs}",
            flush=True,
        )
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics = evaluate_predictions(
        data.y, predictions, data.endpoint_id, data.endpoint_names
    )
    metrics["folds"] = fold_metrics
    metrics["mean_outer_fold_r2"] = float(np.mean([x["r2"] for x in fold_metrics]))
    metrics["std_outer_fold_r2"] = float(np.std([x["r2"] for x in fold_metrics], ddof=1))
    return {
        "kind": "dcnn",
        "metrics": metrics,
        "configs": configurations,
        "predictions": predictions,
    }


def _make_xgb(params: dict[str, Any], device: torch.device):
    from xgboost import XGBRegressor

    options: dict[str, Any] = {
        "objective": "reg:squarederror",
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "learning_rate": float(params["learning_rate"]),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "tree_method": "hist",
        "eval_metric": "rmse",
        "random_state": 42,
        "verbosity": 0,
    }
    if device.type == "cuda":
        options["device"] = "cuda"
        options["n_jobs"] = 1
    else:
        options["n_jobs"] = -1
    try:
        return XGBRegressor(**options)
    except TypeError:
        options.pop("device", None)
        return XGBRegressor(**options)


def run_xgb(
    data: Any,
    folds: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    target_mode: str = "endpoint_zscore",
    tta: bool = True,
) -> dict[str, Any]:
    predictions = np.full(data.n_rows, np.nan, dtype=np.float32)
    fold_metrics: list[dict[str, Any]] = []
    configurations: list[dict[str, Any]] = []
    for outer_number, (outer_train, outer_test) in enumerate(folds, start=1):
        transform = fit_fold_transform(data, outer_train, fingerprint_mode="raw")
        target_transform = fit_target_transform(
            data.y, data.endpoint_id, outer_train, mode=target_mode
        )
        train_features = transform_features(data, outer_train, transform)
        test_features = transform_features(data, outer_test, transform)
        train_target = target_transform.transform(
            data.y[outer_train], data.endpoint_id[outer_train]
        )
        model = _make_xgb(XGB_PARAMS, device)
        model.fit(train_features, train_target)
        prediction_transformed = np.asarray(model.predict(test_features), dtype=np.float32)
        prediction = target_transform.inverse(
            prediction_transformed, data.endpoint_id[outer_test]
        )
        if tta:
            prediction = aggregate_augmented_predictions(prediction, outer_test)
        predictions[outer_test] = prediction
        metric = evaluate_predictions(
            data.y[outer_test],
            prediction,
            data.endpoint_id[outer_test],
            data.endpoint_names,
        )
        metric["fold"] = outer_number
        metric["n_groups_test"] = int(np.unique(data.groups[outer_test]).size)
        fold_metrics.append(metric)
        configurations.append({"fold": outer_number, "params": XGB_PARAMS})
        print(f"[XGBoost] outer fold {outer_number}: R2={metric['r2']:.4f}", flush=True)
        del model
    metrics = evaluate_predictions(
        data.y, predictions, data.endpoint_id, data.endpoint_names
    )
    metrics["folds"] = fold_metrics
    metrics["mean_outer_fold_r2"] = float(np.mean([x["r2"] for x in fold_metrics]))
    metrics["std_outer_fold_r2"] = float(np.std([x["r2"] for x in fold_metrics], ddof=1))
    return {
        "kind": "xgboost",
        "metrics": metrics,
        "configs": configurations,
        "predictions": predictions,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_results(
    root: Path,
    data: Any,
    results: dict[str, dict[str, Any]],
    folds: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    seed: int,
    target_mode: str,
    tta: bool,
) -> None:
    output_dir = root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(results)
    predictions = np.vstack([results[name]["predictions"] for name in names]).astype(
        np.float32
    )
    outer_fold = np.full(data.n_rows, -1, dtype=np.int64)
    for number, (_, test_idx) in enumerate(folds, start=1):
        outer_fold[test_idx] = number
    np.savez_compressed(
        output_dir / "grouped_dcnn_xgboost_predictions.npz",
        y_true=data.y,
        predictions=predictions,
        model_names=np.asarray(names),
        endpoint_id=data.endpoint_id,
        endpoint_names=np.asarray(data.endpoint_names),
        canonical_group=data.groups,
        outer_fold=outer_fold,
    )
    payload = {
        "experiment": "canonical-IL-grouped five-fold OOF evaluation",
        "protocol": {
            "seed": seed,
            "rows": data.n_rows,
            "canonical_groups": int(np.unique(data.groups).size),
            "outer_folds": len(folds),
            "fingerprint_mode": "raw_binary",
            "continuous_scaling": "fold-local StandardScaler",
            "target_mode": target_mode,
            "tta": bool(tta),
            "hyperparameter_selection": "fixed configurations from existing multimodal implementations; no outer-test tuning",
        },
        "device": str(device),
        "models": {
            name: {
                "kind": result["kind"],
                "metrics": _json_safe(result["metrics"]),
                "configs": _json_safe(result["configs"]),
            }
            for name, result in results.items()
        },
    }
    with (output_dir / "grouped_dcnn_xgboost_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Canonical-IL-grouped cross-validation for multimodal DCNN and XGBoost models.')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--dcnn-only", action="store_true")
    parser.add_argument("--xgb-only", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dcnn_only and args.xgb_only:
        raise SystemExit("Choose at most one of --dcnn-only and --xgb-only")
    data = load_grouped_data(ROOT)
    folds = make_group_folds(data.groups, n_splits=5, seed=args.seed)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    tta = not args.no_tta
    print(
        f"Using device={device}; rows={data.n_rows}; "
        f"canonical_groups={np.unique(data.groups).size}; tta={tta}",
        flush=True,
    )
    started = time.time()
    results: dict[str, dict[str, Any]] = {}
    if not args.xgb_only:
        results["dcnn"] = run_dcnn(data, folds, args.seed, device, tta=tta)
    if not args.dcnn_only:
        results["xgboost"] = run_xgb(data, folds, device, tta=tta)
    save_results(
        ROOT,
        data,
        results,
        folds,
        device,
        args.seed,
        "endpoint_zscore",
        tta,
    )
    print(f"saved grouped DCNN/XGBoost results in {time.time() - started:.1f}s")
    for name, result in results.items():
        metric = result["metrics"]
        print(
            f"{name}: pooled R2={metric['r2']:.4f}, RMSE={metric['rmse']:.4f}, "
            f"MAE={metric['mae']:.4f}, Pearson r={metric['pearson_r']:.4f}, "
            f"macro_endpoint_R2={metric['macro_endpoint_r2']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
