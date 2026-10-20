from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = 42
N_ENDPOINTS = 9


@dataclass
class GroupedData:
    x_fp: np.ndarray
    x_img: np.ndarray
    x_cont: np.ndarray
    endpoint_id: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    endpoint_names: tuple[str, ...]

    @property
    def n_rows(self) -> int:
        return int(self.y.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.x_fp.shape[1] + self.x_img.shape[1] + self.x_cont.shape[1] + len(self.endpoint_names))


@dataclass
class FoldTransform:
    image_scaler: StandardScaler
    continuous_scaler: StandardScaler
    fingerprint_scaler: StandardScaler | None = None
    fingerprint_clip: float | None = None


@dataclass
class TargetTransform:


    mode: str
    means: np.ndarray
    scales: np.ndarray

    def transform(self, values: np.ndarray, endpoint_id: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        endpoint_id = np.asarray(endpoint_id, dtype=np.int64)
        if self.mode == "raw":
            return values.copy()
        return ((values - self.means[endpoint_id]) / self.scales[endpoint_id]).astype(np.float32)

    def inverse(self, values: np.ndarray, endpoint_id: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        endpoint_id = np.asarray(endpoint_id, dtype=np.int64)
        if self.mode == "raw":
            return values.copy()
        return (values * self.scales[endpoint_id] + self.means[endpoint_id]).astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _canonical_smiles(value: Any) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(value))
    if mol is None:
        raise ValueError(f"Unable to parse SMILES while reconstructing groups: {value!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def load_grouped_data(root: Path = ROOT) -> GroupedData:


    root = Path(root)
    npz_path = root / "data" / "processed_data.npz"
    image_path = root / "data" / "img_features_resnet18.npz"
    descriptor_path = root / "data" / "rdkit_descriptors.npz"
    excel_path = root / "data" / "used_dataset.xlsx"

    processed = np.load(npz_path, allow_pickle=True)
    x_fp = np.asarray(processed["X_num"], dtype=np.float32)
    y = np.asarray(processed["y"], dtype=np.float32)
    mw = np.asarray(processed["Mw"], dtype=np.float32).reshape(-1, 1)
    cell_labels = np.asarray(processed["cell_labels"], dtype=str)
    endpoint_names = tuple(str(x) for x in processed["cell_categories"].tolist())
    endpoint_lookup = {name: i for i, name in enumerate(endpoint_names)}
    if len(endpoint_lookup) != len(endpoint_names):
        raise ValueError("Endpoint categories are not unique in processed_data.npz")
    try:
        endpoint_id = np.asarray([endpoint_lookup[x] for x in cell_labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Processed endpoint label is absent from cell_categories: {exc}") from exc

    image_features = np.load(image_path, allow_pickle=True)["features"]
    x_img = np.asarray(image_features, dtype=np.float32)
    descriptors = np.load(descriptor_path, allow_pickle=True)["descriptors"]
    x_cont = np.concatenate([mw, np.asarray(descriptors, dtype=np.float32)], axis=1)

    excel = pd.read_excel(excel_path)
    required = {"SMILES", "细胞系"}
    missing = required.difference(excel.columns)
    if missing:
        raise ValueError(f"used_dataset.xlsx is missing columns: {sorted(missing)}")
    if len(excel) != len(y):
        raise ValueError(f"Row mismatch: Excel has {len(excel)}, processed_data has {len(y)}")
    excel_cells = excel["细胞系"].astype(str).to_numpy()
    if not np.array_equal(excel_cells, cell_labels):
        mismatch = int(np.flatnonzero(excel_cells != cell_labels)[0])
        raise ValueError(
            f"Row alignment mismatch at row {mismatch}: Excel={excel_cells[mismatch]!r}, "
            f"processed={cell_labels[mismatch]!r}"
        )
    canonical = np.asarray([_canonical_smiles(value) for value in excel["SMILES"]], dtype=str)
    groups, unique_groups = pd.factorize(canonical, sort=True)
    groups = groups.astype(np.int64)

    expected = {
        "rows": 11016,
        "fingerprint_dim": 2214,
        "image_dim": 512,
        "continuous_dim": 21,
        "endpoint_count": 9,
        "group_count": 1223,
    }
    actual = {
        "rows": len(y),
        "fingerprint_dim": x_fp.shape[1],
        "image_dim": x_img.shape[1],
        "continuous_dim": x_cont.shape[1],
        "endpoint_count": len(endpoint_names),
        "group_count": len(unique_groups),
    }
    if actual != expected:
        raise ValueError(f"Grouped data contract mismatch: expected {expected}, got {actual}")
    if not (x_fp.shape[0] == x_img.shape[0] == x_cont.shape[0] == len(y)):
        raise ValueError("Feature matrices and target have different row counts")

    return GroupedData(
        x_fp=x_fp,
        x_img=x_img,
        x_cont=x_cont,
        endpoint_id=endpoint_id,
        y=y,
        groups=groups,
        endpoint_names=endpoint_names,
    )


def make_group_folds(groups: np.ndarray, n_splits: int = 5, seed: int = DEFAULT_SEED):


    groups = np.asarray(groups)
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros((len(groups), 1), dtype=np.float32)
    folds = [(tr.astype(np.int64), te.astype(np.int64)) for tr, te in splitter.split(dummy, groups=groups)]
    seen = np.concatenate([te for _, te in folds])
    if not np.array_equal(np.sort(seen), np.arange(len(groups))):
        raise RuntimeError("GroupKFold did not cover every row exactly once")
    for tr, te in folds:
        if set(groups[tr]).intersection(set(groups[te])):
            raise RuntimeError("Group overlap detected between train and test fold")
    return folds


def make_inner_folds(
    train_idx: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 3,
    seed: int = DEFAULT_SEED,
):


    train_idx = np.asarray(train_idx, dtype=np.int64)
    local_groups = np.asarray(groups)[train_idx]
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    local_dummy = np.zeros((len(train_idx), 1), dtype=np.float32)
    folds = []
    for local_tr, local_te in splitter.split(local_dummy, groups=local_groups):
        tr, te = train_idx[local_tr], train_idx[local_te]
        if set(groups[tr]).intersection(set(groups[te])):
            raise RuntimeError("Group overlap detected in inner fold")
        folds.append((tr.astype(np.int64), te.astype(np.int64)))
    return folds


def fit_fold_scaler(x_cont: np.ndarray, train_idx: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(np.asarray(x_cont)[np.asarray(train_idx, dtype=np.int64)])
    return scaler


def fit_target_transform(
    y: np.ndarray,
    endpoint_id: np.ndarray,
    train_idx: np.ndarray,
    mode: str = "endpoint_zscore",
) -> TargetTransform:


    if mode not in {"raw", "endpoint_center", "endpoint_zscore"}:
        raise ValueError(f"Unsupported target transform: {mode}")
    endpoint_id = np.asarray(endpoint_id, dtype=np.int64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    n_endpoints = int(endpoint_id.max()) + 1 if endpoint_id.size else N_ENDPOINTS
    means = np.zeros(n_endpoints, dtype=np.float32)
    scales = np.ones(n_endpoints, dtype=np.float32)
    y_train = np.asarray(y, dtype=np.float32)[train_idx]
    e_train = endpoint_id[train_idx]
    global_mean = float(np.mean(y_train))
    global_std = float(np.std(y_train))
    if not np.isfinite(global_std) or global_std < 1e-6:
        global_std = 1.0
    for endpoint in range(n_endpoints):
        values = y_train[e_train == endpoint]
        if values.size:
            means[endpoint] = float(np.mean(values))
            spread = float(np.std(values))
            scales[endpoint] = spread if np.isfinite(spread) and spread >= 1e-6 else global_std
        else:
            means[endpoint] = global_mean
            scales[endpoint] = global_std
    if mode == "raw":
        means.fill(0.0)
        scales.fill(1.0)
    elif mode == "endpoint_center":
        scales.fill(1.0)
    return TargetTransform(mode=mode, means=means, scales=scales)


def fit_fold_transform(
    data: GroupedData,
    train_idx: np.ndarray,
    fingerprint_mode: str = "raw",
) -> FoldTransform:
    if fingerprint_mode not in {"raw", "zscore", "clipped_zscore"}:
        raise ValueError(f"Unsupported fingerprint mode: {fingerprint_mode}")
    fingerprint_scaler = None
    fingerprint_clip = None
    if fingerprint_mode != "raw":
        fingerprint_scaler = fit_fold_scaler(data.x_fp, train_idx)
        if fingerprint_mode == "clipped_zscore":
            fingerprint_clip = 5.0
    return FoldTransform(
        image_scaler=fit_fold_scaler(data.x_img, train_idx),
        continuous_scaler=fit_fold_scaler(data.x_cont, train_idx),
        fingerprint_scaler=fingerprint_scaler,
        fingerprint_clip=fingerprint_clip,
    )


def transform_features(data: GroupedData, indices: np.ndarray, transform: FoldTransform) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    endpoint_oh = np.eye(len(data.endpoint_names), dtype=np.float32)[data.endpoint_id[indices]]
    fingerprints = data.x_fp[indices]
    if transform.fingerprint_scaler is not None:
        fingerprints = transform.fingerprint_scaler.transform(fingerprints).astype(np.float32)
        if transform.fingerprint_clip is not None:
            fingerprints = np.clip(
                fingerprints,
                -float(transform.fingerprint_clip),
                float(transform.fingerprint_clip),
            )
    return np.concatenate(
        [
            fingerprints,
            transform.image_scaler.transform(data.x_img[indices]).astype(np.float32),
            transform.continuous_scaler.transform(data.x_cont[indices]).astype(np.float32),
            endpoint_oh,
        ],
        axis=1,
    ).astype(np.float32)


def group_balanced_weights(group_values: np.ndarray) -> np.ndarray:


    group_values = np.asarray(group_values)
    _, inverse, counts = np.unique(group_values, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float32)
    return (weights / weights.mean()).astype(np.float32)


class FNN(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden1), nn.Sigmoid()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden1, hidden2), nn.Softplus()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class ModernFNN(nn.Module):


    def __init__(self, input_dim: int, hidden1: int, hidden2: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden1, hidden2), nn.LayerNorm(hidden2), nn.GELU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class EndpointConditionedFNN(nn.Module):


    def __init__(self, input_dim: int, hidden1: int, hidden2: int, n_endpoints: int, dropout: float = 0.0):
        super().__init__()
        self.uses_endpoint_head = True
        trunk: list[nn.Module] = [nn.Linear(input_dim, hidden1), nn.Sigmoid()]
        if dropout > 0:
            trunk.append(nn.Dropout(dropout))
        trunk.extend([nn.Linear(hidden1, hidden2), nn.Softplus()])
        if dropout > 0:
            trunk.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*trunk)
        self.heads = nn.Linear(hidden2, n_endpoints)

    def forward(self, x: torch.Tensor, endpoint_id: torch.Tensor) -> torch.Tensor:
        all_heads = self.heads(self.trunk(x))
        return all_heads.gather(1, endpoint_id.long().view(-1, 1)).squeeze(1)


class BlockFNN(nn.Module):


    def __init__(
        self,
        hidden1: int,
        hidden2: int,
        n_endpoints: int,
        dropout: float = 0.0,
        endpoint_heads: bool = False,
    ):
        super().__init__()
        self.uses_endpoint_head = bool(endpoint_heads)
        self.endpoint_heads = bool(endpoint_heads)
        self.fingerprint = nn.Sequential(
            nn.Linear(2214, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
        )
        self.image = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.continuous = nn.Sequential(
            nn.Linear(21, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.endpoint = nn.Sequential(
            nn.Linear(n_endpoints, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
        )
        fusion_layers: list[nn.Module] = [
            nn.Linear(176, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
        ]
        if dropout > 0:
            fusion_layers.append(nn.Dropout(dropout))
        fusion_layers.extend(
            [nn.Linear(hidden1, hidden2), nn.LayerNorm(hidden2), nn.GELU()]
        )
        if dropout > 0:
            fusion_layers.append(nn.Dropout(dropout))
        self.fusion = nn.Sequential(*fusion_layers)
        self.output = nn.Linear(hidden2, n_endpoints if endpoint_heads else 1)

    def forward(self, x: torch.Tensor, endpoint_id: torch.Tensor | None = None) -> torch.Tensor:
        fingerprint = self.fingerprint(x[:, :2214])
        image = self.image(x[:, 2214:2726])
        continuous = self.continuous(x[:, 2726:2747])
        endpoint = self.endpoint(x[:, 2747:])
        fused = self.fusion(torch.cat([fingerprint, image, continuous, endpoint], dim=1))
        output = self.output(fused)
        if self.endpoint_heads:
            if endpoint_id is None:
                raise ValueError("endpoint_id is required for endpoint-head BlockFNN")
            return output.gather(1, endpoint_id.long().view(-1, 1)).squeeze(1)
        return output.squeeze(1)


def _loss_vector(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "rmse":

        return (pred - target).pow(2)
    if loss_name == "mse":
        return (pred - target).pow(2)
    if loss_name == "huber":
        return nn.functional.huber_loss(pred, target, reduction="none", delta=1.0)
    raise ValueError(f"Unsupported loss: {loss_name}")


def _reduce_loss(values: torch.Tensor, weights: torch.Tensor, loss_name: str) -> torch.Tensor:
    weighted = (values * weights).sum() / weights.sum().clamp_min(1e-8)
    if loss_name == "rmse":
        return torch.sqrt(weighted + 1e-8)
    return weighted


def _make_model(kind: str, input_dim: int, config: dict[str, Any], n_endpoints: int) -> nn.Module:
    if kind == "fnn":
        return FNN(input_dim, int(config["hidden1"]), int(config["hidden2"]), float(config.get("dropout", 0.0)))
    if kind == "modern_fnn":
        return ModernFNN(
            input_dim,
            int(config["hidden1"]),
            int(config["hidden2"]),
            float(config.get("dropout", 0.0)),
        )
    if kind == "endpoint_fnn":
        return EndpointConditionedFNN(
            input_dim,
            int(config["hidden1"]),
            int(config["hidden2"]),
            n_endpoints,
            float(config.get("dropout", 0.0)),
        )
    if kind in {"block_fnn", "block_endpoint_fnn"}:
        return BlockFNN(
            int(config["hidden1"]),
            int(config["hidden2"]),
            n_endpoints,
            float(config.get("dropout", 0.0)),
            endpoint_heads=kind == "block_endpoint_fnn",
        )
    raise ValueError(f"Unknown neural model kind: {kind}")


def _predict(model: nn.Module, x: np.ndarray, endpoint_id: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            xb = torch.from_numpy(x[start:stop]).to(device)
            if getattr(model, "uses_endpoint_head", False):
                eb = torch.from_numpy(endpoint_id[start:stop]).to(device)
                out = model(xb, eb)
            else:
                out = model(xb)
            predictions.append(out.detach().cpu().numpy())
    return np.concatenate(predictions).astype(np.float32)


def fit_neural_model(
    kind: str,
    data: GroupedData,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    transform: FoldTransform,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    group_balance: bool,
    target_transform: TargetTransform | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed)
    x_train = transform_features(data, train_idx, transform)
    x_valid = transform_features(data, valid_idx, transform)
    y_train = data.y[train_idx]
    y_valid = data.y[valid_idx]
    endpoint_train = data.endpoint_id[train_idx]
    endpoint_valid = data.endpoint_id[valid_idx]
    if target_transform is not None:
        y_train = target_transform.transform(y_train, endpoint_train)
        y_valid = target_transform.transform(y_valid, endpoint_valid)
    if group_balance:
        train_weights = group_balanced_weights(data.groups[train_idx])
    else:
        train_weights = np.ones(len(train_idx), dtype=np.float32)

    tensor_x = torch.from_numpy(x_train)
    tensor_y = torch.from_numpy(y_train)
    tensor_e = torch.from_numpy(endpoint_train.astype(np.int64))
    tensor_w = torch.from_numpy(train_weights)
    batch_size = int(config.get("batch_size", 128))
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y, tensor_e, tensor_w)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = _make_model(kind, data.input_dim, config, len(data.endpoint_names)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    loss_name = str(config.get("loss", "rmse"))
    clip_norm = float(config.get("clip_norm", 1.0))
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(int(max_epochs)):
        model.train()
        train_losses = []
        for xb, yb, eb, wb in loader:
            xb, yb, eb, wb = xb.to(device), yb.to(device), eb.to(device), wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, eb) if getattr(model, "uses_endpoint_head", False) else model(xb)
            loss = _reduce_loss(_loss_vector(pred, yb, loss_name), wb, loss_name)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        valid_pred = _predict(model, x_valid, endpoint_valid, device)
        valid_error = float(np.mean((valid_pred - y_valid) ** 2))
        train_error = float(np.mean(train_losses)) if train_losses else float("nan")
        history.append({"epoch": epoch + 1, "train_loss": train_error, "valid_mse": valid_error})
        if valid_error < best_loss - 1e-8:
            best_loss = valid_error
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_valid_mse": best_loss,
        "epochs_run": len(history),
        "history": history,
    }


def fit_neural_model_fixed_epochs(
    kind: str,
    data: GroupedData,
    train_idx: np.ndarray,
    transform: FoldTransform,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    epochs: int,
    group_balance: bool,
    target_transform: TargetTransform | None = None,
) -> tuple[nn.Module, dict[str, Any]]:


    if int(epochs) < 1:
        raise ValueError("epochs must be at least one for a fixed-epoch refit")
    set_seed(seed)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    x_train = transform_features(data, train_idx, transform)
    y_train = data.y[train_idx]
    endpoint_train = data.endpoint_id[train_idx]
    if target_transform is not None:
        y_train = target_transform.transform(y_train, endpoint_train)
    if group_balance:
        train_weights = group_balanced_weights(data.groups[train_idx])
    else:
        train_weights = np.ones(len(train_idx), dtype=np.float32)

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(endpoint_train.astype(np.int64)),
        torch.from_numpy(train_weights),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 128)),
        shuffle=True,
        num_workers=0,
    )
    model = _make_model(kind, data.input_dim, config, len(data.endpoint_names)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    loss_name = str(config.get("loss", "rmse"))
    clip_norm = float(config.get("clip_norm", 1.0))
    history = []
    for epoch in range(int(epochs)):
        model.train()
        train_losses = []
        for xb, yb, eb, wb in loader:
            xb, yb, eb, wb = xb.to(device), yb.to(device), eb.to(device), wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, eb) if getattr(model, "uses_endpoint_head", False) else model(xb)
            loss = _reduce_loss(_loss_vector(pred, yb, loss_name), wb, loss_name)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses))})
    return model, {"epochs_run": int(epochs), "history": history}


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, endpoint_id: np.ndarray, endpoint_names: Iterable[str]) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        metrics["pearson_r"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        metrics["pearson_r"] = float("nan")
    per_endpoint: dict[str, Any] = {}
    endpoint_r2 = []
    for endpoint, name in enumerate(endpoint_names):
        mask = endpoint_id == endpoint
        if mask.sum() < 3:
            continue
        value = {
            "n": int(mask.sum()),
            "r2": float(r2_score(y_true[mask], y_pred[mask])),
            "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
            "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        }
        per_endpoint[str(name)] = value
        endpoint_r2.append(value["r2"])
    metrics["per_endpoint"] = per_endpoint
    metrics["macro_endpoint_r2"] = float(np.mean(endpoint_r2)) if endpoint_r2 else float("nan")
    return metrics


def mean_fold_r2_score(fold_scores: Iterable[float]) -> float:


    values = np.asarray(list(fold_scores), dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("fold_scores must contain at least one finite value")
    return float(np.mean(values))


def pooled_r2_score(y_parts: Iterable[np.ndarray], prediction_parts: Iterable[np.ndarray]) -> float:


    y_values = np.concatenate([np.asarray(values, dtype=float) for values in y_parts])
    predictions = np.concatenate([np.asarray(values, dtype=float) for values in prediction_parts])
    if y_values.shape != predictions.shape or y_values.size == 0:
        raise ValueError("Pooled R2 inputs must be non-empty arrays with matching shapes")
    return float(r2_score(y_values, predictions))


def fit_linear_calibration(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:


    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if y_true.shape != prediction.shape or y_true.size < 2:
        raise ValueError("Calibration inputs must have matching shapes and at least two rows")
    variance = float(np.var(prediction))
    if not np.isfinite(variance) or variance < 1e-12:
        return {"slope": 1.0, "intercept": float(np.mean(y_true) - np.mean(prediction))}
    slope = float(np.cov(prediction, y_true, ddof=0)[0, 1] / variance)
    intercept = float(np.mean(y_true) - slope * np.mean(prediction))
    return {"slope": slope, "intercept": intercept}


def apply_linear_calibration(prediction: np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    return (
        float(calibration["slope"]) * np.asarray(prediction, dtype=np.float32)
        + float(calibration["intercept"])
    ).astype(np.float32)


def fit_blend_weight(
    y_true: np.ndarray,
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
) -> float:


    y_true = np.asarray(y_true, dtype=float)
    first_prediction = np.asarray(first_prediction, dtype=float)
    second_prediction = np.asarray(second_prediction, dtype=float)
    if not (y_true.shape == first_prediction.shape == second_prediction.shape) or y_true.size < 2:
        raise ValueError("Blend inputs must have matching shapes and at least two rows")
    difference = first_prediction - second_prediction
    denominator = float(np.dot(difference, difference))
    if not np.isfinite(denominator) or denominator < 1e-12:
        return 0.5
    weight = float(np.dot(y_true - second_prediction, difference) / denominator)
    return float(np.clip(weight, 0.0, 1.0))


def fit_endpoint_blend_weights(
    y_true: np.ndarray,
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
    endpoint_id: np.ndarray,
    n_endpoints: int,
    shrinkage: float = 0.0,
) -> np.ndarray:


    if not 0.0 <= float(shrinkage) <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    endpoint_id = np.asarray(endpoint_id, dtype=np.int64)
    weights = np.full(int(n_endpoints), 0.5, dtype=np.float32)
    for endpoint in range(int(n_endpoints)):
        mask = endpoint_id == endpoint
        if int(mask.sum()) >= 2:
            weights[endpoint] = fit_blend_weight(
                np.asarray(y_true)[mask],
                np.asarray(first_prediction)[mask],
                np.asarray(second_prediction)[mask],
            )
    return ((1.0 - float(shrinkage)) * weights + float(shrinkage) * 0.5).astype(np.float32)


def select_endpoint_blend_shrinkage(
    y_true: np.ndarray,
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
    endpoint_id: np.ndarray,
    fold_id: np.ndarray,
    n_endpoints: int,
    candidates: Iterable[float] = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
) -> tuple[float, np.ndarray, dict[str, Any]]:


    y_true = np.asarray(y_true, dtype=float)
    first_prediction = np.asarray(first_prediction, dtype=float)
    second_prediction = np.asarray(second_prediction, dtype=float)
    endpoint_id = np.asarray(endpoint_id, dtype=np.int64)
    fold_id = np.asarray(fold_id, dtype=np.int64)
    if not (
        y_true.shape
        == first_prediction.shape
        == second_prediction.shape
        == endpoint_id.shape
        == fold_id.shape
    ):
        raise ValueError("Inner blend arrays must have matching shapes")
    unique_folds = np.unique(fold_id)
    if unique_folds.size == 0:
        raise ValueError("fold_id must contain at least one fold")
    best: tuple[float, float, np.ndarray] | None = None
    trial_rows = []
    for candidate in candidates:
        shrinkage = float(candidate)
        weights = fit_endpoint_blend_weights(
            y_true,
            first_prediction,
            second_prediction,
            endpoint_id,
            n_endpoints,
            shrinkage=shrinkage,
        )
        prediction = weights[endpoint_id] * first_prediction + (1.0 - weights[endpoint_id]) * second_prediction
        fold_scores = [
            float(r2_score(y_true[fold_id == fold], prediction[fold_id == fold]))
            for fold in unique_folds
        ]
        objective = mean_fold_r2_score(fold_scores)
        trial_rows.append(
            {"shrinkage": shrinkage, "mean_inner_fold_r2": objective, "fold_r2": fold_scores}
        )

        if best is None or objective > best[0] + 1e-10 or (
            abs(objective - best[0]) <= 1e-10 and shrinkage < best[1]
        ):
            best = (objective, shrinkage, weights)
    assert best is not None
    return best[1], best[2], {"objective": best[0], "trials": trial_rows}


def aggregate_augmented_predictions(
    prediction: np.ndarray,
    indices: np.ndarray,
    n_original: int = 3672,
) -> np.ndarray:


    prediction = np.asarray(prediction, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    if prediction.ndim != 1 or prediction.shape[0] != indices.shape[0]:
        raise ValueError("prediction and indices must be one-dimensional arrays of equal length")
    if n_original < 1:
        raise ValueError("n_original must be positive")
    local = {int(index): position for position, index in enumerate(indices.tolist())}
    averaged = prediction.copy()
    for original in range(int(n_original)):
        triplet = (original, n_original + 2 * original, n_original + 2 * original + 1)
        positions = [local.get(index) for index in triplet]
        if all(position is not None for position in positions):
            mean_prediction = float(np.mean(prediction[np.asarray(positions, dtype=np.int64)]))
            averaged[np.asarray(positions, dtype=np.int64)] = mean_prediction
    return averaged


def select_refit_epochs(selection: dict[str, Any], max_epochs: int, rule: str = "median") -> int:


    if rule not in {"median", "mean"}:
        raise ValueError(f"Unsupported refit epoch rule: {rule}")
    epochs = [
        int(item["fit"]["best_epoch"])
        for item in selection.get("folds", [])
        if int(item.get("fit", {}).get("best_epoch", 0)) > 0
    ]
    selected = float(np.median(epochs) if rule == "median" else np.mean(epochs)) if epochs else float(max_epochs)
    return max(1, min(int(round(selected)), int(max_epochs)))


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _default_baseline() -> dict[str, Any]:
    return {
        "hidden1": 512,
        "hidden2": 32,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "loss": "rmse",
        "batch_size": 128,
        "clip_norm": 1.0,
    }


def _mean_fold_candidate_configs() -> dict[str, dict[str, Any]]:


    common = {
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "loss": "mse",
        "batch_size": 256,
        "clip_norm": 1.0,
    }
    return {
        "modern_fnn": {**common, "hidden1": 256, "hidden2": 32},
        "endpoint_fnn": {**common, "hidden1": 256, "hidden2": 64},
        "block_fnn": {**common, "hidden1": 128, "hidden2": 32},
        "block_endpoint_fnn": {**common, "hidden1": 128, "hidden2": 32},
    }


def _sample_config(trial: Any, kind: str) -> dict[str, Any]:
    config = {
        "hidden1": trial.suggest_categorical("hidden1", [128, 256, 384, 512]),
        "hidden2": trial.suggest_categorical("hidden2", [32, 64, 96, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.35),
        "learning_rate": trial.suggest_float("learning_rate", 2e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 2e-3, log=True),
        "loss": trial.suggest_categorical("loss", ["rmse", "huber"]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "clip_norm": 1.0,
    }
    if kind == "fnn":
        config["dropout"] = 0.0
    return config


def _inner_score(
    kind: str,
    data: GroupedData,
    outer_train_idx: np.ndarray,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    quick: bool,
    group_balance: bool,
    target_mode: str = "raw",
    tta: bool = False,
    include_oof: bool = False,
    selection_objective: str = "pooled",
    fingerprint_mode: str = "raw",
) -> tuple[float, float, list[dict[str, Any]], dict[str, Any]]:
    if selection_objective not in {"pooled", "mean_fold"}:
        raise ValueError(f"Unsupported selection objective: {selection_objective}")
    inner_folds = make_inner_folds(outer_train_idx, data.groups, n_splits=3, seed=seed)
    y_parts = []
    prediction_parts = []
    index_parts = []
    fold_parts = []
    details = []
    max_epochs = 45 if quick else 90
    patience = 10 if quick else 15
    for inner_number, (inner_train, inner_valid) in enumerate(inner_folds, start=1):
        transform = fit_fold_transform(data, inner_train, fingerprint_mode=fingerprint_mode)
        target_transform = fit_target_transform(
            data.y, data.endpoint_id, inner_train, mode=target_mode
        )


        model, fit_info = fit_neural_model(
            kind,
            data,
            inner_train,
            inner_valid,
            transform,
            config,
            seed + inner_number,
            device,
            max_epochs=max_epochs,
            patience=patience,
            group_balance=group_balance,
            target_transform=target_transform,
        )
        pred_transformed = _predict(
            model,
            transform_features(data, inner_valid, transform),
            data.endpoint_id[inner_valid],
            device,
        )
        pred = target_transform.inverse(pred_transformed, data.endpoint_id[inner_valid])
        if tta:
            pred = aggregate_augmented_predictions(pred, inner_valid)
        score = float(r2_score(data.y[inner_valid], pred))
        y_parts.append(data.y[inner_valid])
        prediction_parts.append(pred)
        index_parts.append(inner_valid)
        fold_parts.append(np.full(len(inner_valid), inner_number, dtype=np.int64))
        details.append({"fold": inner_number, "r2": score, "fit": fit_info})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pooled_score = pooled_r2_score(y_parts, prediction_parts)
    mean_score = mean_fold_r2_score([item["r2"] for item in details])
    objective_score = mean_score if selection_objective == "mean_fold" else pooled_score
    calibration = fit_linear_calibration(
        np.concatenate(y_parts), np.concatenate(prediction_parts)
    )
    details_meta = {
        "calibration": calibration,
        "inner_pooled_r2": pooled_score,
        "inner_mean_fold_r2": mean_score,
        "selection_objective": selection_objective,
        "fingerprint_mode": fingerprint_mode,
    }
    if include_oof:
        details_meta.update(
            {
                "y_oof": np.concatenate(y_parts).astype(np.float32),
                "prediction_oof": np.concatenate(prediction_parts).astype(np.float32),
                "indices_oof": np.concatenate(index_parts).astype(np.int64),
                "fold_oof": np.concatenate(fold_parts).astype(np.int64),
            }
        )
    return (
        objective_score,
        float(np.mean([d["fit"]["best_valid_mse"] for d in details])),
        details,
        details_meta,
    )


def select_config(
    kind: str,
    data: GroupedData,
    outer_train_idx: np.ndarray,
    seed: int,
    device: torch.device,
    n_trials: int,
    quick: bool,
    group_balance: bool,
    target_mode: str = "raw",
    tta: bool = False,
    include_oof: bool = False,
    selection_objective: str = "pooled",
    fingerprint_mode: str = "raw",
    default_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import optuna

    if n_trials <= 0:
        config = copy.deepcopy(default_config) if default_config is not None else _default_baseline()
        score, valid_loss, details, metadata = _inner_score(
            kind,
            data,
            outer_train_idx,
            config,
            seed,
            device,
            quick,
            group_balance,
            target_mode,
            tta,
            include_oof,
            selection_objective,
            fingerprint_mode,
        )
        return config, {
            "method": "fixed_baseline",
            "inner_r2": score,
            "inner_pooled_r2": metadata["inner_pooled_r2"],
            "inner_mean_fold_r2": metadata["inner_mean_fold_r2"],
            "inner_valid_mse": valid_loss,
            "folds": details,
            **metadata,
        }

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: Any) -> float:
        config = _sample_config(trial, kind)
        score, _, _, _ = _inner_score(
            kind,
            data,
            outer_train_idx,
            config,
            seed + trial.number + 1,
            device,
            quick,
            group_balance,
            target_mode,
            tta,
            False,
            selection_objective,
            fingerprint_mode,
        )
        return score

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best["clip_norm"] = 1.0
    if kind == "fnn":
        best["dropout"] = 0.0
    best_score, best_valid_loss, details, metadata = _inner_score(
        kind,
        data,
        outer_train_idx,
        best,
        seed + 10000,
        device,
        quick,
        group_balance,
        target_mode,
        tta,
        include_oof,
        selection_objective,
        fingerprint_mode,
    )
    return best, {
        "method": "inner_grouped_optuna",
        "n_trials": n_trials,
        "best_trial": int(study.best_trial.number),
        "study_best_inner_r2": float(study.best_value),
        "selected_inner_r2": best_score,
        "selected_inner_pooled_r2": metadata["inner_pooled_r2"],
        "selected_inner_mean_fold_r2": metadata["inner_mean_fold_r2"],
        "selected_inner_valid_mse": best_valid_loss,
        "folds": details,
        **metadata,
        "trial_values": [
            {"number": int(t.number), "value": None if t.value is None else float(t.value), "params": dict(t.params)}
            for t in study.trials
        ],
    }


def run_outer_neural(
    kind: str,
    data: GroupedData,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    device: torch.device,
    n_trials: int,
    quick: bool,
    group_balance: bool,
    refit_all: bool = False,
    target_mode: str = "raw",
    refit_epoch_rule: str = "median",
    calibrate_predictions: bool = False,
    tta: bool = False,
    selection_objective: str = "pooled",
    fingerprint_mode: str = "raw",
    default_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    y_pred = np.full(data.n_rows, np.nan, dtype=np.float32)
    fold_rows = []
    configs = []
    for outer_number, (outer_train, outer_test) in enumerate(folds, start=1):
        fold_seed = seed + outer_number * 100
        print(f"[{kind}] outer fold {outer_number}/{len(folds)}: selecting configuration", flush=True)
        config, selection = select_config(
            kind,
            data,
            outer_train,
            fold_seed,
            device,
            n_trials,
            quick,
            group_balance,
            target_mode,
            tta,
            False,
            selection_objective,
            fingerprint_mode,
            default_config,
        )
        max_epochs = 80 if quick else 180
        patience = 12 if quick else 25


        refit_epochs = select_refit_epochs(selection, max_epochs, refit_epoch_rule)
        target_transform = fit_target_transform(
            data.y, data.endpoint_id, outer_train, mode=target_mode
        )
        if refit_all and not quick:
            transform = fit_fold_transform(data, outer_train, fingerprint_mode=fingerprint_mode)
            model, fit_info = fit_neural_model_fixed_epochs(
                kind,
                data,
                outer_train,
                transform,
                config,
                fold_seed + 1,
                device,
                epochs=refit_epochs,
                group_balance=group_balance,
                target_transform=target_transform,
            )
            fit_info["refit_epochs_selected_from_inner"] = refit_epochs
        else:

            split = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=fold_seed)
            local_train, local_valid = next(split.split(outer_train, groups=data.groups[outer_train]))
            fit_train = outer_train[local_train]
            fit_valid = outer_train[local_valid]
            transform = fit_fold_transform(data, fit_train, fingerprint_mode=fingerprint_mode)
            model, fit_info = fit_neural_model(
                kind,
                data,
                fit_train,
                fit_valid,
                transform,
                config,
                fold_seed + 1,
                device,
                max_epochs=max_epochs,
                patience=patience,
                group_balance=group_balance,
                target_transform=target_transform,
            )
        test_features = transform_features(data, outer_test, transform)
        pred_transformed = _predict(model, test_features, data.endpoint_id[outer_test], device)
        pred = target_transform.inverse(pred_transformed, data.endpoint_id[outer_test])
        if tta:
            pred = aggregate_augmented_predictions(pred, outer_test)
        if calibrate_predictions and "calibration" in selection:
            pred = apply_linear_calibration(pred, selection["calibration"])
        y_pred[outer_test] = pred
        fold_metric = evaluate_predictions(data.y[outer_test], pred, data.endpoint_id[outer_test], data.endpoint_names)
        fold_metric["fold"] = outer_number
        fold_metric["n_groups_test"] = int(np.unique(data.groups[outer_test]).size)
        fold_rows.append(fold_metric)
        configs.append({"fold": outer_number, "config": config, "selection": selection, "fit": fit_info})
        print(f"[{kind}] outer fold {outer_number}: R2={fold_metric['r2']:.4f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics = evaluate_predictions(data.y, y_pred, data.endpoint_id, data.endpoint_names)
    metrics["folds"] = fold_rows
    metrics["mean_outer_fold_r2"] = mean_fold_r2_score(item["r2"] for item in fold_rows)
    metrics["std_outer_fold_r2"] = float(np.std([item["r2"] for item in fold_rows], ddof=1))
    return {"kind": kind, "metrics": metrics, "configs": configs, "predictions": y_pred}


def run_outer_neural_ensemble(
    data: GroupedData,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    device: torch.device,
    target_mode: str = "endpoint_zscore",
    group_balance: bool = False,
    refit_all: bool = True,
    refit_epoch_rule: str = "median",
    tta: bool = True,
    endpoint_specific_weights: bool = False,
    selection_objective: str = "pooled",
    fingerprint_mode: str = "raw",
    n_trials: int = 0,
    default_configs: dict[str, dict[str, Any]] | None = None,
    kinds: tuple[str, ...] = ("modern_fnn", "endpoint_fnn"),
) -> dict[str, Any]:


    if len(kinds) != 2:
        raise ValueError("The current convex ensemble requires exactly two model kinds")
    y_pred = np.full(data.n_rows, np.nan, dtype=np.float32)
    fold_rows = []
    configurations = []
    for outer_number, (outer_train, outer_test) in enumerate(folds, start=1):
        fold_seed = seed + outer_number * 100
        selected = []
        for kind in kinds:
            config, selection = select_config(
                kind,
                data,
                outer_train,


                fold_seed,
                device,
                n_trials=n_trials,
                quick=False,
                group_balance=group_balance,
                target_mode=target_mode,
                tta=tta,
                include_oof=True,
                selection_objective=selection_objective,
                fingerprint_mode=fingerprint_mode,
                default_config=(default_configs or {}).get(kind),
            )
            selected.append((kind, config, selection))

        def materialize_inner(
            selection: dict[str, Any],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            indices = np.asarray(selection["indices_oof"], dtype=np.int64)
            predictions = np.asarray(selection["prediction_oof"], dtype=np.float32)
            y_values = np.asarray(selection["y_oof"], dtype=np.float32)
            fold_values = np.asarray(selection["fold_oof"], dtype=np.int64)
            full_prediction = np.full(data.n_rows, np.nan, dtype=np.float32)
            full_y = np.full(data.n_rows, np.nan, dtype=np.float32)
            full_prediction[indices] = predictions
            full_y[indices] = y_values
            mask = np.isfinite(full_prediction) & np.isfinite(full_y)
            full_fold = np.full(data.n_rows, -1, dtype=np.int64)
            full_fold[indices] = fold_values
            return full_y[mask], full_prediction[mask], data.endpoint_id[mask], full_fold[mask]

        y_a, p_a, e_a, f_a = materialize_inner(selected[0][2])
        y_b, p_b, e_b, f_b = materialize_inner(selected[1][2])
        if (
            y_a.shape != y_b.shape
            or not np.allclose(y_a, y_b)
            or not np.array_equal(e_a, e_b)
            or not np.array_equal(f_a, f_b)
        ):
            raise RuntimeError("Inner OOF rows are not aligned between ensemble members")
        if endpoint_specific_weights:
            blend_shrinkage, blend_weight, blend_selection = select_endpoint_blend_shrinkage(
                y_a,
                p_a,
                p_b,
                e_a,
                f_a,
                len(data.endpoint_names),
            )
        else:
            blend_shrinkage = 0.0
            blend_selection = None
            blend_weight = fit_blend_weight(y_a, p_a, p_b)

        outer_predictions = []
        fit_infos = []
        for kind, config, _selection in selected:
            target_transform = fit_target_transform(
                data.y, data.endpoint_id, outer_train, mode=target_mode
            )
            max_epochs = 180
            if refit_all:
                transform = fit_fold_transform(data, outer_train, fingerprint_mode=fingerprint_mode)
                epochs = select_refit_epochs(_selection, max_epochs, refit_epoch_rule)
                model, fit_info = fit_neural_model_fixed_epochs(
                    kind,
                    data,
                    outer_train,
                    transform,
                    config,
                    fold_seed + 1,
                    device,
                    epochs=epochs,
                    group_balance=group_balance,
                    target_transform=target_transform,
                )
                fit_info["refit_epochs_selected_from_inner"] = epochs
            else:
                split = GroupShuffleSplit(
                    n_splits=1, test_size=0.1, random_state=fold_seed
                )
                local_train, local_valid = next(
                    split.split(outer_train, groups=data.groups[outer_train])
                )
                fit_train = outer_train[local_train]
                fit_valid = outer_train[local_valid]
                transform = fit_fold_transform(data, fit_train, fingerprint_mode=fingerprint_mode)
                model, fit_info = fit_neural_model(
                    kind,
                    data,
                    fit_train,
                    fit_valid,
                    transform,
                    config,
                    fold_seed + 1,
                    device,
                    max_epochs=max_epochs,
                    patience=25,
                    group_balance=group_balance,
                    target_transform=target_transform,
                )
            prediction = _predict(
                model,
                transform_features(data, outer_test, transform),
                data.endpoint_id[outer_test],
                device,
            )
            prediction = target_transform.inverse(
                prediction, data.endpoint_id[outer_test]
            )
            if tta:
                prediction = aggregate_augmented_predictions(prediction, outer_test)
            outer_predictions.append(prediction)
            fit_infos.append(fit_info)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if endpoint_specific_weights:
            outer_weights = blend_weight[data.endpoint_id[outer_test]]
        else:
            outer_weights = blend_weight
        blended = outer_weights * outer_predictions[0] + (1.0 - outer_weights) * outer_predictions[1]
        y_pred[outer_test] = blended
        metric = evaluate_predictions(
            data.y[outer_test], blended, data.endpoint_id[outer_test], data.endpoint_names
        )
        metric["fold"] = outer_number
        metric["n_groups_test"] = int(np.unique(data.groups[outer_test]).size)
        metric["modern_weight"] = blend_weight
        metric["blend_shrinkage"] = blend_shrinkage
        if blend_selection is not None:
            metric["blend_selection"] = blend_selection
        fold_rows.append(metric)
        sanitized = []
        for kind, config, selection in selected:
            sanitized_selection = {
                key: value
                for key, value in selection.items()
                if key not in {"y_oof", "prediction_oof", "indices_oof", "fold_oof"}
            }
            sanitized.append(
                {"kind": kind, "config": config, "selection": sanitized_selection}
            )
        configurations.append(
            {"fold": outer_number, "modern_weight": blend_weight, "members": sanitized, "fits": fit_infos}
        )
        print(
            f"[ensemble] outer fold {outer_number}: R2={metric['r2']:.4f}, "
            f"modern_weight={np.mean(np.asarray(blend_weight)):.3f}",
            flush=True,
        )
    metrics = evaluate_predictions(data.y, y_pred, data.endpoint_id, data.endpoint_names)
    metrics["folds"] = fold_rows
    metrics["mean_outer_fold_r2"] = mean_fold_r2_score(item["r2"] for item in fold_rows)
    metrics["std_outer_fold_r2"] = float(np.std([item["r2"] for item in fold_rows], ddof=1))
    return {
        "kind": "modern_endpoint_ensemble",
        "metrics": metrics,
        "configs": configurations,
        "predictions": y_pred,
    }


def save_results(root: Path, data: GroupedData, results: dict[str, dict[str, Any]], folds: list[tuple[np.ndarray, np.ndarray]], args: argparse.Namespace) -> None:
    output_dir = root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(results)
    prediction_matrix = np.vstack([results[name]["predictions"] for name in model_names]).astype(np.float32)
    fold_id = np.full(data.n_rows, -1, dtype=np.int64)
    for number, (_, test_idx) in enumerate(folds, start=1):
        fold_id[test_idx] = number
    np.savez_compressed(
        output_dir / "grouped_fnn_predictions.npz",
        y_true=data.y,
        predictions=prediction_matrix,
        model_names=np.asarray(model_names),
        endpoint_id=data.endpoint_id,
        endpoint_names=np.asarray(data.endpoint_names),
        canonical_group=data.groups,
        outer_fold=fold_id,
    )
    payload = {
        "experiment": "canonical-IL-grouped nested-CV optimization",
        "seed": args.seed,
        "device": str(args.device),
        "rows": data.n_rows,
        "canonical_groups": int(np.unique(data.groups).size),
        "outer_folds": len(folds),
        "inner_folds": 3,
        "group_balance": bool(args.group_balance),
        "refit_all": bool(args.refit_all),
        "refit_epoch_rule": str(args.refit_epoch_rule),
        "calibrate_predictions": bool(args.calibrate_predictions),
        "tta": bool(args.tta),
        "ensemble": bool(getattr(args, "ensemble", False)),
        "ensemble_refit_all": bool(getattr(args, "ensemble", False) or args.refit_all),
        "ensemble_tta": bool(getattr(args, "ensemble", False) or args.tta),
        "selection_objective": str(getattr(args, "selection_objective", "pooled")),
        "fingerprint_mode": str(getattr(args, "fingerprint_mode", "raw")),
        "n_trials": int(args.trials),
        "models": {
            name: {"kind": result["kind"], "metrics": _json_safe(result["metrics"]), "configs": _json_safe(result["configs"])}
            for name, result in results.items()
        },
    }
    with (output_dir / "grouped_fnn_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def run_audit(root: Path, seed: int) -> None:
    data = load_grouped_data(root)
    folds = make_group_folds(data.groups, n_splits=5, seed=seed)
    print(f"rows={data.n_rows}")
    print(f"fingerprints={data.x_fp.shape[1]}, images={data.x_img.shape[1]}, continuous={data.x_cont.shape[1]}, endpoints={len(data.endpoint_names)}")
    print(f"canonical_groups={np.unique(data.groups).size}")
    print("folds=" + ", ".join(str((len(tr), len(te), np.unique(data.groups[te]).size)) for tr, te in folds))
    for outer_number, (outer_train, _) in enumerate(folds, start=1):
        inner = make_inner_folds(outer_train, data.groups, 3, seed + outer_number * 100)
        for inner_train, inner_valid in inner:
            assert set(data.groups[inner_train]).isdisjoint(set(data.groups[inner_valid]))
    print("audit=PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Canonical-IL-grouped cross-validation and optimization for multimodal FNN models.')
    parser.add_argument("--baseline-only", action="store_true", help="Run fixed FNN grouped CV without HPO")
    parser.add_argument("--full", action="store_true", help="Run nested tuning for conservative and endpoint-conditioned FNN")
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Run the strict endpoint-specific modern/conditional FNN ensemble",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run a reduced grouped-CV smoke test")
    parser.add_argument("--audit-only", action="store_true", help="Validate rows, groups and fold isolation")
    parser.add_argument("--trials", type=int, default=None, help="Inner Optuna trials per outer fold")
    parser.add_argument(
        "--selection-objective",
        choices=["pooled", "mean_fold"],
        default="pooled",
        help="Inner-CV objective used for configuration selection",
    )
    parser.add_argument(
        "--fingerprint-mode",
        choices=["raw", "zscore", "clipped_zscore"],
        default="raw",
        help="Fold-local fingerprint preprocessing",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], nargs="?")
    parser.add_argument("--no-group-balance", dest="group_balance", action="store_false")
    parser.add_argument(
        "--refit-all",
        action="store_true",
        help="After inner selection, refit on all outer-training rows",
    )
    parser.add_argument(
        "--refit-epoch-rule",
        choices=["median", "mean"],
        default="median",
        help="Aggregate inner-fold checkpoint epochs for an all-training refit",
    )
    parser.add_argument(
        "--calibrate-predictions",
        action="store_true",
        help="Apply affine calibration fitted on inner grouped-OOF predictions",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Average predictions across each original-plus-two-randomized-SMILES triplet",
    )
    parser.set_defaults(group_balance=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.trials = 1
        args.group_balance = True
    elif args.baseline_only:
        args.trials = 0
    elif args.ensemble:
        args.trials = 0
        args.selection_objective = "mean_fold"
        args.fingerprint_mode = "clipped_zscore"
    elif args.full:
        args.trials = 8 if args.trials is None else args.trials
    elif args.audit_only:
        args.trials = 0
    else:
        args.trials = 0
    root = ROOT
    data = load_grouped_data(root)
    if args.audit_only:
        run_audit(root, args.seed)
        return
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    args.device = device_name
    print(f"Using device={device}", flush=True)
    folds = make_group_folds(data.groups, n_splits=5, seed=args.seed)
    quick = bool(args.smoke_test)
    results: dict[str, dict[str, Any]] = {}
    started = time.time()
    baseline_balance = bool(args.group_balance)
    if not args.ensemble:
        results["fnn_baseline"] = run_outer_neural(
            "fnn",
            data,
            folds,
            args.seed,
            device,
            args.trials,
            quick,
            baseline_balance,
            refit_all=bool(args.refit_all),
            refit_epoch_rule=args.refit_epoch_rule,
            calibrate_predictions=bool(args.calibrate_predictions),
            tta=bool(args.tta),
            selection_objective=args.selection_objective,
            fingerprint_mode=args.fingerprint_mode,
        )
    if args.ensemble and not args.smoke_test:
        results["modern_endpoint_ensemble"] = run_outer_neural_ensemble(
            data,
            folds,
            args.seed,
            device,
            target_mode="endpoint_zscore",
            group_balance=False,
            refit_all=bool(args.refit_all or args.ensemble),
            refit_epoch_rule=args.refit_epoch_rule,
            tta=bool(args.tta or args.ensemble),
            endpoint_specific_weights=True,
            selection_objective=args.selection_objective,
            fingerprint_mode=args.fingerprint_mode,
            n_trials=args.trials,
            default_configs=_mean_fold_candidate_configs(),
        )
    elif args.full and not args.smoke_test:
        results["endpoint_fnn"] = run_outer_neural(
            "endpoint_fnn",
            data,
            folds,
            args.seed + 5000,
            device,
            args.trials,
            quick,
            True,
            refit_all=bool(args.refit_all),
            refit_epoch_rule=args.refit_epoch_rule,
            calibrate_predictions=bool(args.calibrate_predictions),
            tta=bool(args.tta),
            selection_objective=args.selection_objective,
            fingerprint_mode=args.fingerprint_mode,
        )
    save_results(root, data, results, folds, args)
    print(f"saved grouped results in {time.time() - started:.1f}s", flush=True)
    for name, result in results.items():
        metric = result["metrics"]
        print(
            f"{name}: pooled R2={metric['r2']:.4f}, RMSE={metric['rmse']:.4f}, "
            f"MAE={metric['mae']:.4f}, macro_endpoint_R2={metric['macro_endpoint_r2']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
