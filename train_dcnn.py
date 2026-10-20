import os, json, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import optuna

from data_utils import load_data, per_cell_line_r2

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
os.makedirs("results", exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.backends.cudnn.benchmark = True

print("Paper1-DESC | DCNN + RDKit descriptors | Swish + Adam(0.999) + L2 | 10fold CV")
print("=" * 64)

D = load_data()
X_num, img_feats, Mw, cell_oh = D["X_num"], D["img_feats"], D["Mw"], D["cell_oh"]
y_all, cell_labels = D["y"], D["cell_labels"]

rd = np.load("data/rdkit_descriptors.npz", allow_pickle=True)
rdkit_desc = rd["descriptors_norm"].astype(np.float32)
print(f"  RDKit descriptors: {rdkit_desc.shape}")
meta = np.concatenate([Mw, cell_oh, rdkit_desc], axis=1).astype(np.float32)
N = len(y_all)
FP_DIM = X_num.shape[1]
IMG_DIM = img_feats.shape[1]
META_DIM = meta.shape[1]
print(f"  samples: {N}  fp:{FP_DIM} img_feat:{IMG_DIM} meta:{META_DIM}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device: {device}")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
PAPER_R2 = 0.965


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DCNN_MultiModal(nn.Module):

    def __init__(self, c1, c2, k, h1, h2, drop, img_hidden=128, meta_hidden=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, k, padding=k // 2), nn.BatchNorm1d(c1), Swish(),
            nn.Dropout(drop), nn.AvgPool1d(2),
            nn.Conv1d(c1, c2, 5, padding=2), nn.BatchNorm1d(c2), Swish(),
            nn.Dropout(drop), nn.AvgPool1d(2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.img_head = nn.Sequential(
            nn.Linear(IMG_DIM, img_hidden), nn.BatchNorm1d(img_hidden), Swish(), nn.Dropout(drop))
        self.meta_head = nn.Sequential(
            nn.Linear(META_DIM, meta_hidden), Swish())
        fin = c2 + img_hidden + meta_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fin, h1), nn.BatchNorm1d(h1), Swish(), nn.Dropout(drop),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), Swish(), nn.Dropout(drop / 2),
            nn.Linear(h2, 1))

    def forward(self, fp, imgf, met):
        c = self.conv(fp.view(fp.size(0), 1, -1))
        i = self.img_head(imgf)
        m = self.meta_head(met)
        return self.fusion(torch.cat([c, i, m], 1)).squeeze(1)


def _tensor(idx, device):

    return (torch.tensor(X_num[idx], dtype=torch.float32),
            torch.tensor(img_feats[idx], dtype=torch.float32),
            torch.tensor(meta[idx], dtype=torch.float32),
            torch.tensor(y_all[idx], dtype=torch.float32))


def make_loaders(tr_idx, val_idx, te_idx, batch=128):
    Xn, Xi, Xm, yb = _tensor(tr_idx, device)
    dl = DataLoader(TensorDataset(Xn, Xi, Xm, yb), batch_size=batch, shuffle=True)

    vn, vi, vm, vy = _tensor(val_idx, device)
    tn, ti, tm, ty = _tensor(te_idx, device)
    return dl, (vn, vi, vm, vy), (tn, ti, tm, ty)


def _eval_on(model, n, i, m, batch=256):

    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(n), batch):
            e = min(s + batch, len(n))
            xb = n[s:e].to(device); ib = i[s:e].to(device); mb = m[s:e].to(device)
            preds.append(model(xb, ib, mb).cpu())
    return torch.cat(preds).numpy()


def train_fold(tr_idx, te_idx, params, quick=False):

    rng = np.random.RandomState(42)
    perm = rng.permutation(tr_idx)
    nv = max(1, int(len(perm) * 0.1))
    val_idx, tr2 = perm[:nv], perm[nv:]
    dl, (vn, vi, vm, vy), (tn, ti, tm, ty) = make_loaders(tr2, val_idx, te_idx)
    model = DCNN_MultiModal(params["c1"], params["c2"], params["k"],
                           params["h1"], params["h2"], params["drop"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"],
                           betas=(0.99, 0.999), weight_decay=params["l2"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    crit = nn.MSELoss()
    epochs = 30 if quick else 100
    best_vl, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        for xn, xi, xm, yb in dl:
            xn, xi, xm, yb = xn.to(device), xi.to(device), xm.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xn, xi, xm), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vl = torch.nn.functional.mse_loss(
            torch.tensor(_eval_on(model, vn, vi, vm)), vy).item()
        sch.step(vl)
        if vl < best_vl:
            best_vl = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    pred = _eval_on(model, tn, ti, tm)
    return float(r2_score(ty.numpy(), pred)), pred, ty.numpy()


def quick_r2(params):

    kf3 = KFold(n_splits=3, shuffle=True, random_state=42)
    r2s = []
    for tr, te in kf3.split(X_num):
        r2, _, _ = train_fold(tr, te, params, quick=True)
        r2s.append(r2)
    return float(np.mean(r2s))


def objective(trial):
    p = {
        "c1": trial.suggest_int("c1", 32, 128, step=32),
        "c2": trial.suggest_int("c2", 64, 128, step=32),
        "k": trial.suggest_categorical("k", [3, 5, 7]),
        "h1": trial.suggest_int("h1", 128, 256, step=64),
        "h2": trial.suggest_int("h2", 64, 128, step=32),
        "drop": trial.suggest_float("drop", 0.1, 0.4),
        "lr": trial.suggest_float("lr", 5e-4, 2e-3, log=True),
        "l2": trial.suggest_float("l2", 1e-4, 1e-3, log=True),
    }
    r2 = quick_r2(p)
    print(f"    trial {trial.number+1}: R2={r2:.4f}  c1={p['c1']} c2={p['c2']} k={p['k']}", flush=True)
    return r2


print("\n  [DCNN] Bayesian HPO (Optuna TPE, 20 trials)...")
t0 = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20, show_progress_bar=False)
best = study.best_params
best = {"c1": best["c1"], "c2": best["c2"], "k": best["k"], "h1": best["h1"],
        "h2": best["h2"], "drop": best["drop"], "lr": best["lr"], "l2": best["l2"]}
print(f"  [DCNN] Bayesian best R2={study.best_value:.4f}  ({time.time()-t0:.0f}s)")
print(f"         params: {best}")


print(f"  [DCNN] local search refinement...")
cur_r2 = quick_r2(best)
ls_space = {
    "c1": [best["c1"] - 32, best["c1"] + 32],
    "c2": [best["c2"] - 32, best["c2"] + 32],
    "k": [3, 5, 7],
    "drop": [best["drop"] - 0.1, best["drop"] + 0.1],
}
for key, vals in ls_space.items():
    for v in vals:
        cand = dict(best)
        cand[key] = v
        if cand["c1"] < 32 or cand["c2"] < 32 or cand["drop"] < 0.05 or cand["drop"] > 0.6:
            continue
        try:
            r2 = quick_r2(cand)
            if r2 > cur_r2:
                cur_r2, best = r2, cand
        except Exception:
            continue
print(f"  [DCNN] final params: {best}")


print(f"  [DCNN] final 10-fold CV...")
folds = []
r2s = []
for i, (tr, te) in enumerate(kf.split(X_num)):
    r2, pred, yte = train_fold(tr, te, best, quick=False)
    pcl = per_cell_line_r2(yte, pred, cell_labels[te])
    folds.append({"fold": i + 1, "r2": round(r2, 4), "per_cell_line": pcl})
    r2s.append(r2)
    print(f"    fold{i+1}: R2={r2:.4f}")

avg, std = round(float(np.mean(r2s)), 4), round(float(np.std(r2s)), 4)
print(f"\n  [DCNN] 10-fold R2={avg:.4f} +/- {std:.4f}  vs paper(single-modal)={PAPER_R2}")

res = {"model": "DCNN (Paper1 multimodal, Bayesian HPO)",
       "paper": "Paper1: Conv1D(fp)+ResNet(img)+Meta + Swish + Adam(0.999)+L2 + 10fold CV + Optuna Bayesian",
       "paper_published_r2_single_modal": PAPER_R2,
       "avg_r2": avg, "std_r2": std, "best_params": best,
       "n_valid": len(folds), "n_total": len(folds), "folds": folds}
with open("results/dcnn_metrics.json", "w") as f:
    json.dump(res, f, indent=2)
print("  [DCNN] saved results/dcnn_metrics.json")
print("\n" + "=" * 64)
print("Paper1 complete!")
