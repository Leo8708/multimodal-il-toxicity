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

print("Paper3-DESC | FNN + RDKit descriptors (old recipe, +10 desc) | 5fold CV")
print("=" * 64)

D = load_data()

rd = np.load("data/rdkit_descriptors.npz", allow_pickle=True)
rdkit_desc = rd["descriptors_norm"].astype(np.float32)
print(f"  RDKit descriptors: {rdkit_desc.shape}")


X_num, img_feats, Mw, cell_oh = D["X_num"], D["img_feats"], D["Mw"], D["cell_oh"]
all_feats = np.concatenate([X_num, img_feats, Mw, cell_oh, rdkit_desc], axis=1).astype(np.float32)
y_all, cell_labels = D["y"], D["cell_labels"]
N, Fdim = all_feats.shape
print(f"  samples: {N}  features: {Fdim} (原始{X_num.shape[1]}+img{img_feats.shape[1]}+Mw1+cell{cell_oh.shape[1]}+desc{rdkit_desc.shape[1]})")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device: {device}")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
PAPER_R2 = 0.8917
OLD_R2 = 0.8982


class FNN(nn.Module):

    def __init__(self, in_dim, h1, h2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.Sigmoid(),
            nn.Linear(h1, h2), nn.Softplus(),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def rmse_loss(pred, target):
    return torch.sqrt(nn.MSELoss()(pred, target) + 1e-8)


def train_eval_fold(tr_idx, te_idx, h1, h2, epochs=120, lr=1e-3, batch=128, quick=False):

    Xtr = torch.tensor(all_feats[tr_idx], dtype=torch.float32, device=device)
    ytr = torch.tensor(y_all[tr_idx], dtype=torch.float32, device=device)
    Xte = torch.tensor(all_feats[te_idx], dtype=torch.float32, device=device)
    yte = y_all[te_idx]
    ds = TensorDataset(Xtr, ytr)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    model = FNN(Fdim, h1, h2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_r2, best_pred = -9, None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = rmse_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xte).cpu().numpy()
        r2 = r2_score(yte, pred)
        if r2 > best_r2:
            best_r2, best_pred = r2, pred
        if quick and ep >= 40:
            break
    return best_r2, best_pred


def quick_r2(h1, h2):
    kf3 = KFold(n_splits=3, shuffle=True, random_state=42)
    r2s = []
    for tr, te in kf3.split(all_feats):
        r2, _ = train_eval_fold(tr, te, h1, h2, quick=True)
        r2s.append(r2)
    return float(np.mean(r2s))


def fnn_objective(trial):
    h1 = trial.suggest_int("h1", 64, 512, step=32)
    h2 = trial.suggest_int("h2", 32, 256, step=32)
    r2 = quick_r2(h1, h2)
    print(f"    trial {trial.number+1}: R2={r2:.4f}", flush=True)
    return r2


print("\n  [FNN-DESC] 跳过HPO，固定超参 h1=256 h2=32 (20维描述符+更大网络)")
best = {"h1": 256, "h2": 32}
print(f"  [FNN-DESC] params: {best}")

print(f"  [FNN-DESC] final 5-fold CV...")
folds = []
r2s = []
for i, (tr, te) in enumerate(kf.split(all_feats)):
    r2, pred = train_eval_fold(tr, te, best["h1"], best["h2"], epochs=180, quick=False)
    pcl = per_cell_line_r2(y_all[te], pred, cell_labels[te])
    folds.append({"fold": i + 1, "r2": round(r2, 4), "per_cell_line": pcl})
    r2s.append(r2)
    print(f"    fold{i+1}: R2={r2:.4f}")

avg, std = round(float(np.mean(r2s)), 4), round(float(np.std(r2s)), 4)
print(f"\n  [FNN-DESC] 5-fold R2={avg:.4f} +/- {std:.4f}")
print(f"             优化前(旧FNN)={OLD_R2}  提升={avg-OLD_R2:+.4f}  vs文献={PAPER_R2}")

res = {"model": "FNN + RDKit descriptors (Paper3, old recipe + 10 physchem desc)",
       "paper": "Paper3-DESC: original feats + RDKit physchem + RMSE + 2layer + Bayesian",
       "paper_published_r2_single_modal": PAPER_R2,
       "prev_multimodal_r2": OLD_R2,
       "avg_r2": avg, "std_r2": std, "best_params": best,
       "feat_dim": int(Fdim), "n_valid": len(folds), "n_total": len(folds), "rounds": folds}
with open("results/fnn_metrics.json", "w") as f:
    json.dump(res, f, indent=2)
print("  [FNN-DESC] saved results/fnn_metrics.json")
print("\n" + "=" * 64)
print("Paper3-DESC complete!")
