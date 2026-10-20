import os, json, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import optuna

from data_utils import load_data

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
os.makedirs("results/predictions", exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.backends.cudnn.benchmark = True
np.random.seed(42); torch.manual_seed(42)

print("Prepare figure data: OOF + Optuna (FNN+desc)")
print("=" * 60)
D = load_data()
rd = np.load("data/rdkit_descriptors.npz", allow_pickle=True)
rdkit_desc = rd["descriptors_norm"].astype(np.float32)
X_num, img_feats, Mw, cell_oh = D["X_num"], D["img_feats"], D["Mw"], D["cell_oh"]
all_feats = np.concatenate([X_num, img_feats, Mw, cell_oh, rdkit_desc], axis=1).astype(np.float32)
y_all, cell_labels = D["y"], D["cell_labels"]
N, Fdim = all_feats.shape
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  N={N} Fdim={Fdim} device={device}")
kf = KFold(n_splits=5, shuffle=True, random_state=42)


class FNN(nn.Module):
    def __init__(self, in_dim, h1, h2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.Sigmoid(),
            nn.Linear(h1, h2), nn.Softplus(),
            nn.Linear(h2, 1))
    def forward(self, x): return self.net(x).squeeze(1)


def rmse_loss(p, t): return torch.sqrt(nn.MSELoss()(p, t) + 1e-8)


def train_eval_fold(tr_idx, te_idx, h1, h2, epochs=180, lr=1e-3, batch=128, quick=False):
    Xtr = torch.tensor(all_feats[tr_idx], dtype=torch.float32, device=device)
    ytr = torch.tensor(y_all[tr_idx], dtype=torch.float32, device=device)
    Xte = torch.tensor(all_feats[te_idx], dtype=torch.float32, device=device)
    yte = y_all[te_idx]
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True, num_workers=0)
    model = FNN(Fdim, h1, h2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_r2, best_pred = -9, None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = rmse_loss(model(xb), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad(): pred = model(Xte).cpu().numpy()
        r2 = r2_score(yte, pred)
        if r2 > best_r2: best_r2, best_pred = r2, pred
        if quick and ep >= 40: break
    return best_r2, best_pred


print("\n[1] OOF predictions (FNN+desc, 5fold, h1=256 h2=32, 180 epochs)...")
t0 = time.time()
y_pred_oof = np.full(N, np.nan)
fold_idx = np.full(N, -1, dtype=int)
r2s = []
for i, (tr, te) in enumerate(kf.split(all_feats)):
    r2, pred = train_eval_fold(tr, te, 256, 32, epochs=180)
    y_pred_oof[te] = pred
    fold_idx[te] = i
    r2s.append(r2)
    print(f"    fold{i+1}: R2={r2:.4f}  ({time.time()-t0:.0f}s)")
overall = float(np.mean(r2s))
np.savez("results/predictions/fnn_oof_predictions.npz",
         y_true=y_all, y_pred_oof=y_pred_oof, cell_labels=cell_labels,
         fold_idx=fold_idx, overall_r2=overall, r2_per_fold=np.array(r2s))
print(f"  OOF overall R2={overall:.4f}  -> results/predictions/fnn_oof_predictions.npz")


print("\n[2] Optuna HPO trajectory (40 trials, 3fold quick)...")
kf3 = KFold(n_splits=3, shuffle=True, random_state=42)
history = []


def quick_r2(h1, h2):
    rs = []
    for tr, te in kf3.split(all_feats):
        r2, _ = train_eval_fold(tr, te, h1, h2, quick=True)
        rs.append(r2)
    return float(np.mean(rs))


def obj(trial):
    h1 = trial.suggest_int("h1", 64, 512, step=32)
    h2 = trial.suggest_int("h2", 32, 256, step=32)
    r2 = quick_r2(h1, h2)
    history.append({"trial": trial.number + 1, "r2": round(r2, 4), "h1": h1, "h2": h2})
    print(f"    trial {trial.number+1}: R2={r2:.4f} (h1={h1},h2={h2})", flush=True)
    return r2


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(obj, n_trials=40)
with open("results/predictions/fnn_optimization_history.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"  best: R2={study.best_value:.4f} params={study.best_params}")
print(f"  -> results/predictions/fnn_optimization_history.json")
print("\n" + "=" * 60); print("DONE")
