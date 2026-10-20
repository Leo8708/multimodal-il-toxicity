import os, json, time, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import optuna

from data_utils import load_data, per_cell_line_r2

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
os.makedirs("results", exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Paper2-DESC | RF + XGBoost + RDKit descriptors | 5-fold CV | Bayesian HPO")
print("=" * 64)

D = load_data()
X_num, img_feats, Mw, cell_oh = D["X_num"], D["img_feats"], D["Mw"], D["cell_oh"]
y, cell_labels = D["y"], D["cell_labels"]

rd = np.load("data/rdkit_descriptors.npz", allow_pickle=True)
rdkit_desc = rd["descriptors_norm"].astype(np.float32)
print(f"  RDKit descriptors: {rdkit_desc.shape}")
all_feats = np.concatenate([X_num, img_feats, Mw, cell_oh, rdkit_desc], axis=1).astype(np.float32)
N, F = all_feats.shape
print(f"  samples: {N}  combined features: {F}")

scaler = StandardScaler()
all_s = scaler.fit_transform(all_feats)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

PAPER_R2 = 0.957


def cv_eval(model_factory, all_s_arr, y_arr, c_labels, kf_obj, n_jobs=1):

    folds = []
    r2s = []
    for i, (tr, te) in enumerate(kf_obj.split(all_s_arr)):
        m = model_factory()
        m.fit(all_s_arr[tr], y_arr[tr])
        pred = m.predict(all_s_arr[te])
        r2 = float(r2_score(y_arr[te], pred))
        mae = float(mean_absolute_error(y_arr[te], pred))
        rmse = float(np.sqrt(mean_squared_error(y_arr[te], pred)))
        pcl = per_cell_line_r2(y_arr[te], pred, c_labels[te])
        folds.append({"fold": i + 1, "r2": round(r2, 4), "mae": round(mae, 4),
                      "rmse": round(rmse, 4), "per_cell_line": pcl})
        r2s.append(r2)
    return folds, round(float(np.mean(r2s)), 4), round(float(np.std(r2s)), 4)


def _quick_r2(model_factory, all_s_arr, y_arr, c_labels):

    kf3 = KFold(n_splits=3, shuffle=True, random_state=42)
    r2s = []
    for tr, te in kf3.split(all_s_arr):
        m = model_factory()
        m.fit(all_s_arr[tr], y_arr[tr])
        r2s.append(float(r2_score(y_arr[te], m.predict(all_s_arr[te]))))
    return float(np.mean(r2s))


def local_search_rf(best):

    space = {
        "n_estimators": [best["n_estimators"] - 100, best["n_estimators"], best["n_estimators"] + 100],
        "max_depth": [best["max_depth"] - 2, best["max_depth"], best["max_depth"] + 2],
        "min_samples_leaf": [max(1, best["min_samples_leaf"] - 2), best["min_samples_leaf"], best["min_samples_leaf"] + 2],
    }
    cur = dict(best)
    cur_r2 = _quick_r2(lambda c=cur: RandomForestRegressor(
        n_estimators=cur["n_estimators"], max_depth=cur["max_depth"],
        min_samples_leaf=cur["min_samples_leaf"], max_features="sqrt",
        n_jobs=-1, random_state=42), all_s, y, cell_labels)
    for key, vals in space.items():
        for v in vals:
            if v is None or (key == "max_depth" and v < 4) or (key == "n_estimators" and v < 100) or (key == "min_samples_leaf" and v < 1):
                continue
            cand = dict(cur)
            cand[key] = v
            try:
                r2 = _quick_r2(lambda c=cand: RandomForestRegressor(
                    n_estimators=c["n_estimators"], max_depth=c["max_depth"],
                    min_samples_leaf=c["min_samples_leaf"], max_features="sqrt",
                    n_jobs=-1, random_state=42), all_s, y, cell_labels)
                if r2 > cur_r2:
                    cur_r2, cur = r2, cand
            except Exception:
                continue
    return cur, cur_r2


def local_search_xgb(best):
    space = {
        "n_estimators": [best["n_estimators"] - 100, best["n_estimators"], best["n_estimators"] + 100],
        "max_depth": [best["max_depth"] - 1, best["max_depth"], best["max_depth"] + 1],
        "learning_rate": [best["learning_rate"] * 0.7, best["learning_rate"], best["learning_rate"] * 1.3],
        "reg_alpha": [best["reg_alpha"] * 0.5, best["reg_alpha"], best["reg_alpha"] * 1.5],
    }
    cur = dict(best)
    cur_r2 = _quick_r2(lambda c=cur: _make_xgb(cur), all_s, y, cell_labels)
    for key, vals in space.items():
        for v in vals:
            if v is None or (key == "max_depth" and v < 3) or (key == "n_estimators" and v < 100) or (key == "learning_rate" and v <= 0):
                continue
            cand = dict(cur)
            cand[key] = v
            try:
                r2 = _quick_r2(lambda c=cand: _make_xgb(c), all_s, y, cell_labels)
                if r2 > cur_r2:
                    cur_r2, cur = r2, cand
            except Exception:
                continue
    return cur, cur_r2


def _make_xgb(c):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=c["n_estimators"], max_depth=c["max_depth"],
                       learning_rate=c["learning_rate"], reg_alpha=c["reg_alpha"],
                       reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
                       random_state=42, n_jobs=-1, eval_metric="rmse", verbosity=0)


def rf_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 6, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }
    return _quick_r2(lambda p=params: RandomForestRegressor(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"], max_features="sqrt",
        n_jobs=-1, random_state=42), all_s, y, cell_labels)


print("\n  [RF] Bayesian HPO (Optuna TPE, 40 trials)...")
t0 = time.time()
study_rf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_rf.optimize(rf_objective, n_trials=40, show_progress_bar=False)
rf_best = study_rf.best_params
print(f"  [RF] Bayesian best R2={study_rf.best_value:.4f}  ({time.time()-t0:.0f}s)")
print(f"  [RF] local search refinement...")
rf_best, rf_ls_r2 = local_search_rf(rf_best)
print(f"  [RF] final params: {rf_best}")
rf_folds, rf_avg, rf_std = cv_eval(lambda: RandomForestRegressor(
    n_estimators=rf_best["n_estimators"], max_depth=rf_best["max_depth"],
    min_samples_leaf=rf_best["min_samples_leaf"], max_features="sqrt",
    n_jobs=-1, random_state=42), all_s, y, cell_labels, kf)
print(f"  [RF] 5-fold R2={rf_avg:.4f} +/- {rf_std:.4f}  vs paper(single-modal)={PAPER_R2}")

with open("results/random_forest_metrics.json", "w") as f:
    json.dump({"model": "RF (Paper2 multimodal, Bayesian HPO)",
               "paper": "Paper2: RF on multimodal feats + Optuna Bayesian + 5fold CV",
               "paper_published_r2_single_modal": PAPER_R2,
               "avg_r2": rf_avg, "std_r2": rf_std, "best_params": rf_best,
               "n_valid": len(rf_folds), "n_total": len(rf_folds), "folds": rf_folds}, f, indent=2)
print("  [RF] saved results/random_forest_metrics.json")


def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
    }
    return _quick_r2(lambda p=params: _make_xgb(p), all_s, y, cell_labels)


print("\n  [XGB] Bayesian HPO (Optuna TPE, 40 trials)...")
t0 = time.time()
study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(xgb_objective, n_trials=40, show_progress_bar=False)
xgb_best = study_xgb.best_params
print(f"  [XGB] Bayesian best R2={study_xgb.best_value:.4f}  ({time.time()-t0:.0f}s)")
print(f"  [XGB] local search refinement...")
xgb_best, xgb_ls_r2 = local_search_xgb(xgb_best)
print(f"  [XGB] final params: {xgb_best}")
xgb_folds, xgb_avg, xgb_std = cv_eval(lambda: _make_xgb(xgb_best), all_s, y, cell_labels, kf)
print(f"  [XGB] 5-fold R2={xgb_avg:.4f} +/- {xgb_std:.4f}  vs paper(single-modal)={PAPER_R2}")

with open("results/xgboost_metrics.json", "w") as f:
    json.dump({"model": "XGBoost (Paper2 multimodal, Bayesian HPO)",
               "paper": "Paper2: XGBoost on multimodal feats + Optuna Bayesian + 5fold CV",
               "paper_published_r2_single_modal": PAPER_R2,
               "avg_r2": xgb_avg, "std_r2": xgb_std, "best_params": xgb_best,
               "n_valid": len(xgb_folds), "n_total": len(xgb_folds), "folds": xgb_folds}, f, indent=2)
print("  [XGB] saved results/xgboost_metrics.json")

print("\n" + "=" * 64)
print("Paper2 complete!")
