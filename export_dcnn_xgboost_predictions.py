import os, json, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb
from data_utils import load_data

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
os.makedirs("results/predictions", exist_ok=True)
torch.backends.cudnn.benchmark = True
np.random.seed(42); torch.manual_seed(42)

D = load_data()
rd = np.load("data/rdkit_descriptors.npz", allow_pickle=True)
rdkit_desc = rd["descriptors_norm"].astype(np.float32)
X_num, img_feats, Mw, cell_oh = D["X_num"], D["img_feats"], D["Mw"], D["cell_oh"]
y_all, cell_labels = D["y"], D["cell_labels"]
N = len(y_all)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"N={N} device={device}")


meta = np.concatenate([Mw, cell_oh, rdkit_desc], axis=1).astype(np.float32)

all_feats = np.concatenate([X_num, img_feats, Mw, cell_oh, rdkit_desc], axis=1).astype(np.float32)


class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class DCNN_MultiModal(nn.Module):
    def __init__(self, c1, c2, k, h1, h2, drop, img_hidden=128, meta_hidden=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, k, padding=k//2), nn.BatchNorm1d(c1), Swish(), nn.Dropout(drop), nn.AvgPool1d(2),
            nn.Conv1d(c1, c2, 5, padding=2), nn.BatchNorm1d(c2), Swish(), nn.Dropout(drop), nn.AvgPool1d(2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.img_head = nn.Sequential(nn.Linear(512, img_hidden), nn.BatchNorm1d(img_hidden), Swish(), nn.Dropout(drop))
        self.meta_head = nn.Sequential(nn.Linear(30, meta_hidden), Swish())
        self.fusion = nn.Sequential(
            nn.Linear(c2+img_hidden+meta_hidden, h1), nn.BatchNorm1d(h1), Swish(), nn.Dropout(drop),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), Swish(), nn.Dropout(drop/2), nn.Linear(h2, 1))
    def forward(self, fp, imgf, met):
        c = self.conv(fp.view(fp.size(0),1,-1)); i = self.img_head(imgf); m = self.meta_head(met)
        return self.fusion(torch.cat([c,i,m],1)).squeeze(1)


def _eval_on(model, n, i, m, batch=256):
    model.eval(); preds=[]
    with torch.no_grad():
        for s in range(0,len(n),batch):
            e=min(s+batch,len(n))
            preds.append(model(n[s:e].to(device), i[s:e].to(device), m[s:e].to(device)).cpu())
    return torch.cat(preds).numpy()


def dcnn_train_fold(tr_idx, te_idx, p):

    rng = np.random.RandomState(42); perm = rng.permutation(tr_idx)
    nv = max(1, int(len(perm)*0.1)); val_idx, tr2 = perm[:nv], perm[nv:]
    Xn=torch.tensor(X_num[tr2],dtype=torch.float32); Xi=torch.tensor(img_feats[tr2],dtype=torch.float32)
    Xm=torch.tensor(meta[tr2],dtype=torch.float32); yb=torch.tensor(y_all[tr2],dtype=torch.float32)
    dl=DataLoader(TensorDataset(Xn,Xi,Xm,yb),batch_size=128,shuffle=True)
    vn=torch.tensor(X_num[val_idx],dtype=torch.float32); vi=torch.tensor(img_feats[val_idx],dtype=torch.float32)
    vm=torch.tensor(meta[val_idx],dtype=torch.float32); vy=torch.tensor(y_all[val_idx],dtype=torch.float32)
    tn=torch.tensor(X_num[te_idx],dtype=torch.float32); ti=torch.tensor(img_feats[te_idx],dtype=torch.float32)
    tm=torch.tensor(meta[te_idx],dtype=torch.float32); ty=y_all[te_idx]
    model=DCNN_MultiModal(p["c1"],p["c2"],p["k"],p["h1"],p["h2"],p["drop"]).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=p["lr"],betas=(0.99,0.999),weight_decay=p["l2"])
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=8,factor=0.5); crit=nn.MSELoss()
    best_vl,best_state=float("inf"),None
    for ep in range(100):
        model.train()
        for xn,xi,xm,yb in dl:
            xn,xi,xm,yb=xn.to(device),xi.to(device),xm.to(device),yb.to(device)
            opt.zero_grad(); crit(model(xn,xi,xm),yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        vl=torch.nn.functional.mse_loss(torch.tensor(_eval_on(model,vn,vi,vm)),vy).item(); sch.step(vl)
        if vl<best_vl: best_vl=vl; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    pred=_eval_on(model,tn,ti,tm)
    return float(r2_score(ty,pred)), pred, ty


print("\n[DCNN] 10-fold OOF...")
dcnn_params = json.load(open("results/dcnn_metrics.json",encoding="utf-8"))["best_params"]
t0=time.time()
y_pred_dcnn=np.full(N,np.nan); r2s=[]
kf10=KFold(n_splits=10,shuffle=True,random_state=42)
for i,(tr,te) in enumerate(kf10.split(X_num)):
    r2,pred,ty=dcnn_train_fold(tr,te,dcnn_params)
    y_pred_dcnn[te]=pred; r2s.append(r2)
    print(f"  DCNN fold{i+1}: R2={r2:.4f} ({time.time()-t0:.0f}s)", flush=True)
print(f"DCNN OOF R2={np.mean(r2s):.4f}")
np.savez("results/predictions/dcnn_oof_predictions.npz", y_true=y_all, y_pred_oof=y_pred_dcnn,
         cell_labels=cell_labels, overall_r2=float(np.mean(r2s)))


print("\n[XGBoost] 5-fold OOF...")
xgb_params = json.load(open("results/xgboost_metrics.json",encoding="utf-8"))["best_params"]
y_pred_xgb=np.full(N,np.nan); r2s=[]
kf5=KFold(n_splits=5,shuffle=True,random_state=42)
t0=time.time()
for i,(tr,te) in enumerate(kf5.split(all_feats)):
    model=xgb.XGBRegressor(n_estimators=xgb_params["n_estimators"],max_depth=xgb_params["max_depth"],
        learning_rate=xgb_params["learning_rate"],reg_alpha=xgb_params["reg_alpha"],
        tree_method="hist",n_jobs=-1,random_state=42)
    model.fit(all_feats[tr],y_all[tr])
    pred=model.predict(all_feats[te])
    y_pred_xgb[te]=pred; r2s.append(r2_score(y_all[te],pred))
    print(f"  fold{i+1}: R2={r2s[-1]:.4f} ({time.time()-t0:.0f}s)")
print(f"XGBoost OOF R2={np.mean(r2s):.4f}")
np.savez("results/predictions/xgboost_oof_predictions.npz", y_true=y_all, y_pred_oof=y_pred_xgb,
         cell_labels=cell_labels, overall_r2=float(np.mean(r2s)))
print("\nDONE")
