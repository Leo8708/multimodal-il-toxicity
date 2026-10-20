import os, numpy as np, torch
from torchvision import transforms, models
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(BASE, "data", "processed_data.npz")
FEAT = os.path.join(BASE, "data", "img_features_resnet18.npz")

_IMG_SIZE = 224
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def compute_image_features(img_paths, device, batch_size=128):

    tf = transforms.Compose([
        transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet = nn_sequential_no_fc(resnet).to(device).eval()
    for p in resnet.parameters():
        p.requires_grad = False

    n = len(img_paths)
    feats = np.zeros((n, 512), dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        batch = []
        for p in img_paths[s:e]:
            fp = p if os.path.isabs(p) else os.path.join(BASE, p)
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                img = Image.new("RGB", (_IMG_SIZE, _IMG_SIZE), (255, 255, 255))
            batch.append(tf(img))
        t = torch.stack(batch).to(device)
        out = resnet(t).cpu().squeeze(-1).squeeze(-1).numpy()
        feats[s:e] = out
        if (s // batch_size + 1) % 20 == 0:
            print(f"    features {e}/{n}")
    return feats


def nn_sequential_no_fc(resnet):
    import torch.nn as nn
    return nn.Sequential(*list(resnet.children())[:-1])


def clean_features(X_num, img_feats, Mw, cell_oh, variance_threshold=0.01, verbose=True):

    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler


    n0 = X_num.shape[1]
    sel = VarianceThreshold(threshold=variance_threshold)
    Xc = sel.fit_transform(X_num).astype(np.float32)
    keep_mask = sel.get_support()
    if verbose:
        print(f"  [clean] 指纹 {n0} -> {Xc.shape[1]} 维 (删掉 {n0-Xc.shape[1]} 维常数/低方差)")


    img_norm = StandardScaler().fit_transform(img_feats).astype(np.float32)
    Mw_norm = StandardScaler().fit_transform(Mw).astype(np.float32)

    return {"X_num_clean": Xc, "img_norm": img_norm, "Mw_norm": Mw_norm,
            "cell_oh": cell_oh.astype(np.float32), "keep_mask": keep_mask}


def load_data(device=None, use_cache=True):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    d = np.load(NPZ, allow_pickle=True)
    X_num = d["X_num"].astype(np.float32)
    Mw = d["Mw"].reshape(-1, 1).astype(np.float32) if d["Mw"].ndim == 1 else d["Mw"].astype(np.float32)
    cell_oh = d["cell_oh"].astype(np.float32)
    y = d["y"].astype(np.float32)
    cell_labels = d["cell_labels"]
    img_paths = d["img_paths"]

    if use_cache and os.path.exists(FEAT):
        img_feats = np.load(FEAT)["features"]
        print(f"  cached image features: {img_feats.shape}")
    else:
        print(f"  computing frozen ResNet-18 features for {len(img_paths)} images...")
        img_feats = compute_image_features(img_paths, device)
        np.savez(FEAT, features=img_feats)
        print(f"  saved -> {FEAT}")

    return {
        "X_num": X_num,
        "img_feats": img_feats,
        "Mw": Mw,
        "cell_oh": cell_oh,
        "y": y,
        "cell_labels": cell_labels,
        "train_idx": d["train_idx"],
        "val_idx": d["val_idx"],
        "test_idx": d["test_idx"],
    }


def per_cell_line_r2(y_true, y_pred, labels):

    from sklearn.metrics import r2_score
    pcl = {}
    for cl in np.unique(labels):
        m = labels == cl
        if m.sum() >= 3:
            try:
                pcl[str(cl)] = round(float(r2_score(y_true[m], y_pred[m])), 4)
            except Exception:
                pass
    return pcl


if __name__ == "__main__":
    load_data()
