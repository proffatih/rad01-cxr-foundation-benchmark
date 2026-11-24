"""
common.py — shared utilities.

* NIH ChestX-ray14 label table + official patient-disjoint split
  (test_list.txt, 25,596 images) + a fixed-seed patient-level train/val partition
  of train_val_list.txt (seed 20260531; the resulting split is in data/nih_split.csv).
* Embedding loader (shards written by extract_full.py).
* GPU weighted L2 logistic regression, equivalent to scikit-learn's
  LogisticRegression(C, class_weight) objective  0.5||w||^2 + C * sum_i s_i * logloss_i
  (intercept not penalised), solved per label with full-batch L-BFGS. The 14 labels
  are independent problems solved in one batched call.
* Metrics: AUROC (rank-based, ties averaged), AUPRC, ECE (equal-width and
  equal-mass), Brier, operating points.

Paths: RUN_OUT (large intermediate files: dataset cache, embeddings, probe outputs);
results/ (the JSON result files that every table, figure and number is built from).
"""
from __future__ import annotations
import os, glob, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RUN = Path(os.environ.get("RUN_OUT", "run"))
RESULTS = Path(__file__).resolve().parent.parent / "results"
PREDS = Path(os.environ.get("PREDS", Path(__file__).resolve().parent.parent / "preds"))  # probe outputs (main.py)
_snap = glob.glob(str(RUN / "hf_cache/datasets--alkzar90--NIH-Chest-X-ray-dataset/snapshots/*/data"))
SNAP = Path(os.environ.get("NIH_DATA", _snap[0] if _snap else str(RUN / "nih_data")))
SEED = 20260531
DEV = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
          "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
          "Pleural_Thickening", "Hernia"]

MAIN = ["rad_dino_518", "biomedclip_224", "dinov2_small_224", "clip_b16_224",
        "resnet50_224", "vit_b16_224"]
CONTROLS = ["rad_dino_224", "dinov2_base_224", "dinov2_base_518"]
ALL = MAIN + CONTROLS
PRETTY = {"rad_dino_518": "RAD-DINO", "biomedclip_224": "BiomedCLIP",
          "dinov2_small_224": "DINOv2-S", "clip_b16_224": "CLIP-B/16",
          "resnet50_224": "ResNet-50", "vit_b16_224": "ViT-B/16",
          "rad_dino_224": "RAD-DINO@224", "dinov2_base_224": "DINOv2-B",
          "dinov2_base_518": "DINOv2-B@518"}


# --------------------------------------------------------------------------- data
SPLIT_CSV = Path(__file__).resolve().parent.parent / "data" / "nih_split.csv"


def label_table() -> pd.DataFrame:
    """NIH labels + split. Built from the NIH metadata when available (and checked against
    data/nih_split.csv); otherwise read from data/nih_split.csv, so tables and figures can
    be rebuilt from results/ without downloading the images."""
    if not (SNAP / "Data_Entry_2017_v2020.csv").exists():
        return pd.read_csv(SPLIT_CSV, dtype={l: np.int8 for l in LABELS + ["abnormal"]})
    df = pd.read_csv(SNAP / "Data_Entry_2017_v2020.csv")
    df = df.rename(columns={"Image Index": "name", "Finding Labels": "findings",
                            "Patient ID": "pid", "Patient Age": "age",
                            "Patient Gender": "sex", "View Position": "view"})
    for l in LABELS:
        df[l] = df["findings"].str.split("|").map(lambda t, l=l: int(l in t)).astype(np.int8)
    df["abnormal"] = (df[LABELS].sum(1) > 0).astype(np.int8)       # = not "No Finding"
    test = set(open(SNAP / "test_list.txt").read().split())
    df["split"] = np.where(df["name"].isin(test), "test", "trainval")
    # fixed-seed patient-level 90/10 partition of train_val into train/val
    rng = np.random.default_rng(SEED)
    tv = np.array(sorted(df.loc[df.split == "trainval", "pid"].unique()))
    rng.shuffle(tv)
    val_p = set(tv[: int(round(0.10 * len(tv)))])
    df.loc[(df.split == "trainval") & df.pid.isin(val_p), "split"] = "val"
    df.loc[df.split == "trainval", "split"] = "train"
    assert not (set(df.pid[df.split == "test"]) & set(df.pid[df.split != "test"]))
    assert not (set(df.pid[df.split == "val"]) & set(df.pid[df.split == "train"]))
    return df.reset_index(drop=True)


def load_emb(cfg: str, names: np.ndarray) -> np.ndarray:
    """Return float32 L2-normalised embeddings aligned with `names`."""
    Xs, Ns = [], []
    for f in sorted(glob.glob(str(RUN / "emb" / cfg / "images_*.npz"))):
        d = np.load(f)
        Xs.append(d["X"]); Ns.append(d["names"])
    X = np.concatenate(Xs).astype(np.float32); N = np.concatenate(Ns)
    pos = pd.Series(np.arange(len(N)), index=N)
    X = X[pos.loc[names].values]
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


# --------------------------------------------------------------------------- probe
def fit_lr(X: np.ndarray, Y: np.ndarray, C: float, balanced: bool = True,
           max_iter: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Independent per-column weighted L2 logistic regressions.
    Objective per label k:  0.5||w_k||^2 + C * sum_i s_ik * logloss_ik ,
    s_ik = n / (2 n_c) for class c of sample i (sklearn 'balanced'), else 1.
    Returns W [D,K], b [K]."""
    dev = DEV if len(X) >= 20000 else "cpu"     # small problems: CPU avoids GPU launch latency
    Xt = torch.as_tensor(X, device=dev, dtype=torch.float32)
    Yt = torch.as_tensor(Y, device=dev, dtype=torch.float32)
    n, d = Xt.shape; K = Yt.shape[1]
    if balanced:
        npos = Yt.sum(0).clamp(min=1); nneg = (n - Yt.sum(0)).clamp(min=1)
        S = torch.where(Yt > 0, n / (2 * npos), n / (2 * nneg))
    else:
        S = torch.ones_like(Yt)
    W = torch.zeros(d, K, device=dev, requires_grad=True)
    p0 = (Yt.mean(0).clamp(1e-4, 1 - 1e-4))
    b = torch.log(p0 / (1 - p0)).detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, tolerance_grad=1e-7,
                            tolerance_change=1e-10, history_size=20,
                            line_search_fn="strong_wolfe")
    scale = 1.0 / n   # divide the whole objective by n for conditioning

    def closure():
        opt.zero_grad()
        z = Xt @ W + b
        ll = torch.nn.functional.binary_cross_entropy_with_logits(z, Yt, reduction="none")
        loss = scale * (0.5 * (W * W).sum() + C * (S * ll).sum())
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def predict_logit(X, W, b):
    return X @ W + b


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------- metrics
def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_matrix(Y: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.array([auroc(Y[:, k], S[:, k]) for k in range(Y.shape[1])])


def auprc(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def ece(y, p, bins=15, mode="width"):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if mode == "width":
        edges = np.linspace(0, 1, bins + 1)
    else:  # equal mass
        edges = np.quantile(p, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = 0, 1
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    e = 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def threshold_for_sensitivity(y, s, target):
    """Largest threshold whose sensitivity on (y, s) is >= target."""
    pos = np.sort(s[y.astype(bool)])
    if len(pos) == 0:
        return np.nan
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[k])


def op_metrics(y, s, thr):
    y = y.astype(bool); pred = s >= thr
    tp = (pred & y).sum(); fn = (~pred & y).sum(); tn = (~pred & ~y).sum(); fp = (pred & ~y).sum()
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    npv = tn / max(tn + fn, 1)
    return dict(sens=float(sens), spec=float(spec), fnr=float(1 - sens), npv=float(npv),
                frac_neg=float((~pred).mean()), tp=int(tp), fn=int(fn), tn=int(tn), fp=int(fp))


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
