"""
original_subset.py — small-sample sensitivity analysis (Table A1 of the article).

Evaluates three pipelines on one fixed 3,000-image subset of NIH ChestX-ray14 and its
2,000/500/500 split (data/original_subset/labels.csv):
  A  224x224-stored images, generic normalisation, standardised features + liblinear probe;
  B  full-resolution source, official normalisation, projected CLIP embedding, same probe as A;
  C  embeddings as in B with the probe used in this study (no feature standardisation,
     L-BFGS L2 logistic regression).
Pipeline-A embeddings are not distributed (set ORIG_EMB to their location to re-run);
the published Pipeline-A values are in data/original_subset/pipeline_a_reported.json.
Output: results/original_subset.json, results/decomposition.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import common as rc

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "original_subset")
ORIG = os.path.join(DATA, "labels.csv")
ORIG_MACRO = json.load(open(os.path.join(DATA, "pipeline_a_reported.json")))


def main():
    o = pd.read_csv(ORIG)
    o["name"] = o["filename"].str.split("_", n=1).str[1]
    Y = o[rc.LABELS].values.astype(np.float32)
    sp = {s: np.where(o.split.values == s)[0] for s in ("train", "val", "test")}
    out = {"original_reported": ORIG_MACRO, "corrected_on_original_subset": {}}
    for c in rc.MAIN:
        X = rc.load_emb(c, o["name"].values)
        fits, aucs = [], []
        for C in [0.01, 0.1, 1.0, 10.0]:
            W, b = rc.fit_lr(X[sp["train"]], Y[sp["train"]], C)
            fits.append((W, b)); aucs.append(rc.auroc_matrix(Y[sp["val"]], X[sp["val"]] @ W + b))
        A = np.where(np.isnan(np.array(aucs)), -1, np.array(aucs)); best = A.argmax(0)
        W = np.stack([fits[best[k]][0][:, k] for k in range(14)], 1); b = np.array([fits[best[k]][1][k] for k in range(14)])
        a = rc.auroc_matrix(Y[sp["test"]], X[sp["test"]] @ W + b)
        out["corrected_on_original_subset"][c] = float(np.nanmean(a))
        print(f"{rc.PRETTY[c]:11s} corrected-preprocessing macro AUROC on original 500-image test: {np.nanmean(a):.3f}", flush=True)
    rc.save_json(out, rc.RESULTS / "original_subset.json")


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def decompose():
    """2x2: {Pipeline-A embeddings, Pipeline-B embeddings} x {Pipeline-A probe, study probe} on the
    fixed 3,000-image subset. Pipeline-A probe = StandardScaler + liblinear balanced LR,
    C in {0.01,0.1,1,10} by val AUROC."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    o = pd.read_csv(ORIG); o["name"] = o["filename"].str.split("_", n=1).str[1]
    Y = o[rc.LABELS].values.astype(np.float32)
    sp = {s: np.where(o.split.values == s)[0] for s in ("train", "val", "test")}
    old_id = {"rad_dino_518": "rad_dino", "biomedclip_224": "biomedclip", "dinov2_small_224": "dinov2_small",
              "clip_b16_224": "clip_b16", "resnet50_224": "resnet50", "vit_b16_224": "vit_b16"}
    root = os.environ.get("ORIG_EMB", str(rc.RUN / "pipeline_a_emb"))

    def old_probe(Xtr, Ytr, Xva, Yva, Xte, Yte):
        sc = StandardScaler().fit(Xtr); a, b_, c_ = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
        res = []
        for k in range(14):
            if Ytr[:, k].sum() < 5 or Yva[:, k].sum() < 1 or Yte[:, k].sum() < 1:
                continue
            best, bs = None, -np.inf
            for C in [0.01, 0.1, 1.0, 10.0]:
                m = LogisticRegression(C=C, max_iter=2000, solver="liblinear", random_state=rc.SEED,
                                       class_weight="balanced").fit(a, Ytr[:, k])
                s = rc.auroc(Yva[:, k], m.decision_function(b_))
                if s > bs:
                    best, bs = m, s
            res.append(rc.auroc(Yte[:, k], best.decision_function(c_)))
        return float(np.mean(res))

    def new_probe(Xtr, Ytr, Xva, Yva, Xte, Yte):
        fits, aucs = [], []
        for C in [0.01, 0.1, 1.0, 10.0]:
            W, b = rc.fit_lr(Xtr, Ytr, C); fits.append((W, b)); aucs.append(rc.auroc_matrix(Yva, Xva @ W + b))
        A = np.where(np.isnan(np.array(aucs)), -1, np.array(aucs)); best = A.argmax(0)
        W = np.stack([fits[best[k]][0][:, k] for k in range(14)], 1); b = np.array([fits[best[k]][1][k] for k in range(14)])
        keep = [k for k in range(14) if Ytr[:, k].sum() >= 5 and Yva[:, k].sum() >= 1 and Yte[:, k].sum() >= 1]
        return float(np.mean(rc.auroc_matrix(Yte, Xte @ W + b)[keep]))

    tab = {}
    for c in rc.MAIN:
        Xn = rc.load_emb(c, o["name"].values)
        old = {s: np.load(f"{root}/{old_id[c]}/{s}.npz", allow_pickle=True) for s in ("train", "val", "test")}
        # align old embeddings to the same row order
        Xo = np.zeros((len(o), old["train"]["X"].shape[1]), np.float32)
        for s in ("train", "val", "test"):
            pos = {f: i for i, f in enumerate(o["filename"].values)}
            Xo[[pos[f] for f in old[s]["fnames"]]] = old[s]["X"]
        r = {}
        for en, X in (("orig_emb", Xo), ("corr_emb", Xn)):
            for pn, P in (("orig_probe", old_probe), ("study_probe", new_probe)):
                r[f"{en}|{pn}"] = P(X[sp["train"]], Y[sp["train"]], X[sp["val"]], Y[sp["val"]], X[sp["test"]], Y[sp["test"]])
        tab[c] = r
        print(rc.PRETTY[c], {k: round(v, 3) for k, v in r.items()}, flush=True)
    rc.save_json(tab, rc.RESULTS / "decomposition.json")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "decompose":
    decompose()
