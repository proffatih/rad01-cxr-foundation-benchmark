"""
replicates.py — sampling robustness and ranking stability.

(a) 20 independent re-draws of a small-sample protocol
    (2,000 train / 500 val from train_val patients, 500 test from official-test
    patients; 55 % any-finding enrichment; patient-disjoint). Each replicate is scored
    on its own 500-image test set AND on the full official test set, separating
    train-sample from test-sample noise.
(b) Learning curve: patient-level train subsamples of 1k ... 40k images + full,
    3 seeds below full; fixed validation partition; official test.
(c) 5 alternative patient-level train/val partitions of train_val (full size).
Kendall's W quantifies agreement of the macro-AUROC ranking across replicates.
Output: results/replicates.json
"""
from __future__ import annotations
import sys, time
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import common as rc

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
CFGS = rc.MAIN + ["dinov2_base_224"]


def fit_select(Xtr, Ytr, Xva, Yva):
    fits, aucs = [], []
    for C in C_GRID:
        W, b = rc.fit_lr(Xtr, Ytr, C)
        fits.append((W, b)); aucs.append(rc.auroc_matrix(Yva, rc.predict_logit(Xva, W, b)))
    A = np.array(aucs); A[np.isnan(A)] = -1
    best = A.argmax(0)
    W = np.stack([fits[best[k]][0][:, k] for k in range(Ytr.shape[1])], 1)
    b = np.array([fits[best[k]][1][k] for k in range(Ytr.shape[1])])
    return W, b


def macro(Y, Z):
    a = rc.auroc_matrix(Y, Z)
    return float(np.nanmean(a)), a


def kendall_w(R):  # R: [m raters, n items] ranks
    m, n = R.shape
    S = ((R.sum(0) - m * (n + 1) / 2) ** 2).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n)))


def enrich_sample(d, n, rng):
    pos = d[d.abnormal == 1]; neg = d[d.abnormal == 0]
    k = int(n * 0.55)
    return np.concatenate([rng.choice(pos.index, k, replace=False),
                           rng.choice(neg.index, n - k, replace=False)])


def patients_until(d, n_img, rng):
    pts = d.pid.unique().copy(); rng.shuffle(pts)
    cnt = d.groupby("pid").size().loc[pts].cumsum()
    keep = set(cnt.index[: int(np.searchsorted(cnt.values, n_img)) + 1])
    return d.index[d.pid.isin(keep)]


def main():
    df = rc.label_table()
    E = {c: rc.load_emb(c, df["name"].values) for c in CFGS}
    Y = df[rc.LABELS].values.astype(np.float32)
    te = df.index[df.split == "test"]; tr = df.index[df.split == "train"]; va = df.index[df.split == "val"]
    out = {"cfgs": CFGS}

    # (a) small-sample re-draws --------------------------------------------------
    t0 = time.time()
    rep = {c: {"own": [], "full": [], "own_labels": [], "full_labels": []} for c in CFGS}
    prev = []
    tv = df[df.split != "test"]; tst = df[df.split == "test"]
    for r in range(20):
        rng = np.random.default_rng(rc.SEED + 100 + r)
        pts = tv.pid.unique().copy(); rng.shuffle(pts)
        cut = int(len(pts) * 65 / 82)
        trp, vap = set(pts[:cut]), set(pts[cut:])
        i_tr = enrich_sample(tv[tv.pid.isin(trp)], 2000, rng)
        i_va = enrich_sample(tv[tv.pid.isin(vap)], 500, rng)
        i_te = enrich_sample(tst, 500, rng)
        prev.append({"train_pos": Y[i_tr].sum(0).tolist(), "val_pos": Y[i_va].sum(0).tolist(),
                     "test_pos": Y[i_te].sum(0).tolist()})
        for c in CFGS:
            X = E[c]
            W, b = fit_select(X[i_tr], Y[i_tr], X[i_va], Y[i_va])
            m_own, a_own = macro(Y[i_te], rc.predict_logit(X[i_te], W, b))
            m_full, a_full = macro(Y[te], rc.predict_logit(X[te], W, b))
            rep[c]["own"].append(m_own); rep[c]["full"].append(m_full)
            rep[c]["own_labels"].append(a_own.tolist()); rep[c]["full_labels"].append(a_full.tolist())
        print(f"[a] replicate {r+1}/20  {(time.time()-t0)/60:.1f} min", flush=True)
    main_idx = [CFGS.index(c) for c in rc.MAIN]
    for key in ("own", "full"):
        M = np.array([[rep[c][key][r] for c in rc.MAIN] for r in range(20)])
        R = (-M).argsort(1).argsort(1) + 1
        out[f"small_{key}_kendall_w"] = kendall_w(R)
        out[f"small_{key}_rank_counts"] = {c: np.bincount(R[:, i], minlength=7)[1:].tolist()
                                           for i, c in enumerate(rc.MAIN)}
    out["small_replicates"] = rep
    out["small_prevalence"] = prev

    # (b) learning curve ----------------------------------------------------------
    lc = {c: {} for c in CFGS}
    dtr = df.loc[tr]
    for n in [1000, 2000, 5000, 10000, 20000, 40000, None]:
        seeds = [0] if n is None else [0, 1, 2]
        for s in seeds:
            idx = tr if n is None else patients_until(dtr, n, np.random.default_rng(rc.SEED + 1000 * s + (n or 0)))
            for c in CFGS:
                X = E[c]
                W, b = fit_select(X[idx], Y[idx], X[va], Y[va])
                m, _ = macro(Y[te], rc.predict_logit(X[te], W, b))
                lc[c].setdefault(str(n or len(tr)), []).append({"n_img": int(len(idx)), "macro": m})
        print(f"[b] n={n} done  {(time.time()-t0)/60:.1f} min", flush=True)
    out["learning_curve"] = lc

    # (c) alternative train/val partitions ----------------------------------------
    alt = {c: [] for c in CFGS}
    for s in range(1, 6):
        rng = np.random.default_rng(rc.SEED + 7 * s)
        pts = tv.pid.unique().copy(); rng.shuffle(pts)
        vp = set(pts[: int(round(0.1 * len(pts)))])
        i_va = tv.index[tv.pid.isin(vp)]; i_tr = tv.index[~tv.pid.isin(vp)]
        for c in CFGS:
            X = E[c]
            W, b = fit_select(X[i_tr], Y[i_tr], X[i_va], Y[i_va])
            alt[c].append(macro(Y[te], rc.predict_logit(X[te], W, b))[0])
        print(f"[c] partition {s}/5  {(time.time()-t0)/60:.1f} min", flush=True)
    M = np.array([[alt[c][s] for c in rc.MAIN] for s in range(5)])
    out["partitions"] = alt
    out["partitions_kendall_w"] = kendall_w((-M).argsort(1).argsort(1) + 1)
    rc.save_json(out, rc.RESULTS / "replicates.json")
    print("[ok] replicates written", flush=True)


if __name__ == "__main__":
    main()
