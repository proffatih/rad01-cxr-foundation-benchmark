"""
stats.py — inference on the full-data probes.

Patient-level cluster bootstrap (B = 2,000): every replicate resamples the 2,797 test
patients with replacement; each image inherits its patient's multiplicity as a weight.
Weighted AUROC is computed exactly from a single sort per (encoder, target).
Pairwise differences: Holm-adjusted p values and a pre-specified equivalence margin
of 0.01 macro AUROC. Also calibration under five probe/calibration conditions and
clinical operating points.

Output: results/stats.json
"""
from __future__ import annotations
import sys, itertools
import numpy as np
import torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
import common as rc

B = 2000
DELTA = 0.01
TARGETS = rc.LABELS + ["abnormal"]
rng = np.random.default_rng(rc.SEED)


def boot_weights(pids: np.ndarray, B: int) -> torch.Tensor:
    up, inv = np.unique(pids, return_inverse=True)
    cnt = rng.multinomial(len(up), np.full(len(up), 1 / len(up)), size=B)   # [B, P]
    return torch.as_tensor(cnt[:, inv], dtype=torch.float32, device=rc.DEV)  # [B, n]


def weighted_auroc(y: np.ndarray, s: np.ndarray, Wt: torch.Tensor) -> np.ndarray:
    """Exact weighted AUROC for each row of Wt (ties broken by average)."""
    order = np.argsort(s, kind="mergesort")
    ys = torch.as_tensor(y[order], dtype=torch.float32, device=rc.DEV)
    ss = s[order]
    # group ties
    _, first = np.unique(ss, return_index=True)
    grp = np.zeros(len(ss), dtype=np.int64); grp[first[1:]] = 1; grp = np.cumsum(grp)
    g = torch.as_tensor(grp, device=rc.DEV)
    out = []
    for W in torch.split(Wt[:, order], 250):
        wp, wn = W * ys, W * (1 - ys)
        G = int(grp[-1]) + 1
        gp = torch.zeros(W.shape[0], G, device=rc.DEV).index_add_(1, g, wp)
        gn = torch.zeros(W.shape[0], G, device=rc.DEV).index_add_(1, g, wn)
        cn = torch.cumsum(gn, 1) - gn                       # negatives strictly below
        num = (gp * (cn + 0.5 * gn)).sum(1)
        den = gp.sum(1) * gn.sum(1)
        out.append((num / den).cpu().numpy())
    return np.concatenate(out)


def main():
    df = rc.label_table()
    te = df[df.split == "test"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    Y = te[TARGETS].values; Yv = va[TARGETS].values
    P = {}
    for cfg in rc.ALL:
        for sch in ("balanced", "unweighted"):
            d = np.load(rc.PREDS / f"{cfg}__{sch}.npz")
            P[cfg, sch] = d
    Wb = boot_weights(te.pid.values, B)
    res = {"n_test": len(te), "n_test_patients": int(te.pid.nunique()), "B": B, "delta": DELTA}

    # ---------------------------------------------------------- discrimination
    point, boot = {}, {}
    for cfg in rc.ALL:
        z = P[cfg, "balanced"]["z_test"]
        point[cfg] = rc.auroc_matrix(Y, z)
        boot[cfg] = np.stack([weighted_auroc(Y[:, k], z[:, k], Wb) for k in range(len(TARGETS))], 1)
    disc = {}
    for cfg in rc.ALL:
        mb = boot[cfg][:, :14].mean(1)
        disc[cfg] = {
            "macro_auroc": float(point[cfg][:14].mean()),
            "macro_ci": np.nanpercentile(mb, [2.5, 97.5]).tolist(),
            "per_label": dict(zip(TARGETS, point[cfg].tolist())),
            "per_label_ci": {t: np.nanpercentile(boot[cfg][:, k], [2.5, 97.5]).tolist()
                             for k, t in enumerate(TARGETS)},
            "macro_auprc": float(np.mean([rc.auprc(Y[:, k], P[cfg, "balanced"]["z_test"][:, k])
                                          for k in range(14)])),
            "per_label_auprc": {t: rc.auprc(Y[:, k], P[cfg, "balanced"]["z_test"][:, k])
                                for k, t in enumerate(TARGETS)},
            "triage_auroc": float(point[cfg][14]),
            "triage_ci": np.nanpercentile(boot[cfg][:, 14], [2.5, 97.5]).tolist(),
            "C_selected": dict(zip(TARGETS, P[cfg, "balanced"]["C"].tolist())),
        }
    res["discrimination"] = disc

    # pairwise among MAIN (Holm) + equivalence decision
    pairs = list(itertools.combinations(rc.MAIN, 2))
    raw = []
    for a, b in pairs:
        d = boot[a][:, :14].mean(1) - boot[b][:, :14].mean(1)
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        raw.append((a, b, float(point[a][:14].mean() - point[b][:14].mean()),
                    np.nanpercentile(d, [2.5, 97.5]).tolist(), p))
    order = np.argsort([r[4] for r in raw]); m = len(raw); adj = np.zeros(m); run = 0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * raw[j][4])); adj[j] = run
    res["pairwise"] = []
    for (a, b, dlt, ci, p), pa in zip(raw, adj):
        within = ci[0] > -DELTA and ci[1] < DELTA
        verdict = ("different, below margin" if (pa < 0.05 and within) else
                   "equivalent" if within else
                   "different" if pa < 0.05 else "inconclusive")
        res["pairwise"].append(dict(a=a, b=b, diff=dlt, ci=ci, p=p, p_holm=float(pa), verdict=verdict))

    # matched contrasts (R3-Q1)
    contrasts = [("rad_dino_224", "dinov2_base_224", "data domain (radiology vs natural), ViT-B/14 DINOv2, 224"),
                 ("rad_dino_518", "dinov2_base_518", "data domain (radiology vs natural), ViT-B/14 DINOv2, 518"),
                 ("biomedclip_224", "clip_b16_224", "data domain (biomedical vs web), ViT-B/16 contrastive, 224"),
                 ("rad_dino_518", "rad_dino_224", "input resolution 518 vs 224, RAD-DINO"),
                 ("dinov2_base_518", "dinov2_base_224", "input resolution 518 vs 224, DINOv2-B"),
                 ("dinov2_base_224", "dinov2_small_224", "model scale ViT-B vs ViT-S, DINOv2, 224"),
                 ("clip_b16_224", "vit_b16_224", "objective (contrastive web vs supervised ImageNet), ViT-B/16, 224")]
    res["contrasts"] = []
    for a, b, what in contrasts:
        if a not in boot or b not in boot:
            continue
        d = boot[a][:, :14].mean(1) - boot[b][:, :14].mean(1)
        dl = boot[a][:, :14] - boot[b][:, :14]
        res["contrasts"].append(dict(a=a, b=b, what=what,
                                     diff=float(point[a][:14].mean() - point[b][:14].mean()),
                                     ci=np.nanpercentile(d, [2.5, 97.5]).tolist(),
                                     p=float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                                     labels_a_better=int(((point[a][:14] - point[b][:14]) > 0).sum()),
                                     labels_ci_excl0_a_better=int((np.nanpercentile(dl, 2.5, 0) > 0).sum()),
                                     labels_ci_excl0_b_better=int((np.nanpercentile(dl, 97.5, 0) < 0).sum())))

    # rank stability over bootstrap (MAIN)
    M = np.stack([boot[c][:, :14].mean(1) for c in rc.MAIN], 1)          # [B, 6]
    ranks = (-M).argsort(1).argsort(1) + 1
    res["rank_bootstrap"] = {c: np.bincount(ranks[:, i], minlength=7)[1:].tolist()
                             for i, c in enumerate(rc.MAIN)}
    res["rank_point"] = {c: int(r) for c, r in zip(rc.MAIN, (-M.mean(0)).argsort().argsort() + 1)}

    # ---------------------------------------------------------- calibration (R3-Q3)
    def cal_conditions(cfg):
        db, du = P[cfg, "balanced"], P[cfg, "unweighted"]
        pri = db["prior_train"]
        out = {"balanced_raw": rc.sigmoid(db["z_test"]),
               "unweighted_raw": rc.sigmoid(du["z_test"]),
               "balanced_prior_corrected": rc.sigmoid(db["z_test"] + np.log(pri / (1 - pri)))}
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        pl, iso = np.zeros_like(db["z_test"]), np.zeros_like(db["z_test"])
        for k in range(len(TARGETS)):
            if Yv[:, k].min() == Yv[:, k].max():      # no positives in validation
                pl[:, k] = iso[:, k] = out["balanced_raw"][:, k]; continue
            lr = LogisticRegression(C=1e6).fit(db["z_val"][:, k:k + 1], Yv[:, k])
            pl[:, k] = lr.predict_proba(db["z_test"][:, k:k + 1])[:, 1]
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(db["z_val"][:, k], Yv[:, k])
            iso[:, k] = ir.predict(db["z_test"][:, k])
        out["balanced_platt_val"] = pl; out["balanced_isotonic_val"] = iso
        return out

    Wc = Wb[:500]
    cal = {}
    for cfg in rc.ALL:
        cal[cfg] = {}
        for cond, Pm in cal_conditions(cfg).items():
            e = [rc.ece(Y[:, k], Pm[:, k]) for k in range(14)]
            ea = [rc.ece(Y[:, k], Pm[:, k], mode="mass") for k in range(14)]
            br = [rc.brier(Y[:, k], Pm[:, k]) for k in range(14)]
            # bootstrap macro ECE / Brier
            eb = np.zeros((Wc.shape[0], 14)); bb = np.zeros((Wc.shape[0], 14))
            for k in range(14):
                idx = np.clip(np.floor(Pm[:, k] * 15).astype(int), 0, 14)
                O = torch.zeros(len(idx), 15, device=rc.DEV); O[torch.arange(len(idx)), torch.as_tensor(idx)] = 1
                yt = torch.as_tensor(Y[:, k], dtype=torch.float32, device=rc.DEV)
                pt = torch.as_tensor(Pm[:, k], dtype=torch.float32, device=rc.DEV)
                n_ = Wc @ O; sy = (Wc * yt) @ O; sp = (Wc * pt) @ O
                tot = Wc.sum(1, keepdim=True)
                eb[:, k] = ((sy - sp).abs() / tot).sum(1).cpu().numpy()
                bb[:, k] = ((Wc * (pt - yt) ** 2).sum(1) / tot[:, 0]).cpu().numpy()
            cal[cfg][cond] = dict(ece=float(np.mean(e)), ece_ci=np.nanpercentile(eb.mean(1), [2.5, 97.5]).tolist(),
                                  ece_mass=float(np.mean(ea)), brier=float(np.mean(br)),
                                  brier_ci=np.nanpercentile(bb.mean(1), [2.5, 97.5]).tolist(),
                                  triage_ece=rc.ece(Y[:, 14], Pm[:, 14]), triage_brier=rc.brier(Y[:, 14], Pm[:, 14]),
                                  mean_pred=float(Pm[:, :14].mean()), mean_obs=float(Y[:, :14].mean()))
    res["calibration"] = cal

    # ---------------------------------------------------------- operating points (R3-Q6)
    ops = {}
    for cfg in rc.ALL:
        db = P[cfg, "balanced"]; ops[cfg] = {}
        for tgt in (0.90, 0.95):
            rows = {}
            for k, t in enumerate(TARGETS):
                thr = rc.threshold_for_sensitivity(Yv[:, k], db["z_val"][:, k], tgt)
                m_val_thr = rc.op_metrics(Y[:, k], db["z_test"][:, k], thr)
                thr_te = rc.threshold_for_sensitivity(Y[:, k], db["z_test"][:, k], tgt)
                m_oracle = rc.op_metrics(Y[:, k], db["z_test"][:, k], thr_te)
                rows[t] = dict(val_threshold=m_val_thr, spec_at_sens_test=m_oracle["spec"],
                               n_pos=int(Y[:, k].sum()))
            mac = lambda key, sub=None: float(np.mean([rows[t]["val_threshold"][key] for t in rc.LABELS]))
            ops[cfg][f"sens{int(tgt*100)}"] = dict(
                per_target=rows,
                macro_sens=mac("sens"), macro_spec=mac("spec"), macro_fnr=mac("fnr"),
                macro_spec_at_sens_test=float(np.mean([rows[t]["spec_at_sens_test"] for t in rc.LABELS])))
    res["operating_points"] = ops

    rc.save_json(res, rc.RESULTS / "stats.json")
    print("[ok] stats written")
    for c in rc.ALL:
        print(f"{rc.PRETTY[c]:13s} macro {disc[c]['macro_auroc']:.4f} {np.round(disc[c]['macro_ci'],4)}  "
              f"triage {disc[c]['triage_auroc']:.4f}")
    for r in res["pairwise"]:
        print(f"{rc.PRETTY[r['a']]:>11s} - {rc.PRETTY[r['b']]:<11s} {r['diff']:+.4f} {np.round(r['ci'],4)} "
              f"p_holm={r['p_holm']:.3g} {r['verdict']}")
    for r in res["contrasts"]:
        print(f"{r['what']:62s} {r['diff']:+.4f} {np.round(r['ci'],4)}")


if __name__ == "__main__":
    main()
