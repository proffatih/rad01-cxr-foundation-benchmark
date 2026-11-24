"""
main.py — full-data linear probes.

For every embedding config and for two training-weight schemes (class-balanced and
unweighted):
  * fit per-label L2 logistic regressions on the train partition for
    C in {0.01, 0.1, 1, 10, 100, 1000};
  * select C per label by validation AUROC;
  * store validation/test logits for the 14 pathologies + the binary triage target
    "abnormal" (any of the 14 findings vs. No Finding), and the selected weights so
    that the same probes can be applied to the external cohort.
Output: RUN_OUT/preds/<cfg>__<scheme>.npz
"""
from __future__ import annotations
import sys, time
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import common as rc

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
TARGETS = rc.LABELS + ["abnormal"]


def main():
    df = rc.label_table()
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    print(f"[i] train {len(tr)} ({tr.pid.nunique()} pts)  val {len(va)} ({va.pid.nunique()} pts)  "
          f"test {len(te)} ({te.pid.nunique()} pts)", flush=True)
    out = rc.PREDS; out.mkdir(exist_ok=True)
    Ytr, Yva, Yte = (d[TARGETS].values.astype(np.float32) for d in (tr, va, te))
    cfgs = sys.argv[1:] or rc.ALL
    for cfg in cfgs:
        X = {s: rc.load_emb(cfg, d["name"].values) for s, d in (("tr", tr), ("va", va), ("te", te))}
        for scheme in ("balanced", "unweighted"):
            f = out / f"{cfg}__{scheme}.npz"
            if f.exists():
                print(f"[skip] {f.name}"); continue
            t0 = time.time()
            val_auc = np.zeros((len(C_GRID), len(TARGETS)))
            fits = []
            for i, C in enumerate(C_GRID):
                W, b = rc.fit_lr(X["tr"], Ytr, C, balanced=(scheme == "balanced"))
                zv = rc.predict_logit(X["va"], W, b)
                val_auc[i] = rc.auroc_matrix(Yva, zv)
                fits.append((W, b))
            best = np.where(np.isnan(val_auc), -1, val_auc).argmax(0)
            W = np.stack([fits[best[k]][0][:, k] for k in range(len(TARGETS))], 1)
            b = np.array([fits[best[k]][1][k] for k in range(len(TARGETS))])
            np.savez(f, W=W, b=b, C=np.array(C_GRID)[best], val_auc_grid=val_auc,
                     z_val=rc.predict_logit(X["va"], W, b).astype(np.float32),
                     z_test=rc.predict_logit(X["te"], W, b).astype(np.float32),
                     prior_train=Ytr.mean(0))
            te_auc = rc.auroc_matrix(Yte, rc.predict_logit(X["te"], W, b))
            print(f"[ok] {cfg:17s} {scheme:10s} macro14 test AUROC {np.nanmean(te_auc[:14]):.4f}  "
                  f"triage {te_auc[14]:.4f}  C={np.array(C_GRID)[best].tolist()}  "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
