"""
make_figures.py — result figures (all numbers from results/*.json).

Colour: Okabe-Ito, fixed per encoder (never by rank); every encoder also has its own
marker shape and is direct-labelled, so identity never relies on colour alone
(deuteranopia separation of one pair is 7.6 OKLab dE; secondary encoding required).
Controls: grey, hollow markers.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as rc
import pro_style
pro_style.apply()

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "figures"); os.makedirs(OUT, exist_ok=True)
ST = rc.RESULTS
COL = {"rad_dino_518": "#0072B2", "biomedclip_224": "#D55E00", "dinov2_small_224": "#009E73",
       "clip_b16_224": "#E69F00", "resnet50_224": "#CC79A7", "vit_b16_224": "#56B4E9",
       "rad_dino_224": "#555555", "dinov2_base_224": "#555555", "dinov2_base_518": "#555555"}
MRK = {"rad_dino_518": "o", "biomedclip_224": "s", "dinov2_small_224": "^", "clip_b16_224": "D",
       "resnet50_224": "v", "vit_b16_224": "P", "rad_dino_224": "o", "dinov2_base_224": "h", "dinov2_base_518": "H"}
HOLLOW = set(rc.CONTROLS)


def J(n):
    p = ST / n
    return json.load(open(p)) if p.exists() else None


def save(fig, name):
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{e}"))
    plt.close(fig); print("[ok]", name)


def mk(ax, x, y, c, s=36, **kw):
    ax.scatter(x, y, s=s, marker=MRK[c], facecolors="white" if c in HOLLOW else COL[c],
               edgecolors=COL[c], linewidths=1.2, zorder=3, **kw)


S, L, E, R = J("stats.json"), J("latency.json"), J("external.json"), J("replicates.json")

# ---------------------------------------------------------------- macro + triage forest
if S:
    D = S["discrimination"]; order = rc.MAIN + rc.CONTROLS
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, (k, kci, title) in zip(axs, [("macro_auroc", "macro_ci", "Macro AUROC (14 findings)"),
                                         ("triage_auroc", "triage_ci", "Triage AUROC (any finding)")]):
        for i, c in enumerate(order):
            y = len(order) - i - (0.6 if i >= 6 else 0)
            lo, hi = D[c][kci]
            ax.plot([lo, hi], [y, y], color=COL[c], lw=1.6, zorder=2)
            mk(ax, D[c][k], y, c)
        ax.set_title(title, fontsize=9.5); ax.grid(axis="y", visible=False)
        ax.set_xlabel("AUROC (95 % CI, patient-level bootstrap)", fontsize=8.5)
    axs[0].set_yticks([len(order) - i - (0.6 if i >= 6 else 0) for i in range(len(order))])
    axs[0].set_yticklabels([rc.PRETTY[c] for c in order], fontsize=8.5)
    axs[0].axhline(len(order) - 5.8, color="#BBBBBB", lw=0.8, ls=":")
    axs[1].axhline(len(order) - 5.8, color="#BBBBBB", lw=0.8, ls=":")
    axs[0].text(axs[0].get_xlim()[0], len(order) - 6.25, "controls", fontsize=7, color="#777777", va="center")
    fig.tight_layout(); save(fig, "fig_macro")

# ---------------------------------------------------------------- per-label heatmap
if S:
    D = S["discrimination"]; order = rc.MAIN + rc.CONTROLS
    M = np.array([[D[c]["per_label"][l] for l in rc.LABELS] for c in order])
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    im = ax.imshow(M, cmap="Blues", vmin=np.floor(M.min() * 20) / 20, vmax=np.ceil(M.max() * 20) / 20, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.2,
                    color="white" if M[i, j] > np.percentile(M, 60) else "#222222")
    ax.set_xticks(range(14)); ax.set_xticklabels([l.replace("_", " ") for l in rc.LABELS], rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([rc.PRETTY[c] for c in order], fontsize=7.5)
    ax.axhline(5.5, color="white", lw=2.5); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("Test AUROC", fontsize=8)
    fig.tight_layout(); save(fig, "fig_heatmap")

# ---------------------------------------------------------------- robustness: re-draws + learning curve
if R and S:
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    rng = np.random.default_rng(0)
    for ax, key, title in [(axs[0], "own", "20 re-draws, own 500-image test"),
                           (axs[1], "full", "Same probes, full official test")]:
        for i, c in enumerate(rc.MAIN):
            v = np.array(R["small_replicates"][c][key])
            x = i + rng.uniform(-0.18, 0.18, len(v))
            ax.scatter(x, v, s=12, marker=MRK[c], color=COL[c], alpha=0.75, linewidths=0, zorder=3)
            ax.plot([i - 0.3, i + 0.3], [S["discrimination"][c]["macro_auroc"]] * 2, color="#222222", lw=1.4, zorder=4)
        ax.set_xticks(range(len(rc.MAIN))); ax.set_xticklabels([rc.PRETTY[c] for c in rc.MAIN], rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=8.5); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("Macro AUROC", fontsize=8.5)
    lo = min(min(R["small_replicates"][c][k]) for c in rc.MAIN for k in ("own", "full")) - 0.01
    hi = max(max(R["small_replicates"][c][k]) for c in rc.MAIN for k in ("own", "full")) + 0.01
    hi = max(hi, max(S["discrimination"][c]["macro_auroc"] for c in rc.MAIN) + 0.01)
    for ax in axs[:2]:
        ax.set_ylim(lo, hi)
    axs[1].text(len(rc.MAIN) - 0.6, hi - 0.004, "— full-data\n   estimate", fontsize=6.5, ha="right", va="top", color="#222222")
    ax = axs[2]
    ends = {}
    for c in R["cfgs"]:
        pts = sorted(((np.mean([r["n_img"] for r in v]), np.mean([r["macro"] for r in v])) for v in R["learning_curve"][c].values()))
        x, y = zip(*pts)
        ax.plot(x, y, color=COL[c], lw=1.4, ls="--" if c in HOLLOW else "-")
        for xi, yi in pts:
            mk(ax, xi, yi, c, s=14)
        ends[c] = (x[-1], y[-1])
    # label repel: keep at least `gap` between label baselines (top-down)
    gap = 0.011; placed = []
    for c in sorted(ends, key=lambda k: -ends[k][1]):
        yl = ends[c][1]
        if placed and placed[-1] - yl < gap:
            yl = placed[-1] - gap
        placed.append(yl)
        ax.text(ends[c][0] * 1.15, yl, rc.PRETTY[c], fontsize=6.3, va="center", color="#222222")
    ax.set_xscale("log"); ax.set_xlim(right=ax.get_xlim()[1] * 4)
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}")); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Training images (log scale)", fontsize=8.5); ax.set_title("Learning curve, official test", fontsize=8.5)
    fig.tight_layout(); save(fig, "fig_robustness")

# ---------------------------------------------------------------- calibration: reliability small multiples
if S:
    df = rc.label_table(); te = df[df.split == "test"]
    Y = te[rc.LABELS].values.ravel()
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
    styles = [("Balanced", "#888888", "--"), ("Balanced + prior shift", "#0072B2", "-"),
              ("Unweighted", "#D55E00", ":")]
    bins = np.linspace(0, 1, 16)
    for ax, c in zip(axs.ravel(), rc.MAIN):
        db = np.load(rc.PREDS / f"{c}__balanced.npz"); du = np.load(rc.PREDS / f"{c}__unweighted.npz")
        pri = db["prior_train"][:14]
        ps = [rc.sigmoid(db["z_test"][:, :14]), rc.sigmoid(db["z_test"][:, :14] + np.log(pri / (1 - pri))),
              rc.sigmoid(du["z_test"][:, :14])]
        for (lab, colr, ls), P in zip(styles, ps):
            p = P.ravel(); idx = np.clip(np.digitize(p, bins) - 1, 0, 14)
            xs = [p[idx == k].mean() for k in range(15) if (idx == k).sum() >= 30]
            ys = [Y[idx == k].mean() for k in range(15) if (idx == k).sum() >= 30]
            ax.plot(xs, ys, color=colr, ls=ls, lw=1.5, marker="o", ms=2.5, label=lab)
        ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8)
        ax.set_title(rc.PRETTY[c], fontsize=8.5)
    for ax in axs[1]:
        ax.set_xlabel("Mean predicted probability", fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel("Observed frequency", fontsize=8)
    axs[0, 0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); save(fig, "fig_reliability")

# ---------------------------------------------------------------- latency vs AUROC with strict Pareto set
if S and L:
    D = S["discrimination"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, lk, title in [(axs[0], "gpu_fp32_b1_ms", f"GPU ({L['gpu'].replace('NVIDIA GeForce ', '')}), batch 1, fp32"),
                          (axs[1], "cpu_fp32_b1_ms", f"CPU ({L['cpu_threads']} threads), batch 1, fp32")]:
        pts = {c: (L["configs"][c][lk], D[c]["macro_auroc"]) for c in rc.ALL}
        front = [c for c in pts if not any((pts[o][0] <= pts[c][0] and pts[o][1] >= pts[c][1]) and
                                           (pts[o][0] < pts[c][0] or pts[o][1] > pts[c][1]) for o in pts)]
        fr = sorted(front, key=lambda c: pts[c][0])
        xs = [pts[c][0] for c in fr]; ys = [pts[c][1] for c in fr]
        ax.step(xs + [xs[-1] * 3], ys + [ys[-1]], where="post", color="#999999", lw=1, ls="--", zorder=1)
        for c, (x, y) in pts.items():
            lo, hi = D[c]["macro_ci"]
            ax.plot([x, x], [lo, hi], color=COL[c], lw=1, alpha=0.6, zorder=2)
            mk(ax, x, y, c, s=40 if c in front else 26)
            off = (4, -10) if c == "dinov2_base_224" else (4, 3)
            ax.annotate(rc.PRETTY[c], (x, y), xytext=off, textcoords="offset points", fontsize=6.3,
                        fontweight="bold" if c in front else "normal", color="#222222")
        ax.set_xscale("log"); ax.set_title(title, fontsize=8.5); ax.set_xlabel("Forward-pass latency (ms / image, log)", fontsize=8)
        from matplotlib.ticker import FuncFormatter, NullFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(min(xs) * 0.6, max(p[0] for p in pts.values()) * 2.2)
    axs[0].set_ylabel("Macro AUROC (official test)", fontsize=8.5)
    fig.tight_layout(); save(fig, "fig_pareto")

# ---------------------------------------------------------------- external: per-label NIH vs Open-i
if E:
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for c in rc.MAIN:
        e = E["encoders"][c]
        x = [e["per_label_nih_test"][l] for l in e["per_label"]]; y = [e["per_label"][l] for l in e["per_label"]]
        mk(ax, x, y, c, s=22, label=rc.PRETTY[c])
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="#BBBBBB", lw=0.8, zorder=1); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("AUROC, NIH official test", fontsize=8.5); ax.set_ylabel("AUROC, external (Open-i)", fontsize=8.5)
    ax.legend(fontsize=6.8, loc="upper left", ncol=1)
    ax.set_aspect("equal"); fig.tight_layout(); save(fig, "fig_external")
