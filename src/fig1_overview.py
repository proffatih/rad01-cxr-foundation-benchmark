"""
fig1_overview.py — Figure 1 (study overview). Every text artist is checked
programmatically against its enclosing box; the script fails if any label overflows.
"""
from __future__ import annotations
import os, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_style
pro_style.apply()
plt.rcParams.update({"axes.grid": False})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "figures")
RUN = os.environ.get("RUN_OUT", "run")
os.makedirs(OUT, exist_ok=True)
n_ext = "—"
try:
    n_ext = f"{json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'external.json')))['n_reports_with_frontal']:,}"
except Exception:
    pass

C_DATA, C_ENC, C_CTRL, C_PROBE, C_EVAL = "#EEF2F7", "#FFFFFF", "#F6F0E6", "#EEF6EE", "#F3EFF8"
EDGE = "#3A3A3A"
fig = plt.figure(figsize=(7.4, 3.5))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")
checks = []   # (text artist, box (x0,y0,x1,y1))


def box(x, y, w, h, fc, lw=0.9, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                       fc=fc, ec=EDGE, lw=lw, ls=ls)
    ax.add_patch(p)
    return (x, y, x + w, y + h)


def text(x, y, s, bx, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("fontsize", 6.6)
    t = ax.text(x, y, s, **kw)
    checks.append((t, bx))
    return t


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 lw=0.9, color=EDGE))


# ---------------- data
b = box(1, 25, 18, 21, C_DATA)
text(10, 43.3, "NIH ChestX-ray14", b, fontsize=7.4, fontweight="bold")
text(10, 35.0, "112,120 frontal CXRs\n30,805 patients\nofficial patient-disjoint\nsplit", b)
text(10, 27.6, "train 78,097 · val 8,427\ntest 25,596", b, fontsize=6.0)
b = box(1, 4, 18, 17, C_DATA, ls="--")
text(10, 18.0, "External cohort", b, fontsize=7.4, fontweight="bold")
text(10, 11.0, f"Indiana Univ. (Open-i)\n{n_ext} reports, one\nfrontal image each\n(no re-training)", b)

# ---------------- encoder grid: rows = data domain, cols = objective
gx, gy, gw, gh = 23, 4, 42, 42
b = box(gx, gy, gw, gh, "#FAFAFA")
text(gx + gw / 2, gy + gh - 2.2, "Frozen encoders: data domain × pre-training objective", b,
     fontsize=6.9, fontweight="bold")
cols = ["Self-supervised\n(DINOv2)", "Vision–language\n(contrastive)", "Supervised\n(ImageNet)"]
rows = ["Radiology /\nbiomedical", "Natural\nimage"]
x0, cw = gx + 8.5, (gw - 9.5) / 3
y_top, rh = gy + gh - 9.2, 15.0
for j, c in enumerate(cols):
    text(x0 + cw * (j + 0.5), y_top + 3.3, c, b, fontsize=6.0, style="italic")
for i, r in enumerate(rows):
    text(gx + 4.3, y_top - rh * (i + 0.5), r, b, fontsize=6.0, style="italic")
cells = {
    (0, 0): [("RAD-DINO", "ViT-B/14 · 518\n(+224*)", C_ENC)],
    (0, 1): [("BiomedCLIP", "ViT-B/16 · 224", C_ENC)],
    (0, 2): [],
    (1, 0): [("DINOv2-S", "ViT-S/14 · 224", C_ENC), ("DINOv2-B*", "ViT-B/14\n224 / 518", C_CTRL)],
    (1, 1): [("CLIP-B/16", "ViT-B/16 · 224", C_ENC)],
    (1, 2): [("ResNet-50", "CNN · 224", C_ENC), ("ViT-B/16", "ViT-B/16 · 224", C_ENC)],
}
for (i, j), items in cells.items():
    cx, cy = x0 + cw * j, y_top - rh * (i + 1)
    if not items:
        cb = (cx + 0.6, cy + 0.6, cx + cw - 0.6, cy + rh - 0.6)
        text(cx + cw / 2, cy + rh / 2, "(no public\nencoder)", cb, fontsize=5.8, color="#888888")
        continue
    hh = (rh - 1.2 - 0.8 * (len(items) - 1)) / len(items)
    for k, (name, sub, fc) in enumerate(items):
        yy = cy + rh - 0.6 - hh * (k + 1) - 0.8 * k
        cb = box(cx + 0.6, yy, cw - 1.2, hh, fc, lw=0.7)
        text(cx + cw / 2, yy + hh * 0.64, name, cb, fontsize=6.6, fontweight="bold")
        text(cx + cw / 2, yy + hh * 0.28, sub, cb, fontsize=5.4)
text(gx + gw / 2, gy + 1.6, "* architecture-matched control", b, fontsize=5.6, color="#555555")

# ---------------- probe
b = box(68, 14, 13, 22, C_PROBE)
text(74.5, 32.6, "Linear probe", b, fontsize=7.2, fontweight="bold")
text(74.5, 23.0, "L2 logistic\nregression\nper target\n(14 findings\n+ triage)\nC by val AUROC", b, fontsize=5.9)

# ---------------- evaluation
b = box(84, 4, 15, 42, C_EVAL)
text(91.5, 43.4, "Evaluation", b, fontsize=7.2, fontweight="bold")
ev = ["Discrimination\nAUROC · AUPRC", "Calibration\nECE · Brier", "Operating points\nspecificity at\n90/95 % sens.",
      "Latency\nforward pass\nGPU · CPU", "Robustness\nre-draws · learning\ncurve · external"]
ys = [38.6, 32.6, 25.6, 17.6, 9.6]
for s, y in zip(ev, ys):
    text(91.5, y, s, b, fontsize=5.7)

arrow(19, 35.5, 23, 35.5); arrow(19, 12.5, 23, 12.5)
arrow(65, 25, 68, 25); arrow(81, 25, 84, 25)

# ---------------- overflow check
fig.canvas.draw()
r = fig.canvas.get_renderer()
inv = ax.transData.inverted()
bad = []
for t, (bx0, by0, bx1, by1) in checks:
    e = inv.transform(t.get_window_extent(r))
    (tx0, ty0), (tx1, ty1) = e
    if tx0 < bx0 + 0.2 or tx1 > bx1 - 0.2 or ty0 < by0 + 0.2 or ty1 > by1 - 0.2:
        bad.append((t.get_text().replace("\n", " | "), round(tx0, 1), round(tx1, 1), (bx0, bx1)))
if bad:
    for x in bad:
        print("OVERFLOW", x)
    sys.exit(1)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig1_overview.{ext}"))
print("[ok] fig1 written; all", len(checks), "labels inside their boxes")
