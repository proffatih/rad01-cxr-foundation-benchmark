"""
fig_examples.py — qualitative figure: six official-test radiographs
(single-label positives + one No Finding) read from the original 1024x1024 PNGs, with the
RAD-DINO last-block class-token attention (max over heads) at its native 518x518 input and
official normalisation. Qualitative only.
"""
from __future__ import annotations
import os, sys, io, glob, zipfile
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as rc
import pro_style
pro_style.apply(); plt.rcParams.update({"axes.grid": False})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "figures")
TARGETS = ["Cardiomegaly", "Effusion", "Pneumothorax", "Mass", "Atelectasis", None]


def main():
    df = rc.label_table(); te = df[df.split == "test"].copy()
    te["npos"] = te[rc.LABELS].sum(1)
    rng = np.random.default_rng(rc.SEED)
    picks = []
    for t in TARGETS:
        pool = te[(te.npos == 0)] if t is None else te[(te[t] == 1) & (te.npos == 1)]
        r = pool.iloc[int(rng.integers(len(pool)))]
        picks.append((t or "No Finding", r["name"]))
    zips = sorted(glob.glob(str(rc.SNAP / "images" / "images_*.zip")))
    where = {}
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.endswith(".png") and "__MACOSX" not in m:
                    where[os.path.basename(m)] = (z, m)
    from transformers import AutoModel
    model = AutoModel.from_pretrained("microsoft/rad-dino", attn_implementation="eager").eval().to("cuda")
    mu, sd = 0.5307, 0.2583
    fig, axs = plt.subplots(2, 6, figsize=(7.4, 3.0))
    for j, (lab, name) in enumerate(picks):
        z, m = where[name]
        with zipfile.ZipFile(z) as zf:
            img = Image.open(io.BytesIO(zf.read(m))).convert("L")
        a = np.asarray(img.resize((518, 518), Image.BICUBIC), dtype=np.float32) / 255
        x = torch.tensor((a - mu) / sd).expand(1, 3, 518, 518).cuda()
        with torch.inference_mode():
            att = model(pixel_values=x, output_attentions=True).attentions[-1][0]   # [heads, T, T]
        cls = att[:, 0, 1:].max(0).values.cpu().numpy()
        n = int(np.sqrt(cls.size)); cls = cls[-n * n:].reshape(n, n)
        lo, hi = np.percentile(cls, 5), np.percentile(cls, 99)
        g = np.clip((cls - lo) / (hi - lo + 1e-12), 0, 1) ** 0.7
        g = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((518, 518), Image.BICUBIC)) / 255
        axs[0, j].imshow(a, cmap="gray"); axs[0, j].set_title(lab, fontsize=7.5)
        axs[1, j].imshow(a, cmap="gray"); axs[1, j].imshow(g, cmap="inferno", alpha=0.45)
        for ax in axs[:, j]:
            ax.set_axis_off()
    axs[0, 0].text(-0.08, 0.5, "radiograph", transform=axs[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=7)
    axs[1, 0].text(-0.08, 0.5, "RAD-DINO attention", transform=axs[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=7)
    fig.tight_layout(w_pad=0.3, h_pad=0.4)
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_examples.{e}"))
    print("[ok] examples:", picks)


if __name__ == "__main__":
    main()
