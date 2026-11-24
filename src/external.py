"""
external.py — external validation on the Indiana University chest X-ray collection
(NLM Open-i; Demner-Fushman et al., JAMIA 2016).

Steps
  extract : frozen embeddings for all Open-i PNGs with the same 9 encoder configs
            (whole image resized to the square native input, official normalisation).
  views   : frontal/lateral separation by 2-means on DINOv2-B embeddings (verified by
            visual inspection of a random montage, saved to RUN_OUT/openi/view_check.png;
            the resulting assignment is in data/openi_views.csv).
  eval    : NIH-trained probes (main.py, class-balanced, no re-fitting) applied to one
            frontal image per report; labels from the report's MeSH 'major' terms.
            Thresholds for operating points are the NIH-validation thresholds
            (transported unchanged), so threshold portability is measured too.
Output: results/external.json
"""
from __future__ import annotations
import os, sys, re, glob, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import common as rc

OPENI = rc.RUN / "openi"
PNG_DIR = OPENI / "png"
MAP = {  # NIH label -> Open-i MeSH major (top-level term)
    "Cardiomegaly": ["Cardiomegaly"],
    "Atelectasis": ["Pulmonary Atelectasis"],
    "Effusion": ["Pleural Effusion"],
    "Nodule": ["Nodule"],
    "Emphysema": ["Emphysema", "Pulmonary Emphysema"],
    "Infiltration": ["Infiltrate"],
    "Hernia": ["Hernia, Hiatal"],
    "Edema": ["Pulmonary Edema"],
    "Pneumonia": ["Pneumonia"],
}
MINOR_MAP = {"Mass": ["Mass"], "Pneumothorax": ["Pneumothorax"], "Consolidation": ["Consolidation"],
             "Fibrosis": ["Pulmonary Fibrosis"]}   # <40 positives: used only for the triage target
EXT = list(MAP)
B = 2000


def reports() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(OPENI / "rep/ecgen-radiology/*.xml")):
        s = open(f).read()
        terms = {m.split("/")[0].strip() for m in re.findall(r"<major>(.*?)</major>", s)}
        imgs = re.findall(r'<parentImage id="(.*?)"', s)
        r = {"report": Path(f).stem, "images": imgs, "normal": int("normal" in terms)}
        for k, v in {**MAP, **MINOR_MAP}.items():
            r[k] = int(bool(terms & set(v)))
        rows.append(r)
    return pd.DataFrame(rows)


class DirDS(torch.utils.data.Dataset):
    def __init__(self, files): self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("L")
        a518 = np.asarray(img.resize((518, 518), Image.BICUBIC), dtype=np.uint8)
        a224 = np.asarray(img.resize((224, 224), Image.BICUBIC), dtype=np.uint8)
        return torch.from_numpy(a518.copy()), torch.from_numpy(a224.copy()), i


def extract():
    import extract_full as ex
    files = sorted(glob.glob(str(PNG_DIR / "*.png")))
    names = np.array([Path(f).stem for f in files])
    out = rc.RUN / "emb_openi"; out.mkdir(exist_ok=True)
    fns, norm = {}, {}
    for cid, (b, size, mean, std) in ex.CONFIGS.items():
        key = cid.rsplit("_", 1)[0]
        fns[cid] = fns.get(key + "_shared") or b(); fns[key + "_shared"] = fns[cid]
        norm[cid] = (size, torch.tensor(mean, device="cuda").view(1, 3, 1, 1),
                     torch.tensor(std, device="cuda").view(1, 3, 1, 1))
    feats = {c: np.zeros((len(files), 0), np.float16) for c in ex.CONFIGS}
    buf = {c: [None] * len(files) for c in ex.CONFIGS}
    dl = torch.utils.data.DataLoader(DirDS(files), batch_size=32, num_workers=8)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for a518, a224, idx in dl:
            g = {518: a518.cuda().float().div_(255).unsqueeze(1).expand(-1, 3, -1, -1),
                 224: a224.cuda().float().div_(255).unsqueeze(1).expand(-1, 3, -1, -1)}
            for c in ex.CONFIGS:
                size, mu, sd = norm[c]
                o = fns[c]((g[size] - mu) / sd).float().cpu().numpy().astype(np.float16)
                for j, k in enumerate(idx.tolist()):
                    buf[c][k] = o[j]
    for c in ex.CONFIGS:
        np.savez(out / f"{c}.npz", X=np.stack(buf[c]), names=names)
    print(f"[ok] Open-i embeddings: {len(files)} images", flush=True)


def load_openi(cfg):
    d = np.load(rc.RUN / "emb_openi" / f"{cfg}.npz")
    X = d["X"].astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, d["names"]


def views():
    from sklearn.cluster import KMeans
    X, names = load_openi("dinov2_base_224")
    km = KMeans(2, n_init=10, random_state=rc.SEED).fit(X)
    lab = km.labels_
    # frontal cluster = the one closer to the NIH (all-frontal) centroid
    df = rc.label_table(); rng = np.random.default_rng(0)
    nih = rc.load_emb("dinov2_base_224", rng.choice(df["name"].values, 2000, replace=False)).mean(0)
    d = [np.linalg.norm(km.cluster_centers_[k] - nih) for k in (0, 1)]
    frontal_k = int(np.argmin(d))
    is_frontal = lab == frontal_k
    # montage for visual verification
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 12, figsize=(18, 6.5))
    for row, want in enumerate([True, True, False, False]):
        pick = rng.choice(np.where(is_frontal == want)[0], 12, replace=False)
        for j, i in enumerate(pick):
            ax[row, j].imshow(Image.open(PNG_DIR / f"{names[i]}.png").convert("L"), cmap="gray")
            ax[row, j].set_axis_off()
        ax[row, 0].set_title("frontal cluster" if want else "lateral cluster", fontsize=8, loc="left")
    plt.tight_layout(); plt.savefig(OPENI / "view_check.png", dpi=70)
    pd.DataFrame({"name": names, "frontal": is_frontal.astype(int)}).to_csv(OPENI / "views.csv", index=False)
    print(f"[ok] views: frontal {is_frontal.sum()} / lateral {(~is_frontal).sum()}", flush=True)


def evaluate():
    rep = reports()
    v = pd.read_csv(OPENI / "views.csv").set_index("name")["frontal"]
    rows = []
    for _, r in rep.iterrows():
        fr = [i for i in r.images if v.get(i, 0) == 1]
        if fr:
            rows.append({**r.drop("images").to_dict(), "name": sorted(fr)[0]})
    ext = pd.DataFrame(rows)
    mapped_any = ext[list(MAP) + list(MINOR_MAP)].sum(1) > 0
    ext["triage_eligible"] = (mapped_any | (ext.normal == 1)).astype(int)
    ext["abnormal"] = mapped_any.astype(int)
    df = rc.label_table(); te = df[df.split == "test"]; va = df[df.split == "val"]
    kidx = [rc.LABELS.index(l) for l in EXT]
    Y = ext[EXT].values; Yt = ext["abnormal"].values; elig = ext.triage_eligible.values == 1
    rng = np.random.default_rng(rc.SEED)
    boots = rng.integers(0, len(ext), size=(B, len(ext)))
    res = {"n_reports_with_frontal": len(ext), "n_triage_eligible": int(elig.sum()),
           "n_triage_abnormal": int(Yt[elig].sum()), "positives": {l: int(ext[l].sum()) for l in EXT},
           "encoders": {}}
    macro_b = {}
    for cfg in rc.ALL:
        X, names = load_openi(cfg)
        pos = pd.Series(np.arange(len(names)), index=names)
        Xe = X[pos.loc[ext["name"]].values]
        d = np.load(rc.PREDS / f"{cfg}__balanced.npz")
        Z = rc.predict_logit(Xe, d["W"], d["b"])
        a = np.array([rc.auroc(Y[:, j], Z[:, k]) for j, k in enumerate(kidx)])
        # same 9 labels on NIH test for a like-for-like drop
        a_nih = np.array([rc.auroc(te[l].values, d["z_test"][:, rc.LABELS.index(l)]) for l in EXT])
        mb = np.array([np.mean([rc.auroc(Y[bi, j], Z[bi, k]) for j, k in enumerate(kidx)]) for bi in boots[:500]])
        macro_b[cfg] = mb
        tri = rc.auroc(Yt[elig], Z[elig, 14])
        ops = {}
        for tgt in (0.90, 0.95):
            thr = rc.threshold_for_sensitivity(va["abnormal"].values, d["z_val"][:, 14], tgt)
            ops[f"triage_sens{int(tgt*100)}_nih_threshold"] = rc.op_metrics(Yt[elig], Z[elig, 14], thr)
            thr_e = rc.threshold_for_sensitivity(Yt[elig], Z[elig, 14], tgt)
            ops[f"triage_spec_at_sens{int(tgt*100)}_external"] = rc.op_metrics(Yt[elig], Z[elig, 14], thr_e)["spec"]
            per = {}
            for j, k in enumerate(kidx):
                thr = rc.threshold_for_sensitivity(va[EXT[j]].values, d["z_val"][:, k], tgt)
                per[EXT[j]] = rc.op_metrics(Y[:, j], Z[:, k], thr)
            ops[f"labels_sens{int(tgt*100)}_nih_threshold"] = per
        pri = d["prior_train"]
        P = rc.sigmoid(Z + np.log(pri / (1 - pri)))
        res["encoders"][cfg] = dict(per_label=dict(zip(EXT, a.tolist())), macro9=float(a.mean()),
                                    macro9_ci=np.nanpercentile(mb, [2.5, 97.5]).tolist(),
                                    macro9_nih_test=float(a_nih.mean()),
                                    per_label_nih_test=dict(zip(EXT, a_nih.tolist())),
                                    triage_auroc=float(tri), ops=ops,
                                    ece_prior_corrected=float(np.mean([rc.ece(Y[:, j], P[:, k]) for j, k in enumerate(kidx)])))
        print(f"{rc.PRETTY[cfg]:13s} ext macro9 {a.mean():.4f} (NIH same labels {a_nih.mean():.4f})  triage {tri:.4f}", flush=True)
    pairs = []
    for x, y in itertools.combinations(rc.MAIN, 2):
        dd = macro_b[x] - macro_b[y]
        pairs.append(dict(a=x, b=y, diff=res["encoders"][x]["macro9"] - res["encoders"][y]["macro9"],
                          ci=np.nanpercentile(dd, [2.5, 97.5]).tolist(),
                          p=float(min(1, 2 * min((dd <= 0).mean(), (dd >= 0).mean())))))
    order = np.argsort([p["p"] for p in pairs]); run = 0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(pairs) - i) * pairs[j]["p"])); pairs[j]["p_holm"] = run
    res["pairwise_main"] = pairs
    M = np.stack([macro_b[c] for c in rc.MAIN], 1); R = (-M).argsort(1).argsort(1) + 1
    res["rank_bootstrap"] = {c: np.bincount(R[:, i], minlength=7)[1:].tolist() for i, c in enumerate(rc.MAIN)}
    rc.save_json(res, rc.RESULTS / "external.json")


if __name__ == "__main__":
    step = sys.argv[1]
    {"extract": extract, "views": views, "eval": evaluate}[step]()
