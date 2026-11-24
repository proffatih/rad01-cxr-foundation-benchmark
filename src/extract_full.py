"""
extract_full.py — frozen-embedding extraction on the full NIH ChestX-ray14 release
(112,120 images, 12 shards).

  * Source images are decoded once from the original 1024x1024 PNGs and resized
    (bicubic, whole image, no crop) directly to each model's native input size.
  * Each model receives its official normalisation (HF processor / timm data config /
    open_clip transform).
  * CLIP-B/16 and BiomedCLIP both use the 512-d projected image embedding, so the
    vision-language pair is compared on the same representation.
  * Architecture-matched controls: DINOv2-base (same ViT-B/14 architecture and DINOv2
    objective as RAD-DINO, natural-image data) and both ViT-B/14 models at 224 and 518
    (resolution as an explicit factor).
  * Latency is not measured here (wall time would include PNG decoding); see
    latency.py for the forward-pass-only protocol.

Embeddings are extracted under fp16 autocast and stored as float16 (raw, not
L2-normalised; normalisation is applied in the probing script).

Output (RUN_OUT):  emb/<config>/<shard>.npz  (X float16 [N,D], names)
Resumable: a shard is skipped when all configs already have it.
"""
from __future__ import annotations
import os, io, sys, time, json, zipfile, glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RUN = Path(os.environ.get("RUN_OUT", "run"))
ZIP_DIR = Path(os.environ.get("NIH_ZIPS", ""))
OUT = RUN / "emb"
DEV = "cuda"
BATCH = int(os.environ.get("BATCH", "32"))
WORKERS = int(os.environ.get("WORKERS", "8"))

MEAN_IMNET, STD_IMNET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MEAN_CLIP, STD_CLIP = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
MEAN_RAD, STD_RAD = (0.5307,) * 3, (0.2583,) * 3
MEAN_HALF, STD_HALF = (0.5,) * 3, (0.5,) * 3


def build_hf_cls(name):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(name, attn_implementation="sdpa").eval().to(DEV)
    def f(x):
        return m(pixel_values=x).last_hidden_state[:, 0]   # post-LN CLS token
    return f


def build_clip():
    from transformers import CLIPVisionModelWithProjection
    m = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16").eval().to(DEV)
    def f(x):
        return m(pixel_values=x).image_embeds                # 512-d projected
    return f


def build_biomedclip():
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    v = model.visual.eval().to(DEV)
    def f(x):
        return v(x)                                          # 512-d projected
    return f


def build_timm(name):
    import timm
    m = timm.create_model(name, pretrained=True, num_classes=0).eval().to(DEV)
    def f(x):
        return m(x)
    return f


# config id -> (builder, input size, mean, std)
CONFIGS = {
    "rad_dino_518":    (lambda: build_hf_cls("microsoft/rad-dino"), 518, MEAN_RAD, STD_RAD),
    "rad_dino_224":    (lambda: build_hf_cls("microsoft/rad-dino"), 224, MEAN_RAD, STD_RAD),
    "dinov2_base_518": (lambda: build_hf_cls("facebook/dinov2-base"), 518, MEAN_IMNET, STD_IMNET),
    "dinov2_base_224": (lambda: build_hf_cls("facebook/dinov2-base"), 224, MEAN_IMNET, STD_IMNET),
    "dinov2_small_224":(lambda: build_hf_cls("facebook/dinov2-small"), 224, MEAN_IMNET, STD_IMNET),
    "biomedclip_224":  (build_biomedclip, 224, MEAN_CLIP, STD_CLIP),
    "clip_b16_224":    (build_clip, 224, MEAN_CLIP, STD_CLIP),
    "resnet50_224":    (lambda: build_timm("resnet50.a1_in1k"), 224, MEAN_IMNET, STD_IMNET),
    "vit_b16_224":     (lambda: build_timm("vit_base_patch16_224.augreg2_in21k_ft_in1k"), 224, MEAN_HALF, STD_HALF),
}


class ShardDS(torch.utils.data.Dataset):
    def __init__(self, zpath, members):
        self.zpath, self.members, self.zf = zpath, members, None

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zpath, "r")
        img = Image.open(io.BytesIO(self.zf.read(self.members[i]))).convert("L")
        a518 = np.asarray(img.resize((518, 518), Image.BICUBIC), dtype=np.uint8)
        a224 = np.asarray(img.resize((224, 224), Image.BICUBIC), dtype=np.uint8)
        return torch.from_numpy(a518.copy()), torch.from_numpy(a224.copy()), i


def main():
    zips = sorted(glob.glob(str(ZIP_DIR / "images_*.zip")))
    if not os.environ.get("LIMIT"):
        assert len(zips) == 12, f"expected 12 shards in {ZIP_DIR}, found {len(zips)}"
    only = os.environ.get("ONLY_SHARDS")
    if only:
        zips = [z for z in zips if Path(z).stem in only.split(",")]
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"[i] building {len(CONFIGS)} configs ...", flush=True)
    fns = {}
    norm = {}
    for cid, (b, size, mean, std) in CONFIGS.items():
        # share the same model object between the two resolutions of one backbone
        key = cid.rsplit("_", 1)[0]
        fns[cid] = fns.get(key + "_shared") or b()
        fns[key + "_shared"] = fns[cid]
        norm[cid] = (size,
                     torch.tensor(mean, device=DEV).view(1, 3, 1, 1),
                     torch.tensor(std, device=DEV).view(1, 3, 1, 1))
        (OUT / cid).mkdir(parents=True, exist_ok=True)
    print("[i] models ready", flush=True)

    for z in zips:
        shard = Path(z).stem
        if all((OUT / c / f"{shard}.npz").exists() for c in CONFIGS):
            print(f"[skip] {shard}", flush=True)
            continue
        with zipfile.ZipFile(z) as zf:
            members = sorted(m for m in zf.namelist()
                             if m.lower().endswith(".png") and "__MACOSX" not in m
                             and not Path(m).name.startswith("._"))
        if os.environ.get("LIMIT"):
            members = members[:int(os.environ["LIMIT"])]
        names = np.array([Path(m).name for m in members])
        feats = {c: [None] * len(members) for c in CONFIGS}
        dl = torch.utils.data.DataLoader(ShardDS(z, members), batch_size=BATCH,
                                         num_workers=WORKERS, pin_memory=True,
                                         persistent_workers=False, prefetch_factor=4)
        t0, n = time.time(), 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for a518, a224, idx in dl:
                g518 = a518.to(DEV, non_blocking=True).float().div_(255).unsqueeze(1).expand(-1, 3, -1, -1)
                g224 = a224.to(DEV, non_blocking=True).float().div_(255).unsqueeze(1).expand(-1, 3, -1, -1)
                for c in CONFIGS:
                    size, mu, sd = norm[c]
                    x = ((g518 if size == 518 else g224) - mu) / sd
                    out = fns[c](x).float().cpu().numpy().astype(np.float16)
                    for j, k in enumerate(idx.tolist()):
                        feats[c][k] = out[j]
                n += len(idx)
                if (n // BATCH) % 50 == 0:
                    print(f"   {shard}: {n}/{len(members)}  {n/(time.time()-t0):.1f} img/s", flush=True)
        for c in CONFIGS:
            X = np.stack(feats[c])
            assert np.isfinite(X.astype(np.float32)).all(), f"non-finite in {c}/{shard}"
            np.savez(OUT / c / f"{shard}.npz", X=X, names=names)
        print(f"[ok] {shard}: {len(members)} imgs in {(time.time()-t0)/60:.1f} min", flush=True)
    json.dump({c: {"input": v[1], "mean": v[2], "std": v[3]} for c, v in CONFIGS.items()},
              open(OUT / "configs.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
