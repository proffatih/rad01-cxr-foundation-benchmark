"""
make_tables.py — LaTeX tables + number macros.

Every number quoted in the article text is emitted here as a \\newcommand macro
(outputs/numbers.tex), so the text cannot drift from the result files.
Best value per column is set in bold automatically.
Inputs: results/*.json      Outputs: outputs/tables/*.tex, outputs/numbers.tex
"""
from __future__ import annotations
import os, sys, json, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as rc

HERE = os.path.dirname(os.path.abspath(__file__))
MS = os.path.join(HERE, "..", "outputs"); TAB = os.path.join(MS, "tables")
os.makedirs(TAB, exist_ok=True)
ST = rc.RESULTS


def J(name):
    p = ST / name
    return json.load(open(p)) if p.exists() else None


S, L, E, R = J("stats.json"), J("latency.json"), J("external.json"), J("replicates.json")
macros = {}


DIG = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
       "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def mac(name, val):
    import re
    name = "".join(DIG.get(ch, ch) for ch in name)
    assert re.fullmatch(r"[A-Za-z]+", name), name
    macros[name] = val


def f3(x): return f"{x:.3f}"
def f1(x): return f"{x:.1f}"
def ci(c, f=f3): return f"[{f(c[0])}, {f(c[1])}]"
def pct(x): return f"{100*x:.1f}"


def bold_best(rows, col, best="max"):
    vals = [r[col][0] for r in rows if r[col][0] is not None]
    if not vals:
        return
    tgt = max(vals) if best == "max" else min(vals)
    for r in rows:
        v, s = r[col]
        if v is not None and np.isclose(v, tgt, atol=5e-7):
            r[col] = (v, r"\textbf{" + s + "}")


def write_table(fname, rows, cols, header, caption, label, note=None, groups=None, size=r"\footnotesize"):
    lines = [r"\begin{table}[H]\centering", size, rf"\caption{{{caption}}}\label{{{label}}}",
             r"\begin{adjustbox}{max width=\linewidth}",
             r"\begin{tabular}{" + "l" + "c" * len(cols) + "}", r"\toprule", header + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if groups and i in groups:
            lines.append(r"\midrule" if i else "")
            lines.append(rf"\multicolumn{{{len(cols)+1}}}{{l}}{{\emph{{{groups[i]}}}}} \\")
        lines.append(" & ".join([r["name"]] + [r[c][1] for c in cols]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if note:
        lines.append(rf"\par\smallskip\parbox{{0.95\linewidth}}{{\scriptsize {note}}}")
    lines.append(r"\end{table}")
    open(os.path.join(TAB, fname), "w").write("\n".join(l for l in lines if l is not None) + "\n")


def key(c):  # macro-safe name
    return {"rad_dino_518": "RadDino", "biomedclip_224": "BiomedClip", "dinov2_small_224": "DinoS",
            "clip_b16_224": "Clip", "resnet50_224": "ResNet", "vit_b16_224": "ViT",
            "rad_dino_224": "RadDinoLow", "dinov2_base_224": "DinoB", "dinov2_base_518": "DinoBHigh"}[c]


# ------------------------------------------------------------------ dataset table (always)
df = rc.label_table()
g = df.groupby("split")
rows = []
ext_pos = (E or {}).get("positives", {})
for lab in rc.LABELS + ["abnormal"]:
    r = {"name": lab.replace("_", " ") if lab != "abnormal" else "Any finding (triage)"}
    for s in ("train", "val", "test"):
        n = int(g[lab].sum()[s]); r[s] = (n, f"{n:,} ({pct(n/g.size()[s])})")
    r["ext"] = (None, f"{ext_pos[lab]:,}" if lab in ext_pos else "--")
    rows.append(r)
n_tr, n_va, n_te = (int(g.size()[s]) for s in ("train", "val", "test"))
p_tr, p_va, p_te = (int(g.pid.nunique()[s]) for s in ("train", "val", "test"))
mac("PrevAnyTrain", pct(g["abnormal"].sum()["train"] / n_tr)); mac("PrevAnyTest", pct(g["abnormal"].sum()["test"] / n_te))
for k, v in dict(NTrain=n_tr, NVal=n_va, NTest=n_te, PTrain=p_tr, PVal=p_va, PTest=p_te).items():
    mac(k, f"{v:,}")
write_table("tab_dataset.tex", rows, ["train", "val", "test", "ext"],
            r"Finding & Train & Validation & Test & External",
            rf"Positive images per finding and split (percentage of the split in parentheses). "
            rf"Train {n_tr:,} images / {p_tr:,} patients; validation {n_va:,} / {p_va:,}; official test "
            rf"{n_te:,} / {p_te:,}. External: Indiana University (Open-i) reports with a frontal image; "
            rf"`--' = finding not mapped (fewer than 40 positive reports).",
            "tab:dataset")

# ------------------------------------------------------------------ encoder / latency table
if L:
    meta = {"rad_dino_518": ("Radiology", "SSL (DINOv2)", "ViT-B/14"), "biomedclip_224": ("Biomedical", "VL contrastive", "ViT-B/16"),
            "dinov2_small_224": ("Natural", "SSL (DINOv2)", "ViT-S/14"), "clip_b16_224": ("Natural (web)", "VL contrastive", "ViT-B/16"),
            "resnet50_224": ("Natural", "Supervised", "ResNet-50"), "vit_b16_224": ("Natural", "Supervised", "ViT-B/16"),
            "rad_dino_224": ("Radiology", "SSL (DINOv2)", "ViT-B/14"), "dinov2_base_224": ("Natural", "SSL (DINOv2)", "ViT-B/14"),
            "dinov2_base_518": ("Natural", "SSL (DINOv2)", "ViT-B/14")}
    rows = []
    for c in rc.ALL:
        l = L["configs"][c]; d, o, a = meta[c]
        rows.append({"name": rc.PRETTY[c], "dom": (None, d), "obj": (None, o), "arch": (None, a),
                     "par": (None, f1(l["params_M"])), "res": (None, f"{l['input']}$^2$"),
                     "g1": (l["gpu_fp32_b1_ms"], f1(l["gpu_fp32_b1_ms"])),
                     "g32": (l["gpu_fp32_b32_ms"], f"{l['gpu_fp32_b32_ms']:.2f}"),
                     "h32": (l["gpu_fp16_b32_ms"], f"{l['gpu_fp16_b32_ms']:.2f}"),
                     "cpu": (l["cpu_fp32_b1_ms"], f"{l['cpu_fp32_b1_ms']:.0f}")})
        for k in ("g1", "g32", "h32", "cpu"):
            mac(f"Lat{k.capitalize()}{key(c)}", rows[-1][k][1])
    for k in ("g1", "g32", "h32", "cpu"):
        bold_best(rows[:6], k, "min"); bold_best(rows[6:], k, "min")
    mac("GPUName", L["gpu"].replace("NVIDIA GeForce ", "")); mac("CPUName", L["cpu"]); mac("CPUThreads", str(L["cpu_threads"]))
    write_table("tab_encoders.tex", rows, ["dom", "obj", "arch", "par", "res", "g1", "g32", "h32", "cpu"],
                r"Encoder & Data & Objective & Arch. & Params (M) & Input & GPU b1 & GPU b32 & GPU b32 fp16 & CPU b1",
                rf"Encoders and forward-pass latency (ms per image). GPU: {L['gpu'].replace('NVIDIA GeForce ', '')}, fp32 "
                rf"unless stated; CPU: {L['cpu']}, {L['cpu_threads']} threads, batch 1, fp32. Latency is the median over "
                rf"timed iterations of the encoder forward pass on a pre-processed tensor already on the device "
                rf"(image decoding and resizing excluded), divided by the batch size. Bold: fastest per column "
                rf"within each block. The last three rows are architecture-matched controls.",
                "tab:encoders", groups={0: "Main encoders", 6: "Controls"}, size=r"\scriptsize")

# ------------------------------------------------------------------ discrimination
if S:
    D = S["discrimination"]
    rows = []
    for c in rc.ALL:
        d = D[c]
        rows.append({"name": rc.PRETTY[c],
                     "auc": (d["macro_auroc"], f3(d["macro_auroc"])),
                     "ci": (None, ci(d["macro_ci"])),
                     "prc": (d["macro_auprc"], f3(d["macro_auprc"])),
                     "tri": (d["triage_auroc"], f3(d["triage_auroc"])),
                     "tci": (None, ci(d["triage_ci"]))})
        mac(f"Auc{key(c)}", f3(d["macro_auroc"])); mac(f"AucCi{key(c)}", ci(d["macro_ci"]))
        mac(f"Prc{key(c)}", f3(d["macro_auprc"]))
        mac(f"Tri{key(c)}", f3(d["triage_auroc"])); mac(f"TriCi{key(c)}", ci(d["triage_ci"]))
    for k in ("auc", "prc", "tri"):
        bold_best(rows[:6], k); bold_best(rows[6:], k)
    G1 = J("stats_gridv1.json")
    if G1:
        dd = [abs(S["discrimination"][c]["macro_auroc"] - G1["discrimination"][c]["macro_auroc"]) for c in rc.ALL]
        mac("GridSensMaxDiff", f"{max(dd):.3f}")
    for c in rc.ALL:
        for lab in rc.LABELS:
            mac(f"Pl{key(c)}{lab.replace('_', '')}", f3(S["discrimination"][c]["per_label"][lab]))
    mac("NBoot", f"{S['B']:,}"); mac("EqMargin", f"{S['delta']}")
    write_table("tab_macro.tex", rows, ["auc", "ci", "prc", "tri", "tci"],
                r"Encoder & Macro AUROC & 95\,\% CI & Macro AUPRC & Triage AUROC & 95\,\% CI",
                rf"Discrimination on the official NIH ChestX-ray14 test split ({S['n_test']:,} images, "
                rf"{S['n_test_patients']:,} patients). Macro values average the 14 findings; triage is the binary "
                rf"any-finding vs.\ No Finding target. CIs: patient-level cluster bootstrap, B\,=\,{S['B']:,}. "
                rf"Bold: best per column within each block.",
                "tab:macro", groups={0: "Main encoders", 6: "Controls"})

    # pairwise
    rows = []
    for r in S["pairwise"]:
        within = r["ci"][0] > -S["delta"] and r["ci"][1] < S["delta"]
        r["verdict"] = ("different, below margin" if (r["p_holm"] < 0.05 and within) else
                        "equivalent" if within else "different" if r["p_holm"] < 0.05 else "inconclusive")
        rows.append({"name": f"{rc.PRETTY[r['a']]} $-$ {rc.PRETTY[r['b']]}",
                     "d": (None, f"{r['diff']:+.3f}"), "ci": (None, ci(r["ci"])),
                     "p": (None, "$<$0.001" if r["p_holm"] < 0.001 else f"{r['p_holm']:.3f}"),
                     "v": (None, r["verdict"])})
        mac(f"Diff{key(r['a'])}{key(r['b'])}", f"{r['diff']:+.3f}")
        mac(f"DiffCi{key(r['a'])}{key(r['b'])}", ci(r["ci"]))
        mac(f"Verdict{key(r['a'])}{key(r['b'])}", r["verdict"])
    write_table("tab_pairwise.tex", rows, ["d", "ci", "p", "v"],
                r"Pair (A $-$ B) & $\Delta$ macro AUROC & 95\,\% CI & $p_{\mathrm{Holm}}$ & Decision",
                rf"All pairwise macro-AUROC differences among the six main encoders (patient-level cluster bootstrap; "
                rf"two-sided bootstrap $p$-values, Holm-adjusted over 15 comparisons). Equivalence margin fixed before the "
                rf"analysis; the \emph{{below margin}} qualifier was added afterwards (Section~\ref{{sec:metrics}}). Rule: \emph{{different}} if $p_{{\mathrm{{Holm}}}}<0.05$, qualified as \emph{{below margin}} when the "
                rf"whole 95\,\% CI also lies within $\pm${S['delta']}; \emph{{equivalent}} if the CI lies within "
                rf"$\pm${S['delta']} and $p_{{\mathrm{{Holm}}}}\geq0.05$; otherwise \emph{{inconclusive}}.",
                "tab:pairwise", size=r"\scriptsize")

    # contrasts
    rows = []
    for r in S["contrasts"]:
        rows.append({"name": f"{rc.PRETTY[r['a']]} vs.\\ {rc.PRETTY[r['b']]}",
                     "w": (None, r["what"].split(",")[0]), "d": (None, f"{r['diff']:+.3f}"),
                     "ci": (None, ci(r["ci"])), "n": (None, f"{r['labels_ci_excl0_a_better']} / {r['labels_ci_excl0_b_better']}")})
        mac(f"Con{key(r['a'])}{key(r['b'])}", f"{r['diff']:+.3f}"); mac(f"ConCi{key(r['a'])}{key(r['b'])}", ci(r["ci"]))
        mac(f"ConN{key(r['a'])}{key(r['b'])}", rows[-1]["n"][1])
        mac(f"ConNA{key(r['a'])}{key(r['b'])}", str(r["labels_ci_excl0_a_better"]))
        mac(f"ConNB{key(r['a'])}{key(r['b'])}", str(r["labels_ci_excl0_b_better"]))
    write_table("tab_contrasts.tex", rows, ["w", "d", "ci", "n"],
                r"Contrast (A vs.\ B) & Factor varied & $\Delta$ macro AUROC & 95\,\% CI & Findings A$>$B / B$>$A",
                r"Controlled contrasts in which only one factor differs. "
                r"Last column: number of the 14 findings whose per-label 95\,\% CI of the difference excludes zero in "
                r"favour of A / of B.", "tab:contrasts", size=r"\scriptsize")

    # calibration
    conds = ["balanced_raw", "unweighted_raw", "balanced_prior_corrected", "balanced_platt_val", "balanced_isotonic_val"]
    rows = []
    for c in rc.MAIN:
        r = {"name": rc.PRETTY[c]}
        for k in conds:
            v = S["calibration"][c][k]; r[k] = (v["ece"], f3(v["ece"]))
            mac(f"Ece{key(c)}{''.join(w.capitalize() for w in k.split('_'))}", f3(v["ece"]))
            mac(f"Brier{key(c)}{''.join(w.capitalize() for w in k.split('_'))}", f3(v["brier"]))
        r["bb"] = (S["calibration"][c]["balanced_raw"]["brier"], f3(S["calibration"][c]["balanced_raw"]["brier"]))
        r["bp"] = (S["calibration"][c]["balanced_prior_corrected"]["brier"], f3(S["calibration"][c]["balanced_prior_corrected"]["brier"]))
        rows.append(r)
    for k in conds + ["bb", "bp"]:
        bold_best(rows, k, "min")
    write_table("tab_calibration.tex", rows, conds + ["bb", "bp"],
                r"Encoder & \multicolumn{5}{c}{Macro ECE} & \multicolumn{2}{c}{Macro Brier} \\ "
                r" & Balanced & Unweighted & Bal.+prior & Bal.+Platt & Bal.+isotonic & Balanced & Bal.+prior",
                r"Calibration on the official test split under five probe/post-processing conditions (macro over the 14 "
                r"findings; ECE with 15 equal-width bins). \emph{Balanced}: class-balanced training weights (a common "
                r"default); \emph{Unweighted}: no class weights; \emph{Bal.+prior}: balanced logits shifted by the "
                r"training log-prior odds; Platt and isotonic maps were fitted on the validation split. Bold: best "
                r"(lowest) per column.", "tab:calibration", size=r"\scriptsize")

    # operating points
    rows = []
    for c in rc.ALL:
        o90, o95 = S["operating_points"][c]["sens90"], S["operating_points"][c]["sens95"]
        t95 = o95["per_target"]["abnormal"]["val_threshold"]
        r = {"name": rc.PRETTY[c],
             "s95": (o95["macro_sens"], f3(o95["macro_sens"])), "sp95": (o95["macro_spec"], f3(o95["macro_spec"])),
             "or95": (o95["macro_spec_at_sens_test"], f3(o95["macro_spec_at_sens_test"])),
             "ts": (t95["sens"], f3(t95["sens"])), "tsp": (t95["spec"], f3(t95["spec"])),
             "npv": (t95["npv"], f3(t95["npv"])), "def": (t95["frac_neg"], pct(t95["frac_neg"]))}
        rows.append(r)
        for k in r:
            if k != "name":
                mac(f"Op{k.capitalize()}{key(c)}", r[k][1])
        mac(f"OpMissed{key(c)}", f"{1000*(1-t95['sens']):.0f}")
        mac(f"OpTriOr{key(c)}", f3(o95["per_target"]["abnormal"]["spec_at_sens_test"]))
        o90t = o90["per_target"]["abnormal"]
        mac(f"OpTriOrNinety{key(c)}", f3(o90t["spec_at_sens_test"]))
        mac(f"OpSpNinety{key(c)}", f3(o90["macro_spec"])); mac(f"OpOrNinety{key(c)}", f3(o90["macro_spec_at_sens_test"]))
    for k in ("sp95", "or95", "tsp", "npv", "def"):
        bold_best(rows[:6], k); bold_best(rows[6:], k)
    write_table("tab_operating.tex", rows, ["s95", "sp95", "or95", "ts", "tsp", "npv", "def"],
                r" & \multicolumn{3}{c}{14 findings (macro)} & \multicolumn{4}{c}{Triage: any finding vs.\ No Finding} \\ "
                r"Encoder & Sens. & Spec. & Spec.@95\,\% (test) & Sens. & Spec. & NPV & Below thr. (\%)",
                r"High-sensitivity operating points on the official test split. Thresholds were fixed on the "
                r"validation split to reach 95\,\% sensitivity and transported unchanged to the test split; "
                r"`Spec.@95\,\%~(test)' is the threshold-free specificity at exactly 95\,\% test sensitivity. "
                r"Missed-diagnosis rate = 1 $-$ sensitivity. `Below thr.' = share of test studies that fall below the "
                r"triage threshold (candidate for deprioritisation). Bold: best per column within each block.",
                "tab:operating", groups={0: "Main encoders", 6: "Controls"}, size=r"\scriptsize")

# ------------------------------------------------------------------ external
if E:
    rows = []
    for c in rc.ALL:
        e = E["encoders"][c]; t = e["ops"]["triage_sens95_nih_threshold"]
        r = {"name": rc.PRETTY[c], "nih": (e["macro9_nih_test"], f3(e["macro9_nih_test"])),
             "ext": (e["macro9"], f3(e["macro9"])), "ci": (None, ci(e["macro9_ci"])),
             "tri": (e["triage_auroc"], f3(e["triage_auroc"])),
             "ts": (t["sens"], f3(t["sens"])), "tsp": (t["spec"], f3(t["spec"]))}
        rows.append(r)
        for k in r:
            if k != "name":
                mac(f"Ext{k.capitalize()}{key(c)}", r[k][1])
    for k in ("nih", "ext", "tri", "tsp"):
        bold_best(rows[:6], k); bold_best(rows[6:], k)
    for c in rc.ALL:
        e = E["encoders"][c]
        mac(f"ExtOr{key(c)}", f3(e["ops"]["triage_spec_at_sens95_external"]))
        mac(f"ExtEce{key(c)}", f3(e["ece_prior_corrected"]))
        mac(f"ExtDrop{key(c)}", f"{e['macro9'] - e['macro9_nih_test']:+.3f}")
    for c in rc.MAIN:
        mac(f"ExtFirst{key(c)}", pct(E["rank_bootstrap"][c][0] / sum(E["rank_bootstrap"][c])))
    for a, b in (("rad_dino_224", "dinov2_base_224"), ("biomedclip_224", "clip_b16_224"), ("rad_dino_518", "rad_dino_224")):
        mac(f"ExtCon{key(a)}{key(b)}", f"{E['encoders'][a]['macro9'] - E['encoders'][b]['macro9']:+.3f}")
    for p_ in E["pairwise_main"]:
        mac(f"ExtDiff{key(p_['a'])}{key(p_['b'])}", f"{p_['diff']:+.3f}"); mac(f"ExtDiffCi{key(p_['a'])}{key(p_['b'])}", ci(p_["ci"]))
    nsig = sum(1 for p in E["pairwise_main"] if p["p_holm"] < 0.05)
    mac("ExtNSig", str(nsig))
    mac("ExtN", f"{E['n_reports_with_frontal']:,}"); mac("ExtTriN", f"{E['n_triage_eligible']:,}")
    mac("ExtTriPos", f"{E['n_triage_abnormal']:,}")
    write_table("tab_external.tex", rows, ["nih", "ext", "ci", "tri", "ts", "tsp"],
                r" & \multicolumn{3}{c}{Macro AUROC, 9 mapped findings} & \multicolumn{3}{c}{Triage} \\ "
                r"Encoder & NIH test & External & 95\,\% CI & AUROC & Sens.\ (NIH thr.) & Spec.\ (NIH thr.)",
                rf"External validation on the Indiana University collection (Open-i; {E['n_reports_with_frontal']:,} "
                rf"reports, one frontal image each; probes trained on NIH and applied without re-fitting). Triage: "
                rf"{E['n_triage_eligible']:,} reports that are either tagged normal or carry a mapped finding "
                rf"({E['n_triage_abnormal']:,} abnormal). Thresholds are the NIH-validation thresholds for 95\,\% "
                rf"sensitivity. CI: report-level bootstrap, B\,=\,500. Bold: best per column within each block.",
                "tab:external", groups={0: "Main encoders", 6: "Controls"}, size=r"\scriptsize")

# ------------------------------------------------------------------ per-finding AUROC with CI (appendix)
if S:
    D = S["discrimination"]
    lines = [r"\begin{table}[H]\centering", r"\scriptsize",
             r"\caption{Per-finding test AUROC $\pm$ half-width of the 95\,\% patient-level cluster-bootstrap CI "
             r"(main encoders; official test split). $n_+$: positive test images. Bold: best encoder per finding.}"
             r"\label{tab:perlabel}",
             r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lr" + "c" * len(rc.MAIN) + "}", r"\toprule",
             "Finding & $n_+$ & " + " & ".join(rc.PRETTY[c] for c in rc.MAIN) + r" \\", r"\midrule"]
    te_ = df[df.split == "test"]
    for lab in rc.LABELS + ["abnormal"]:
        vals = [D[c]["per_label"][lab] for c in rc.MAIN]; best = max(vals)
        cells = []
        for c, v in zip(rc.MAIN, vals):
            lo, hi = D[c]["per_label_ci"][lab]
            txt = f"{v:.3f}$\\pm${(hi - lo) / 2:.3f}"
            cells.append(r"\textbf{" + txt + "}" if np.isclose(v, best) else txt)
        nm = "Any finding (triage)" if lab == "abnormal" else lab.replace("_", " ")
        lines.append(f"{nm} & {int(te_[lab].sum()):,} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    open(os.path.join(TAB, "tab_perlabel.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ small-sample implementation sensitivity
DEC = J("decomposition.json")
if DEC and S:
    rows = []
    for c in rc.MAIN:
        d = DEC[c]
        r = {"name": rc.PRETTY[c]}
        for k, col in (("orig_emb|orig_probe", "oo"), ("corr_emb|orig_probe", "co"), ("corr_emb|study_probe", "cr")):
            r[col] = (d[k], f3(d[k]))
            mac(f"Dec{col.upper()}{key(c)}", f3(d[k]))
        r["full"] = (S["discrimination"][c]["macro_auroc"], f3(S["discrimination"][c]["macro_auroc"]))
        rows.append(r)
    for k in ("oo", "co", "cr", "full"):
        bold_best(rows, k)
    write_table("tab_decomposition.tex", rows, ["oo", "co", "cr", "full"],
                r" & \multicolumn{3}{c}{3,000-image subset (2,000 / 500 / 500)} & Full data \\ "
                r"Encoder & Pipeline A & Pipeline B & Pipeline C & Pipeline C, official test",
                r"Sensitivity of a small-sample benchmark to implementation details (macro AUROC on a fixed 3,000-image "
                r"subset of the first NIH image shard with a 2,000/500/500 patient-disjoint split and 55\,\% any-finding "
                r"enrichment). Pipeline A: images stored at $224^2$ (upsampled to $518^2$ for RAD-DINO), ImageNet "
                r"normalisation for all encoders except the CLIP models, un-projected CLIP-B/16 feature, and features "
                r"standardised to zero mean and unit variance before a \texttt{liblinear} logistic regression. Pipeline B: "
                r"as A but with the image preparation, normalisation and representations of Section~\ref{sec:backbones}. "
                r"Pipeline C: as B with the probe of Section~\ref{sec:probe} (no feature standardisation). Last column: "
                r"pipeline C on the full data and official test split. Bold: best per column.", "tab:decomposition", size=r"\scriptsize")

# ------------------------------------------------------------------ Pareto sets
if S and L:
    for lk, nm in (("gpu_fp32_b1_ms", "GPU"), ("cpu_fp32_b1_ms", "CPU"), ("gpu_fp32_b32_ms", "GPUb")):
        pts = {c: (L["configs"][c][lk], S["discrimination"][c]["macro_auroc"]) for c in rc.ALL}
        front = [c for c in pts if not any((pts[o][0] <= pts[c][0] and pts[o][1] >= pts[c][1]) and
                                           (pts[o][0] < pts[c][0] or pts[o][1] > pts[c][1]) for o in pts)]
        front = sorted(front, key=lambda c: pts[c][0])
        mac(f"Pareto{nm}", ", ".join(rc.PRETTY[c] for c in front))
        frm = [c for c in front if c in rc.MAIN]
        ptsm = {c: pts[c] for c in rc.MAIN}
        fm = [c for c in ptsm if not any((ptsm[o][0] <= ptsm[c][0] and ptsm[o][1] >= ptsm[c][1]) and
                                         (ptsm[o][0] < ptsm[c][0] or ptsm[o][1] > ptsm[c][1]) for o in ptsm)]
        mac(f"ParetoMain{nm}", ", ".join(rc.PRETTY[c] for c in sorted(fm, key=lambda c: ptsm[c][0])))

# ------------------------------------------------------------------ replicates
if R:
    for c in R["cfgs"]:
        own = np.array(R["small_replicates"][c]["own"]); full = np.array(R["small_replicates"][c]["full"])
        mac(f"RepOwnMean{key(c)}", f3(own.mean())); mac(f"RepOwnSd{key(c)}", f3(own.std(ddof=1)))
        mac(f"RepOwnRange{key(c)}", f"{own.min():.3f}--{own.max():.3f}")
        mac(f"RepFullMean{key(c)}", f3(full.mean())); mac(f"RepFullSd{key(c)}", f3(full.std(ddof=1)))
    lcm = lambda c, n: float(np.mean([r["macro"] for r in R["learning_curve"][c][n]]))
    full_n = max(R["learning_curve"]["rad_dino_518"], key=int)
    for tag, n in (("One", "1000"), ("Full", full_n)):
        mac(f"LcRadBio{tag}", f3(lcm("rad_dino_518", n) - lcm("biomedclip_224", n)))
        mac(f"LcBioDinoS{tag}", f3(lcm("biomedclip_224", n) - lcm("dinov2_small_224", n)))
    mac("LcGainDinoS", f3(lcm("dinov2_small_224", full_n) - lcm("dinov2_small_224", "1000")))
    mac("LcGainBiomedClip", f3(lcm("biomedclip_224", full_n) - lcm("biomedclip_224", "1000")))
    mac("KendallSmallOwn", f"{R['small_own_kendall_w']:.2f}"); mac("KendallSmallFull", f"{R['small_full_kendall_w']:.2f}")
    mac("KendallPartitions", f"{R['partitions_kendall_w']:.2f}")
    for c in rc.MAIN:
        mac(f"RepFirstOwn{key(c)}", str(R["small_own_rank_counts"][c][0]))
        mac(f"RepFirstFull{key(c)}", str(R["small_full_rank_counts"][c][0]))
if S:
    for c in rc.MAIN:
        mac(f"BootFirst{key(c)}", pct(S["rank_bootstrap"][c][0] / S["B"]))

with open(os.path.join(MS, "numbers.tex"), "w") as f:
    f.write("% AUTO-GENERATED by src/make_tables.py — do not edit by hand\n")
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"[ok] {len(macros)} macros; tables:", sorted(os.listdir(TAB)))
