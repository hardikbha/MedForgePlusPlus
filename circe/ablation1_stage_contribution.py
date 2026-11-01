"""
Ablation 1 (highest priority): stage contribution.

Model A: Stage 1 only, no replay          -> gen_mfp_crossforgery.jsonl
Model B: Stage 1 + Failure-Targeted Replay -> gen_mfp_crossforgery_implant_replay_sft_8b.jsonl
Model C: Stage 1 + Replay + GRPO (Stage 2) -> gen_v2_classdom_beta0_final.jsonl
Model D: Stage 1 + Replay + GRPO + Interp (final BRIDGE, alpha=0.10) -> gen_interp_alpha0.10.jsonl

All four files are generations from the identical 2,994-image Cross-Forgery test set,
using the same harness (eval_medforge.py / circe_reward.py parser), so this is a clean,
apples-to-apples comparison requiring no new training or evaluation runs.

Reports: Accuracy, Macro-F1, Precision, Recall (over the binary real/deepfake verdict),
mean IoU, and Localization Accuracy (IoU >= 0.5 among boxed predictions with a GT box).
No GPU / no training -- reads only already-saved generation files.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json
import os
import sys
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from circe_reward import parse_output

DF = {"remove", "inject", "edit", "deepfake", "removal", "injection", "implant"}
BASE = _release_path('AAAI/MedForge++')
OUT_DIR = _release_path('CIRCE/paper/figures')
RESULTS_DIR = _release_path('CIRCE/results_ablation')
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = [
    ("A: Stage 1 only",              f"{BASE}/gen_mfp_crossforgery.jsonl"),
    ("B: +Failure-Targeted Replay",  f"{BASE}/gen_mfp_crossforgery_implant_replay_sft_8b.jsonl"),
    ("C: +GRPO (Stage 2)",           f"{BASE}/gen_v2_classdom_beta0_final.jsonl"),
    ("D: +Interpolation (BRIDGE)",   f"{BASE}/gen_interp_alpha0.10.jsonl"),
]


def iou(a, b):
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def score(path):
    tp = fp = tn = fn = 0
    ious, loc_ok, loc_n = [], 0, 0
    n = 0
    for line in open(path):
        r = json.loads(line)
        gt = "deepfake" if r["type"] in DF else "real"
        parsed = parse_output(r["output"])
        pred = parsed["verdict"] or "real"
        n += 1
        if gt == "deepfake" and pred == "deepfake":
            tp += 1
        elif gt == "real" and pred == "real":
            tn += 1
        elif gt == "real" and pred == "deepfake":
            fp += 1
        else:
            fn += 1
        gt_box = r.get("gt_box")
        pred_box = parsed.get("box")
        if gt_box is not None:
            v = iou(gt_box, pred_box) if pred_box else 0.0
            ious.append(v)
            loc_n += 1
            if v >= 0.5:
                loc_ok += 1

    acc = (tp + tn) / n
    prec_pos = tp / (tp + fp) if (tp + fp) else 0.0
    rec_pos = tp / (tp + fn) if (tp + fn) else 0.0
    f1_pos = 2 * prec_pos * rec_pos / (prec_pos + rec_pos) if (prec_pos + rec_pos) else 0.0
    prec_neg = tn / (tn + fn) if (tn + fn) else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) else 0.0
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) else 0.0
    macro_f1 = (f1_pos + f1_neg) / 2
    macro_prec = (prec_pos + prec_neg) / 2
    macro_rec = (rec_pos + rec_neg) / 2
    miou = sum(ious) / len(ious) if ious else 0.0
    loc_acc = loc_ok / loc_n if loc_n else 0.0
    return dict(n=n, acc=acc, macro_f1=macro_f1, precision=macro_prec, recall=macro_rec,
                miou=miou, loc_acc=loc_acc)


def main():
    rows = []
    print(f"{'Model':<32}{'Acc':>8}{'MacroF1':>10}{'Prec':>8}{'Rec':>8}{'mIoU':>8}{'LocAcc':>9}")
    for name, path in MODELS:
        if not os.path.exists(path):
            print(f"{name:<32} MISSING: {path}")
            continue
        m = score(path)
        print(f"{name:<32}{m['acc']*100:>7.2f}%{m['macro_f1']*100:>9.2f}%{m['precision']*100:>7.2f}%"
              f"{m['recall']*100:>7.2f}%{m['miou']*100:>7.2f}%{m['loc_acc']*100:>8.2f}%")
        rows.append(dict(model=name, **m))

    with open(f"{RESULTS_DIR}/ablation1_stage_contribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[saved] {RESULTS_DIR}/ablation1_stage_contribution.csv")

    # LaTeX table
    with open(f"{RESULTS_DIR}/ablation1_table.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Stage contribution ablation, Cross-Forgery test set ($n{=}2{,}994$). "
                "Each row adds one stage of the \\bridge{} recipe.}\n")
        f.write("\\label{tab:ablation-stage-contribution}\n")
        f.write("\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("Model & Acc. & F1 & Prec. & Rec. & mIoU & Loc.\\ Acc. \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['model']} & {r['acc']*100:.2f} & {r['macro_f1']*100:.2f} & "
                     f"{r['precision']*100:.2f} & {r['recall']*100:.2f} & {r['miou']*100:.2f} & "
                     f"{r['loc_acc']*100:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")
    print(f"[saved] {RESULTS_DIR}/ablation1_table.tex")

    # bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [r["model"].split(": ")[1] for r in rows]
    x = np.arange(len(rows))
    axes[0].bar(x, [r["acc"] * 100 for r in rows], color="#2b6cb0", width=0.55, label="Accuracy")
    axes[0].bar(x, [r["macro_f1"] * 100 for r in rows], color="#c05621", width=0.28, label="Macro-F1")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=7.5)
    axes[0].set_ylabel("%"); axes[0].set_ylim(90, 100)
    axes[0].set_title("Classification", fontsize=10); axes[0].legend(fontsize=8)

    axes[1].bar(x, [r["miou"] * 100 for r in rows], color="#2f855a", width=0.55, label="mIoU")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=7.5)
    axes[1].set_ylabel("%"); axes[1].set_title("Localization (mIoU)", fontsize=10)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ablation1_stage_contribution.png", dpi=200)
    print(f"[saved] {OUT_DIR}/ablation1_stage_contribution.png")


if __name__ == "__main__":
    main()
