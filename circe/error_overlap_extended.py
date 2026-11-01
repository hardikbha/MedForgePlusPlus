"""
Ablation 7 (extended): per-class error complementarity, agreement matrix, and confusion
matrices for the Stage-1 (replay-SFT) vs Stage-2 (best RL) checkpoints on the Cross-Forgery
test set. Builds directly on error_overlap_analysis.py's pairing logic and its verified
93.35/2.34/2.40/1.90 split -- no GPU, reads only already-saved generation files.

Outputs:
  - console table: per-class (Real/Implant/Remove) complementarity breakdown
  - figures/ablation7_agreement_heatmap.png  (2x2 agreement heatmap, overall + per-class)
  - figures/ablation7_confusion_sft.png, ablation7_confusion_rl.png  (verdict confusion matrices)
  - results/ablation7_overlap_extended.csv
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json
import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from circe_reward import parse_output

DF = {"remove", "inject", "edit", "deepfake", "removal", "injection", "implant"}
BASE = _release_path('AAAI/MedForge++')
SFT_GEN = f"{BASE}/gen_mfp_crossforgery_implant_replay_sft_8b.jsonl"
RL_GEN = f"{BASE}/gen_v2_classdom_beta0_final.jsonl"
OUT_DIR = _release_path('CIRCE/paper/figures')
RESULTS_DIR = _release_path('CIRCE/results_ablation')
os.makedirs(RESULTS_DIR, exist_ok=True)


def gt_class(rec):
    if rec["type"] not in DF:
        return "real"
    low = rec["image"].lower()
    if "remov" in low or rec["type"] in {"remove", "removal"}:
        return "remove"
    return "implant"


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        gt = "deepfake" if r["type"] in DF else "real"
        pred = parse_output(r["output"])["verdict"] or "real"
        out[r["image"]] = {
            "correct": pred == gt,
            "gt": gt,
            "pred": pred,
            "cls": gt_class(r),
        }
    return out


def main():
    sft = load(SFT_GEN)
    rl = load(RL_GEN)
    common = sorted(set(sft) & set(rl))
    n = len(common)

    # ---- per-class complementarity ----
    classes = ["real", "implant", "remove"]
    rows = []
    print(f"{'Class':<10}{'n':>6}{'SFT acc':>10}{'RL acc':>10}{'Both OK':>10}{'SFT-only':>10}{'RL-only':>10}{'Both wrong':>12}{'Oracle':>9}")
    for c in classes + ["ALL"]:
        keys = common if c == "ALL" else [k for k in common if sft[k]["cls"] == c]
        m = len(keys)
        both_ok = sum(1 for k in keys if sft[k]["correct"] and rl[k]["correct"])
        both_wrong = sum(1 for k in keys if not sft[k]["correct"] and not rl[k]["correct"])
        sft_only = sum(1 for k in keys if sft[k]["correct"] and not rl[k]["correct"])
        rl_only = sum(1 for k in keys if not sft[k]["correct"] and rl[k]["correct"])
        sft_acc = sum(sft[k]["correct"] for k in keys) / m
        rl_acc = sum(rl[k]["correct"] for k in keys) / m
        oracle = (both_ok + sft_only + rl_only) / m
        print(f"{c:<10}{m:>6}{sft_acc*100:>9.2f}%{rl_acc*100:>9.2f}%{both_ok/m*100:>9.2f}%{sft_only/m*100:>9.2f}%{rl_only/m*100:>9.2f}%{both_wrong/m*100:>11.2f}%{oracle*100:>8.2f}%")
        rows.append(dict(cls=c, n=m, sft_acc=sft_acc, rl_acc=rl_acc,
                          both_ok=both_ok / m, sft_only=sft_only / m, rl_only=rl_only / m,
                          both_wrong=both_wrong / m, oracle=oracle))

    with open(f"{RESULTS_DIR}/ablation7_overlap_extended.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[saved] {RESULTS_DIR}/ablation7_overlap_extended.csv")

    # ---- agreement heatmap (per-class outcome distribution) ----
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = ["Both\ncorrect", "SFT-only\ncorrect", "RL-only\ncorrect", "Both\nwrong"]
    data = np.array([[r["both_ok"], r["sft_only"], r["rl_only"], r["both_wrong"]] for r in rows[:3]]) * 100
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(["Real", "Implant", "Remove"], fontsize=9)
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{data[i,j]:.1f}%", ha="center", va="center",
                     color="white" if data[i, j] > 50 else "black", fontsize=8.5)
    ax.set_title("Per-class agreement: Stage-1 (SFT) vs Stage-2 (RL), Cross-Forgery", fontsize=10)
    fig.colorbar(im, ax=ax, label="% of class")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ablation7_agreement_heatmap.png", dpi=200)
    plt.close(fig)
    print(f"[saved] {OUT_DIR}/ablation7_agreement_heatmap.png")

    # ---- confusion matrices ----
    for name, d in [("sft", sft), ("rl", rl)]:
        cm = np.zeros((2, 2), dtype=int)
        idx = {"real": 0, "deepfake": 1}
        for k in common:
            cm[idx[d[k]["gt"]], idx[d[k]["pred"]]] += 1
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["real", "deepfake"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["real", "deepfake"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
        ax.set_title(f"{'Stage-1 SFT' if name=='sft' else 'Stage-2 RL'} confusion matrix", fontsize=9.5)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/ablation7_confusion_{name}.png", dpi=200)
        plt.close(fig)
        print(f"[saved] {OUT_DIR}/ablation7_confusion_{name}.png")


if __name__ == "__main__":
    main()
