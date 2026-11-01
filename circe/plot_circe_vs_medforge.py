"""CIRCE (8B, full-data) vs MedForge — accuracy bar chart, In-Domain + Cross-Model.
Per-category accuracy computed DIRECTLY from the eval gens (no hand-transcription)."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, sys
sys.path.insert(0, _release_path('CIRCE/circe'))
from circe_reward import parse_output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = _release_path('AAAI/MedForge++/')
DF = {"edit", "remove", "inject"}
GRP = {"edit": "Implant", "remove": "Remove", "real": "Real"}


def cat_acc(fn):
    from collections import defaultdict
    tot, cor = defaultdict(int), defaultdict(int)
    for l in open(D + fn):
        g = json.loads(l)
        want = "deepfake" if g["type"] in DF else "real"
        v = parse_output(g["output"])["verdict"] or "real"
        k = GRP.get(g["type"], g["type"])
        tot[k] += 1; cor[k] += int(v == want)
    return {k: 100 * cor[k] / tot[k] for k in tot}


id_ours = cat_acc("gen_sft8full_indomain.jsonl")
cm_ours = cat_acc("gen_sft8full_crossmodel.jsonl")
id_ours["AVG"] = np.mean([id_ours[k] for k in ("Real", "Implant", "Remove")])
cm_ours["AVG"] = np.mean([cm_ours[k] for k in ("Real", "Implant", "Remove")])

# MedForge-Reasoner reported accuracy (paper table)
MF_ID = {"Real": 99.24, "Implant": 99.24, "Remove": 99.21, "AVG": 99.23}
MF_CM = {"Real": 92.86, "Implant": 94.86, "Remove": 94.09, "AVG": 93.94}

TEAL, GOLD = "#1a7f7f", "#c98a1a"
cats = ["Real", "Implant", "Remove", "AVG"]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("CIRCE-8B (Ours, full-data) vs MedForge  (red = gap in points)",
             fontsize=15, fontweight="bold")

for ax, ours, mf, title in [(axes[0], id_ours, MF_ID, "Accuracy — In-Domain"),
                            (axes[1], cm_ours, MF_CM, "Accuracy — Cross-Model")]:
    x = np.arange(len(cats)); w = 0.38
    ov = [ours[c] for c in cats]; mv = [mf[c] for c in cats]
    ax.bar(x - w/2, ov, w, color=TEAL, label="Ours (CIRCE-8B, full-data SFT)")
    ax.bar(x + w/2, mv, w, color=GOLD, label="MedForge-Reasoner (8B, 90K)")
    for i, (o, m) in enumerate(zip(ov, mv)):
        ax.text(i - w/2, o + 0.6, f"{o:.1f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(i + w/2, m + 0.6, f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")
        gap = o - m  # positive => we win
        col = "green" if gap >= 0 else "red"
        ax.text(i, max(o, m) + 4.2, f"{gap:+.1f}", ha="center", fontsize=11,
                fontweight="bold", color=col)
    ax.axhline(100, ls=":", c="grey", lw=1)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 112)
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.legend(loc="lower left", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = D + "circe8b_full_vs_medforge.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
print("In-Domain  Ours:", {k: round(id_ours[k], 1) for k in cats})
print("Cross-Model Ours:", {k: round(cm_ours[k], 1) for k in cats})
