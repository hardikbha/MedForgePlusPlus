"""Assemble the EXACT MedForge 3-tier matrix (In-Domain / Cross-Forgery / Cross-Model) with our
protocol-faithful 8B row, computed directly from the shared-test gens. Emits CSV + a comparison chart."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, csv, sys
import numpy as np
sys.path.insert(0, _release_path('CIRCE/circe'))
from circe_reward import parse_output
D = _release_path('AAAI/MedForge++/')
DF = {"edit", "remove", "inject"}

# MedForge-Reasoner published accuracy/F1 (Real/Implant/Remove x ID/XF/XM)
MF_ACC = {"Real": (99.24, 95.24, 92.86), "Implant": (99.24, 93.39, 94.86), "Remove": (99.21, 99.15, 94.09)}
MF_F1  = {"Real": (98.86, 93.29, 92.07), "Implant": (98.86, 92.97, 94.63), "Remove": (99.21, 99.14, 93.76)}

def metrics(fn):
    gens = [json.loads(l) for l in open(D + fn)]
    reals = [g for g in gens if g["type"] == "real"]
    def verd(g): return parse_output(g["output"])["verdict"] or "real"
    out = {}
    # accuracy per category
    for grp, key in [("real", "Real"), ("edit", "Implant"), ("remove", "Remove")]:
        items = [g for g in gens if g["type"] == grp]
        want = "real" if grp == "real" else "deepfake"
        out[key + "_acc"] = 100 * sum(verd(g) == want for g in items) / len(items) if items else None
    # binary F1 per fake group (positive=deepfake, vs reals)
    for grp, key in [("edit", "Implant"), ("remove", "Remove")]:
        pos = [g for g in gens if g["type"] == grp]
        tp = sum(verd(g) == "deepfake" for g in pos); fn_ = len(pos) - tp
        fp = sum(verd(g) == "deepfake" for g in reals)
        prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn_) if tp+fn_ else 0
        out[key + "_f1"] = 100*2*prec*rec/(prec+rec) if prec+rec else 0
    # Real F1 (positive=real vs all deepfakes)
    df = [g for g in gens if g["type"] in DF]
    tp = sum(verd(g) == "real" for g in reals); fn_ = len(reals) - tp
    fp = sum(verd(g) == "real" for g in df)
    prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn_) if tp+fn_ else 0
    out["Real_f1"] = 100*2*prec*rec/(prec+rec) if prec+rec else 0
    return out

tiers = {"ID": metrics("gen_mfp_indomain.jsonl"),
         "XF": metrics("gen_mfp_crossforgery.jsonl"),
         "XM": metrics("gen_mfp_crossmodel.jsonl")}

def ours(metric):  # metric 'acc' or 'f1'
    row = {}
    for cat in ["Real", "Implant", "Remove"]:
        row[cat] = tuple(round(tiers[t][f"{cat}_{metric}"], 1) for t in ["ID", "XF", "XM"])
    return row

OA, OF = ours("acc"), ours("f1")

# ---- print head-to-head ----
def show(title, ours_d, mf_d):
    print(f"\n=== {title} ===")
    print(f"{'Category':9s} | {'ID (ours/MF/Δ)':22s} | {'XF (ours/MF/Δ)':22s} | {'XM (ours/MF/Δ)':22s}")
    for cat in ["Real", "Implant", "Remove"]:
        cells = []
        for i, t in enumerate(["ID", "XF", "XM"]):
            o, m = ours_d[cat][i], mf_d[cat][i]
            cells.append(f"{o:5.1f}/{m:5.1f}/{o-m:+5.1f}")
        print(f"{cat:9s} | {cells[0]:22s} | {cells[1]:22s} | {cells[2]:22s}")
    # avg row
    for i, t in enumerate(["ID", "XF", "XM"]):
        oa = np.mean([ours_d[c][i] for c in ["Real","Implant","Remove"]])
        ma = np.mean([mf_d[c][i] for c in ["Real","Implant","Remove"]])
        print(f"{'AVG '+t:9s} ours={oa:.1f}  MedForge={ma:.1f}  Δ={oa-ma:+.1f}")

show("ACCURACY", OA, MF_ACC)
show("F1", OF, MF_F1)

# ---- CSV ----
HEAD = ["Method","Metric","Real_ID","Real_XF","Real_XM","Implant_ID","Implant_XF","Implant_XM",
        "Remove_ID","Remove_XF","Remove_XM"]
def flat(d): return [d["Real"][0],d["Real"][1],d["Real"][2],d["Implant"][0],d["Implant"][1],d["Implant"][2],
                     d["Remove"][0],d["Remove"][1],d["Remove"][2]]
out = D + "medforge_protocol_matrix.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(HEAD)
    w.writerow(["MedForge-Reasoner (8B, SFT+GSPO)","Accuracy"]+flat(MF_ACC))
    w.writerow(["MedForge-Reasoner (8B, SFT+GSPO)","F1"]+flat(MF_F1))
    w.writerow(["Ours (CIRCE-8B, SFT-only, exact protocol)","Accuracy"]+flat(OA))
    w.writerow(["Ours (CIRCE-8B, SFT-only, exact protocol)","F1"]+flat(OF))
print("\nwrote", out)
