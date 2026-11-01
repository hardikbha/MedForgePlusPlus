"""Build the MedForge main-results table as CSV, with our model ('Ours') added."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, csv, sys
sys.path.insert(0, _release_path('CIRCE/circe'))
from circe_reward import parse_output

D = _release_path('AAAI/MedForge++')
DF = {"edit", "remove", "inject"}

# ---- transcribed from MedForge paper table (main numbers) ----
# order: Real[ID,XF,XM], Implant[ID,XF,XM], Remove[ID,XF,XM], Avg[ID,XF,XM]
ACC = {
 "SIDA-7B":       [77.56,73.14,73.42, 79.83,76.18,76.57, 82.31,77.92,75.68, 79.90,75.75,75.22],
 "SIDA-13B":      [89.37,85.68,84.25, 91.52,87.24,85.86, 93.84,90.17,86.53, 91.58,87.70,85.55],
 "FakeVLM":       [86.48,81.86,76.94, 88.93,83.57,79.82, 91.27,86.35,82.51, 88.89,83.93,79.76],
 "AIGI-Holmes":   [90.24,86.53,84.37, 88.12,84.46,80.91, 90.58,90.64,87.23, 89.65,87.21,84.17],
 "Qwen3-VL-Flash":[57.60,55.50,54.90, 47.83,50.47,50.93, 54.17,54.17,54.61, 53.20,53.38,53.48],
 "Qwen3-VL-Plus": [54.14,54.28,55.10, 57.42,55.78,56.03, 55.80,55.86,56.10, 55.79,55.31,55.74],
 "Gemini 3 Flash":[71.26,73.39,72.57, 57.33,62.46,60.14, 57.14,62.27,60.34, 61.91,66.04,64.35],
 "Gemini 3 Pro":  [67.29,65.11,64.49, 75.34,75.24,74.80, 65.05,66.49,65.70, 69.23,68.95,68.33],
 "MedForge-Reasoner":[99.24,95.24,92.86, 99.24,93.39,94.86, 99.21,99.15,94.09, 99.23,95.93,93.94],
}
F1 = {
 "SIDA-7B":       [76.84,72.47,67.73, 86.47,74.06,69.85, 81.64,77.28,72.94, 81.65,74.60,70.17],
 "SIDA-13B":      [88.69,84.95,80.58, 90.86,86.57,83.17, 93.18,89.46,85.87, 90.91,86.99,83.21],
 "FakeVLM":       [85.73,81.18,76.29, 88.27,82.86,79.16, 90.58,85.67,81.84, 88.19,83.24,79.10],
 "AIGI-Holmes":   [88.57,85.84,86.68, 90.46,88.73,84.25, 89.82,85.97,83.58, 89.62,86.85,84.84],
 "Qwen3-VL-Flash":[32.48,38.62,40.26, 55.08,53.70,52.69, 63.13,59.51,58.78, 50.23,50.61,50.58],
 "Qwen3-VL-Plus": [47.73,46.36,46.18, 54.70,53.91,54.99, 52.42,54.26,55.32, 51.62,51.51,52.16],
 "Gemini 3 Flash":[24.42,41.14,34.78, 70.22,72.05,71.11, 70.07,71.91,71.34, 54.90,61.70,59.08],
 "Gemini 3 Pro":  [61.82,62.64,62.22, 74.22,71.53,71.02, 59.47,57.62,56.19, 65.17,63.93,63.14],
 "MedForge-Reasoner":[98.86,93.29,92.07, 98.86,92.97,94.63, 99.21,99.14,93.76, 98.98,95.13,93.49],
}

def load(f):
    return [json.loads(l) for l in open(os.path.join(D, f))] if os.path.exists(os.path.join(D, f)) else []

def group_metrics(gens):
    """per-group accuracy + binary-F1 (reals+group, positive=deepfake)."""
    reals=[g for g in gens if g["type"]=="real"]
    def acc(items, want): return 100*sum(int((parse_output(g["output"])["verdict"] or "real")==want) for g in items)/len(items) if items else None
    out={}
    out["real_acc"]=acc(reals,"real")
    for grp,key in [("edit","implant"),("remove","remove")]:
        items=[g for g in gens if g["type"]==grp]
        out[key+"_acc"]=acc(items,"deepfake")
        # binary F1 on reals+group
        pos=items; neg=reals
        tp=sum(int((parse_output(g["output"])["verdict"] or "real")=="deepfake") for g in pos)
        fn=len(pos)-tp
        fp=sum(int((parse_output(g["output"])["verdict"] or "real")=="deepfake") for g in neg)
        prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
        out[key+"_f1"]=100*2*prec*rec/(prec+rec) if prec+rec else 0
    # real F1 (positive=real, on reals+all deepfakes)
    df=[g for g in gens if g["type"] in DF]
    tp=sum(int((parse_output(g["output"])["verdict"] or "real")=="real") for g in reals)
    fn=len(reals)-tp
    fp=sum(int((parse_output(g["output"])["verdict"] or "real")=="real") for g in df)
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    out["real_f1"]=100*2*prec*rec/(prec+rec) if prec+rec else None
    return out

NA="n/a"
def avg(*xs):
    v=[x for x in xs if isinstance(x,(int,float))]
    return round(sum(v)/len(v),2) if v else NA

def ours_rows(id_gen, cm_gen):
    idm=group_metrics(load(id_gen)); cmm=group_metrics(load(cm_gen))
    if not idm.get("real_acc") and not idm.get("implant_acc"):
        return None, None
    acc=[round(idm["real_acc"],1),NA,NA,
         round(idm["implant_acc"],1),NA,round(cmm["implant_acc"],1),
         round(idm["remove_acc"],1),NA,round(cmm["remove_acc"],1),
         avg(idm["real_acc"],idm["implant_acc"],idm["remove_acc"]),NA,avg(cmm["implant_acc"],cmm["remove_acc"])]
    f1=[round(idm["real_f1"],1) if idm["real_f1"] else NA,NA,NA,
        round(idm["implant_f1"],1),NA,round(cmm["implant_f1"],1),
        round(idm["remove_f1"],1),NA,round(cmm["remove_f1"],1),
        avg(idm.get("real_f1"),idm["implant_f1"],idm["remove_f1"]),NA,avg(cmm["implant_f1"],cmm["remove_f1"])]
    return acc,f1

# our models (each: label, indomain_gen, crossmodel_gen) — full-data runs on the fixed splits
OURS = [
 ("Ours (CIRCE-8B, full-data SFT)", "gen_sft8full_indomain.jsonl", "gen_sft8full_crossmodel.jsonl"),
 ("Ours (CIRCE-4B, 9K-subset SFT)", "gen_par_indomain.jsonl",     "gen_par_crossmodel.jsonl"),
 ("Ours (CIRCE-4B, 9K SFT + calibration-GRPO)", "gen_calib_indomain.jsonl", "gen_calib_crossmodel.jsonl"),
]

CAT={"SIDA-7B":"Specialized Detectors","SIDA-13B":"Specialized Detectors",
 "FakeVLM":"Specialized Detectors","AIGI-Holmes":"Specialized Detectors",
 "Qwen3-VL-Flash":"Generic MLLMs","Qwen3-VL-Plus":"Generic MLLMs",
 "Gemini 3 Flash":"Generic MLLMs","Gemini 3 Pro":"Generic MLLMs",
 "MedForge-Reasoner":"MedForge (SOTA)"}
HEAD=["Category","Method","Metric",
 "Real_InDomain","Real_CrossForgery","Real_CrossModel",
 "Implant_InDomain","Implant_CrossForgery","Implant_CrossModel",
 "Remove_InDomain","Remove_CrossForgery","Remove_CrossModel",
 "Avg_InDomain","Avg_CrossForgery","Avg_CrossModel"]
out=os.path.join(D,"medforge_results_comparison.csv")
with open(out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(HEAD)
    for m in ACC:
        w.writerow([CAT[m],m,"Accuracy"]+ACC[m])
        w.writerow([CAT[m],m,"F1"]+F1[m])
    for label, idg, cmg in OURS:
        a, f = ours_rows(idg, cmg)
        if a:
            w.writerow(["Ours", label, "Accuracy"] + a)
            w.writerow(["Ours", label, "F1"] + f)
            print(f"{label}: acc={a}")
print("wrote", out)
