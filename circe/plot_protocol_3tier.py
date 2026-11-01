"""3-tier exact-protocol comparison: CIRCE-8B (SFT-only) vs MedForge-Reasoner. Numbers from gens (no transcription)."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, sys, csv
sys.path.insert(0, _release_path('CIRCE/circe'))
from circe_reward import parse_output
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
D = _release_path('AAAI/MedForge++/')
DF = {"edit", "remove", "inject"}
def cats(fn):
    from collections import defaultdict
    tot, cor = defaultdict(int), defaultdict(int)
    for l in open(D+fn):
        g=json.loads(l); t=g["type"]; want="deepfake" if t in DF else "real"
        v=parse_output(g["output"])["verdict"] or "real"
        k={"edit":"Implant","remove":"Remove","real":"Real"}[t]
        tot[k]+=1; cor[k]+=int(v==want)
    r={k:100*cor[k]/tot[k] for k in tot}; r["AVG"]=np.mean([r[k] for k in ("Real","Implant","Remove")]); return r
OURS={t:cats(f) for t,f in [("In-Domain","gen_mfp_indomain.jsonl"),
      ("Cross-Forgery","gen_mfp_crossforgery.jsonl"),("Cross-Model","gen_mfp_crossmodel.jsonl")]}
MF={"In-Domain":{"Real":99.24,"Implant":99.24,"Remove":99.21,"AVG":99.23},
    "Cross-Forgery":{"Real":95.24,"Implant":93.39,"Remove":99.15,"AVG":95.93},
    "Cross-Model":{"Real":92.86,"Implant":94.86,"Remove":94.09,"AVG":93.94}}
cats_o=["Real","Implant","Remove","AVG"]; TEAL,GOLD="#1a7f7f","#c98a1a"
fig,axes=plt.subplots(1,3,figsize=(19,6))
fig.suptitle("CIRCE-8B (Ours, SFT-only) vs MedForge-Reasoner (SFT+GSPO) — MedForge exact protocol",
             fontsize=15,fontweight="bold")
for ax,tier in zip(axes,["In-Domain","Cross-Forgery","Cross-Model"]):
    x=np.arange(4); w=0.38; ov=[OURS[tier][c] for c in cats_o]; mv=[MF[tier][c] for c in cats_o]
    ax.bar(x-w/2,ov,w,color=TEAL,label="Ours (CIRCE-8B, SFT-only)")
    ax.bar(x+w/2,mv,w,color=GOLD,label="MedForge-Reasoner (8B)")
    for i,(o,m) in enumerate(zip(ov,mv)):
        ax.text(i-w/2,o+0.5,f"{o:.1f}",ha="center",fontsize=9,fontweight="bold")
        ax.text(i+w/2,m+0.5,f"{m:.1f}",ha="center",fontsize=9,fontweight="bold")
        g=o-m; ax.text(i,max(o,m)+3.5,f"{g:+.1f}",ha="center",fontsize=10,fontweight="bold",
                       color="green" if g>=0 else "red")
    ax.axhline(100,ls=":",c="grey",lw=1); ax.set_title(f"Accuracy — {tier}",fontsize=13,fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0,113); ax.set_xticks(x); ax.set_xticklabels(cats_o)
    ax.legend(loc="lower left",fontsize=8)
plt.tight_layout(rect=[0,0,1,0.96])
out=D+"circe8b_exact_protocol_3tier.png"; plt.savefig(out,dpi=130,bbox_inches="tight"); print("wrote",out)
# CSV
with open(D+"medforge_exact_protocol.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["Tier","Category","Ours_CIRCE8B_SFT","MedForge_Reasoner","Delta"])
    for tier in ["In-Domain","Cross-Forgery","Cross-Model"]:
        for c in cats_o: w.writerow([tier,c,round(OURS[tier][c],1),MF[tier][c],round(OURS[tier][c]-MF[tier][c],1)])
print("wrote",D+"medforge_exact_protocol.csv")
