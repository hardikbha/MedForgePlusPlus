#!/usr/bin/env python3
"""Build the MedForge++ unified manifest by merging all forgery datasets into one schema:
{id, organ, modality, klass, forgery_type(real/inject/remove), method, image_path,
 real_source_path, ann_type(mask/box/point/none), annotation}.
References files in place (no copying). Kidney is a documented empty slot to fill later."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os, json, csv, hashlib

MF   = _release_path('AAAI/MedForge/data')
MFIMG= os.path.join(MF, "MedForge-90K")
MED  = _release_path('medical defect')
COLON= _release_path('AAAI/colon')
OUT  = _release_path('AAAI/MedForge++')
os.makedirs(OUT, exist_ok=True)
recs = []
def rid(*a): return hashlib.md5("|".join(map(str,a)).encode()).hexdigest()[:12]

# ---------- 1) MedForge-90K (brain-MRI / chest-Xray / fundus) : mask+box ----------
MOD = {"brain-mri":("brain","MRI"), "chest-xray":("chest","Xray"), "fundus":("fundus","Fundus")}
raw = json.load(open(os.path.join(MF,"json_dataset_mapping.json")))
items = raw["items"] if isinstance(raw,dict) else raw
seen=set()
for it in items:
    k=it.get("key") or it.get("image_path")
    if k in seen: continue
    seen.add(k)
    organ,modality = MOD.get(it.get("dataset",""), (it.get("dataset"),"?"))
    ftype = {"edit":"inject","remove":"remove","real":"real"}.get(it.get("type"),"real")
    recs.append(dict(id=rid("mf",k), source="MedForge-90K", organ=organ, modality=modality,
        klass=it.get("disease"), forgery_type=ftype, method=it.get("model"),
        image_path=os.path.join(MFIMG, it["image_path"]),
        real_source_path=(os.path.join(MFIMG,it["real_image_path"]) if it.get("real_image_path") else None),
        ann_type=("mask" if ftype!="real" and it.get("mask_path") else "none"),
        annotation=(os.path.join(MFIMG,it["mask_path"]) if it.get("mask_path") else None)))

# ---------- 2) CT lung nodule (LIDC) : point (x,y) ----------
def ingest_csv(path, organ, modality, klass, ftype):
    if not os.path.exists(path): return 0
    n=0
    for r in csv.DictReader(open(path)):
        p=r.get("path");
        if not p: continue
        method = "SD" if "_SD/" in p else ("CTGAN" if "CTGAN" in p else r.get("type",""))
        recs.append(dict(id=rid(organ,ftype,p), source="medical-defect", organ=organ, modality=modality,
            klass=klass, forgery_type=ftype, method=method,
            image_path=os.path.join(MED,p), real_source_path=None,
            ann_type="point", annotation=[int(float(r["x"])), int(float(r["y"]))] if r.get("x") else None))
        n+=1
    return n
n_ct_i = ingest_csv(f"{MED}/CT/injection/CT_injection.csv","lung","CT","nodule","inject")
n_ct_r = ingest_csv(f"{MED}/CT/removal/CT_removal.csv","lung","CT","nodule","remove")
# ---------- 3) Breast MRI (Duke) : point (x,y) ----------
n_mr_i = ingest_csv(f"{MED}/MRI/MRI_injection.csv","breast","MRI","lesion","inject")
n_mr_r = ingest_csv(f"{MED}/MRI/MRI_removal.csv","breast","MRI","lesion","remove")

# ---------- 4) Colon polyp — FLUX-Kontext injections (1000, with GT box + MedForge CoT) ----------
import glob as _glob
_inj = 0
for m in sorted(_glob.glob(f"{COLON}/flux_injected/manifest_*.jsonl")):
    for line in open(m):
        r = json.loads(line)
        cot = next((mm["content"] for mm in r["messages"] if mm["role"] == "assistant"), None)
        recs.append(dict(id=rid("colon-inj", r["images"][0]), source="colon-kvasir-flux",
            organ="colon", modality="Colonoscopy", klass="polyp", forgery_type="inject",
            method="flux.1-kontext", image_path=r["images"][0], real_source_path=None,
            ann_type="box", annotation=cot, box=r["box"]))
        _inj += 1
# ---------- 4b) Colon polyp — Kvasir-SEG real images (1000, authentic) ----------
_real = 0
for f in sorted(_glob.glob(f"{COLON}/kvasir_extracted/Kvasir-SEG/images/*.jpg")):
    mp = f.replace("/images/", "/masks/")
    recs.append(dict(id=rid("colon-real", f), source="colon-kvasir", organ="colon",
        modality="Colonoscopy", klass="polyp", forgery_type="real", method="real",
        image_path=f, real_source_path=None,
        ann_type="none", annotation=(mp if os.path.exists(mp) else None), box=None))
    _real += 1
print(f"colon: {_inj} inject (FLUX-Kontext, boxed) + {_real} real (Kvasir-SEG)")

# ---------- 5) Kidney MRI : EMPTY SLOT (to source + generate) ----------
KIDNEY_TODO = {"organ":"kidney","modality":"MRI","status":"NOT_PRESENT",
               "note":"source kidney MRI (e.g. TCGA-KIRC / KiTS) then run inject+remove"}

# ---------- harmonize annotations -> uniform `box` [x1,y1,x2,y2] for localization ----------
POINT_R = 32   # px half-size for point->box (CT/breast); tune per modality later
for r in recs:
    if r["ann_type"] == "box" and r.get("box"):
        pass                                   # colon injections already carry explicit [x1,y1,x2,y2]
    elif r["ann_type"] == "point" and r["annotation"]:
        x, y = r["annotation"]; r["box"] = [x-POINT_R, y-POINT_R, x+POINT_R, y+POINT_R]
    elif r["ann_type"] == "mask":
        r["box"] = "from_mask"   # tight box derivable from the mask PNG at load time
    else:
        r["box"] = None

json.dump(recs, open(f"{OUT}/medforgepp_manifest.json","w"))
# summary
from collections import Counter
by_organ = Counter((r["organ"], r["forgery_type"]) for r in recs)
print(f"=== MedForge++ manifest: {len(recs)} records -> {OUT}/medforgepp_manifest.json ===")
print("organs:", sorted(set(r["organ"] for r in recs)))
print("ann types:", dict(Counter(r["ann_type"] for r in recs)))
print("forgery types:", dict(Counter(r["forgery_type"] for r in recs)))
for (org,ft),c in sorted(by_organ.items()):
    print(f"  {org:8s} {ft:8s} {c}")
print("KIDNEY:", KIDNEY_TODO)
json.dump({"n":len(recs),"by_organ_forgery":{f"{o}/{f}":c for (o,f),c in by_organ.items()},
           "kidney":KIDNEY_TODO}, open(f"{OUT}/summary.json","w"), indent=2)
