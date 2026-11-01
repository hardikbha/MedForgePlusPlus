#!/usr/bin/env python3
"""Give the colon + kidney injections the SAME annotation artifacts as MedForge-90K:
  - gt/bbox_mask/<stem>_bbox_mask.png   (solid white rectangle at the box, 1024x1024)
  - gt/explanation/<stem>.txt           (evidence + conclusion text)
  - dataset_mapping_medforgepp.json     (MedForge schema: key,image_path,real_image_path,
                                         mask_path,explain_path,model,dataset,disease,type)
(SFT {messages,images} already exist in each set's annotations.)"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, glob, re
import numpy as np
from PIL import Image

OUT = _release_path('AAAI/MedForge++/annotations_medforge')
CANVAS = 1024
os.makedirs(OUT, exist_ok=True)

def evidence_from_cot(cot):
    ev = re.search(r"<evidence>(.*?)</evidence>", cot or "", re.S)
    cn = re.search(r"<conclusion>(.*?)</conclusion>", cot or "", re.S)
    parts = []
    if ev: parts.append(ev.group(1).strip())
    if cn: parts.append("Conclusion: " + cn.group(1).strip())
    return "\n".join(parts) if parts else (cot or "").strip()

def write_mask(box, path):
    m = np.zeros((CANVAS, CANVAS), np.uint8)
    x1, y1, x2, y2 = [max(0, min(CANVAS - 1, int(v))) for v in box]
    m[y1:y2, x1:x2] = 255                       # solid rectangle = MedForge bbox_mask
    Image.fromarray(m).save(path)

def process(name, records, real_lookup, disease, dataset_tag):
    base = os.path.join(OUT, name)
    md = os.path.join(base, "gt", "bbox_mask"); ex = os.path.join(base, "gt", "explanation")
    os.makedirs(md, exist_ok=True); os.makedirs(ex, exist_ok=True)
    mapping = []
    for r in records:
        img = r["images"][0] if "images" in r else r["image"]
        stem = os.path.splitext(os.path.basename(img))[0]
        box = r.get("box")
        if not box or len(box) != 4:
            continue
        cot = next((m["content"] for m in r.get("messages", []) if m["role"] == "assistant"), r.get("reasoning", ""))
        mpath = os.path.join(md, f"{stem}_bbox_mask.png")
        epath = os.path.join(ex, f"{stem}.txt")
        write_mask(box, mpath)
        open(epath, "w").write(evidence_from_cot(cot))
        mapping.append({
            "key": os.path.basename(img), "image_path": img,
            "real_image_path": real_lookup(stem), "mask_path": mpath, "explain_path": epath,
            "model": "flux.1-kontext", "dataset": dataset_tag, "disease": disease, "type": "edit",
        })
    json.dump(mapping, open(os.path.join(base, "dataset_mapping.json"), "w"))
    print(f"{name}: {len(mapping)} masks+explanations -> {base}")
    return mapping

# ---------- COLON ----------
col = [json.loads(l) for l in open(_release_path('AAAI/MedForge++/colon_injections/annotations.jsonl'))]
KVDIR = _release_path('AAAI/colon/kvasir_extracted/Kvasir-SEG/images')
def colon_real(stem):
    base = stem.replace("_polypinj", "")
    p = os.path.join(KVDIR, base + ".jpg")
    return p if os.path.exists(p) else None
m_col = process("colon", col, colon_real, "polyp", "colon")

# ---------- KIDNEY ----------
kid = []
for m in glob.glob(_release_path('AAAI/kidney/flux_injected/manifest_*.jsonl')):
    kid += [json.loads(l) for l in open(m)]
NORMDIR = _release_path('AAAI/kidney/normal')
norm_index = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(f"{NORMDIR}/*.jpg")}
def kidney_real(stem):
    base = stem.replace("_tumorinj", "")            # "Normal-_(1002)"
    cand = norm_index.get(base) or norm_index.get(base.replace("Normal-_", "Normal- "))
    if cand: return cand
    hits = [v for k, v in norm_index.items() if base.split("(")[-1].rstrip(")") in k]
    return hits[0] if hits else None
m_kid = process("kidney", kid, kidney_real, "tumor", "kidney")

# combined mapping
allm = m_col + m_kid
json.dump(allm, open(os.path.join(OUT, "dataset_mapping_medforgepp.json"), "w"))
print(f"TOTAL: {len(allm)} MedForge-format records -> {OUT}/dataset_mapping_medforgepp.json")
print("real_image_path resolved:", sum(1 for r in allm if r['real_image_path']), "/", len(allm))
