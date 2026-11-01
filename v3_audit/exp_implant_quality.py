#!/usr/bin/env python
"""Implant quality + global-regeneration check (CPU, read-only).
For each FLUX implant, compare to its pre-injection source (resized to the 1024 canvas):
  in-box mean|d|  : lesion visibility (README threshold: 'clear' if >=12)
  out-box mean|d| : how much of the REST of the image changed (global VAE/resize fingerprint)."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os, json, glob, zipfile
import numpy as np, cv2
from multiprocessing import Pool

A = _release_path('AAAI')
KZ = zipfile.ZipFile(f"{A}/colon/kvasir.zip")
KN = {os.path.splitext(os.path.basename(n))[0]: n for n in KZ.namelist() if "/images/" in n and n.lower().endswith(".jpg")}

def load_manifest(globpat):
    recs = []
    for f in sorted(glob.glob(globpat)):
        for l in open(f):
            recs.append(json.loads(l))
    return recs

def one(args):
    organ, rec = args
    try:
        p = rec["images"][0]; box = rec["box"]
        fake = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if organ == "kidney":
            stem = os.path.basename(p).replace("_tumorinj.jpg", "").replace("_", " ")
            src = cv2.imread(f"{A}/kidney/normal/{stem}.jpg", cv2.IMREAD_GRAYSCALE)
        else:
            stem = os.path.basename(p).replace("_polypinj.jpg", "")
            z = zipfile.ZipFile(f"{A}/colon/kvasir.zip")
            src = cv2.imdecode(np.frombuffer(z.read(KN[stem]), np.uint8), cv2.IMREAD_GRAYSCALE)
        if src is None or fake is None: return None
        src = cv2.resize(src, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32)
        d = np.abs(fake.astype(np.float32) - src)
        x1, y1, x2, y2 = [max(0, min(1023, v)) for v in box]
        m = np.zeros_like(d, bool); m[y1:y2, x1:x2] = True
        # ring margin excluded from 'outside' (feathered blend region)
        pad = 40; ring = np.zeros_like(m); ring[max(0, y1-pad):y2+pad, max(0, x1-pad):x2+pad] = True
        return organ, float(d[m].mean()), float(d[~ring].mean()), (x2-x1)*(y2-y1)/1024**2
    except Exception as e:
        return None

if __name__ == "__main__":
    kid = load_manifest(f"{A}/kidney/flux_injected/manifest_*.jsonl")
    col = [json.loads(l) for l in open(f"{A}/MedForge++/colon_injections/annotations.jsonl")]
    # dedupe by image path (redo shards can duplicate)
    kid = list({r["images"][0]: r for r in kid}.values())
    jobs = [("kidney", r) for r in kid] + [("colon", r) for r in col]
    with Pool(12) as pool:
        res = [r for r in pool.map(one, jobs, chunksize=8) if r]
    for organ in ["kidney", "colon"]:
        R = np.array([r[1:] for r in res if r[0] == organ])
        inb, outb, area = R[:, 0], R[:, 1], R[:, 2]
        print(f"{organ}: n={len(R)}  in-box mean|d| median={np.median(inb):.1f}  frac<12 (faint)={np.mean(inb<12):.3f}  "
              f"frac<6={np.mean(inb<6):.3f}  | OUTSIDE-box mean|d| median={np.median(outb):.2f} (p90 {np.percentile(outb,90):.2f})  "
              f"| box area frac median={np.median(area):.4f}")
    # placement diversity: number of distinct box centres
    for organ, recs in [("kidney", kid), ("colon", col)]:
        cen = {((b[0]+b[2])//2//32, (b[1]+b[3])//2//32) for b in (r["box"] for r in recs)}
        print(f"{organ}: distinct box centres on a 32-px grid = {len(cen)} over {len(recs)} implants")
