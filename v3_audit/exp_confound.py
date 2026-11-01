#!/usr/bin/env python
"""Source/format confound probe for the MedForge++ addition (CPU only, read-only).
For each pair (A vs B) compute cheap global features that carry NO forgery information
(file format, JPEG 8x8 blockiness, global gray histogram, aspect of the ORIGINAL source)
and report 5-fold logistic-regression AUC. AUC ~1.0 => the two groups are separable by source/format alone."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os, glob, sys, json
import numpy as np, cv2
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

A = _release_path('AAAI')
G = {
 "kidney_implant(flux,on Kaggle CT-Kidney)": sorted(glob.glob(f"{A}/kidney/flux_injected/images/*.jpg")),
 "kidney_normal(KiTS)": sorted(glob.glob(f"{A}/MedForge++/naitik_annotated/kidney_normal/images/*.png")),
 "kidney_real_tumor(KiTS)": sorted(glob.glob(f"{A}/MedForge++/naitik_annotated/kidney_real_tumor/images/*.png")),
 "kidney_removal(LaMa,KiTS)": sorted(glob.glob(f"{A}/MedForge++/naitik_annotated/kidney_removal/images/*.png")),
 "kidney_src_normal(Kaggle CT-Kidney, pre-injection)": sorted(glob.glob(f"{A}/kidney/normal/*.jpg")),
 "colon_implant(flux,on Kvasir)": sorted(glob.glob(f"{A}/MedForge++/colon_injections/images/*.jpg")),
 "polyp_real(Kvasir)": sorted(glob.glob(f"{A}/MedForge++/naitik_annotated/polyp_real/images/*.png")),
 "polyp_removal(LaMa,Kvasir)": sorted(glob.glob(f"{A}/MedForge++/naitik_annotated/polyp_removal/images/*.png")),
 "polyp_normal(curated-colon)": sorted(glob.glob(f"{A}/MedForge++/naitik_annotated/polyp_normal/images/*.png")),
}

def blockiness(g):
    g = g.astype(np.float32)
    dh = np.abs(np.diff(g, axis=1)); dv = np.abs(np.diff(g, axis=0))
    cols = np.arange(dh.shape[1]); rows = np.arange(dv.shape[0])
    bh = dh[:, (cols % 8) == 7].mean() / (dh[:, (cols % 8) != 7].mean() + 1e-6)
    bv = dv[(rows % 8) == 7, :].mean() / (dv[(rows % 8) != 7, :].mean() + 1e-6)
    return bh, bv

def feat(p):
    try:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None: return None
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bh, bv = blockiness(g)
        small = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA)
        hist = np.histogram(small, bins=16, range=(0, 256))[0] / small.size
        lap = cv2.Laplacian(small, cv2.CV_32F).var()
        sat = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2HSV)[..., 1].mean()
        fmt = 1.0 if p.lower().endswith((".jpg", ".jpeg")) else 0.0
        return np.concatenate([[fmt, bh, bv, lap, sat, small.mean(), small.std()], hist]).astype(np.float32)
    except Exception:
        return None

def load(paths):
    with Pool(12) as pool:
        F = pool.map(feat, paths, chunksize=16)
    return np.array([f for f in F if f is not None])

def auc(Xa, Xb, cols):
    X = np.vstack([Xa[:, cols], Xb[:, cols]]); y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean()

if __name__ == "__main__":
    feats = {}
    for k, v in G.items():
        feats[k] = load(v); print(f"{k}: n={len(feats[k])}  JPEG-fraction={feats[k][:,0].mean():.2f}  "
                                 f"blockiness h/v={feats[k][:,1].mean():.3f}/{feats[k][:,2].mean():.3f}", flush=True)
    pairs = [("kidney_implant(flux,on Kaggle CT-Kidney)", "kidney_normal(KiTS)"),
             ("kidney_implant(flux,on Kaggle CT-Kidney)", "kidney_real_tumor(KiTS)"),
             ("kidney_src_normal(Kaggle CT-Kidney, pre-injection)", "kidney_normal(KiTS)"),
             ("kidney_removal(LaMa,KiTS)", "kidney_real_tumor(KiTS)"),
             ("colon_implant(flux,on Kvasir)", "polyp_real(Kvasir)"),
             ("polyp_removal(LaMa,Kvasir)", "polyp_real(Kvasir)"),
             ("polyp_real(Kvasir)", "polyp_normal(curated-colon)")]
    allc = list(range(feats[pairs[0][0]].shape[1]))
    nofmt = [c for c in allc if c not in (0, 1, 2)]   # drop format + blockiness: histogram/sharpness only
    for a, b in pairs:
        print(f"AUC  {a}  vs  {b}:  all-features={auc(feats[a], feats[b], allc):.3f}   "
              f"histogram+sharpness-only (no format/blockiness)={auc(feats[a], feats[b], nofmt):.3f}", flush=True)
