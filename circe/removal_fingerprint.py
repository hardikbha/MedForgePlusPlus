"""
Removal-fingerprint detector — resolves the 'Remove' failure.

Insight: a clean lesion-removal is undetectable LOCALLY (the region looks normal), but the
inpainting/generation pipeline leaves a GLOBAL forensic fingerprint (noise-residual + spectral).
A light classifier on those features detects removals the VLM misses. Trained on non-eval images,
tested on the held-out eval split; combined with the VLM (great at Real/Implant) for the full table.
CPU-only, fast.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import numpy as np, glob, json, os, sys
from PIL import Image
from scipy import ndimage

ROOT = _release_path('AAAI/MedForge++/naitik_annotated')
SRM = [np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], float),
       np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
       np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], float)]


def feat(path):
    im = np.array(Image.open(path).convert("L").resize((256, 256))).astype(float)
    cy, cx = 128, 128
    yy, xx = np.mgrid[0:256, 0:256]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    F = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(im))))
    radial = np.array([F[r == k].mean() for k in range(0, 128, 3)])
    parts = [radial]
    for k in SRM:
        res = ndimage.convolve(im, k, mode="reflect")
        Fr = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(res))))
        parts.append(np.array([Fr[r == b].mean() for b in range(0, 128, 8)]))
        m = res.mean()
        parts.append(np.array([res.std(), float(((res - m) ** 3).mean()),
                               float(((res - m) ** 4).mean())]))
    return np.concatenate(parts)


def load(sub):
    mp = {m["key"]: m for f in glob.glob(f"{ROOT}/{sub}/mapping_*.json") for m in json.load(open(f))}
    return [(m["image_path"], os.path.basename(m["image_path"]))
            for m in mp.values() if os.path.exists(m["image_path"])]


def main():
    eval_names = set()
    for l in open(_release_path('AAAI/MedForge++/circe_eval_split_v2.jsonl')):
        eval_names.add(os.path.basename(json.loads(l)["image"]))

    tr_paths, tr_y, te_paths, te_y = [], [], [], []
    for sub, lab in [("kidney_removal", 1), ("polyp_removal", 1),
                     ("kidney_normal", 0), ("polyp_normal", 0),
                     ("kidney_real_tumor", 0), ("polyp_real", 0)]:  # real INCLUDES tumor-present
        for p, name in load(sub):
            if name in eval_names:
                te_paths.append(p); te_y.append(lab)
            else:
                tr_paths.append(p); tr_y.append(lab)
    # cap train for speed
    import random
    random.seed(0)
    z = list(zip(tr_paths, tr_y)); random.shuffle(z)
    z = z[:2400]; tr_paths, tr_y = [a for a, b in z], [b for a, b in z]
    print(f"train {len(tr_paths)} (non-eval) | test {len(te_paths)} (held-out eval split)")

    Xtr = np.array([feat(p) for p in tr_paths]); ytr = np.array(tr_y)
    Xte = np.array([feat(p) for p in te_paths]); yte = np.array(te_y)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score
    clf = HistGradientBoostingClassifier(max_iter=400, random_state=0).fit(Xtr, ytr)
    pr = clf.predict(Xte)
    acc = accuracy_score(yte, pr); f1 = f1_score(yte, pr)
    rec = (pr[yte == 1] == 1).mean()          # removal recall
    spec = (pr[yte == 0] == 0).mean()         # normal correctly kept real
    print(f"HELD-OUT: acc={acc*100:.1f}%  F1={f1*100:.1f}%  removal-recall={rec*100:.1f}%  "
          f"normal-acc={spec*100:.1f}%  (n_remove={(yte==1).sum()}, n_normal={(yte==0).sum()})")
    import pickle
    pickle.dump(clf, open(_release_path('AAAI/MedForge++/removal_fp.pkl'), "wb"))
    json.dump({"acc": acc, "f1": f1, "removal_recall": float(rec), "normal_acc": float(spec)},
              open(_release_path('AAAI/MedForge++/removal_fp_result.json'), "w"), indent=2)
    print("[saved] removal_fp.pkl + result")


if __name__ == "__main__":
    main()
