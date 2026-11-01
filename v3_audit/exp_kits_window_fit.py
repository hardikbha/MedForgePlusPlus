"""Infer the HU->PNG mapping Naitik used for KiTS 2D slices (stream one small case into memory; nothing saved)."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import numpy as np, cv2, gzip, struct, urllib.request
import exp_kits_slices as k
k.BASE = "https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/"
vol, pd = k.load("case_00061.nii.gz")
print("imaging", vol.shape, vol.dtype, "HU range", vol.min(), vol.max())
png = cv2.imread(_release_path('AAAI/naitik_data/kidney_ext/Kidney_dataset/real/Normal_kideny/images/case_00061_slice_022.png'), cv2.IMREAD_UNCHANGED).astype(np.float64)
best = None
for ax in range(3):
    if vol.shape[ax] <= 22: continue
    sl = np.take(vol, 22, axis=ax).astype(np.float64)
    for tr in [lambda x: x, lambda x: x.T, lambda x: np.flipud(x), lambda x: np.fliplr(x), lambda x: np.flipud(x.T), lambda x: np.fliplr(x.T), lambda x: np.rot90(x,2), lambda x: np.rot90(x,2).T]:
        s = tr(sl)
        if s.shape != png.shape: continue
        m = (png > 0) & (png < 65535)
        if m.sum() < 1000: continue
        A = np.vstack([s[m], np.ones(m.sum())]).T; coef, res, *_ = np.linalg.lstsq(A, png[m], rcond=None)
        r = np.corrcoef(s[m], png[m])[0, 1]
        if best is None or r > best[0]: best = (r, ax, coef)
r, ax, (a, b) = best
lo, hi = -b / a, (65535 - b) / a
print(f"best corr={r:.4f} axis={ax}  PNG = {a:.3f}*HU + {b:.1f}  => window HU [{lo:.0f}, {hi:.0f}] (W={hi-lo:.0f}, L={(hi+lo)/2:.0f})")
