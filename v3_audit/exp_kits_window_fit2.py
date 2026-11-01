"""Search all slices/orientations of case_00061 for the one matching Naitik's PNG; fit HU->PNG window."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import numpy as np, cv2
import exp_kits_slices as k
k.BASE = "https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/"
vol, pd = k.load("case_00061.nii.gz")
png = cv2.imread(_release_path('AAAI/naitik_data/kidney_ext/Kidney_dataset/real/Normal_kideny/images/case_00061_slice_022.png'), cv2.IMREAD_UNCHANGED).astype(np.float64)
m = (png > 0) & (png < 65535)
trs = {"id": lambda x: x, "T": lambda x: x.T, "ud": np.flipud, "lr": np.fliplr, "udT": lambda x: np.flipud(x.T),
       "lrT": lambda x: np.fliplr(x.T), "r180": lambda x: np.rot90(x, 2), "r180T": lambda x: np.rot90(x, 2).T}
best = None
for i in range(vol.shape[0]):
    sl = vol[i].astype(np.float64)
    for n, tr in trs.items():
        s = tr(sl); r = np.corrcoef(s[m], png[m])[0, 1]
        if best is None or r > best[0]: best = (r, i, n)
r, i, n = best
s = trs[n](vol[i].astype(np.float64)); a, b = np.polyfit(s[m], png[m], 1)
lo, hi = -b / a, (65535 - b) / a
print(f"best slice index {i} transform {n} corr={r:.4f}; PNG = {a:.2f}*HU + {b:.0f}  => window HU [{lo:.0f}, {hi:.0f}]  W={hi-lo:.0f} L={(hi+lo)/2:.0f}")
print("fraction of PNG pixels clipped at 0 / 65535:", (png == 0).mean().round(3), (png == 65535).mean().round(3))
