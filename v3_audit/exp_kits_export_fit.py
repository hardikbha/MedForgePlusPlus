"""Recover the 16-bit 512 -> 8-bit 1024 RGB export used for naitik_annotated kidney PNGs."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import cv2, numpy as np
from PIL import Image
a16 = cv2.imread(_release_path('AAAI/naitik_data/kidney_ext/Kidney_dataset/real/Normal_kideny/images/case_00061_slice_022.png'), cv2.IMREAD_UNCHANGED)
t = cv2.imread(_release_path('AAAI/MedForge++/naitik_annotated/kidney_normal/images/case_00061_slice_022.png')).astype(np.float64)
print("target channels equal:", np.abs(t[...,0]-t[...,1]).max(), np.abs(t[...,0]-t[...,2]).max())
res = []
for sname, s8 in [("/257", (a16/257.0)), ("/256", (a16/256.0)), (">>8", (a16>>8).astype(np.float64))]:
    for rnd in ["round", "floor"]:
        b = np.round(s8) if rnd == "round" else np.floor(s8)
        b = np.clip(b, 0, 255).astype(np.uint8)
        for kn, k in [("nearest", cv2.INTER_NEAREST), ("linear", cv2.INTER_LINEAR), ("cubic", cv2.INTER_CUBIC), ("lanczos", cv2.INTER_LANCZOS4), ("area", cv2.INTER_AREA)]:
            r = cv2.resize(b, (1024, 1024), interpolation=k).astype(np.float64)
            res.append((np.abs(r - t[..., 0]).mean(), f"cv2 {sname} {rnd} {kn}"))
        for kn, k in [("PIL-bilinear", Image.BILINEAR), ("PIL-bicubic", Image.BICUBIC), ("PIL-lanczos", Image.LANCZOS), ("PIL-nearest", Image.NEAREST)]:
            r = np.array(Image.fromarray(b).resize((1024, 1024), k)).astype(np.float64)
            res.append((np.abs(r - t[..., 0]).mean(), f"{sname} {rnd} {kn}"))
res.sort()
for e, n in res[:5]: print(f"  mean|err|={e:.4f}  {n}")
