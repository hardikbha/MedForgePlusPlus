#!/usr/bin/env python
"""Count usable 2D axial slices in KiTS23 from the SEGMENTATIONS only (0.14 GB total, streamed into memory,
nothing written to disk except the summary). Labels: 1 kidney, 2 tumor, 3 cyst.
Per case: #slices with kidney; #slices with tumor area>=thr; #slices with kidney>=thr and NO tumor/cyst."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import gzip, io, json, struct, sys, urllib.request, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor

BASE = "https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/segmentations/"
TREE = "https://huggingface.co/api/datasets/neheller/KiTS-Challenge-Imaging/tree/main/segmentations"
OUT = _release_path('CIRCE/v3_audit/exp_out/kits23_slice_counts.json')
DT = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 256: np.int8, 512: np.uint16, 768: np.uint32}

def load(name):
    raw = gzip.decompress(urllib.request.urlopen(BASE + name, timeout=120).read())
    hdr = raw[:348]
    endian = "<" if struct.unpack("<i", hdr[:4])[0] == 348 else ">"
    dim = struct.unpack(endian + "8h", hdr[40:56]); dtype = struct.unpack(endian + "h", hdr[70:72])[0]
    pixdim = struct.unpack(endian + "8f", hdr[76:108]); off = int(struct.unpack(endian + "f", hdr[108:112])[0])
    shape = tuple(dim[1:1 + dim[0]])
    a = np.frombuffer(raw[off:], dtype=np.dtype(DT[dtype]).newbyteorder(endian), count=int(np.prod(shape)))
    return a.reshape(shape, order="F"), pixdim[1:4]

def per_case(name):
    try:
        seg, pd = load(name)
        # KiTS23 nifti: axial axis is the one whose length != 512 (others are 512x512)
        ax = [i for i, s in enumerate(seg.shape) if s != 512]
        ax = ax[0] if ax else 0
        seg = np.moveaxis(seg, ax, 0)
        kid = (seg == 1).reshape(seg.shape[0], -1).sum(1); tum = (seg == 2).reshape(seg.shape[0], -1).sum(1)
        cys = (seg == 3).reshape(seg.shape[0], -1).sum(1)
        thr = 200
        return name, dict(n_slices=int(seg.shape[0]), slice_mm=float(pd[ax]),
                          kidney=int((kid > 0).sum()), tumor_ge200=int((tum >= thr).sum()),
                          normal_kidney=int(((kid >= 2000) & (tum == 0) & (cys == 0)).sum()),
                          tumor_px_median=float(np.median(tum[tum >= thr])) if (tum >= thr).any() else 0.0)
    except Exception as e:
        return name, {"error": str(e)}

if __name__ == "__main__":
    names = [x["path"].split("/")[-1] for x in json.load(urllib.request.urlopen(TREE, timeout=60))]
    with ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(per_case, names))
    ok = {k: v for k, v in res.items() if "error" not in v}
    print("cases", len(names), "parsed", len(ok), "errors", len(res) - len(ok))
    S = lambda f: sum(v[f] for v in ok.values())
    print("total axial slices", S("n_slices"), "| with kidney", S("kidney"), "| tumor>=200px", S("tumor_ge200"),
          "| kidney>=2000px & no tumor/cyst", S("normal_kidney"))
    sm = sorted(v["slice_mm"] for v in ok.values()); print("slice thickness mm: median", sm[len(sm)//2], "min", sm[0], "max", sm[-1])
    t = sorted(v["tumor_ge200"] for v in ok.values()); n = sorted(v["normal_kidney"] for v in ok.values())
    print("per-case tumor slices median", t[len(t)//2], " per-case normal-kidney slices median", n[len(n)//2])
    print("cases with >=6 tumor slices", sum(x >= 6 for x in t), " cases with >=6 normal-kidney slices", sum(x >= 6 for x in n))
    json.dump(res, open(OUT, "w"))
