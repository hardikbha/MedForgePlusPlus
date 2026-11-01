"""
legacy_meta_v3.py -- group / organ / generator / twin metadata for OLD generation files (CPU only).

README
  Old generation files (~/AAAI/MedForge++/v3_select/gen_*.jsonl, v3_counterfactual/results/gen_*.jsonl) only carry
  {image, gt_verdict, gt_box, subset, type, output}. stats_v3.py needs, per image, the bootstrap GROUP, the organ,
  the generator and the authentic-source TWIN. This script builds that table once for every MedForge-90K image:

    group : connected component over (forgery -> its source scan) + (fundus patient, both eyes; EyePACS id space
            separate) + (Yale brain patient) + (md5-identical reals) + (verified near-duplicate reals: 64-bit
            dHash Hamming <= 4 AND 32px-thumbnail MAE <= 3). These are exactly the edges used by
            build_mfclean.py, and the root of a component is its lexicographically smallest member in both
            union-finds, so  group id = "g" + md5(root)[:12]  is IDENTICAL to the MF-Clean group id.
    dup   : duplicate cluster (md5 + verified near-duplicates only) -> used to find a fake's twin when the file
            contains a duplicate of its source instead of the source itself.
    src   : the forgery's authentic source scan (dataset_mapping.json real_image_path).

  Output: ~/CIRCE/v3/cache/mf90k_legacy_meta.json  {rel_path: [group, dup_root, organ, gen, type, src_or_null]}
  Run:    NUMPY_MADVISE_HUGEPAGE=0 nice ~/.conda/envs/aiface/bin/python legacy_meta_v3.py      (~1-3 min, 1 core)
  MF++ (naitik_annotated) images are resolved by rule inside stats_v3.py (no table needed).
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import hashlib, json, os, re, time
import numpy as np

MF = _release_path('AAAI/MedForge++/medforge90k_full/')
HASHES = _release_path('CIRCE/v3_audit/data/image_hashes.tsv')
THUMBS = _release_path('CIRCE/v3_audit/data/thumbs32.npy')
OUT = _release_path('CIRCE/v3/cache/mf90k_legacy_meta.json')


class UF:
    """Same semantics as build_mfclean.UF: the root of every component is its smallest member."""
    def __init__(self):
        self.p = {}

    def f(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a, b):
        ra, rb = self.f(a), self.f(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main():
    t0 = time.time()
    items = json.load(open(MF + "dataset_mapping.json"))["items"]
    md5, dh, idx_of = {}, {}, {}
    for l in open(HASHES):
        p = l.rstrip("\n").split("\t")
        if p[0] == "idx" or len(p) < 5 or "/medforge90k_full/" not in p[2]:
            continue
        rel = p[2].split("/medforge90k_full/", 1)[1]
        idx_of[rel] = int(p[0])
        if p[3]:
            md5[rel], dh[rel] = p[3], p[4]
    reals = {it["image_path"] for it in items if it["image_path"].startswith("real/")}
    uf, dup = UF(), UF()
    for it in items:
        p_ = it["image_path"]
        if p_ not in md5:
            continue
        uf.f(p_)
        if not p_.startswith("real/"):
            uf.u(p_, it["real_image_path"])
        b_ = os.path.splitext(os.path.basename(p_ if p_.startswith("real/") else it["real_image_path"]))[0]
        if it["dataset"] == "fundus":
            m = re.match(r"^(\d+)_(left|right)(_eyepacs)?$", b_)
            if m:
                uf.u(p_, "PAT:fundus" + ("E" if m.group(3) else "O") + m.group(1))
        if it["dataset"] == "brain-mri":
            m = re.match(r"^(YG_[A-Z0-9]+)_", b_)
            if m:
                uf.u(p_, "PAT:" + m.group(1))
    by_md5 = {}
    for rel in reals:
        if rel in md5:
            dup.f(rel)
            by_md5.setdefault(md5[rel], []).append(rel)
    for grp in by_md5.values():
        for x in grp[1:]:
            uf.u(grp[0], x); dup.u(grp[0], x)
    th = np.load(THUMBS).astype(np.int16)
    LUT = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)
    for org in ("brain-mri", "chest-xray", "fundus"):
        L = sorted(r for r in reals if r in md5 and r.startswith("real/" + org))
        I = np.array([idx_of[r] for r in L])
        D = np.array([np.uint64(int(dh[r], 16)) for r in L], dtype=np.uint64)
        for st in range(0, len(L), 512):
            x = np.bitwise_xor(D[st:st + 512, None], D[None, :])
            H = LUT[x.view(np.uint16).reshape(x.shape[0], x.shape[1], 4)].sum(axis=2)
            ia, jb = np.nonzero(H <= 4)
            for i, j in zip(st + ia, jb):
                if i < j and np.abs(th[I[i]] - th[I[j]]).mean() <= 3:
                    uf.u(L[i], L[j]); dup.u(L[i], L[j])
    out = {}
    for it in items:
        p_ = it["image_path"]
        root = uf.f(p_)
        g = "g" + hashlib.md5(root.encode()).hexdigest()[:12]
        isreal = p_.startswith("real/")
        out[p_] = [g, dup.f(p_) if isreal else None, it["dataset"], "real" if isreal else it["model"],
                   "real" if isreal else it["type"], None if isreal else it.get("real_image_path")]
    # dup root of each fake's source (so stats can match a twin by duplicate cluster)
    for p_, v in out.items():
        if v[5]:
            v[1] = dup.f(v[5])
    tmp = OUT + ".tmp"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, OUT)
    ng = len({v[0] for v in out.values()})
    print(f"[legacy_meta] {len(out)} MF90K items, {ng} groups, wrote {OUT} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
