"""
Carve a balanced held-out eval split for eval_medforge.py.
Emits {image, type, box, subset} per line. Takes the LAST `--per` samples of each
subset (so a matching --skip_last in SFT can later hold them out cleanly).
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, glob, os, argparse

ROOT = _release_path('AAAI/MedForge++/naitik_annotated')
INJECT_SOURCES = [
    ("colon_inject", [_release_path('AAAI/MedForge++/colon_injections/annotations.jsonl')]),
    ("kidney_inject", sorted(glob.glob(_release_path('AAAI/kidney/flux_injected/manifest_*.jsonl')))),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=100)
    ap.add_argument("--out", default=_release_path('AAAI/MedForge++/circe_eval_split.jsonl'))
    args = ap.parse_args()
    subsets = sorted(d for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(d))
    rows = []
    # naitik: removal (deepfake) + real_tumor/real (real) + normal (real)
    for sd in subsets:
        mp = {}
        for f in glob.glob(os.path.join(sd, "mapping_*.json")):
            for m in json.load(open(f)):
                mp[m["key"]] = m
        items = [m for m in mp.values() if os.path.exists(m["image_path"])]
        for m in items[-args.per:]:              # LAST per — matches SFT ordering for a cleaner split
            rows.append({"image": m["image_path"], "type": m["type"],
                         "box": m.get("box"), "subset": os.path.basename(sd)})
        print(f"  {os.path.basename(sd):18s} -> {min(args.per, len(items))}")
    # injection (IMPLANT) sources
    for name, paths in INJECT_SOURCES:
        seen, n = set(), 0
        recs = []
        for p in paths:
            for l in open(p):
                r = json.loads(l); img = r["images"][0]
                if img in seen or not os.path.exists(img):
                    continue
                seen.add(img)
                recs.append({"image": img, "type": "inject", "box": r.get("box"), "subset": name})
        for r in recs[-args.per:]:
            rows.append(r)
        print(f"  {name:18s} -> {min(args.per, len(recs))}  (IMPLANT)")
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"TOTAL eval: {len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
