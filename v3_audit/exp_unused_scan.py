#!/usr/bin/env python
"""Expansion audit: which MF90K images were never referenced by any protocol / eval / gen file.
Read-only. Writes results under ~/CIRCE/v3_audit/exp_out/."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os, re, json, glob, collections, sys

ROOT = _release_path('AAAI/MedForge++')
OUT = _release_path('CIRCE/v3_audit/exp_out')
os.makedirs(OUT, exist_ok=True)

items = json.load(open(f"{ROOT}/medforge90k_full/dataset_mapping.json"))["items"]
keys = {it["image_path"]: it for it in items}
print("mapping items", len(items), "unique image_path", len(keys))

# regex: anything after 'medforge90k_full/' or 'MedForge-90K/' up to a quote / whitespace
pat = re.compile(r'(?:medforge90k_full|MedForge-90K|medforge90k_eval)/((?:real|deepfake)/[^"\s\\]+?\.(?:jpe?g|png|JPE?G|PNG))')

files = sorted(set(glob.glob(f"{ROOT}/*.jsonl") + glob.glob(f"{ROOT}/*/*.jsonl") + glob.glob(f"{ROOT}/*.json")
                   + glob.glob(_release_path('CIRCE/**/*.jsonl'), recursive=True)))
files = [f for f in files if "dataset_mapping.json" not in f and "medforgepp_manifest.json" not in f
         and "/v3_audit/" not in f]
per_file = {}
used = collections.defaultdict(set)
for f in files:
    try:
        sz = os.path.getsize(f)
    except OSError:
        continue
    hits = set()
    with open(f, "r", errors="ignore") as fh:
        for line in fh:
            for m in pat.findall(line):
                hits.add(m)
    if hits:
        per_file[f] = len(hits)
        for h in hits:
            used[h].add(os.path.relpath(f, "${MEDFORGE_WORKSPACE}"))

used_keys = set(used) & set(keys)
unknown = set(used) - set(keys)
print("files scanned", len(files), "files with MF90K refs", len(per_file))
print("distinct referenced MF90K paths", len(used), "matching mapping", len(used_keys), "not in mapping", len(unknown))
never = [k for k in keys if k not in used_keys]
print("never referenced", len(never))

c = collections.Counter((keys[k]["dataset"], keys[k]["type"]) for k in never)
for k, v in sorted(c.items()):
    print("  never", k, v)
cm = collections.Counter(keys[k]["model"] for k in never)
print("never by model", dict(sorted(cm.items())))

# Does every never-used fake have its real source also never used? (pair-disjointness for a test set)
never_set = set(never)
fake_never = [k for k in never if keys[k]["type"] != "real"]
src_used = sum(1 for k in fake_never if keys[k]["real_image_path"] in used_keys)
print("never-used fakes", len(fake_never), "whose real source IS used elsewhere", src_used)

# usage by file category (top-level protocol files only)
cat = collections.Counter()
for f, n in sorted(per_file.items(), key=lambda x: -x[1])[:60]:
    print(f"  {n:6d}  {os.path.relpath(f, '${MEDFORGE_WORKSPACE}')}")

json.dump({"never": never}, open(f"{OUT}/mf90k_never_used.json", "w"))
print("unknown sample", list(unknown)[:5])
