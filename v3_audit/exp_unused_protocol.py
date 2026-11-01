#!/usr/bin/env python
"""Never-used MF90K images relative to the CURRENT training/eval protocol files (not old v1 pools / gen outputs)."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os, re, json, glob, collections, random, math

ROOT = _release_path('AAAI/MedForge++')
OUT = _release_path('CIRCE/v3_audit/exp_out')
items = json.load(open(f"{ROOT}/medforge90k_full/dataset_mapping.json"))["items"]
keys = {it["image_path"]: it for it in items}
pat = re.compile(r'(?:medforge90k_full|MedForge-90K|medforge90k_eval)/((?:real|deepfake)/[^"\s\\]+?\.(?:jpe?g|png|JPE?G|PNG))')

def refs(f):
    s = set()
    with open(f, errors="ignore") as fh:
        for line in fh:
            s.update(pat.findall(line))
    return s

protocol = sorted(set(
    glob.glob(f"{ROOT}/mfp_train_*.jsonl") + glob.glob(f"{ROOT}/mfp_v2_*.jsonl") +
    glob.glob(f"{ROOT}/mfp_*replay*.jsonl") + glob.glob(f"{ROOT}/*replay*.jsonl") +
    glob.glob(f"{ROOT}/mfp_test_shared.jsonl") + glob.glob(f"{ROOT}/v3_select/mfp_v3_select_clean.jsonl") +
    glob.glob(f"{ROOT}/circe_*.jsonl") + glob.glob(f"{ROOT}/raft_*.jsonl") + glob.glob(f"{ROOT}/mix8.jsonl") +
    glob.glob(f"{ROOT}/genmix.jsonl") + glob.glob(f"{ROOT}/mf90k_replay.jsonl") +
    glob.glob(_release_path('CIRCE/v3/*.jsonl'))))
old = [f"{ROOT}/mf90k_train.jsonl", f"{ROOT}/mf90k_train_eval600.jsonl", f"{ROOT}/mf90k_test_crossmodel.jsonl",
       f"{ROOT}/mf90k_test_crossmodel_f.jsonl", f"{ROOT}/mf90k_test_crossmodel_fixed.jsonl",
       f"{ROOT}/mf90k_test_indomain.jsonl", f"{ROOT}/mf90k_test_indomain_f.jsonl"]
used = set(); per = {}
for f in protocol:
    r = refs(f) & set(keys); per[os.path.basename(f)] = len(r); used |= r
print("protocol files", len(protocol))
for k, v in sorted(per.items(), key=lambda x: -x[1]):
    if v: print(f"  {v:6d} {k}")
never = [k for k in keys if k not in used]
print("NEVER used by current protocol files:", len(never))
c = collections.Counter((keys[k]["dataset"], keys[k]["type"]) for k in never)
for k, v in sorted(c.items()): print("  ", k, v)
print(" by type", collections.Counter(keys[k]["type"] for k in never))
print(" by model", dict(sorted(collections.Counter(keys[k]["model"] for k in never).items())))

oldused = set()
for f in old:
    if os.path.exists(f):
        r = refs(f) & set(keys); print(f"  old-protocol file {os.path.basename(f)}: {len(r)} MF90K refs, overlap with never-used: {len(r & set(never))}")
        oldused |= r
print("never-used that appear in OLD v1 pools (mf90k_train/test):", len(set(never) & oldused))

sel = refs(f"{ROOT}/v3_select/mfp_v3_select_clean.jsonl")
print("v3_select size", len(sel))
# real-source pairing: never-used fakes whose real counterpart is in the training sets
train_files = [f for f in protocol if ("train" in os.path.basename(f) or "replay" in os.path.basename(f) or "_rl_" in os.path.basename(f))]
train_used = set()
for f in train_files: train_used |= refs(f) & set(keys)
nf = [k for k in never if keys[k]["type"] != "real"]
print("never-used fakes", len(nf), "; whose real source image is in some TRAIN/RL/replay file:", sum(keys[k]["real_image_path"] in train_used for k in nf))
nr = [k for k in never if keys[k]["type"] == "real"]
fake_of_real = collections.defaultdict(list)
for it in items:
    if it["type"] != "real": fake_of_real[it["real_image_path"]].append(it["image_path"])
print("never-used reals", len(nr), "; with >=1 fake derivative in TRAIN:", sum(any(x in train_used for x in fake_of_real.get(k, [])) for k in nr))

# CI half-width at acc ~0.94 for various n
for n in [2994, 3000, 6000, 10000, 15000, 20000, 36227]:
    print(f"  95% Wald CI half-width at p=0.94, n={n}: +/-{1.96*math.sqrt(0.94*0.06/n)*100:.2f} pp")
json.dump({"never": never}, open(f"{OUT}/mf90k_never_used_protocol.json", "w"))
