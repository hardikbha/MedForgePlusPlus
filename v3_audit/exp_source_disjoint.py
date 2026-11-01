#!/usr/bin/env python
"""How many never-used MF90K images form a SOURCE-DISJOINT test pool (neither the real source image nor
any of its derivatives appear in any train/val/RL/replay file of a given regime)? Also: does the
existing shared test (n=2,994) already share sources with training?"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os, re, json, glob, collections, math

ROOT = _release_path('AAAI/MedForge++')
items = json.load(open(f"{ROOT}/medforge90k_full/dataset_mapping.json"))["items"]
keys = {it["image_path"]: it for it in items}
pat = re.compile(r'(?:medforge90k_full|MedForge-90K|medforge90k_eval)/((?:real|deepfake)/[^"\s\\]+?\.(?:jpe?g|png|JPE?G|PNG))')
def refs(f):
    s = set()
    with open(f, errors="ignore") as fh:
        for line in fh: s.update(pat.findall(line))
    return s & set(keys)
def src(k):
    it = keys[k]; return k if it["type"] == "real" else it["real_image_path"]

regimes = {
 "CF": ["mfp_train_crossforgery.jsonl", "mfp_v2_train_crossforgery_strict.jsonl", "mfp_v2_val_crossforgery_strict.jsonl",
        "mfp_v2_rl_crossforgery_strict.jsonl", "mfp_v2_replay_crossforgery_strict.jsonl", "mfp_replay_implant_focus_sft.jsonl"],
 "CM": ["mfp_train_crossmodel.jsonl", "mfp_v2_train_crossmodel.jsonl", "mfp_v2_val_crossmodel.jsonl",
        "mfp_v2_rl_crossmodel.jsonl", "mfp_crossmodel_replay.jsonl"],
 "ID": ["mfp_train_indomain.jsonl", "mfp_v2_train_indomain.jsonl", "mfp_v2_val_indomain.jsonl",
        "mfp_v2_rl_indomain.jsonl", "mfp_indomain_replay.jsonl"],
}
never = set(json.load(open(_release_path('CIRCE/v3_audit/exp_out/mf90k_never_used_protocol.json')))["never"])
test = refs(f"{ROOT}/mfp_v2_test_shared.jsonl")
sel = refs(f"{ROOT}/v3_select/mfp_v3_select_clean.jsonl")
# group: source real -> all images (real + derivatives)
groups = collections.defaultdict(set)
for k in keys: groups[src(k)].add(k)
print("source groups", len(groups), " mean images/group %.2f" % (len(keys)/len(groups)))
allused_train = set()
for r, fl in regimes.items():
    tr = set()
    for f in fl:
        p = f"{ROOT}/{f}"
        if os.path.exists(p): tr |= refs(p)
    allused_train |= tr
    trsrc = {src(k) for k in tr}
    t_overlap = sum(src(k) in trsrc for k in test)
    s_overlap = sum(src(k) in trsrc for k in sel)
    disj = [k for k in never if src(k) not in trsrc]
    print(f"[{r}] train/val/rl/replay images {len(tr)} from {len(trsrc)} sources; "
          f"shared test items whose source is in {r}-train: {t_overlap}/{len(test)}; "
          f"v3_select: {s_overlap}/{len(sel)}; never-used AND source-disjoint from {r}-train: {len(disj)}")
    print("     by (organ,type):", dict(sorted(collections.Counter((keys[k]['dataset'], keys[k]['type']) for k in disj).items())))
# strictest: source-disjoint from the union of ALL regimes' training files AND from test/select
allsrc = {src(k) for k in allused_train} | {src(k) for k in test} | {src(k) for k in sel}
strict = [k for k in never if src(k) not in allsrc]
c = collections.Counter((keys[k]['dataset'], keys[k]['type']) for k in strict)
print("STRICT pool (never-used, source-disjoint from ALL regimes' train/val/rl/replay + test + select):", len(strict))
for k, v in sorted(c.items()): print("   ", k, v)
print("   by model:", dict(sorted(collections.Counter(keys[k]['model'] for k in strict).items())))
# held-out-generator subset of the strict pool (CM held-out gens)
held = {"gemini", "gpt", "stable-diffusion-3.5-medium", "stable-diffusion-xl-1.0-inpainting-0.1"}
print("   strict pool fakes from CM held-out generators:", sum(keys[k]['model'] in held for k in strict))
json.dump({"strict": strict}, open(_release_path('CIRCE/v3_audit/exp_out/mf90k_strict_pool.json'), "w"))
