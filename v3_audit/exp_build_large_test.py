#!/usr/bin/env python
"""Build an enlarged, image-disjoint MF90K test file from images never referenced by the current protocol
(train/val/rl/replay/test/v3_select). CPU only; writes under ~/CIRCE/v3_audit/exp_out/.
Format identical to mfp_v3_select_clean.jsonl: {"image","type","gen","box"}.
Stratified: N_PER_TYPE reals (equal per organ) / edits / removes, fakes balanced over the 10 generators."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os, json, random, collections
import numpy as np
from PIL import Image
from multiprocessing import Pool

ROOT = _release_path('AAAI/MedForge++/medforge90k_full')
OUT = _release_path('CIRCE/v3_audit/exp_out')
N_PER_TYPE = 4000
SEED = 2027
HELD = {"gemini", "gpt", "stable-diffusion-3.5-medium", "stable-diffusion-xl-1.0-inpainting-0.1"}

def box_from_mask(mask_rel):          # same as CIRCE/circe/build_medforge_protocol.py
    p = os.path.join(ROOT, mask_rel or "")
    if not mask_rel or not os.path.exists(p): return None
    m = np.array(Image.open(p).convert("L")); ys, xs = np.where(m > 127)
    if len(xs) == 0: return None
    H, W = m.shape
    return [int(xs.min()*1024/W), int(ys.min()*1024/H), int(xs.max()*1024/W), int(ys.max()*1024/H)]

def rec(it):
    img = os.path.join(ROOT, it["image_path"])
    if not os.path.exists(img): return None
    t = it["type"]
    box = None if t == "real" else box_from_mask(it.get("mask_path"))
    if t != "real" and box is None: return None
    return {"image": img, "type": t, "gen": it["model"], "box": box, "organ": it["dataset"]}

if __name__ == "__main__":
    items = {it["image_path"]: it for it in json.load(open(f"{ROOT}/dataset_mapping.json"))["items"]}
    never = json.load(open(f"{OUT}/mf90k_never_used_protocol.json"))["never"]
    rng = random.Random(SEED)
    pick = []
    reals = [items[k] for k in never if items[k]["type"] == "real"]
    by_org = collections.defaultdict(list)
    for it in reals: by_org[it["dataset"]].append(it)
    per_org = N_PER_TYPE // len(by_org)
    for o, L in sorted(by_org.items()):
        rng.shuffle(L); pick += L[:per_org]
    for t in ["edit", "remove"]:
        by_gen = collections.defaultdict(list)
        for k in never:
            if items[k]["type"] == t: by_gen[items[k]["model"]].append(items[k])
        per_gen = N_PER_TYPE // len(by_gen)
        for g, L in sorted(by_gen.items()):
            rng.shuffle(L); pick += L[:per_gen]
    with Pool(12) as pool:
        recs = [r for r in pool.map(rec, pick, chunksize=32) if r]
    rng.shuffle(recs)
    fn = f"{OUT}/mfp_v3_test_large12k.jsonl"
    with open(fn, "w") as f:
        for r in recs: f.write(json.dumps({k: r[k] for k in ("image", "type", "gen", "box")}) + "\n")
    c = collections.Counter((r["organ"], r["type"]) for r in recs)
    print("wrote", fn, len(recs)); [print("  ", k, v) for k, v in sorted(c.items())]
    print("  CM held-out-generator fakes:", sum(r["gen"] in HELD for r in recs))
    print("  per-generator:", dict(sorted(collections.Counter(r["gen"] for r in recs).items())))
    # sanity: disjoint from shared test and v3_select
    def imgs(p): return {json.loads(l)["image"] for l in open(p)}
    A = _release_path('AAAI/MedForge++')
    s = {r["image"] for r in recs}
    print("  overlap with mfp_v2_test_shared:", len(s & imgs(f"{A}/mfp_v2_test_shared.jsonl")),
          " with v3_select:", len(s & imgs(f"{A}/v3_select/mfp_v3_select_clean.jsonl")))
    # strict (source-disjoint) real pool for a future source-disjoint test
    strict = json.load(open(f"{OUT}/mf90k_strict_pool.json"))["strict"]
    with open(f"{OUT}/mf90k_strict_real_pool.jsonl", "w") as f:
        for k in strict:
            f.write(json.dumps({"image": os.path.join(ROOT, k), "type": "real", "gen": "real", "box": None,
                                "folder": "/".join(k.split("/")[1:3])}) + "\n")
    print("wrote strict source-disjoint real pool:", len(strict))
