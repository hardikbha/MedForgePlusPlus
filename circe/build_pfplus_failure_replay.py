"""
Failure-targeted replay SFT for PF+ (MedForge++ kidney/colon addition), mirroring the recipe
that produced the Cross-Forgery replay-SFT win (95.76/95.13) -- but built from-scratch for PF+
because its directory naming (naitik_annotated/<subset>/, colon_injections/, flux_injected/)
doesn't match build_targeted_sft_replay.py's MedForge-style path regexes.

v2 -- rebalanced after v1's over-aggressive single-subset oversample (kidney_removal 5x, all
others 1x) fixed the target (3%->86% acc) but destroyed both the model's "real kidney" prior
(kidney_normal 98%->8%, kidney_real_tumor 95%->10%) and the unrelated Cross-Forgery/MedForge
domain (95.76%->63.8% acc) via distribution shift. v2 uses a flatter, broader oversample (max
2x, spread across the weak AND their sibling real classes) plus a MedForge-domain anchor slice
mixed in specifically to prevent the cross-domain collapse seen in v1.

Per-subset accuracy of the current best PF+ checkpoint (circe_sft_v0.3_8b, 82.25% overall),
measured directly from gen_v0.3_8b.jsonl on circe_eval_split_v2.jsonl (n=800):
  kidney_normal 98%, kidney_real_tumor 95%, kidney_removal 3%(!), polyp_normal 100%,
  polyp_real 88%, polyp_removal 75%, colon_inject 99%, kidney_inject 100%.

NOTE on data hygiene: circe_sft_v3.jsonl (the PF+ train pool) was found to contain all 800
images in circe_eval_split_v2.jsonl (100% overlap) -- the "held-out" eval set was carved out
of the train pool without being removed from it, so the current 82.25% baseline is itself
evaluated on partially-memorized data. This script excludes all eval images from the replay
set (so this continued-training step doesn't make that worse), but the starting checkpoint
was already trained on them -- flagged, not silently fixed, matching how the MedForge-side
replay leak was handled earlier in this project.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json
import re
import random
from collections import Counter

TRAIN = _release_path('AAAI/MedForge++/circe_sft_v3.jsonl')
EVAL = _release_path('AAAI/MedForge++/circe_eval_split_v2.jsonl')
MEDFORGE_ANCHOR = _release_path('AAAI/MedForge++/mfp_replay_implant_focus_sft.jsonl')
OUT = _release_path('AAAI/MedForge++/pfplus_replay_v2_balanced_sft.jsonl')

# v2: flatter, max 2x, spread across the weak subset AND its sibling "real" class so the
# real-kidney/real-polyp prior isn't washed out this time.
FACTOR = {
    "kidney_removal": 2,     # 3% acc originally -- still the primary target, gentler this time
    "kidney_normal": 2,      # collapsed to 8% in v1 -- oversampled this round to re-anchor it
    "kidney_real_tumor": 2,  # collapsed to 10% in v1 -- oversampled this round to re-anchor it
    "polyp_removal": 2,      # 75% acc -- secondary weak spot
    "polyp_real": 2,         # dropped 88%->72% in v1 -- oversampled this round to re-anchor it
    "polyp_normal": 1,
    "colon_inject": 1,
    "kidney_inject": 1,
}
N_MEDFORGE_ANCHOR = 3000    # MedForge-domain rows mixed in to guard against the Cross-Forgery
                             # collapse seen in v1 (95.76%->63.8%)


def get_subset(path):
    m = re.search(r"naitik_annotated/([^/]+)/", path)
    if m:
        return m.group(1)
    if "colon_injections" in path:
        return "colon_inject"
    if "flux_injected" in path:
        return "kidney_inject"
    return "unknown"


def main():
    eval_images = {json.loads(l)["image"] for l in open(EVAL)}
    print(f"[eval] {len(eval_images)} held-out images excluded from replay pool")

    by_subset = {}
    for l in open(TRAIN):
        rec = json.loads(l)
        img = rec["images"][0]
        if img in eval_images:
            continue
        by_subset.setdefault(get_subset(img), []).append(rec)

    print("[pool] non-eval records per subset:", {k: len(v) for k, v in by_subset.items()})

    rng = random.Random(0)
    out = []
    for subset, records in by_subset.items():
        factor = FACTOR.get(subset, 1)
        for _ in range(factor):
            out.extend(records)

    anchor_rows = [json.loads(l) for l in open(MEDFORGE_ANCHOR)]
    rng.shuffle(anchor_rows)
    anchor_rows = anchor_rows[:N_MEDFORGE_ANCHOR]
    out.extend(anchor_rows)

    rng.shuffle(out)

    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    counts = Counter(get_subset(r["images"][0]) for r in out)
    print(f"[out] {len(out)} records ({len(anchor_rows)} MedForge anchor rows) -> {OUT}")
    print("[composition]", dict(counts))


if __name__ == "__main__":
    main()
