"""
Per-sample error-overlap analysis between the Stage-1 replay-SFT checkpoint and the Stage-2
(best classification-dominant, beta=0, 400-step) RL checkpoint on the Cross-Forgery test set.

Backs the interpolation-mechanism claim in the paper (Section 4.3 / Limitations): if the two
checkpoints make *complementary* verdict errors -- i.e. a non-trivial fraction of the test set is
gotten right by exactly one of them -- then a weight blend has room to recover cases neither
endpoint alone dominates, which is consistent with the interpolated model beating both endpoints.
This is the empirical check the paper currently flags as not-yet-done.

No GPU / no training: reads the two already-saved generation files and cross-tabulates their
parsed verdicts against ground truth.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from circe_reward import parse_output

DF = {"remove", "inject", "edit", "deepfake", "removal", "injection", "implant"}
BASE = _release_path('AAAI/MedForge++')
SFT_GEN = f"{BASE}/gen_mfp_crossforgery_implant_replay_sft_8b.jsonl"
RL_GEN = f"{BASE}/gen_v2_classdom_beta0_final.jsonl"


def load_correct(path):
    """image -> bool (verdict correct) for every row in a gen file."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        gt = "deepfake" if r["type"] in DF else "real"
        pred = parse_output(r["output"])["verdict"] or "real"
        out[r["image"]] = (pred == gt)
    return out


def main():
    sft = load_correct(SFT_GEN)
    rl = load_correct(RL_GEN)
    common = sorted(set(sft) & set(rl))
    n = len(common)

    both_right = sum(1 for k in common if sft[k] and rl[k])
    both_wrong = sum(1 for k in common if not sft[k] and not rl[k])
    sft_only = sum(1 for k in common if sft[k] and not rl[k])
    rl_only = sum(1 for k in common if not sft[k] and rl[k])

    sft_acc = sum(sft[k] for k in common) / n
    rl_acc = sum(rl[k] for k in common) / n
    # oracle: correct if *either* checkpoint is correct -- the ceiling any blend of the two could reach
    oracle = sum(1 for k in common if sft[k] or rl[k]) / n
    disagreement = (sft_only + rl_only) / n

    print(f"paired test images:        {n}")
    print(f"Stage-1 (replay-SFT) acc:  {sft_acc*100:.2f}%")
    print(f"Stage-2 (best RL)    acc:  {rl_acc*100:.2f}%")
    print("-" * 42)
    print(f"both correct:              {both_right:5d}  ({both_right/n*100:.2f}%)")
    print(f"both wrong:                {both_wrong:5d}  ({both_wrong/n*100:.2f}%)")
    print(f"SFT-only correct:          {sft_only:5d}  ({sft_only/n*100:.2f}%)")
    print(f"RL-only correct:           {rl_only:5d}  ({rl_only/n*100:.2f}%)")
    print("-" * 42)
    print(f"total disagreement:        {disagreement*100:.2f}%")
    print(f"oracle (either correct):   {oracle*100:.2f}%   <- ceiling for any 2-way blend")
    print(f"headroom over best single: {(oracle-max(sft_acc,rl_acc))*100:.2f} points")
    # interpolation (alpha=0.10) actually reached 96.76; report gap to oracle for context
    print(f"interp alpha=0.10 reached:  96.76%   (captures "
          f"{(0.9676-max(sft_acc,rl_acc))/(oracle-max(sft_acc,rl_acc))*100:.0f}% of the headroom)")


if __name__ == "__main__":
    main()
