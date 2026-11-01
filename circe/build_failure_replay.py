"""
Typed failure-ledger + replay (task #4).

Tag each training example by the current model's failure type (SOLVED / LOCALIZE / VERDICT / BOTH /
HALLUCINATE) using classify_failure, then build a REPLAY training set that oversamples the hard
(non-SOLVED) cases. A follow-up GRPO round on this set = the failure-replay mechanism.

Input:
  --gens : jsonl of the current model's outputs on train prompts {image, gt_verdict, gt_box, type, output}
  --sft  : the SFT-format train jsonl (mf90k_train.jsonl) — source of {messages, images} records
Output:
  --out  : replay jsonl (SFT format) with failures oversampled --factor x
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, argparse, sys
sys.path.insert(0, os.path.dirname(__file__))
from circe_reward import parse_output, classify_failure

DF = {"remove", "inject", "edit"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--sft", default=_release_path('AAAI/MedForge++/mf90k_train.jsonl'))
    ap.add_argument("--out", default=_release_path('AAAI/MedForge++/mf90k_replay.jsonl'))
    ap.add_argument("--factor", type=int, default=3, help="oversample multiplier for failures")
    args = ap.parse_args()

    # 1) tag failures from gens
    ledger = {}  # image -> failure type
    from collections import Counter
    counts = Counter()
    for l in open(args.gens):
        r = json.loads(l)
        gt_v = "deepfake" if r["type"] in DF else "real"
        gt_box = r.get("gt_box"); has_box = bool(gt_box)
        p = parse_output(r["output"])
        ftype = classify_failure(p, gt_box, has_box, (p["verdict"] or "real") == gt_v)
        ledger[r["image"]] = ftype        # key by FULL path (basenames collide across generators)
        counts[ftype] += 1
    print("failure ledger:", dict(counts))

    # 2) build FOCUSED replay from only the TAGGED records, oversampling non-SOLVED failures
    out = []
    for l in open(args.sft):
        rec = json.loads(l)
        name = rec["images"][0]
        if name not in ledger:            # only replay what the model actually saw+failed/solved
            continue
        ftype = ledger[name]
        reps = 1 if ftype == "SOLVED" else args.factor
        out.extend([rec] * reps)
    import random
    random.seed(0); random.shuffle(out)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    solved = sum(1 for v in ledger.values() if v == "SOLVED")
    print(f"replay set: {len(out)} records (from {len(ledger)} tagged; {solved} solved, "
          f"{len(ledger)-solved} failures oversampled {args.factor}x) -> {args.out}")


if __name__ == "__main__":
    main()
