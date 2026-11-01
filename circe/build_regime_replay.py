"""
Failure-targeted replay for the In-Domain / Cross-Model regimes, mirroring the
Stage-1 replay mechanism described in the paper (Section 4.1): mine weak
(manipulation type, organ) strata from the current SFT checkpoint's errors on
a held-out validation split, then oversample matching strata in the real
training pool 2x, mixed with the unweighted remainder.

Unlike build_targeted_sft_replay.py (which targets one excluded manipulation
type for the Cross-Forgery regime), this script has no excluded type -- every
type is present in In-Domain/Cross-Model training -- so it ranks ALL strata by
validation failure rate and oversamples the weakest ones generically.

Inputs:
  --val_eval : eval-format jsonl of the held-out validation split {image,type,box}
  --val_gens : eval_medforge.py --save_gen output on that validation split
  --train    : the real SFT training pool (jsonl, {images, messages} format)
  --out      : replay training jsonl (same format as --train)
  --factor   : oversample multiplier for weak strata (default 2, matching the paper)
  --top_k    : how many weakest strata to target (default 12)
"""
import argparse
import json
import random
import sys
import os
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from circe_reward import parse_output

DEEPFAKE_TYPES = {"remove", "inject", "deepfake", "removal", "injection", "edit"}


def image_type(path):
    low = path.lower()
    if "/real/" in low:
        return "real"
    if "-edit/" in low or "/edit/" in low or "-inject/" in low or "/inject/" in low:
        return "edit"
    if "-remove/" in low or "/remove/" in low:
        return "remove"
    return "deepfake"


def strata(path):
    parts = path.split("/")
    typ = image_type(path)
    organ = "unknown"
    for i, part in enumerate(parts):
        low = part.lower()
        if typ in {"edit", "remove"} and (low.endswith(f"-{typ}") or low == typ):
            organ = low.rsplit("-", 1)[0] if "-" in low else "unknown"
            break
    if typ == "real":
        try:
            j = parts.index("real")
            organ = parts[j + 1].lower()
        except (ValueError, IndexError):
            pass
    return typ, organ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_eval", required=True)
    ap.add_argument("--val_gens", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    val_rows = {json.loads(l)["image"]: json.loads(l) for l in open(args.val_eval)}

    fail, seen = Counter(), Counter()
    for line in open(args.val_gens):
        gen = json.loads(line)
        row = val_rows.get(gen["image"])
        if not row:
            continue
        key = strata(gen["image"])
        gt = "deepfake" if row["type"] in DEEPFAKE_TYPES else "real"
        pred = parse_output(gen["output"])["verdict"] or "real"
        seen[key] += 1
        if pred != gt:
            fail[key] += 1

    ranked = sorted(
        ((fail[k] / n if n else 0.0, fail[k], n, k) for k, n in seen.items()),
        reverse=True,
    )
    print("[weak strata on held-out validation]")
    for rate, f, n, key in ranked[:args.top_k]:
        print(f"  {key}: fail={f}/{n} rate={rate:.3f}")
    weak_strata = {key for _, f, _, key in ranked[:args.top_k] if f > 0}

    rng = random.Random(args.seed)
    out = []
    counts = Counter()
    for line in open(args.train):
        rec = json.loads(line)
        img = rec["images"][0]
        key = strata(img)
        reps = args.factor if key in weak_strata else 1
        out.extend([rec] * reps)
        counts[key] += reps
    rng.shuffle(out)

    with open(args.out, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"[out] {len(out)} records ({len(weak_strata)} strata oversampled {args.factor}x) -> {args.out}")


if __name__ == "__main__":
    main()
