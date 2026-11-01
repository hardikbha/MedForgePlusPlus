"""
Build a train-side SFT replay set targeted at held-out failure strata.

The eval generations are used only to identify weak strata (for example
implant/edit cases by organ/disease). The output records are copied from SFT
training jsonl files, and any image present in the eval jsonl is excluded.
"""
import argparse
import json
import os
import random
import re
import sys
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
    organ, disease = "unknown", "unknown"
    for i, part in enumerate(parts):
        low = part.lower()
        if typ in {"edit", "remove"} and (low.endswith(f"-{typ}") or low == typ):
            organ = low.rsplit("-", 1)[0] if "-" in low else "unknown"
            if i + 1 < len(parts):
                disease = parts[i + 1].lower()
            break
    if typ == "real":
        try:
            j = parts.index("real")
            organ = parts[j + 1].lower()
            disease = parts[j + 2].lower() if j + 2 < len(parts) else "normal"
        except ValueError:
            pass
    return typ, organ, disease


def load_eval_fail_strata(eval_path, gens_path, target_type):
    eval_rows = {}
    for line in open(eval_path):
        row = json.loads(line)
        eval_rows[row["image"]] = row

    fail = Counter()
    seen = Counter()
    for line in open(gens_path):
        gen = json.loads(line)
        row = eval_rows.get(gen["image"])
        if not row:
            continue
        typ, organ, disease = strata(gen["image"])
        if typ != target_type:
            continue
        gt = "deepfake" if row["type"] in DEEPFAKE_TYPES else "real"
        pred = parse_output(gen["output"])["verdict"] or "real"
        key = (typ, organ, disease)
        seen[key] += 1
        if pred != gt:
            fail[key] += 1

    ranked = []
    for key, n in seen.items():
        ranked.append((fail[key] / n if n else 0.0, fail[key], n, key))
    ranked.sort(reverse=True)
    return ranked


def load_sft_records(paths, eval_images):
    buckets = defaultdict(list)
    duplicates = 0
    seen = set()
    for path in paths:
        for line in open(path):
            rec = json.loads(line)
            img = rec["images"][0]
            if img in eval_images:
                continue
            if img in seen:
                duplicates += 1
                continue
            seen.add(img)
            buckets[strata(img)].append(rec)
    return buckets, duplicates


def choose_records(buckets, ranked, target_type, n_target, n_remove, n_real, seed):
    rng = random.Random(seed)
    out = []
    used = set()

    def add(records, n, reps=1):
        rng.shuffle(records)
        added = 0
        for rec in records:
            img = rec["images"][0]
            if img in used:
                continue
            used.add(img)
            for _ in range(reps):
                out.append(rec)
            added += 1
            if added >= n:
                break
        return added

    # Allocate most target examples to the weakest eval strata, while retaining
    # a fallback pool of all target examples to avoid too narrow a replay set.
    target_added = 0
    top = [key for _, fails, _, key in ranked if fails > 0][:12]
    per = max(1, int(n_target * 0.75 / max(1, len(top))))
    for key in top:
        target_added += add(list(buckets.get(key, [])), per, reps=2)
    all_target = []
    for key, records in buckets.items():
        if key[0] == target_type:
            all_target.extend(records)
    target_added += add(all_target, max(0, n_target - target_added), reps=2)

    all_remove, all_real = [], []
    for key, records in buckets.items():
        if key[0] == "remove":
            all_remove.extend(records)
        elif key[0] == "real":
            all_real.extend(records)
    remove_added = add(all_remove, n_remove)
    real_added = add(all_real, n_real)

    rng.shuffle(out)
    counts = Counter(image_type(rec["images"][0]) for rec in out)
    return out, {"target_unique": target_added, "remove_unique": remove_added,
                 "real_unique": real_added, "output_type_counts": dict(counts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--gens", required=True)
    ap.add_argument("--sft_sources", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_type", default="edit")
    ap.add_argument("--n_target", type=int, default=3600)
    ap.add_argument("--n_remove", type=int, default=1200)
    ap.add_argument("--n_real", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    eval_images = {json.loads(line)["image"] for line in open(args.eval)}
    ranked = load_eval_fail_strata(args.eval, args.gens, args.target_type)
    print("[weak strata]")
    for rate, fails, n, key in ranked[:12]:
        print(f"  {key}: fail={fails}/{n} rate={rate:.3f}")

    buckets, duplicates = load_sft_records(args.sft_sources, eval_images)
    out, summary = choose_records(buckets, ranked, args.target_type,
                                  args.n_target, args.n_remove, args.n_real,
                                  args.seed)
    with open(args.out, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"[sources] buckets={len(buckets)} duplicates_skipped={duplicates}")
    print(f"[out] {len(out)} records -> {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
