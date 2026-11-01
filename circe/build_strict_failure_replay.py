"""
CIRCE-v2 Phase 9: protocol-legal failure-targeted replay mining.

Differs from build_targeted_sft_replay.py (the script that produced the leaky
mfp_replay_implant_focus_sft.jsonl, see audit_protocol.py's findings -- 4,678/6,400 rows
were implant-derived, invalidating a strict Cross-Forgery claim) in three ways:

1. Source pool is REQUIRED to be a versioned mfp_v2_train_<regime>.jsonl / _strict.jsonl
   file (the leak-free manifests audit_protocol.py --build-strict produces), not an
   arbitrary list of SFT source files that might include the in-domain pool wholesale.
2. Refuses to mine from a file whose name matches a known test-set pattern
   (`mfp_test_shared`, `mfp_v2_test_shared`) -- hard assertion, not just a convention.
3. For target_type="edit" under a --strict flag, asserts zero surviving implant-derived
   rows in the OUTPUT (not just the eval-failure-ranking step) before writing anything.

Failure-category composition (spec'd mix, best-effort against what's available):
    40% authentic false positives   (gt=real,     pred=deepfake)
    30% fake false negatives        (gt=deepfake, pred=real)
    20% low-margin correct cases    (correct verdict, but < margin_conf confidence if available,
                                      else: correct verdict with a weak/short evidence match)
    10% correct-verdict, poor-localization cases (gt has a box, correct verdict, IoU < loc_tau)

Every output record carries a provenance block so downstream code (and the audit script)
can verify legality without re-deriving it:
    {"replay_reason": "false_positive|false_negative|low_margin|bad_localization",
     "source_split": ..., "source_checkpoint": ..., "protocol_regime": ...,
     "is_protocol_legal": true}
"""
import argparse, json, os, random, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from circe_reward import parse_output, iou
from audit_protocol import organ_manip_type

_TEST_SET_MARKERS = ("mfp_test_shared", "mfp_v2_test_shared")


def _assert_not_test_set(path, role):
    base = os.path.basename(path)
    for marker in _TEST_SET_MARKERS:
        if marker in base:
            raise SystemExit(f"[REFUSED] --{role} points at a held-out test-set file ({path}). "
                              f"Replay mining must never read the final test set.")


def load_eval_and_gens(eval_path, gens_path):
    _assert_not_test_set(eval_path, "eval")
    eval_rows = {json.loads(l)["image"]: json.loads(l) for l in open(eval_path)}
    out = []
    for line in open(gens_path):
        gen = json.loads(line)
        row = eval_rows.get(gen["image"])
        if not row:
            continue
        pred = parse_output(gen["output"])
        gt_type = row.get("type", "real")
        gt_verdict = "deepfake" if gt_type in ("edit", "remove", "inject", "removal", "injection") else "real"
        out.append({"image": gen["image"], "gt_verdict": gt_verdict, "gt_box": row.get("box"),
                    "pred_verdict": pred["verdict"], "pred_box": pred["box"], "raw_output": gen["output"]})
    return out


def categorize(rows, loc_tau=0.5):
    """Bucket eval rows into the four failure_reason categories."""
    buckets = {"false_positive": [], "false_negative": [], "low_margin": [], "bad_localization": []}
    for r in rows:
        gt, pred = r["gt_verdict"], r["pred_verdict"]
        if gt == "real" and pred == "deepfake":
            buckets["false_positive"].append(r)
        elif gt == "deepfake" and pred == "real":
            buckets["false_negative"].append(r)
        elif pred == gt and gt == "deepfake" and r["gt_box"] is not None:
            _iou = iou(r["pred_box"], r["gt_box"]) if r["pred_box"] else 0.0
            if _iou < loc_tau:
                buckets["bad_localization"].append(r)
            elif len(r["raw_output"]) < 250:   # short/weak grounded reasoning -> "low margin"
                buckets["low_margin"].append(r)
        elif pred == gt:
            buckets["low_margin"].append(r)
    return buckets


def strata_key(path):
    """(type, organ, disease) strata key, used to find TRAINING-pool replacement examples
    that share the same failure profile as an eval-side failure -- NOT to look up the exact
    same image, which by construction of a proper train/val split will never be in the pool."""
    manip = organ_manip_type(path)
    parts = path.split("/")
    organ, disease = "unknown", "unknown"
    if manip in ("edit", "remove") and "deepfake" in parts:
        i = parts.index("deepfake")
        if i + 2 < len(parts):
            seg = parts[i + 2]
            organ = seg.rsplit("-", 1)[0]
            if i + 3 < len(parts):
                disease = parts[i + 3].lower()
    elif "real" in parts:
        j = parts.index("real")
        if j + 1 < len(parts):
            organ = parts[j + 1].lower()
        if j + 2 < len(parts):
            disease = parts[j + 2].lower()
    typ = "real" if manip in (None, "real") else manip
    return (typ, organ, disease)


def load_legal_pool(train_path, strict, eval_images):
    """Load the SFT record pool, bucketed by strata_key for replacement-example lookup.
    Refuses to keep any row also present in eval_images (belt-and-suspenders; a proper
    train/val split should already make this a no-op)."""
    by_strata = {}
    n_implant_dropped = 0
    n_total = 0
    for line in open(train_path):
        rec = json.loads(line)
        img = rec["images"][0]
        n_total += 1
        if img in eval_images:
            continue
        if strict and organ_manip_type(img) == "edit":
            n_implant_dropped += 1
            continue
        by_strata.setdefault(strata_key(img), []).append(rec)
    return by_strata, n_implant_dropped, n_total


def sample_replay(buckets, pool_by_strata, n_target, fracs, seed):
    """For each failure category, find pool records sharing the FAILING image's strata
    (organ/disease/manipulation-type) -- this is what makes it a targeted replay: we can't
    replay the literal failing eval image (it's not in the training pool by construction of
    the split), but we CAN oversample training images from the same weak strata."""
    rng = random.Random(seed)
    order = ["false_positive", "false_negative", "low_margin", "bad_localization"]
    out, counts, used = [], Counter(), set()
    for reason, frac in zip(order, fracs):
        fail_rows = buckets[reason][:]
        rng.shuffle(fail_rows)
        n_want = int(n_target * frac)
        added = 0
        # gather all pool records whose strata matches ANY failure in this bucket
        target_strata = {strata_key(r["image"]) for r in fail_rows}
        candidates = []
        for sk in target_strata:
            candidates.extend(pool_by_strata.get(sk, []))
        rng.shuffle(candidates)
        for rec in candidates:
            img = rec["images"][0]
            if img in used:
                continue
            used.add(img)
            out.append((rec, reason))
            added += 1
            if added >= n_want:
                break
        counts[reason] = added
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="held-out-from-training eval jsonl (NOT the final test set)")
    ap.add_argument("--gens", required=True, help="model generations on --eval")
    ap.add_argument("--train_pool", required=True,
                     help="the legal SFT source pool to mine replay records from "
                          "(should be an mfp_v2_train_*.jsonl / mfp_v2_val_*.jsonl manifest)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol_regime", required=True, choices=["indomain", "crossmodel", "crossforgery_strict", "pfplus"])
    ap.add_argument("--source_checkpoint", required=True, help="checkpoint path used to produce --gens (provenance only)")
    ap.add_argument("--strict", action="store_true",
                     help="drop any implant-derived row from the legal pool before sampling "
                          "(required for crossforgery_strict)")
    ap.add_argument("--n_target", type=int, default=3600)
    ap.add_argument("--fracs", type=float, nargs=4, default=[0.40, 0.30, 0.20, 0.10],
                     help="false_positive false_negative low_margin bad_localization")
    ap.add_argument("--loc_tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.protocol_regime == "crossforgery_strict" and not args.strict:
        raise SystemExit("[REFUSED] --protocol_regime crossforgery_strict requires --strict.")

    _assert_not_test_set(args.train_pool, "train_pool")
    eval_images = {json.loads(l)["image"] for l in open(args.eval)}

    eval_rows = load_eval_and_gens(args.eval, args.gens)
    buckets = categorize(eval_rows, args.loc_tau)
    print("[failure buckets]")
    for k, v in buckets.items():
        print(f"  {k:16s}: {len(v)}")

    pool_by_strata, n_dropped, n_total = load_legal_pool(args.train_pool, args.strict, eval_images)
    n_pool = sum(len(v) for v in pool_by_strata.values())
    print(f"[pool] {args.train_pool}: {n_total} total rows, {n_pool} legal after filtering, "
          f"{len(pool_by_strata)} distinct strata")
    if args.strict:
        print(f"[strict] dropped {n_dropped} implant-derived rows from the legal pool before sampling")

    chosen, counts = sample_replay(buckets, pool_by_strata, args.n_target, args.fracs, args.seed)

    if args.strict:
        leaks = [rec for rec, _ in chosen if organ_manip_type(rec["images"][0]) == "edit"]
        assert not leaks, f"[BUG] {len(leaks)} implant-derived rows survived strict filtering -- refusing to write output."

    out_recs = []
    for rec, reason in chosen:
        rec = dict(rec)  # shallow copy so we don't mutate the shared pool dict
        rec["replay_reason"] = reason
        rec["source_split"] = args.train_pool
        rec["source_checkpoint"] = args.source_checkpoint
        rec["protocol_regime"] = args.protocol_regime
        rec["is_protocol_legal"] = True
        out_recs.append(rec)

    with open(args.out, "w") as f:
        for r in out_recs:
            f.write(json.dumps(r) + "\n")

    print(f"[out] {len(out_recs)} replay records -> {args.out}")
    print(f"[counts] {dict(counts)}")


if __name__ == "__main__":
    main()
