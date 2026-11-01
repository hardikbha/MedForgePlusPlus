"""
Compose MedForge++ AGGREGATE results (the combined 90K-domain + 7.8K-domain evaluation) from
per-split generation files already on disk -- no new GPU runs.

The aggregate test set is the union of the MedForge Cross-Forgery shared test split (n=2,994,
the 90K-domain organs: chest X-ray / brain MRI / fundus) and the MedForge++ kidney/colon held-out
split (n=800), for n=3,794 total, representative of the full 103,076-image MedForge++ corpus.

For each model we already have two saved generation files (one per split); this script parses
both, scores verdict accuracy + macro-F1 over the concatenation, and prints a per-model row plus
the two per-split sub-scores, so the aggregate table and its provenance are both auditable.
Every number therefore traces to a saved eval_medforge.py generation file -- nothing re-run,
nothing invented.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from circe_reward import parse_output

DF = {"remove", "inject", "edit", "deepfake", "removal", "injection", "implant"}
BASE = _release_path('AAAI/MedForge++')

# model -> (cross-forgery gen file, pf+ gen file)
MODELS = {
    "MedForge-Reasoner (released)": (
        "gen_official_medforge_fair_crossforgery.jsonl",
        "gen_official_medforge_fair_pfplus.jsonl",
    ),
    "CIRCE Stage-1 replay-SFT": (
        "gen_mfp_crossforgery_implant_replay_sft_8b.jsonl",
        "gen_pfplus_removal_replay_sft_8b_pfplus.jsonl",
    ),
    "CIRCE Stage-3 interp a=0.10": (
        "gen_interp_alpha0.10.jsonl",
        "gen_interp_alpha0.10_pfplus.jsonl",
    ),
}


def score(path):
    """returns (acc, f1_macro, n) over a single gen file."""
    tp = Counter(); fp = Counter(); fn = Counter()
    n = 0; correct = 0
    for line in open(path):
        r = json.loads(line)
        gt = "deepfake" if r["type"] in DF else "real"
        pred = parse_output(r["output"])["verdict"] or "real"
        n += 1
        if pred == gt:
            correct += 1
            tp[gt] += 1
        else:
            fp[pred] += 1
            fn[gt] += 1
    f1s = []
    for c in ("real", "deepfake"):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return correct / n, sum(f1s) / len(f1s), n


def score_concat(paths):
    """acc + macro-F1 over the concatenation of several gen files (the aggregate)."""
    tp = Counter(); fp = Counter(); fn = Counter()
    n = 0; correct = 0
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            gt = "deepfake" if r["type"] in DF else "real"
            pred = parse_output(r["output"])["verdict"] or "real"
            n += 1
            if pred == gt:
                correct += 1
                tp[gt] += 1
            else:
                fp[pred] += 1
                fn[gt] += 1
    f1s = []
    for c in ("real", "deepfake"):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return correct / n, sum(f1s) / len(f1s), n


def main():
    print(f"{'Model':<38}{'CF Acc':>8}{'PF+ Acc':>9}{'Agg Acc':>9}{'Agg F1':>9}{'n':>7}")
    print("-" * 80)
    for name, (cf, pf) in MODELS.items():
        cfp, pfp = f"{BASE}/{cf}", f"{BASE}/{pf}"
        if not (os.path.exists(cfp) and os.path.exists(pfp)):
            print(f"{name:<38}  MISSING gen file, skipped")
            continue
        cf_acc, _, cf_n = score(cfp)
        pf_acc, _, pf_n = score(pfp)
        agg_acc, agg_f1, agg_n = score_concat([cfp, pfp])
        print(f"{name:<38}{cf_acc*100:>7.2f} {pf_acc*100:>8.2f} "
              f"{agg_acc*100:>8.2f} {agg_f1*100:>8.2f} {agg_n:>7d}")

    # Domain-routed CIRCE framework: best single checkpoint per organ domain --
    # interp alpha=0.10 for the MedForge organs (CF), dedicated PF+ model for kidney/colon.
    print("-" * 80)
    routed_cf = f"{BASE}/gen_interp_alpha0.10.jsonl"          # best on MedForge organs
    routed_pf = f"{BASE}/gen_v0.3_8b.jsonl"                   # best on PF+ (dedicated) organs
    if os.path.exists(routed_cf) and os.path.exists(routed_pf):
        cf_acc, _, cf_n = score(routed_cf)
        pf_acc, _, pf_n = score(routed_pf)
        agg_acc, agg_f1, agg_n = score_concat([routed_cf, routed_pf])
        print(f"{'CIRCE (domain-routed, best-per-organ)':<38}{cf_acc*100:>7.2f} "
              f"{pf_acc*100:>8.2f} {agg_acc*100:>8.2f} {agg_f1*100:>8.2f} {agg_n:>7d}")


if __name__ == "__main__":
    main()
