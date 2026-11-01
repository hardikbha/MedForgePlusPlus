"""
End-to-end smoke test for the CIRCE reward + failure-ledger core, run on REAL
MedForge++ samples (Naitik annotation set). No GPU / RL stack needed.

It proves the two central claims the design rests on:
  (A) CIRCE r_loc strictly PENALIZES a lazy-large box that MedForge coverage REWARDS
      (coverage=1.0 for a box covering the whole image; r_loc collapses via area penalty).
  (B) The typed failure ledger labels each (verdict, box) outcome correctly:
      SOLVED / LOCALIZE / VERDICT / BOTH / HALLUCINATE.
Also exercises the real Qwen-format parser and the zoom Δ-IoU process reward.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, glob, sys
sys.path.insert(0, os.path.dirname(__file__))
from circe_reward import (parse_output, iou, diou, r_loc, r_delta,
                          medforge_cov, classify_failure)

ROOT = _release_path('AAAI/MedForge++/naitik_annotated')
W, H = 1024, 1024


def load_samples(subset, n=3):
    mp = glob.glob(os.path.join(ROOT, subset, "mapping_*.json"))[0]
    sf = glob.glob(os.path.join(ROOT, subset, "sft_*.jsonl"))[0]
    mapping = {m["key"]: m for m in json.load(open(mp))}
    out = []
    for line in open(sf):
        rec = json.loads(line)
        key = os.path.basename(rec["images"][0])
        if key in mapping:
            out.append((mapping[key], rec["messages"][-1]["content"]))
        if len(out) >= n:
            break
    return out


def big_box(gt):
    """A lazy-large box that fully contains gt (MedForge's exploit)."""
    return [0, 0, W, H] if gt is None else [max(0, gt[0]-300), max(0, gt[1]-300),
                                            min(W, gt[2]+300), min(H, gt[3]+300)]


def jitter(gt, d):
    return [gt[0]+d, gt[1]+d, gt[2]+d, gt[3]+d]


def tight(gt):
    """A genuinely tight box: shift by ~8% of the smaller side (>=1px)."""
    d = max(1, round(0.08 * min(gt[2]-gt[0], gt[3]-gt[1])))
    return jitter(gt, d)


DEEPFAKE_TYPES = {"remove", "inject", "deepfake", "removal", "injection"}


def main():
    ok = True
    print("=" * 78)
    print("STAGE 1 — parse the REAL assistant traces (Qwen3-VL MedForge format)")
    print("=" * 78)
    subsets = ["kidney_real_tumor", "kidney_removal", "polyp_real",
               "polyp_removal", "kidney_normal", "polyp_normal"]
    parsed_samples = {}
    for s in subsets:
        try:
            samples = load_samples(s, 2)
        except (IndexError, FileNotFoundError):
            print(f"  [skip] {s}: not found"); continue
        parsed_samples[s] = []
        for mp, txt in samples:
            p = parse_output(txt)
            gt_has_box = bool(mp.get("box"))
            gt_v = "deepfake" if mp["type"] in DEEPFAKE_TYPES else "real"
            match_v = (p["verdict"] == gt_v)
            match_b = (not gt_has_box) or (p["box"] is not None and iou(p["box"], mp["box"]) > 0.99)
            print(f"  {s:18s} {mp['key'][:26]:26s} "
                  f"verdict={str(p['verdict']):8s}(gt {gt_v}, {'OK' if match_v else 'MISS'})  "
                  f"box={str(p['box']!=None):5s}(gt {gt_has_box}, {'OK' if match_b else 'MISS'})")
            parsed_samples[s].append((mp, p, gt_v, gt_has_box))
            if not match_v or not match_b:
                ok = False
    print()

    print("=" * 78)
    print("STAGE 2 — CLAIM (A): CIRCE penalizes the lazy-large box MedForge rewards")
    print("=" * 78)
    print(f"  {'sample':30s} {'box type':11s} {'IoU':>6s} {'MedForge_cov':>12s} {'CIRCE_rloc':>11s}")
    for s in ["kidney_real_tumor", "polyp_removal"]:
        for mp, p, gt_v, gt_has_box in parsed_samples.get(s, [])[:1]:
            gt = mp["box"]
            cands = {"perfect (=gt)": gt,
                     "tight jitter": tight(gt),
                     "lazy-large": big_box(gt)}
            for name, pb in cands.items():
                cov = medforge_cov(pb, gt)
                rl = r_loc(pb, gt)
                print(f"  {s[:18]+'/'+mp['key'][:10]:30s} {name:11s} "
                      f"{iou(pb,gt):6.3f} {cov:12.3f} {rl:11.3f}")
            # the assertion: lazy-large gets ~full coverage but LOW/负 r_loc vs perfect
            cov_lazy = medforge_cov(big_box(gt), gt)
            rl_lazy = r_loc(big_box(gt), gt)
            rl_perfect = r_loc(gt, gt)
            claimA = (cov_lazy > 0.99) and (rl_lazy < rl_perfect) and (rl_lazy < 0.5)
            print(f"    -> coverage rewards lazy-large ({cov_lazy:.2f}) but CIRCE does NOT "
                  f"({rl_lazy:.2f} < perfect {rl_perfect:.2f}): {'PASS' if claimA else 'FAIL'}")
            ok = ok and claimA
    print()

    print("=" * 78)
    print("STAGE 3 — zoom-and-fix Δ-IoU process reward (only tightening pays)")
    print("=" * 78)
    gt = parsed_samples["kidney_real_tumor"][0][0]["box"]
    turn1 = big_box(gt)            # coarse turn-1 box
    turn2 = jitter(gt, 4)          # tightened turn-2 box on hi-res crop
    i1, i2 = iou(turn1, gt), iou(turn2, gt)
    rd_good = r_delta(i1, i2)
    rd_bad = r_delta(i2, i1)       # loosening again
    print(f"  turn1 IoU={i1:.3f} -> turn2 IoU={i2:.3f}   R_delta(tighten)={rd_good:.3f}")
    print(f"  loosening back                              R_delta(loosen) ={rd_bad:.3f} (should be 0)")
    claimD = rd_good > 0 and rd_bad == 0
    print(f"    -> tightening earns reward, loosening earns 0: {'PASS' if claimD else 'FAIL'}")
    ok = ok and claimD
    print()

    print("=" * 78)
    print("STAGE 4 — CLAIM (B): typed failure ledger labels outcomes correctly")
    print("=" * 78)
    # build synthetic (pred, gt) scenarios anchored on a real boxed sample + a normal one
    gt = parsed_samples["kidney_real_tumor"][0][0]["box"]; gtv = "real"
    normal_mp = parsed_samples["kidney_normal"][0][0]
    scenarios = [
        ("SOLVED",      {"box": tight(gt)},    gt, True,  True),
        ("LOCALIZE",    {"box": big_box(gt)},  gt, True,  True),   # right verdict, sloppy box
        ("VERDICT",     {"box": tight(gt)},    gt, True,  False),  # tight box, wrong verdict
        ("BOTH",        {"box": [10,10,40,40]},gt, True,  False),  # far box + wrong verdict
        ("HALLUCINATE", {"box": [500,500,540,540]}, None, False, True),  # box on a normal img
    ]
    for expect, pred, gt_b, gt_has_box, vok in scenarios:
        got = classify_failure(pred, gt_b, gt_has_box, vok)
        status = "PASS" if got == expect else f"FAIL(got {got})"
        print(f"  expect {expect:12s} -> {got:12s} {status}")
        ok = ok and (got == expect)
    print()

    print("=" * 78)
    print(f"SMOKE TEST: {'ALL PASS — CIRCE reward+ledger core works end to end' if ok else 'FAILURES ABOVE'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
