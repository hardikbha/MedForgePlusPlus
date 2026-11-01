"""
Training-pipeline smoke test (CPU, real MedForge++ data, no GPU/RL stack).

Exercises the full chain the trainer will run, minus the GPU model:
  real sample -> build 2-turn SELF-QUESTIONING SFT trace -> simulate an 8-rollout
  GRPO group -> compute per-rollout verifiable reward -> GRPO group advantages
  -> assert a lazy-large rollout ends up with NEGATIVE advantage vs a tight one.

This proves the data->format->reward->advantage wiring is self-consistent before
we spend a GPU on real rollouts.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, glob, sys, statistics
sys.path.insert(0, os.path.dirname(__file__))
from circe_reward import iou, r_loc, r_delta, medforge_cov, classify_failure

ROOT = _release_path('AAAI/MedForge++/naitik_annotated')
DEEPFAKE_TYPES = {"remove", "inject", "deepfake", "removal", "injection"}


def load_one(subset):
    mp = glob.glob(os.path.join(ROOT, subset, "mapping_*.json"))[0]
    return json.load(open(mp))[0]


# ---------- S0: SFT trace builder (data -> two-turn self-questioning format) ----------
SELF_QS = [
    "Is there a boundary discontinuity around any structure?",
    "Does local texture match the surrounding-tissue noise statistics?",
    "Is the density/intensity physiologically plausible for this organ?",
]

def build_sft_trace(mp):
    gt_v = "deepfake" if mp["type"] in DEEPFAKE_TYPES else "real"
    organ, disease = mp["dataset"], mp["disease"]
    box = mp.get("box")
    ref = "deepfake" if gt_v == "deepfake" else disease
    qa = "\n    ".join(f"Q{i+1}: {q}  A{i+1}: <reasoned>" for i, q in enumerate(SELF_QS))
    box_tok = (f'<|object_ref_start|>"{ref}"<|object_ref_end|>'
               f'<|box_start|>x1="{box[0]}" y1="{box[1]}" x2="{box[2]}" y2="{box[3]}"<|box_end|>'
               if box else "(no manipulated region)")
    turn1 = (f"<think>\n  <description>{organ} image</description>\n"
             f"  <self_ask>\n    {qa}\n  </self_ask>\n"
             f"  <evidence cue=\"boundary_discontinuity\">...</evidence>\n"
             f"  <conclusion>...</conclusion>\n  {box_tok}\n</think>\n\n"
             f"This image is a {'deepfake.' if gt_v=='deepfake' else 'real image.'}")
    turn2 = "<sc><repair>box loose</repair><fix>tightened box on hi-res crop</fix></sc>"
    return {"gt_verdict": gt_v, "gt_box": box, "turn1": turn1, "turn2": turn2}


# ---------- S1: simulate an 8-rollout GRPO group + reward + advantage ----------
def big_box(gt):
    return [max(0, gt[0]-300), max(0, gt[1]-300),
            min(1024, gt[2]+300), min(1024, gt[3]+300)]

def jit(gt, d):
    return [gt[0]+d, gt[1]+d, gt[2]+d, gt[3]+d]

def rollout_reward(pred_box, pred_verdict, gt_box, gt_verdict,
                   w_acc=1.0, w_loc=1.0):
    r_v = w_acc * (1.0 if pred_verdict == gt_verdict else 0.0)
    r_l = w_loc * r_loc(pred_box, gt_box)
    return r_v + r_l

def group_advantages(rewards):
    mu = statistics.mean(rewards)
    sd = statistics.pstdev(rewards) or 1.0
    return [(r - mu) / sd for r in rewards]


def main():
    ok = True
    print("=" * 78)
    print("STAGE A — S0 trace builder: real sample -> two-turn self-questioning SFT")
    print("=" * 78)
    for s in ["kidney_removal", "polyp_real", "kidney_normal"]:
        mp = load_one(s)
        t = build_sft_trace(mp)
        has_selfask = "<self_ask>" in t["turn1"]
        has_verdict = t["gt_verdict"] in ("real", "deepfake")
        box_ok = ("box_start" in t["turn1"]) == bool(mp.get("box"))
        print(f"  {s:15s} verdict={t['gt_verdict']:8s} self_ask={has_selfask} "
              f"box_in_trace={('box_start' in t['turn1'])} (gt_box={bool(mp.get('box'))}) "
              f"{'OK' if (has_selfask and has_verdict and box_ok) else 'FAIL'}")
        ok = ok and has_selfask and has_verdict and box_ok
    print()

    print("=" * 78)
    print("STAGE B — S1 GRPO group: 8 rollouts -> verifiable reward -> group advantage")
    print("=" * 78)
    mp = load_one("kidney_removal")
    gt_box, gt_v = mp["box"], "deepfake"
    # 8 synthetic rollouts spanning the behaviours GRPO must separate
    rollouts = [
        ("perfect+correct",   gt_box,        "deepfake"),
        ("tight+correct",     jit(gt_box,1), "deepfake"),
        ("tight+correct2",    jit(gt_box,2), "deepfake"),
        ("lazy-large+correct",big_box(gt_box),"deepfake"),  # MedForge would reward this
        ("lazy-large+wrong",  big_box(gt_box),"real"),
        ("wrongbox+correct",  [10,10,50,50], "deepfake"),
        ("wrongbox+wrong",    [10,10,50,50], "real"),
        ("nobox+wrong",       None,          "real"),
    ]
    rewards = [rollout_reward(b, v, gt_box, gt_v) for _, b, v in rollouts]
    advs = group_advantages(rewards)
    print(f"  {'rollout':22s} {'MedForge_cov':>12s} {'reward':>8s} {'advantage':>10s} {'ledger':>11s}")
    idx = {}
    for (name, b, v), r, a in zip(rollouts, rewards, advs):
        cov = medforge_cov(b, gt_box) if b else 0.0
        lab = classify_failure({"box": b}, gt_box, True, v == gt_v)
        idx[name] = a
        print(f"  {name:22s} {cov:12.3f} {r:8.3f} {a:+10.3f} {lab:>11s}")

    # assertions: the trainer must push tight boxes UP and lazy-large / wrong DOWN
    claim1 = idx["tight+correct"] > idx["lazy-large+correct"]     # tight beats lazy-large
    claim2 = idx["lazy-large+correct"] > idx["lazy-large+wrong"]  # correct verdict still helps
    claim3 = idx["perfect+correct"] == max(advs)                  # best rollout tops the group
    claim4 = idx["wrongbox+wrong"] < 0                            # worst rollout is negative
    print()
    print(f"  tight > lazy-large advantage ........ {'PASS' if claim1 else 'FAIL'}")
    print(f"  correct-verdict helps even if lazy .. {'PASS' if claim2 else 'FAIL'}")
    print(f"  perfect rollout tops the group ...... {'PASS' if claim3 else 'FAIL'}")
    print(f"  worst rollout has negative advantage  {'PASS' if claim4 else 'FAIL'}")
    ok = ok and claim1 and claim2 and claim3 and claim4
    print()

    print("=" * 78)
    print("STAGE C — zoom-and-fix credit (turn-1 lazy -> turn-2 tight raises reward)")
    print("=" * 78)
    i1 = iou(big_box(gt_box), gt_box); i2 = iou(jit(gt_box, 1), gt_box)
    rd = r_delta(i1, i2)
    print(f"  turn1 IoU={i1:.3f} -> turn2 IoU={i2:.3f}  R_delta={rd:.3f}  "
          f"{'PASS' if rd > 0 else 'FAIL'}")
    ok = ok and rd > 0
    print()

    print("=" * 78)
    print(f"TRAINING SMOKE TEST: {'ALL PASS — data->format->reward->advantage wiring is consistent' if ok else 'FAILURES ABOVE'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
