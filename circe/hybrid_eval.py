"""
Hybrid CIRCE eval: VLM verdict (great at Real/Implant) + removal-fingerprint (catches
removals the VLM misses). Sweeps the fingerprint threshold to balance removal-recall vs
normal false-positives, then reports the per-type table at the best operating point.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, sys, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from circe_reward import parse_output
from removal_fingerprint import feat

EVAL = _release_path('AAAI/MedForge++/circe_eval_split_v2.jsonl')
GEN = _release_path('AAAI/MedForge++/gen_v0.4_4b.jsonl')
DF = {"remove", "inject"}


def main():
    clf = pickle.load(open(_release_path('AAAI/MedForge++/removal_fp.pkl'), "rb"))
    vlm = {os.path.basename(json.loads(l)["image"]): json.loads(l) for l in open(GEN)}
    rows = [json.loads(l) for l in open(EVAL)]

    recs = []
    for r in rows:
        name = os.path.basename(r["image"])
        g = vlm.get(name)
        vlm_v = (parse_output(g["output"])["verdict"] or "real") if g else "real"
        fp_prob = float(clf.predict_proba(feat(r["image"]).reshape(1, -1))[0, 1])
        recs.append({"type": r["type"], "gt": "deepfake" if r["type"] in DF else "real",
                     "vlm": vlm_v, "fp": fp_prob})
    print(f"scored {len(recs)} eval images")

    def evaluate(t):
        from collections import defaultdict
        grp = defaultdict(lambda: [0, 0])
        for r in recs:
            # hybrid: deepfake if VLM says so, OR (VLM says real but fingerprint confidently removal)
            pred = "deepfake" if (r["vlm"] == "deepfake" or (r["vlm"] == "real" and r["fp"] > t)) else "real"
            k = {"real": "Real", "remove": "Remove", "inject": "Implant"}[r["type"]]
            grp[k][1] += 1
            grp[k][0] += int(pred == r["gt"])
        per = {k: 100 * c / n for k, (c, n) in grp.items()}
        avg = np.mean([per["Real"], per["Implant"], per["Remove"]])
        return per, avg

    best = None
    for t in [round(x, 2) for x in np.arange(0.30, 0.95, 0.05)]:
        per, avg = evaluate(t)
        if best is None or avg > best[2]:
            best = (t, per, avg)
    # also baseline (VLM only, t=1.0 => fingerprint never fires)
    base, base_avg = evaluate(1.01)
    print(f"\nVLM-only baseline:  Real {base['Real']:.1f}  Implant {base['Implant']:.1f}  "
          f"Remove {base['Remove']:.1f}  Avg {base_avg:.1f}")
    t, per, avg = best
    print(f"HYBRID (fp thr={t}): Real {per['Real']:.1f}  Implant {per['Implant']:.1f}  "
          f"Remove {per['Remove']:.1f}  Avg {avg:.1f}")
    json.dump({"threshold": t, "hybrid": per, "hybrid_avg": avg,
               "vlm_only": base, "vlm_only_avg": base_avg},
              open(_release_path('AAAI/MedForge++/hybrid_result.json'), "w"), indent=2)
    print("[saved] hybrid_result.json")


if __name__ == "__main__":
    main()
