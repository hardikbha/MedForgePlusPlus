"""Are the evidence/format bonuses of reward_classification_dominant constant on real model outputs?
Uses the exact reward code; READ-ONLY."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, sys, collections
sys.path.insert(0, _release_path('CIRCE/circe'))
from circe_reward import parse_output, iou
from circe_grpo_reward import _cue_of, FORENSIC_CUES, reward_classification_dominant
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f)]
    ev = collections.Counter(); fm = collections.Counter(); contra = collections.Counter(); R = collections.Counter()
    for r in rows:
        t = r["output"].replace("<|im_end|>", "").replace("<|endoftext|>", "")
        p = parse_output(t); tl = t.lower(); cue = _cue_of(t)
        grounded = (cue in FORENSIC_CUES) or any(k in tl for k in ("boundary", "smooth", "inpaint", "blend", "texture", "homogeneous", "seamless", "consistent", "normal", "authentic", "plausible"))
        ev[grounded] += 1
        fm["</think>" in t or "<verdict>" in t] += 1
        v = p["verdict"]
        contra[(v == "deepfake" and p["box"] is None) or (v == "real" and p["box"] is not None)] += 1
        rr = reward_classification_dominant(completions=[t], gt_verdict=[r["gt_verdict"]], gt_box=[r["gt_box"]])[0]
        R[round(rr, 1)] += 1
    print(f"{f.split('/')[-1]}: n={len(rows)} grounded={dict(ev)} format_ok={dict(fm)} contradiction={dict(contra)} cue_of_first={_cue_of(rows[0]['output'])!r}")
    print("   reward histogram (rounded 0.1):", dict(sorted(R.items())))
