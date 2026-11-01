"""Counterfactual-pair leakage audit: do sources (real_image_path) of training fakes appear in test/selection sets,
and do sources of test fakes appear as training reals? READ-ONLY."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, collections
R = _release_path('AAAI/MedForge++/')
P = R + "medforge90k_full/"
mp = json.load(open(P + "dataset_mapping.json"))["items"]
src_of = {}
for x in mp:
    if x["type"] in ("edit", "remove") and x.get("real_image_path"):
        src_of.setdefault(P + x["image_path"], P + x["real_image_path"])
def imgs(fn):
    out = []
    for l in open(R + fn):
        r = json.loads(l)
        out.append(r["images"][0] if "images" in r else r["image"])
    return out
sets = {
  "test_shared": imgs("mfp_v2_test_shared.jsonl"),
  "select_clean": imgs("v3_select/mfp_v3_select_clean.jsonl"),
  "rl_cf_strict": imgs("mfp_v2_rl_crossforgery_strict.jsonl"),
  "sft_cf_base": imgs("mfp_train_crossforgery.jsonl"),
  "replay_cf_strict": imgs("mfp_v2_replay_crossforgery_strict.jsonl"),
}
S = {k: set(v) for k, v in sets.items()}
for k, v in S.items():
    print(f"{k}: {len(v)} unique images")
train_all = S["rl_cf_strict"] | S["sft_cf_base"] | S["replay_cf_strict"]
for ev in ("test_shared", "select_clean"):
    fakes = [i for i in S[ev] if i in src_of]
    srcs = [src_of[i] for i in fakes]
    print(f"[{ev}] fakes={len(fakes)}; their sources seen as TRAIN images (CF sft/replay/rl)={sum(s in train_all for s in srcs)}; "
          f"sources that are themselves {ev} reals={sum(s in S[ev] for s in srcs)}")
    rl_f = [i for i in S["rl_cf_strict"] if i in src_of]
    print(f"[rl_cf_strict -> {ev}] RL fakes whose source image is in {ev}: {sum(src_of[i] in S[ev] for i in rl_f)} / {len(rl_f)}")
    # sibling fakes: other forgeries of the same source in train
    test_src = set(srcs)
    sib = sum(1 for i in train_all if i in src_of and src_of[i] in test_src)
    print(f"[{ev}] training fakes that share a source with a {ev} fake (siblings): {sib}")
