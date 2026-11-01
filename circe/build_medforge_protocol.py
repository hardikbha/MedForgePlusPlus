"""
Rebuild MedForge's EXACT evaluation protocol (arXiv 2603.18577, Sec. 5).

Split ratio 5:1:3 -> SFT : GSPO : test.  We (SFT-only) train on the SFT portion,
evaluate on ONE shared stratified test subset drawn from the test portion.

Three TRAINING regimes (tiers differ in TRAIN data, share the SAME test set):
  - In-Domain    : train on ALL generators + ALL forgery types
  - Cross-Model  : train EXCLUDING 4 generators (Nano-Banana=gemini, GPT-Image=gpt,
                   SD-3.5-Medium, SDXL-Inpainting) -> test still contains them
  - Cross-Forgery: train EXCLUDING lesion-implant (edit) -> test still contains implant
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, random, argparse, sys, re
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from build_sft_data import build_record

ROOT = _release_path('AAAI/MedForge++/medforge90k_full')
MAP = _release_path('.cache/huggingface/hub/datasets--RichardChenZH--MedForge-90K/snapshots/7bbf54f870584b37d6594593da76b2fc78fe4c5b/dataset_mapping.json')
D = _release_path('AAAI/MedForge++')
CANVAS = 1024
# exact Cross-Model held-out generators (paper names -> our dataset_mapping model strings)
CM_HOLDOUT = {"gemini", "gpt", "stable-diffusion-3.5-medium",
              "stable-diffusion-xl-1.0-inpainting-0.1"}


def box_from_mask(mask_rel):
    p = os.path.join(ROOT, mask_rel or "")
    if not mask_rel or not os.path.exists(p):
        return None
    m = np.array(Image.open(p).convert("L"))
    ys, xs = np.where(m > 127)
    if len(xs) == 0:
        return None
    H, W = m.shape
    return [int(xs.min()*CANVAS/W), int(ys.min()*CANVAS/H),
            int(xs.max()*CANVAS/W), int(ys.max()*CANVAS/H)]


def to_record(it):
    img = os.path.join(ROOT, it["image_path"])
    if not os.path.exists(img):
        return None
    t = it.get("type", "real")                       # real | edit | remove
    ftype = "inject" if t == "edit" else ("remove" if t == "remove" else "real")
    box = None if t == "real" else box_from_mask(it.get("mask_path", ""))
    rec = build_record(img, "", {"type": ftype, "box": box,
                                 "dataset": it.get("dataset", "?"),
                                 "disease": it.get("disease", "lesion")}, "template")
    rec["_meta"] = {"type": t, "gen": it.get("model", "")}
    return rec


def dump_sft(recs, name):
    with open(f"{D}/{name}", "w") as f:
        for r in recs:
            r2 = {k: v for k, v in r.items() if k != "_meta"}
            f.write(json.dumps(r2) + "\n")


def dump_eval(recs, name):
    with open(f"{D}/{name}", "w") as f:
        for r in recs:
            asst = r["messages"][2]["content"]
            m = re.search(r'x1="(\d+)" y1="(\d+)" x2="(\d+)" y2="(\d+)"', asst)
            box = [int(x) for x in m.groups()] if m else None
            f.write(json.dumps({"image": r["images"][0], "type": r["_meta"]["type"],
                                "gen": r["_meta"]["gen"], "box": box}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_per_cat", type=int, default=1000, help="stratified test samples per category")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = json.load(open(MAP))["items"]
    random.seed(42); random.shuffle(items)
    if args.limit:
        items = items[:args.limit]

    # 5:1:3 split -> SFT / GSPO / test  (indices, deterministic)
    n = len(items); n_sft = n*5//9; n_gspo = n//9
    sft_items = items[:n_sft]
    test_items = items[n_sft + n_gspo:]
    print(f"[split] total={n} sft={len(sft_items)} test_pool={len(test_items)}")

    # ---- shared stratified test subset (held out from ALL training) ----
    by_type = {"real": [], "edit": [], "remove": []}
    for it in test_items:
        t = it.get("type", "real")
        if t in by_type:
            by_type[t].append(it)
    test_sel = []
    for t, lst in by_type.items():
        random.shuffle(lst)
        test_sel.extend(lst[:args.test_per_cat])
    test_recs = [r for r in (to_record(it) for it in test_sel) if r]
    dump_eval(test_recs, "mfp_test_shared.jsonl")
    from collections import Counter
    print(f"[test] shared={len(test_recs)} by-type={Counter(r['_meta']['type'] for r in test_recs)}")

    # ---- three TRAIN regimes from the SFT portion ----
    sft_recs = [r for r in (to_record(it) for it in sft_items) if r]
    # In-Domain: everything
    dump_sft(sft_recs, "mfp_train_indomain.jsonl")
    # Cross-Model: drop the 4 held-out generators (reals kept; they have gen='real')
    cm = [r for r in sft_recs if r["_meta"]["gen"] not in CM_HOLDOUT]
    dump_sft(cm, "mfp_train_crossmodel.jsonl")
    # Cross-Forgery: drop lesion-implant (edit) from training
    cf = [r for r in sft_recs if r["_meta"]["type"] != "edit"]
    dump_sft(cf, "mfp_train_crossforgery.jsonl")
    print(f"[train] in-domain={len(sft_recs)} cross-model={len(cm)} cross-forgery={len(cf)}")
    print(f"[train] in-domain types={Counter(r['_meta']['type'] for r in sft_recs)}")
    print(f"[train] cross-model held-out gens excluded={CM_HOLDOUT}")
    print(f"[train] cross-forgery types={Counter(r['_meta']['type'] for r in cf)}")


if __name__ == "__main__":
    main()
