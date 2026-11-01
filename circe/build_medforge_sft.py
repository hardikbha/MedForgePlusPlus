"""
Build CIRCE SFT + eval splits from the FULL MedForge-90K (their images/labels/boxes, OUR format).
Fair head-to-head: our recipe trained on their data, evaluated on their generator-holdout protocol.

Splits:
  - Cross-Model test  = HELD-OUT generators (never seen in training)
  - In-Domain test    = held-out samples from TRAINING generators
  - train             = the rest
Box for deepfakes is extracted from their bbox_mask PNG; real images have no box.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, glob, os, random, argparse
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, os.path.dirname(__file__))
from build_sft_data import build_record

ROOT = _release_path('AAAI/MedForge++/medforge90k_full')
HOLDOUT_GENERATORS = {"seeddream", "step1x-edit"}   # Cross-Model tier (unseen editors)
CANVAS = 1024


def box_from_mask(mask_rel):
    p = os.path.join(ROOT, mask_rel)
    if not mask_rel or not os.path.exists(p):
        return None
    m = np.array(Image.open(p).convert("L"))
    ys, xs = np.where(m > 127)
    if len(xs) == 0:
        return None
    H, W = m.shape
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [int(x1 * CANVAS / W), int(y1 * CANVAS / H),
            int(x2 * CANVAS / W), int(y2 * CANVAS / H)]


def to_record(it):
    img = os.path.join(ROOT, it["image_path"])
    if not os.path.exists(img):
        return None
    t = it.get("type", "real")           # real | edit | remove
    ftype = "inject" if t == "edit" else ("remove" if t == "remove" else "real")
    box = None if t == "real" else box_from_mask(it.get("mask_path", ""))
    mp = {"type": ftype, "box": box, "dataset": it.get("dataset", "?"),
          "disease": it.get("disease", "lesion")}
    return build_record(img, "", mp, "template")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap total (0=all)")
    ap.add_argument("--indomain_frac", type=float, default=0.03)
    args = ap.parse_args()
    mp = glob.glob(os.path.expanduser("~/.cache/huggingface/**/dataset_mapping.json"), recursive=True)[0]
    items = json.load(open(mp))["items"]
    random.seed(0); random.shuffle(items)
    if args.limit:
        items = items[:args.limit]

    train, id_test, cm_test = [], [], []
    n_missing = 0
    for it in items:
        gen = it.get("model", "")
        rec = to_record(it)
        if rec is None:
            n_missing += 1; continue
        rec["_meta"] = {"type": it.get("type", "real"), "gen": gen}
        if gen in HOLDOUT_GENERATORS:
            cm_test.append(rec)
        elif random.random() < args.indomain_frac:
            id_test.append(rec)
        else:
            train.append(rec)
    # balance train to ~2:1 deepfake:real (MedForge's ratio) given limited downloaded reals
    tr_real = [r for r in train if r["_meta"]["type"] == "real"]
    tr_fake = [r for r in train if r["_meta"]["type"] != "real"]
    random.shuffle(tr_fake)
    tr_fake = tr_fake[:2 * len(tr_real)] if tr_real else tr_fake
    train = tr_real + tr_fake
    random.shuffle(train)
    print(f"balanced train: {len(tr_real)} real + {len(tr_fake)} deepfake")

    D = _release_path('AAAI/MedForge++')
    # train = SFT-format; test files = eval-format {image,type,gen,box}
    def dump_eval(recs, name):
        with open(f"{D}/{name}", "w") as f:
            for r in recs:
                meta = r["_meta"]; asst = r["messages"][2]["content"]
                import re
                m = re.search(r'x1="(\d+)" y1="(\d+)" x2="(\d+)" y2="(\d+)"', asst)
                box = [int(x) for x in m.groups()] if m else None
                f.write(json.dumps({"image": r["images"][0], "type": meta["type"],
                                    "gen": meta["gen"], "box": box}) + "\n")
    for r in train: r.pop("_meta", None)
    with open(f"{D}/mf90k_train.jsonl", "w") as f:
        for r in train: f.write(json.dumps(r) + "\n")
    dump_eval(id_test, "mf90k_test_indomain.jsonl")
    dump_eval(cm_test, "mf90k_test_crossmodel.jsonl")
    print(f"train={len(train)}  in-domain_test={len(id_test)}  cross-model_test={len(cm_test)}  missing={n_missing}")
    print(f"held-out generators (cross-model): {HOLDOUT_GENERATORS}")


if __name__ == "__main__":
    main()
