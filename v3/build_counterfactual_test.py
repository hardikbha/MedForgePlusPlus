"""
Build shortcut-controlled counterfactual versions of an MF90K eval split (default: the shared test set).

  norm      : every image decoded, gray modalities (brain-mri, chest-xray) forced to single-channel L,
              all images re-encoded identically (JPEG q90, 4:4:4). Kills the colour-cast and JPEG-quality cues.
  pasteback : reals as in `norm`; each fake replaced by a composite = real source everywhere except inside the
              GT bbox mask, where the generator output is kept; then encoded exactly like the reals.
              Removes the whole-frame re-rendering fingerprint (only the edited region is generator pixels).
Fakes whose real source is missing on disk are dropped from `pasteback` (counted in the manifest).
Writes images under <out>/<variant>/... and <out>/<variant>.jsonl in eval format {image,type,gen,box,orig}.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import argparse, json, os, sys, collections
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

MF = _release_path('AAAI/MedForge++/medforge90k_full/')
GRAY = {"brain-mri", "chest-xray"}


def organ_of(rel):
    for o in ("brain-mri", "chest-xray", "fundus"):
        if f"/{o}" in "/" + rel or rel.startswith(f"real/{o}"):
            return o
    for o, k in (("brain-mri", "brain"), ("chest-xray", "chest"), ("fundus", "fundus")):
        if k in rel:
            return o
    return "?"


def save(arr, organ, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im = Image.fromarray(arr.astype(np.uint8))
    im = im.convert("L") if organ in GRAY else im.convert("RGB")
    im.save(path, "JPEG", quality=90, subsampling=0)


def work(job):
    row, item, out = job
    rel = row["image"].split("medforge90k_full/", 1)[1]
    organ = organ_of(rel)
    stem = os.path.splitext(rel)[0].replace("/", "__") + ".jpg"
    res = {}
    f = np.asarray(Image.open(row["image"]).convert("RGB"))
    p_norm = f"{out}/norm/{stem}"
    save(f, organ, p_norm)
    res["norm"] = {**row, "image": p_norm, "orig": row["image"]}
    if row["type"] == "real":
        res["pasteback"] = {**row, "image": p_norm, "orig": row["image"]}
        return res
    try:
        src = MF + item["real_image_path"]
        if not os.path.exists(src):
            res["pasteback"] = None
            res["why"] = "missing_source"
            return res
        r = np.asarray(Image.open(src).convert("RGB").resize((f.shape[1], f.shape[0]), Image.BICUBIC))
        m = np.asarray(Image.open(MF + item["mask_path"]).convert("L").resize((f.shape[1], f.shape[0]), Image.NEAREST)) > 127
        comp = np.where(m[..., None], f, r)
        p_pb = f"{out}/pasteback/{stem}"
        save(comp, organ, p_pb)
        res["pasteback"] = {**row, "image": p_pb, "orig": row["image"], "mask_frac": float(m.mean())}
    except Exception as e:  # noqa
        res["pasteback"] = None
        res["why"] = repr(e)[:120]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=_release_path('AAAI/MedForge++/mfp_v2_test_shared.jsonl'))
    ap.add_argument("--out", default=_release_path('AAAI/MedForge++/v3_counterfactual/test'))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    items = {it["image_path"]: it for it in json.load(open(MF + "dataset_mapping.json"))["items"]}
    rows = [json.loads(l) for l in open(args.eval) if l.strip()]
    jobs = [(r, items.get(r["image"].split("medforge90k_full/", 1)[1]), args.out) for r in rows]
    assert all(j[1] is not None for j in jobs), "eval row without mapping entry"
    outs = {"norm": [], "pasteback": []}
    why = collections.Counter()
    with ProcessPoolExecutor(args.workers) as ex:
        for res in ex.map(work, jobs, chunksize=8):
            outs["norm"].append(res["norm"])
            if res.get("pasteback"):
                outs["pasteback"].append(res["pasteback"])
            else:
                why[res.get("why", "?")] += 1
    for k, v in outs.items():
        tmp = f"{args.out}/{k}.jsonl.tmp"
        with open(tmp, "w") as fh:
            for r in v:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmp, f"{args.out}/{k}.jsonl")
        print(k, len(v), collections.Counter(r["type"] for r in v))
    print("dropped from pasteback:", dict(why))


if __name__ == "__main__":
    main()
