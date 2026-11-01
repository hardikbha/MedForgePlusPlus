"""
Build a balanced eval sample from the ORIGINAL MedForge-90K and download just those images.
Emits {image, type(real/edit/remove), category, modality, model} for zero-shot eval of CIRCE.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, glob, os, random
from huggingface_hub import hf_hub_download

REPO = "RichardChenZH/MedForge-90K"
OUT_DIR = _release_path('AAAI/MedForge++/medforge90k_eval')
os.makedirs(OUT_DIR + "/imgs", exist_ok=True)


def modality(key):
    for m in ["fundus", "chest", "brain"]:
        if m in key:
            return m
    return "?"


def main(per=70):
    mp = glob.glob(os.path.expanduser("~/.cache/huggingface/**/dataset_mapping.json"), recursive=True)[0]
    items = json.load(open(mp))["items"]
    real = [it for it in items if it["image_path"].startswith("real/")]
    edit = [it for it in items if it.get("type") == "edit"]
    remove = [it for it in items if it.get("type") == "remove"]
    random.seed(0)
    for lst in (real, edit, remove):
        random.shuffle(lst)
    sample = real[:per] + edit[:per] + remove[:per]
    print(f"sampling {len(sample)} ({per} each of real/edit/remove)")

    rows = []
    for i, it in enumerate(sample):
        rp = it["image_path"]
        try:
            local = hf_hub_download(REPO, rp, repo_type="dataset")
        except Exception as e:
            print("  skip", rp, repr(e)[:80]); continue
        rows.append({"image": local, "type": it.get("type", "real"),
                     "category": {"edit": "Implant", "remove": "Remove"}.get(it.get("type"), "Real"),
                     "modality": modality(rp), "model": it.get("model", "?"),
                     "disease": it.get("disease", "")})
        if (i + 1) % 30 == 0:
            print(f"  downloaded {i+1}/{len(sample)}", flush=True)
    outp = OUT_DIR + "/eval.jsonl"
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print("DONE", len(rows), "->", outp)
    print("by type:", dict(Counter(r["type"] for r in rows)))
    print("by modality:", dict(Counter(r["modality"] for r in rows)))


if __name__ == "__main__":
    main()
