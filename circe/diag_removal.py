"""
Decisive removal diagnostic: is lesion-removal detectable on our data at all?

Runs the model with a REMOVAL-SPECIFIC 5-cue forensic prompt on removal images vs
true-normal images. If the model's removal-evidence does NOT separate removals from
normals, the signal isn't in the pixels (data problem, not model/prompt).
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, sys, os, re
sys.path.insert(0, os.path.dirname(__file__))

DIAG_SYS = (
    "You are a forensic radiologist detecting CONTENT-AWARE REMOVAL — a lesion digitally "
    "ERASED by inpainting. Do NOT look for an added object; look for signs something was taken "
    "OUT: (1) a region unnaturally uniform/flat vs matched tissue elsewhere; (2) a faint residual "
    "halo or inpaint-mask boundary; (3) loss of expected fine vascular/parenchymal texture; "
    "(4) anatomical implausibility — a place that should show variation but is blank; (5) asymmetry "
    "vs the contralateral side. For each cue state present/absent. Then output a line exactly: "
    "'REMOVAL_SCORE: <0-100>' and a line 'VERDICT: <real|deepfake>'.")
DIAG_USER = ("Analyze this scan for lesion REMOVAL. Check the 5 cues, then give REMOVAL_SCORE and VERDICT.")


def run(args):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    from PIL import Image

    rows = [json.loads(l) for l in open(args.eval)]
    proc = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        attn_implementation="sdpa")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def score(text):
        m = re.search(r"REMOVAL_SCORE:\s*(\d+)", text)
        return int(m.group(1)) if m else None
    def verdict(text):
        m = re.search(r"VERDICT:\s*(real|deepfake)", text, re.I)
        return m.group(1).lower() if m else None

    buckets = {"remove": [], "real": []}
    B = args.batch
    for bi in range(0, len(rows), B):
        batch = rows[bi:bi + B]
        texts, imgs = [], []
        for m in batch:
            msgs = [{"role": "system", "content": DIAG_SYS},
                    {"role": "user", "content": [{"type": "image", "image": m["image"]},
                                                 {"type": "text", "text": DIAG_USER}]}]
            texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            imgs.append([Image.open(m["image"]).convert("RGB")])
        inp = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=384, do_sample=False)
        outs = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        for m, o in zip(batch, outs):
            grp = "remove" if m["type"] == "remove" else ("real" if m["type"] == "real" else None)
            if grp:
                buckets[grp].append({"score": score(o), "verdict": verdict(o)})
        print(f"  [{min(bi+B,len(rows))}/{len(rows)}]", flush=True)

    import statistics
    for grp, items in buckets.items():
        sc = [i["score"] for i in items if i["score"] is not None]
        flagged = sum(1 for i in items if i["verdict"] == "deepfake")
        med = statistics.median(sc) if sc else None
        print(f"{grp:8s} n={len(items)} median_removal_score={med} "
              f"flagged_deepfake={flagged}/{len(items)}")
    print("\nIf remove median_score ~= real median_score -> NO removal signal (data problem).")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default=_release_path('models/Qwen3-VL-4B-Instruct'))
    ap.add_argument("--batch", type=int, default=8)
    run(ap.parse_args())
