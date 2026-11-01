#!/usr/bin/env python3
"""MedForge-format annotations for Naitik's kidney+polyp removal/real-tumor/normal sets,
using local Qwen3-VL-8B for per-image reasoning. Shardable: --subset --start --count.

Per image -> 1024x1024 image copy + bbox_mask (boxed cases) + explanation.txt +
SFT {messages,images} line + dataset_mapping record. Box coords in the 1024 canvas."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os, json, glob, csv, re, argparse
import numpy as np, cv2, torch
from PIL import Image

M       = _release_path('models/Qwen3-VL-8B-Instruct')
BASE    = _release_path('AAAI/naitik_data')
PROMPTS = _release_path('Prompts')
OUT     = _release_path('AAAI/MedForge++/naitik_annotated')
CANVAS  = 1024
SYS = ("You are an expert in medical image forensics. Analyze the provided image to determine if it is "
       "a deepfake or authentic. First, perform a step-by-step examination of the image content, looking "
       "for artifacts, inconsistencies, or biological implausibilities. Use <think> tags to articulate your "
       "reasoning process. If you identify manipulated regions, localize them using bounding boxes within "
       "your reasoning. Conclude your analysis with a final classification.")
USER = "Is this image deepfake or real?"

K = "kidney_ext/Kidney_dataset"; P = "polyp_ext/Polyp_dataset"
SUBSETS = {
 "kidney_removal":    dict(img=f"{K}/deepfake/removal/inpainted_images", prompt="removal_kidney_tumor.txt",
                           ref="deepfake", verdict="This image is a deepfake.", dataset="kidney", disease="tumor", typ="remove"),
 "kidney_real_tumor": dict(img=f"{K}/real/Real_tumor_kidney/images",     prompt="real_kidney_tumor.txt",
                           ref="tumor",    verdict="This image is a real image.", dataset="kidney", disease="tumor", typ="real"),
 "kidney_normal":     dict(img=f"{K}/real/Normal_kideny/images",         prompt="real_kidney.txt",
                           ref=None,       verdict="This image is a real image.", dataset="kidney", disease="normal", typ="real"),
 "polyp_removal":     dict(img=f"{P}/deepfake/removal/ployps_removal",   prompt="removal_colon_polyps.txt",
                           ref="deepfake", verdict="This image is a deepfake.", dataset="colon", disease="polyp", typ="remove"),
 "polyp_real":        dict(img=f"{P}/real/Real_polyps/images",           prompt="real_colon_polyps.txt",
                           ref="polyp",    verdict="This image is a real image.", dataset="colon", disease="polyp", typ="real"),
 "polyp_normal":      dict(img=f"{P}/real/Normal_colon/Normal_colon",    prompt="real_colon.txt",
                           ref=None,       verdict="This image is a real image.", dataset="colon", disease="normal", typ="real"),
}

def load_boxes(sub):
    """native-pixel boxes keyed by filename stem (None-box subsets return {})."""
    d = {}
    if sub == "kidney_removal":
        for r in csv.DictReader(open(f"{BASE}/{K}/deepfake/removal/tumor_removal_annotations.csv")):
            d[os.path.splitext(r["filename"])[0]] = [int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"])]
    elif sub == "kidney_real_tumor":
        for r in csv.DictReader(open(f"{BASE}/{K}/real/Real_tumor_kidney/annotations.csv")):
            d[os.path.splitext(r["filename"])[0]] = [int(r["xmin"]), int(r["ymin"]), int(r["xmax"]), int(r["ymax"])]
    elif sub == "polyp_removal":
        for f in glob.glob(f"{BASE}/{P}/deepfake/removal/csv_file_polyps_removal/*.csv"):
            rows = list(csv.DictReader(open(f)))
            if rows:
                r = rows[0]; d[os.path.splitext(os.path.basename(f))[0]] = [int(float(r["xmin"])), int(float(r["ymin"])), int(float(r["xmax"])), int(float(r["ymax"]))]
    elif sub == "polyp_real":
        for f in glob.glob(f"{BASE}/{P}/real/Real_polyps/bbox/*"):
            try:
                rows = list(csv.DictReader(open(f)))
                if rows:
                    r = rows[0]; d[os.path.splitext(os.path.basename(f))[0]] = [int(float(r["xmin"])), int(float(r["ymin"])), int(float(r["xmax"])), int(float(r["ymax"]))]
            except Exception:
                pass
    return d

def extract(tag, txt):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", txt or "", re.S)
    return m.group(1).strip() if m else None

def build_cot(ref, box, desc, evid, conc, verdict):
    desc = desc or "This is a medical image."
    evid = evid or "The indicated region shows texture and boundary inconsistencies relative to the surrounding tissue."
    conc = conc or ("the region was digitally altered" if ref == "deepfake" else
                    ("a genuine lesion is present" if ref else "no suspicious region is present"))
    if ref is None:                                   # normal / no box
        return (f"<think>\nOkay, let's see. I am starting to analyze this image to identify any lesion regions: "
                f"<description>{desc}</description>. I am now going to carefully scan the entire image for any "
                f"suspicious regions. In my observation, the evidence against a lesion is: <evidence>{evid}</evidence>. "
                f"Based on these findings, I conclude that: <conclusion>{conc}</conclusion>.\n</think>\n\n{verdict}")
    x1, y1, x2, y2 = box
    noun = "deepfake" if ref == "deepfake" else ref
    lead = ("whether this image is a deepfake" if ref == "deepfake" else "this image to identify any lesion regions")
    found = ("suspect deepfake area" if ref == "deepfake" else f"suspect {noun} region")
    eword = ("deepfake" if ref == "deepfake" else f"a {noun}")
    return (f"<think>\nOkay, let's see. I am starting to analyze {lead}: <description>{desc}</description>. "
            f'I have identified the {found} at: <|object_ref_start|>"{noun}"<|object_ref_end|>'
            f'<|box_start|>x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"<|box_end|>. '
            f"Now I am going to perform a detailed analysis of this suspect region. In my observation, the evidence "
            f"for {eword} is: <evidence>{evid}</evidence>. Based on these findings, I conclude that: "
            f"<conclusion>{conc}</conclusion>.\n</think>\n\n{verdict}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=2)
    a = ap.parse_args()
    cfg = SUBSETS[a.subset]
    prompt_tpl = open(f"{PROMPTS}/{cfg['prompt']}").read()
    boxes = load_boxes(a.subset)
    imgs = sorted(glob.glob(f"{BASE}/{cfg['img']}/*"))
    imgs = [f for f in imgs if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    imgs = imgs[a.start:a.start + a.count]

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    proc = AutoProcessor.from_pretrained(M)
    model = Qwen3VLForConditionalGeneration.from_pretrained(M, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0").eval()

    od = f"{OUT}/{a.subset}"; os.makedirs(f"{od}/images", exist_ok=True)
    os.makedirs(f"{od}/gt/bbox_mask", exist_ok=True); os.makedirs(f"{od}/gt/explanation", exist_ok=True)
    sft = open(f"{od}/sft_{a.start}_{a.start+len(imgs)}.jsonl", "w")
    mapping = []
    for ip in imgs:
        try:
            stem = os.path.splitext(os.path.basename(ip))[0]
            im = cv2.imread(ip); H, W = im.shape[:2]
            canvas = cv2.resize(im, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)
            outimg = f"{od}/images/{stem}.png"; cv2.imwrite(outimg, canvas)
            box = None
            if cfg["ref"] is not None:
                nb = boxes.get(stem)
                if nb is None:
                    print(f"[skip {stem}] no box", flush=True); continue
                sx, sy = CANVAS / W, CANVAS / H
                box = [int(nb[0]*sx), int(nb[1]*sy), int(nb[2]*sx), int(nb[3]*sy)]
                filled = prompt_tpl.replace("{x1}", str(box[0])).replace("{y1}", str(box[1])).replace("{x2}", str(box[2])).replace("{y2}", str(box[3]))
            else:
                filled = prompt_tpl
            # ---- Qwen3-VL per-image reasoning ----
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": filled}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = proc(text=[text], images=[Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))], return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=420, do_sample=False)
            raw = proc.tokenizer.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            desc, evid, conc = extract("description", raw), extract("evidence", raw), extract("conclusion", raw)
            cot = build_cot(cfg["ref"], box, desc, evid, conc, cfg["verdict"])
            # ---- artifacts ----
            if box:
                m = np.zeros((CANVAS, CANVAS), np.uint8); m[box[1]:box[3], box[0]:box[2]] = 255
                mpath = f"{od}/gt/bbox_mask/{stem}_bbox_mask.png"; Image.fromarray(m).save(mpath)
            else:
                mpath = None
            epath = f"{od}/gt/explanation/{stem}.txt"; open(epath, "w").write((evid or "") + ("\nConclusion: " + conc if conc else ""))
            rec = {"images": [outimg], "messages": [{"role": "system", "content": SYS},
                    {"role": "user", "content": "<image>" + USER}, {"role": "assistant", "content": cot}]}
            sft.write(json.dumps(rec) + "\n"); sft.flush()
            mapping.append({"key": f"{stem}.png", "image_path": outimg, "mask_path": mpath, "explain_path": epath,
                            "box": box, "model": "real-source", "dataset": cfg["dataset"], "disease": cfg["disease"], "type": cfg["typ"]})
            print(f"[ok {stem}] ref={cfg['ref']} box={box}", flush=True)
        except Exception as e:
            print(f"[ERR {ip}] {e}", flush=True)
    sft.close()
    json.dump(mapping, open(f"{od}/mapping_{a.start}_{a.start+len(imgs)}.json", "w"))
    print(f"SUBSET DONE: {a.subset} {len(mapping)} annotated -> {od}", flush=True)

if __name__ == "__main__":
    main()
