#!/usr/bin/env python3
"""Batch PHOTOREALISTIC polyp injection via FLUX.1-Kontext (GGUF), offline, GPU-shardable.
For each Kvasir-SEG frame: inject a reference-guided synthetic polyp into CLEAN mucosa
(away from the frame's real polyp), export a 1024x1024 image + a MedForge-format
annotation (exact bbox, since WE place the lesion). Writes one JSONL manifest per shard.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python inject_flux_batch.py --start 0 --count 200 --out <dir>
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os, gc, zipfile, json, argparse, math
import numpy as np, cv2, torch
from PIL import Image

BASE = _release_path('AAAI/models/FLUX.1-Kontext-dev')
GGUF = _release_path('AAAI/models/kontext_gguf/flux1-kontext-dev-Q4_K_M.gguf')
ZIP  = _release_path('AAAI/colon/kvasir.zip')
CANVAS = 1024  # MedForge convention: 1024x1024, box in absolute px of this canvas

SYS = ("You are an expert in medical image forensics. Analyze the provided image to determine "
       "if it is a deepfake or authentic. First, perform a step-by-step examination of the image "
       "content, looking for artifacts, inconsistencies, or biological implausibilities. Use "
       "<think> tags to articulate your reasoning process. If you identify manipulated regions, "
       "localize them using bounding boxes within your reasoning. Conclude your analysis with a "
       "final classification.")
USER = "<image>Is this image deepfake or real?"

DESCS = [
    "This is a colonoscopy image showing the colonic mucosa with its characteristic vascular surface pattern and intestinal lumen",
    "This is an endoscopic image of the large intestine, displaying the mucosal folds and superficial capillary network",
    "This is a white-light colonoscopy frame of the colorectal mucosa, showing the lumen and normal haustral folds",
]
EVID = [
    "The highlighted region contains a rounded sessile lesion whose surface texture and microvascular pattern do not match the surrounding mucosa; its boundary shows subtle blending and a lighting profile inconsistent with the true specular highlights of the frame, indicating synthetic insertion",
    "Within the marked area a polypoid structure appears with color and texture statistics that differ from the adjacent tissue; the transition at its margin lacks the natural mucosal continuity and shows soft, artificially feathered edges typical of generative inpainting",
    "The localized lesion exhibits an implausibly uniform surface and a vascular pattern that is not contiguous with the surrounding capillary network; residual blending artifacts along its perimeter reveal that this region was digitally manipulated",
]
CONC = [
    "the localized lesion is a digitally inserted structure inconsistent with the surrounding mucosa, confirming this image is a deepfake",
    "the marked polypoid region was synthetically added and does not correspond to genuine tissue, confirming this image is a deepfake",
    "the highlighted area shows clear signs of generative insertion, confirming this image is a deepfake",
]

def cot(box, i):
    x1, y1, x2, y2 = box
    d, e, c = DESCS[i % len(DESCS)], EVID[i % len(EVID)], CONC[i % len(CONC)]
    return (f"<think>\nOkay, let's see. I am starting to analyze whether this image is a deepfake: "
            f"<description>{d}</description>. I have identified the suspect deepfake area at: "
            f'<|object_ref_start|>"deepfake"<|object_ref_end|>'
            f'<|box_start|>x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"<|box_end|>. '
            f"Now I am going to perform a detailed analysis of this suspect region. In my observation, "
            f"the evidence for deepfake is: <evidence>{e}</evidence>. Based on these findings, I conclude "
            f"that: <conclusion>{c}</conclusion>. \n</think>\n\nThis image is a deepfake.")

def pick_placement(target, polyp_mask, rng):
    """Choose an elliptical placement center in bright mucosa, away from the real polyp / black corners."""
    H, W = target.shape[:2]
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # polyp centroid (if any)
    ys, xs = np.where(polyp_mask > 0)
    pc = (int(xs.mean()), int(ys.mean())) if len(xs) else (W // 2, H // 2)
    ax, ay = int(0.10 * min(H, W)), int(0.09 * min(H, W))
    cands = [(0.35, 0.40), (0.62, 0.62), (0.40, 0.66), (0.62, 0.38), (0.50, 0.50),
             (0.30, 0.55), (0.70, 0.50), (0.50, 0.32)]
    best, bestscore = None, -1
    for fx, fy in cands:
        cx, cy = int(W * fx), int(H * fy)
        if cx - ax < 4 or cy - ay < 4 or cx + ax >= W - 4 or cy + ay >= H - 4:
            continue
        patch = gray[cy - ay:cy + ay, cx - ax:cx + ax]
        if patch.size == 0 or patch.mean() < 55:          # skip dark endoscopic corners
            continue
        pm = polyp_mask[cy - ay:cy + ay, cx - ax:cx + ax]
        if pm.size and pm.mean() > 8:                     # overlaps real polyp -> skip
            continue
        dist = math.hypot(cx - pc[0], cy - pc[1])         # prefer far from real polyp
        if dist > bestscore:
            best, bestscore = (cx, cy, ax, ay), dist
    if best is None:
        best = (int(W * 0.5), int(H * 0.5), ax, ay)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=2)
    ap.add_argument("--out", default=_release_path('AAAI/colon/flux_injected'))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--redo_json", default="", help="list of injected image paths to regenerate (by stem)")
    ap.add_argument("--redo_shard", type=int, default=0, help="which shard of the redo list")
    ap.add_argument("--redo_nshard", type=int, default=1)
    ap.add_argument("--strong", action="store_true", help="boosted params for clearly-visible lesion")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)

    z = zipfile.ZipFile(ZIP)
    imgs = sorted(n for n in z.namelist() if "/images/" in n and n.lower().endswith((".jpg", ".jpeg", ".png")))
    def dec(n, g=False):
        arr = np.frombuffer(z.read(n), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE if g else cv2.IMREAD_COLOR)

    from diffusers import FluxKontextInpaintPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
    print("loading FLUX.1-Kontext (GGUF)...", flush=True)
    transformer = FluxTransformer2DModel.from_single_file(
        GGUF, config=os.path.join(BASE, "transformer"),
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16), torch_dtype=torch.bfloat16)
    pipe = FluxKontextInpaintPipeline.from_pretrained(BASE, transformer=transformer, torch_dtype=torch.bfloat16)
    if os.environ.get("FLUX_NO_OFFLOAD"):    # full model on GPU (fast) when VRAM is ample
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    stem2idx = {os.path.splitext(os.path.basename(n))[0]: i for i, n in enumerate(imgs)}
    if a.redo_json:
        paths = json.load(open(a.redo_json))
        stems = [os.path.basename(p).replace("_polypinj.jpg", "") for p in paths]
        idxs = [stem2idx[s] for s in stems if s in stem2idx]
        index_list = idxs[a.redo_shard::a.redo_nshard]        # round-robin shard of the redo list
        man = None                                            # box unchanged -> keep existing manifests
        print(f"REDO mode: {len(index_list)} of {len(idxs)} weak images (shard {a.redo_shard}/{a.redo_nshard})", flush=True)
    else:
        index_list = list(range(a.start, min(a.start + a.count, len(imgs))))
        man = open(os.path.join(a.out, f"manifest_{a.start}_{a.start + a.count}.jsonl"), "w")
    done = 0
    for idx in index_list:
        try:
            rng = np.random.RandomState(a.seed + idx)
            target = dec(imgs[idx]); H, W = target.shape[:2]
            # target's own polyp mask (to avoid it)
            mrel = imgs[idx].replace("/images/", "/masks/")
            pmask = dec(mrel, g=True)
            if pmask is None or pmask.shape[:2] != (H, W):
                pmask = np.zeros((H, W), np.uint8)
            _, pmask = cv2.threshold(pmask, 127, 255, cv2.THRESH_BINARY)
            # exemplar polyp crop (a DIFFERENT frame)
            exi = imgs[(idx * 7 + 3) % len(imgs)]
            eximg = dec(exi); exmask = dec(exi.replace("/images/", "/masks/"), g=True)
            _, exmask = cv2.threshold(exmask, 127, 255, cv2.THRESH_BINARY)
            eys, exs = np.where(exmask > 0)
            if len(exs) == 0:
                continue
            ex1, ey1, ex2, ey2 = exs.min(), eys.min(), exs.max(), eys.max()
            ref_crop = eximg[max(0, ey1 - 6):ey2 + 6, max(0, ex1 - 6):ex2 + 6]

            cx, cy, ax, ay = pick_placement(target, pmask, rng)
            mask = np.zeros((H, W), np.uint8); cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

            md = 512; sc = min(1.0, md / max(W, H))
            nw = max(16, (int(W * sc) // 16) * 16); nh = max(16, (int(H * sc) // 16) * 16)
            t_rgb = Image.fromarray(cv2.cvtColor(cv2.resize(target, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            m_img = Image.fromarray(cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST))
            rh, rw = ref_crop.shape[:2]
            if max(rh, rw) > 256:
                rs = 256 / max(rh, rw); ref_crop = cv2.resize(ref_crop, (int(rw * rs), int(rh * rs)), interpolation=cv2.INTER_AREA)
            r_rgb = Image.fromarray(cv2.cvtColor(ref_crop, cv2.COLOR_BGR2RGB))
            if a.strong:
                prompt = ("a prominent, clearly visible sessile colorectal polyp with a rounded raised surface and "
                          "distinct pink-red mucosal color and vascular pattern, well-defined against the surrounding "
                          "colon wall, photorealistic, seamlessly blended at the edges")
                steps, guid = 34, 5.0
            else:
                prompt = ("a realistic small sessile colorectal polyp lesion on the colon mucosa, matching the "
                          "surrounding lighting, vascular texture and pink-red color, seamlessly integrated, no borders")
                steps, guid = 28, 3.5
            gc.collect(); torch.cuda.empty_cache()
            res = pipe(prompt=prompt, image=t_rgb, mask_image=m_img, image_reference=r_rgb,
                       height=nh, width=nw, strength=1.0, num_inference_steps=steps, guidance_scale=guid,
                       generator=torch.Generator(device="cpu").manual_seed(a.seed + 1000 + idx if a.strong else a.seed + idx),
                       _auto_resize=False, max_area=nh * nw).images[0]
            flux_full = cv2.cvtColor(np.array(res.resize((W, H))), cv2.COLOR_RGB2BGR)
            # feathered composite: keep the ORIGINAL background untouched, blend only the lesion
            # region with a soft (Gaussian) alpha so the edge matches the surrounding mucosa.
            alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=max(3.0, ax * 0.40))
            alpha = np.clip(alpha, 0, 1)[..., None]
            out = (target.astype(np.float32) * (1 - alpha) + flux_full.astype(np.float32) * alpha).astype(np.uint8)
            # export to 1024x1024 canvas + box in that space
            canvas = cv2.resize(out, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)
            sx, sy = CANVAS / W, CANVAS / H
            box = [int((cx - ax) * sx), int((cy - ay) * sy), int((cx + ax) * sx), int((cy + ay) * sy)]
            stem = os.path.splitext(os.path.basename(imgs[idx]))[0]
            outp = os.path.join(a.out, "images", f"{stem}_polypinj.jpg")
            cv2.imwrite(outp, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if man is not None:                    # normal mode writes annotations; redo keeps existing (box unchanged)
                rec = {"images": [outp], "organ": "colon", "modality": "colonoscopy",
                       "forgery_type": "inject", "klass": "polyp", "method": "flux.1-kontext",
                       "box": box, "canvas": CANVAS,
                       "messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": USER},
                                    {"role": "assistant", "content": cot(box, idx)}]}
                man.write(json.dumps(rec) + "\n"); man.flush()
            done += 1
            print(f"[{idx}] {stem} box={box} -> {outp}", flush=True)
        except Exception as e:
            print(f"[{idx}] SKIP ({e})", flush=True)
    if man is not None:
        man.close()
    print(f"SHARD DONE: {done}/{len(index_list)} injected -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
