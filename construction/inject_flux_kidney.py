#!/usr/bin/env python3
"""Kidney renal-tumor injection via FLUX.1-Kontext (GGUF), offline, GPU-shardable.
Normal kidney CT -> deepfake with a synthetic renal tumor inpainted onto a kidney
(placed in a lateral soft-tissue zone, not random). 1024x1024 export + MedForge annotation."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os, gc, json, argparse, glob, math
import numpy as np, cv2, torch
from PIL import Image

BASE = _release_path('AAAI/models/FLUX.1-Kontext-dev')
GGUF = _release_path('AAAI/models/kontext_gguf/flux1-kontext-dev-Q4_K_M.gguf')
NORMAL = _release_path('AAAI/kidney/normal')
EXEMP  = _release_path('AAAI/kidney/tumor_exemplars')
CANVAS = 1024

SYS = ("You are an expert in medical image forensics. Analyze the provided image to determine "
       "if it is a deepfake or authentic. First, perform a step-by-step examination of the image "
       "content, looking for artifacts, inconsistencies, or biological implausibilities. Use "
       "<think> tags to articulate your reasoning process. If you identify manipulated regions, "
       "localize them using bounding boxes within your reasoning. Conclude your analysis with a "
       "final classification.")
USER = "<image>Is this image deepfake or real?"
DESCS = ["This is an abdominal CT scan showing the kidneys, spine and surrounding soft tissue",
         "This is a computed tomography image of the abdomen depicting the renal and para-spinal anatomy",
         "This is an axial/coronal abdominal CT slice showing the kidneys and retroperitoneal structures"]
EVID = ["The marked region contains a soft-tissue mass whose density and texture are inconsistent with the "
        "adjacent renal parenchyma; its margin shows unnatural blending and a lack of continuity with normal "
        "kidney architecture, indicating a synthetically inserted lesion",
        "Within the highlighted area a rounded lesion appears whose CT attenuation and internal texture differ "
        "from the surrounding kidney tissue, with soft feathered edges characteristic of generative inpainting",
        "The localized mass disrupts the normal corticomedullary pattern of the kidney and shows residual "
        "blending artifacts along its boundary, revealing digital manipulation"]
CONC = ["the localized renal mass is a digitally inserted structure inconsistent with the surrounding kidney, "
        "confirming this image is a deepfake",
        "the marked lesion was synthetically added and does not correspond to genuine renal tissue, confirming "
        "this image is a deepfake",
        "the highlighted region shows clear signs of generative insertion, confirming this image is a deepfake"]

def cot(box, i):
    x1, y1, x2, y2 = box
    return (f"<think>\nOkay, let's see. I am starting to analyze whether this image is a deepfake: "
            f"<description>{DESCS[i%3]}</description>. I have identified the suspect deepfake area at: "
            f'<|object_ref_start|>"deepfake"<|object_ref_end|>'
            f'<|box_start|>x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"<|box_end|>. '
            f"Now I am going to perform a detailed analysis of this suspect region. In my observation, the "
            f"evidence for deepfake is: <evidence>{EVID[i%3]}</evidence>. Based on these findings, I conclude "
            f"that: <conclusion>{CONC[i%3]}</conclusion>. \n</think>\n\nThis image is a deepfake.")

def pick_kidney_spot(gray):
    """Pick a lateral soft-tissue location (a kidney) — bilateral zones lateral to the central spine."""
    H, W = gray.shape
    ax, ay = int(0.062 * min(H, W)), int(0.056 * min(H, W))   # smaller, realistic renal-mass size
    # candidate kidney centers: right & left lateral bands, mid height (coronal + axial both ~here)
    cands = [(0.30, 0.44), (0.70, 0.44), (0.32, 0.52), (0.68, 0.52), (0.28, 0.38), (0.72, 0.38)]
    best, bestscore = None, -1
    for fx, fy in cands:
        cx, cy = int(W * fx), int(H * fy)
        if cx - ax < 4 or cy - ay < 4 or cx + ax >= W - 4 or cy + ay >= H - 4:
            continue
        patch = gray[cy - ay:cy + ay, cx - ax:cx + ax]
        if patch.size == 0:
            continue
        # want soft tissue: not air/background (<25) and not bone (>210); reward mid-gray coverage
        soft = ((patch > 35) & (patch < 205)).mean()
        if soft > bestscore:
            best, bestscore = (cx, cy, ax, ay), soft
    return best or (int(W * 0.30), int(H * 0.44), ax, ay)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--out", default=_release_path('AAAI/kidney/flux_injected'))
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--strong", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    imgs = sorted(glob.glob(f"{NORMAL}/*.jpg"))
    exemplars = sorted(glob.glob(f"{EXEMP}/*.jpg"))
    end = min(a.start + a.count, len(imgs))

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

    man = open(os.path.join(a.out, f"manifest_{a.start}_{end}.jsonl"), "w")
    done = 0
    for idx in range(a.start, end):
        try:
            target = cv2.imread(imgs[idx]); H, W = target.shape[:2]
            gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            cx, cy, ax, ay = pick_kidney_spot(gray)
            mask = np.zeros((H, W), np.uint8); cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
            # exemplar: center crop of a tumor CT (rough tumor-tissue reference)
            # SOFT-TISSUE reference: scan a tumor-CT exemplar for a mid-gray patch (kidney/tumor
            # density ~40-135), avoiding bright bone/contrast/calcification that made lesions white.
            ex = cv2.imread(exemplars[idx % len(exemplars)]); eh, ew = ex.shape[:2]
            exg = cv2.cvtColor(ex, cv2.COLOR_BGR2GRAY)
            ref_crop = None; best_soft = -1
            ph, pw = int(eh * 0.22), int(ew * 0.22)
            for fy in (0.35, 0.45, 0.55, 0.4, 0.5):
                for fx in (0.3, 0.4, 0.5, 0.6, 0.35):
                    cyp, cxp = int(eh * fy), int(ew * fx)
                    patch = exg[cyp:cyp + ph, cxp:cxp + pw]
                    if patch.size == 0:
                        continue
                    soft = ((patch > 40) & (patch < 135)).mean()       # fraction soft-tissue density
                    bright = (patch > 180).mean()                      # penalize bone/contrast
                    score = soft - 2.0 * bright
                    if score > best_soft:
                        best_soft = score; ref_crop = ex[cyp:cyp + ph, cxp:cxp + pw]
            if ref_crop is None or ref_crop.size == 0:
                ref_crop = ex[int(eh*0.4):int(eh*0.6), int(ew*0.35):int(ew*0.55)]

            md = 512; sc = min(1.0, md / max(W, H))
            nw = max(16, (int(W*sc)//16)*16); nh = max(16, (int(H*sc)//16)*16)
            t_rgb = Image.fromarray(cv2.cvtColor(cv2.resize(target,(nw,nh),interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            m_img = Image.fromarray(cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_NEAREST))
            rh, rw = ref_crop.shape[:2]
            if max(rh,rw) > 256:
                rs = 256/max(rh,rw); ref_crop = cv2.resize(ref_crop,(int(rw*rs),int(rh*rs)))
            r_rgb = Image.fromarray(cv2.cvtColor(ref_crop, cv2.COLOR_BGR2RGB))
            # Tuned: force SOFT-TISSUE density (a real renal tumor is iso-/hypodense to kidney, gray,
            # NOT bright/white/calcified). Lower guidance keeps it subtle and blended.
            prompt = ("a well-defined rounded solid renal tumor mass, soft-tissue attenuation slightly lower "
                      "than the surrounding kidney (mildly hypodense and mildly heterogeneous), clearly "
                      "delineated as a distinct mass but blended at its edges into the kidney parenchyma, "
                      "realistic non-contrast abdominal CT texture in shades of gray, not bright, not white, "
                      "not calcified, no bright streaks")
            steps, guid = (32, 4.0) if a.strong else (30, 3.2)
            gc.collect(); torch.cuda.empty_cache()
            res = pipe(prompt=prompt, image=t_rgb, mask_image=m_img, image_reference=r_rgb,
                       height=nh, width=nw, strength=1.0, num_inference_steps=steps, guidance_scale=guid,
                       generator=torch.Generator(device="cpu").manual_seed(a.seed+idx),
                       _auto_resize=False, max_area=nh*nw).images[0]
            out = cv2.cvtColor(np.array(res.resize((W, H))), cv2.COLOR_RGB2BGR)
            canvas = cv2.resize(out, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)
            sx, sy = CANVAS/W, CANVAS/H
            box = [int((cx-ax)*sx), int((cy-ay)*sy), int((cx+ax)*sx), int((cy+ay)*sy)]
            stem = os.path.splitext(os.path.basename(imgs[idx]))[0].replace(" ", "_")
            outp = os.path.join(a.out, "images", f"{stem}_tumorinj.jpg")
            cv2.imwrite(outp, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rec = {"images": [outp], "organ": "kidney", "modality": "CT", "forgery_type": "inject",
                   "klass": "tumor", "method": "flux.1-kontext", "box": box, "canvas": CANVAS,
                   "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": USER},
                                {"role": "assistant", "content": cot(box, idx)}]}
            man.write(json.dumps(rec) + "\n"); man.flush(); done += 1
            print(f"[{idx}] {stem} box={box} -> {outp}", flush=True)
        except Exception as e:
            print(f"[{idx}] SKIP ({e})", flush=True)
    man.close()
    print(f"SHARD DONE: {done}/{end-a.start} injected -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
