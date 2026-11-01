"""Is |forged - source| a usable training-time localisation target? READ-ONLY on data.
For a stratified sample of MF90K fakes: load forged, source (real_image_path), bbox mask; report size match,
mask-bbox vs SFT gt_box coordinate convention, and how well a thresholded difference map agrees with the GT box."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, random, re, sys, os, collections
import numpy as np
from PIL import Image
from multiprocessing import Pool
ROOT = _release_path('AAAI/MedForge++/medforge90k_full/')
mp = json.load(open(ROOT + "dataset_mapping.json"))["items"]
fakes = [x for x in mp if x["type"] in ("edit", "remove") and x.get("real_image_path") and x.get("mask_path")]
rng = random.Random(0)
strata = collections.defaultdict(list)
for x in fakes:
    strata[(x["model"], x["type"])].append(x)
sample = []
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 12
for k in sorted(strata):
    v = strata[k]; rng.shuffle(v); sample += v[:PER]

# SFT gt_box lookup (coordinate convention) from the base CF/ID SFT files
BOX = re.compile(r'x1="(\d+)" y1="(\d+)" x2="(\d+)" y2="(\d+)"')
want = set(ROOT + x["image_path"] for x in sample)
gtbox = {}
for fn in ["mfp_train_indomain.jsonl"]:
    for l in open(_release_path('AAAI/MedForge++/') + fn):
        if '"images"' not in l: continue
        r = json.loads(l); im = r["images"][0]
        if im in want:
            a = " ".join(m["content"] for m in r["messages"] if m["role"] == "assistant")
            b = BOX.search(a)
            if b: gtbox[im] = [int(t) for t in b.groups()]

def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0: return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

def iou(a, b):
    if a is None or b is None: return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - ix*iy
    return ix*iy/u if u > 0 else 0.0

def work(x):
    try:
        f = Image.open(ROOT + x["image_path"]).convert("L"); s = Image.open(ROOT + x["real_image_path"]).convert("L")
        m = Image.open(ROOT + x["mask_path"]).convert("L")
    except Exception as e:
        return {"model": x["model"], "type": x["type"], "err": type(e).__name__}
    out = {"model": x["model"], "type": x["type"], "organ": x["dataset"], "f_size": f.size, "s_size": s.size, "m_size": m.size,
           "same_size": f.size == s.size}
    if s.size != f.size: s = s.resize(f.size, Image.BICUBIC)
    if m.size != f.size: m = m.resize(f.size, Image.NEAREST)
    F = np.asarray(f, np.float32); S = np.asarray(s, np.float32); M = np.asarray(m) > 127
    W, H = f.size
    gb = bbox_of(M); out["mask_bbox_px"] = gb
    out["mask_area_frac"] = float(M.mean())
    D = np.abs(F - S)
    out["mad_in"] = float(D[M].mean()) if M.any() else None
    out["mad_out"] = float(D[~M].mean()) if (~M).any() else None
    for thr in (10, 25):
        T = D > thr
        tp = (T & M).sum()
        out[f"t{thr}_prec"] = float(tp / max(1, T.sum()))      # fraction of changed pixels inside GT box
        out[f"t{thr}_rec"] = float(tp / max(1, M.sum()))       # fraction of GT box that changed
        out[f"t{thr}_frac_changed"] = float(T.mean())
    # box from robust diff: blur + percentile threshold + largest blob bbox
    try:
        import cv2
        Db = cv2.GaussianBlur(D, (0, 0), sigmaX=max(2, W / 256))
        thr = max(10.0, float(np.percentile(Db, 99)))
        T = (Db > thr).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(T, 8)
        if n > 1:
            j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x0, y0, w, h = stats[j, 0], stats[j, 1], stats[j, 2], stats[j, 3]
            db = [int(x0), int(y0), int(x0 + w), int(y0 + h)]
        else:
            db = None
        out["diffbox_iou_vs_maskbox"] = iou(db, gb)
    except Exception as e:
        out["diffbox_err"] = str(e)
    g = gtbox.get(ROOT + x["image_path"])
    if g and gb:
        norm = [round(gb[0] / W * 1000), round(gb[1] / H * 1000), round(gb[2] / W * 1000), round(gb[3] / H * 1000)]
        out["sft_gt_box"] = g; out["mask_box_norm1000"] = norm
        out["iou_sft_vs_px"] = iou(g, gb); out["iou_sft_vs_norm1000"] = iou(g, norm)
    return out

if __name__ == "__main__":
    with Pool(8) as p:
        res = p.map(work, sample, chunksize=4)
    os.makedirs(_release_path('CIRCE/v3_audit/code_scripts/out'), exist_ok=True)
    with open(_release_path('CIRCE/v3_audit/code_scripts/out/diffmap_feasibility.jsonl'), "w") as fh:
        for r in res: fh.write(json.dumps(r) + "\n")
    ok = [r for r in res if "err" not in r]
    print(f"sampled={len(res)} ok={len(ok)} errors={len(res)-len(ok)} same_size={sum(r['same_size'] for r in ok)}")
    w_sft = [r for r in ok if "sft_gt_box" in r]
    if w_sft:
        print(f"SFT gt_box convention (n={len(w_sft)}): mean IoU vs pixel mask-bbox={np.mean([r['iou_sft_vs_px'] for r in w_sft]):.3f}, "
              f"vs 0-1000 normalised mask-bbox={np.mean([r['iou_sft_vs_norm1000'] for r in w_sft]):.3f}")
    print("model,type,n,same_size,mad_in,mad_out,t25_prec,t25_rec,t25_frac_changed,diffbox_IoU(med),diffbox_IoU>0.5")
    grp = collections.defaultdict(list)
    for r in ok: grp[(r["model"], r["type"])].append(r)
    def med(v): v = [t for t in v if t is not None]; return float(np.median(v)) if v else float('nan')
    for k in sorted(grp):
        v = grp[k]
        print(f"{k[0]},{k[1]},{len(v)},{sum(r['same_size'] for r in v)},{med([r['mad_in'] for r in v]):.1f},{med([r['mad_out'] for r in v]):.1f},"
              f"{med([r['t25_prec'] for r in v]):.2f},{med([r['t25_rec'] for r in v]):.2f},{med([r['t25_frac_changed'] for r in v]):.3f},"
              f"{med([r.get('diffbox_iou_vs_maskbox') for r in v]):.2f},{np.mean([r.get('diffbox_iou_vs_maskbox', 0) > 0.5 for r in v]):.2f}")
    for t in ("edit", "remove"):
        v = [r for r in ok if r["type"] == t]
        print(f"ALL {t}: n={len(v)} diffbox_IoU median={med([r.get('diffbox_iou_vs_maskbox') for r in v]):.3f} "
              f">0.5 frac={np.mean([r.get('diffbox_iou_vs_maskbox', 0) > 0.5 for r in v]):.3f} "
              f"t25_prec median={med([r['t25_prec'] for r in v]):.3f} mad_out median={med([r['mad_out'] for r in v]):.2f}")
