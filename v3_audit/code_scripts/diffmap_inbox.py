"""Variant: difference map used only INSIDE the (dilated) GT bbox, after removing global offset.
Measures whether a diff-derived 'edit box' is a tighter subset of the GT bbox (useful as a training-only target)."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, random, collections, sys
import numpy as np, cv2
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, _release_path('CIRCE/v3_audit/code_scripts'))
from diffmap_feasibility import sample, ROOT, bbox_of, iou
def work(x):
    try:
        F = np.asarray(Image.open(ROOT + x["image_path"]).convert("L"), np.float32)
        S = np.asarray(Image.open(ROOT + x["real_image_path"]).convert("L"), np.float32)
        M = np.asarray(Image.open(ROOT + x["mask_path"]).convert("L")) > 127
    except Exception as e:
        return None
    if F.shape != S.shape or F.shape != M.shape: return None
    H, W = F.shape
    gb = bbox_of(M)
    if gb is None: return None
    Dsig = F - S
    Dsig -= np.median(Dsig)                         # remove global intensity offset
    D = cv2.GaussianBlur(np.abs(Dsig), (0, 0), 3)
    # dilate GT box by 25% of its size
    bw, bh = gb[2] - gb[0], gb[3] - gb[1]
    x0, y0 = max(0, int(gb[0] - .25 * bw)), max(0, int(gb[1] - .25 * bh))
    x1, y1 = min(W, int(gb[2] + .25 * bw)), min(H, int(gb[3] + .25 * bh))
    R = D[y0:y1, x0:x1]
    out_med = float(np.median(D[~M])) if (~M).any() else 0.0
    out_p99 = float(np.percentile(D[~M], 99)) if (~M).any() else 0.0
    thr = max(out_p99, 8.0)                          # changed = above the 99th pct of outside-box diff
    T = (R > thr).astype(np.uint8)
    frac_box_changed = float(T[(gb[1]-y0):(gb[3]-y0), (gb[0]-x0):(gb[2]-x0)].mean())
    n, lab, st, _ = cv2.connectedComponentsWithStats(T, 8)
    if n <= 1:
        return dict(type=x["type"], model=x["model"], has_signal=False, frac_box_changed=frac_box_changed, out_p99=out_p99)
    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    db = [x0 + int(st[j, 0]), y0 + int(st[j, 1]), x0 + int(st[j, 0] + st[j, 2]), y0 + int(st[j, 1] + st[j, 3])]
    area = lambda b: (b[2]-b[0])*(b[3]-b[1])
    return dict(type=x["type"], model=x["model"], has_signal=True, iou=iou(db, gb), area_ratio=area(db)/max(1, area(gb)),
                frac_box_changed=frac_box_changed, out_p99=out_p99, inside_gt=float(iou(db, gb) > 0 and db[0] >= gb[0]-2 and db[1] >= gb[1]-2 and db[2] <= gb[2]+2 and db[3] <= gb[3]+2))
if __name__ == "__main__":
    with Pool(8) as p:
        res = [r for r in p.map(work, sample, chunksize=4) if r]
    for t in ("edit", "remove"):
        v = [r for r in res if r["type"] == t]; s = [r for r in v if r["has_signal"]]
        print(f"{t}: n={len(v)} with_signal={len(s)} ({len(s)/len(v):.2f}) median IoU(diffbox,GT)={np.median([r['iou'] for r in s]):.3f} "
              f"median area_ratio={np.median([r['area_ratio'] for r in s]):.3f} frac IoU>0.5={np.mean([r['iou']>0.5 for r in s]):.3f} "
              f"median frac_of_GTbox_changed={np.median([r['frac_box_changed'] for r in v]):.3f} median outside p99={np.median([r['out_p99'] for r in v]):.1f}")
    g = collections.defaultdict(list)
    for r in res: g[r["model"]].append(r)
    for k in sorted(g):
        s = [r for r in g[k] if r["has_signal"]]
        print(f"  {k:42s} n={len(g[k])} signal={len(s)} medIoU={np.median([r['iou'] for r in s]) if s else float('nan'):.2f} "
              f"med_area_ratio={np.median([r['area_ratio'] for r in s]) if s else float('nan'):.2f} med_frac_box_changed={np.median([r['frac_box_changed'] for r in g[k]]):.3f}")
