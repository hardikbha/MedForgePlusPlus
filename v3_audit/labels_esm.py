"""Part 5: proposed Edit-Support Mask (ESM) target vs the bbox mask, on a subsample of
the pairdiff sample (all strata). Read-only. Writes labels_esm.jsonl / prints summary."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, random, collections, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from multiprocessing import Pool
from skimage.registration import phase_cross_correlation

ROOT = _release_path('AAAI/MedForge++/medforge90k_full/')
OUT = _release_path('CIRCE/v3_audit/labels_esm_v2.jsonl')


def box_of(M):
    ys, xs = np.where(M)
    if not len(ys):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def biou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def one(r):
    out = {k: r[k] for k in ('key', 'model', 'type', 'dataset')}
    try:
        F = np.asarray(Image.open(ROOT + r['key']).convert('L'), dtype=np.float32)
        S = np.asarray(Image.open(ROOT + r['real_image_path']).convert('L'), dtype=np.float32)
        M = np.asarray(Image.open(ROOT + r['mask_path']).convert('L')) > 127
        # integer-pixel registration (undo global shift of full-image regenerators)
        sh, _, _ = phase_cross_correlation(S, F, upsample_factor=1)
        dy, dx = int(round(sh[0])), int(round(sh[1]))
        out['shift'] = [dy, dx]
        if dy or dx:
            F = np.roll(F, (dy, dx), axis=(0, 1))
        D = np.abs(F - S)
        Db = ndi.gaussian_filter(D, 2.0)
        dist = ndi.distance_transform_cdt(~M, metric='chessboard')
        far = dist > 64
        near = dist <= 32
        v = Db[far]
        mu = float(np.median(v)); sd = float(1.4826 * np.median(np.abs(v - mu)))
        for name, T in [('esm_z5', max(12.0, mu + 5 * sd)), ('esm_z3', max(8.0, mu + 3 * sd))]:
            B = Db > T
            lab, n = ndi.label(B)
            if n:
                sizes = np.bincount(lab.ravel(), minlength=n + 1)
                touch = np.zeros(n + 1, bool)
                touch[np.unique(lab[near & B])] = True
                good = (sizes >= 64) & touch; good[0] = False
                E = good[lab]
            else:
                E = B
            a = int(E.sum()); bb = int(M.sum())
            rec = {'T': T, 'empty': a == 0, 'area_ratio': a / bb if bb else None,
                   'pix_iou_bbox': float((E & M).sum() / (E | M).sum()) if (E | M).any() else None,
                   'frac_outside_bbox': float((E & ~M).sum() / a) if a else None,
                   'bbox_cov': float((E & M).sum() / bb) if bb else None}
            be = box_of(E); bm = box_of(M)
            if be:
                rec['box_iou_bbox'] = biou(be, bm)
                rec['box_area_ratio'] = ((be[2] - be[0]) * (be[3] - be[1])) / ((bm[2] - bm[0]) * (bm[3] - bm[1]))
                # centre offset in px
                rec['centre_offset_px'] = float(np.hypot((be[0] + be[2] - bm[0] - bm[2]) / 2, (be[1] + be[3] - bm[1] - bm[3]) / 2))
            out[name] = rec
            # clipped variant: ESM restricted to bbox dilated by 32 px (proposed reward target)
            Ec = E & near
            ac = int(Ec.sum())
            rc = {'T': T, 'empty': ac == 0, 'area_ratio': ac / bb if bb else None,
                  'pix_iou_bbox': float((Ec & M).sum() / (Ec | M).sum()) if (Ec | M).any() else None,
                  'frac_outside_bbox': float((Ec & ~M).sum() / ac) if ac else None,
                  'bbox_cov': float((Ec & M).sum() / bb) if bb else None}
            bc = box_of(Ec)
            if bc:
                rc['box_iou_bbox'] = biou(bc, bm)
                rc['box_area_ratio'] = ((bc[2] - bc[0]) * (bc[3] - bc[1])) / ((bm[2] - bm[0]) * (bm[3] - bm[1]))
                rc['centre_offset_px'] = float(np.hypot((bc[0] + bc[2] - bm[0] - bm[2]) / 2, (bc[1] + bc[3] - bm[1] - bm[3]) / 2))
            out[name + '_clip32'] = rc
        out['far_mu'] = mu; out['far_sd'] = sd
    except Exception as e:
        out['error'] = repr(e)
    return out


def main():
    R = [json.loads(l) for l in open(_release_path('CIRCE/v3_audit/labels_pairdiff.jsonl'))]
    rng = random.Random(3)
    st = collections.defaultdict(list)
    for r in R:
        st[(r['model'], r['type'], r['dataset'])].append(r)
    smp = []
    for k in sorted(st):
        smp += rng.sample(st[k], min(10, len(st[k])))
    with Pool(int(os.environ.get('NPROC', '8'))) as p:
        res = p.map(one, smp, chunksize=4)
    with open(OUT, 'w') as f:
        for r in res:
            f.write(json.dumps(r) + '\n')
    ok = [r for r in res if 'error' not in r]
    print('n', len(res), 'ok', len(ok))
    for name in ['esm_z5', 'esm_z3', 'esm_z5_clip32', 'esm_z3_clip32']:
        X = [r[name] for r in ok]
        nz = [x for x in X if not x['empty']]
        f = lambda k: np.array([x[k] for x in nz if x.get(k) is not None])
        print(f'{name}: empty {np.mean([x["empty"] for x in X]):.3f} | median T {np.median([x["T"] for x in X]):.1f} | '
              f'pixel IoU vs bbox med {np.median(f("pix_iou_bbox")):.3f} | frac of ESM outside bbox med {np.median(f("frac_outside_bbox")):.3f} | '
              f'bbox covered med {np.median(f("bbox_cov")):.3f} | ESM area/bbox med {np.median(f("area_ratio")):.3f} (p10 {np.percentile(f("area_ratio"),10):.2f}, p90 {np.percentile(f("area_ratio"),90):.2f}) | '
              f'ESM-box IoU vs bbox med {np.median(f("box_iou_bbox")):.3f}, frac<0.5 {np.mean(f("box_iou_bbox")<0.5):.3f} | ESM-box area/bbox med {np.median(f("box_area_ratio")):.2f} | centre offset med {np.median(f("centre_offset_px")):.1f}px')
        for t in ['edit', 'remove']:
            Y = [r[name] for r in ok if r['type'] == t and not r[name]['empty']]
            print(f'   {t}: n={len(Y)} pixIoU {np.median([y["pix_iou_bbox"] for y in Y]):.3f} boxIoU {np.median([y["box_iou_bbox"] for y in Y if "box_iou_bbox" in y]):.3f} area {np.median([y["area_ratio"] for y in Y]):.2f}')
        for org in ['brain-mri', 'chest-xray', 'fundus']:
            Y = [r[name] for r in ok if r['dataset'] == org and not r[name]['empty']]
            print(f'   {org}: n={len(Y)} pixIoU {np.median([y["pix_iou_bbox"] for y in Y]):.3f} boxIoU {np.median([y["box_iou_bbox"] for y in Y if "box_iou_bbox" in y]):.3f} area {np.median([y["area_ratio"] for y in Y]):.2f} cov {np.median([y["bbox_cov"] for y in Y]):.2f}')
    print('nonzero registration shift frac', np.mean([r['shift'] != [0, 0] for r in ok]))


if __name__ == '__main__':
    main()
