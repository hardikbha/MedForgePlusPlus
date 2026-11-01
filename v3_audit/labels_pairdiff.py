"""Labels/mask/edit-locality audit: forged vs source difference analysis on a
stratified sample of MedForge-90K forgeries. Read-only on data.
Writes ~/CIRCE/v3_audit/labels_pairdiff.jsonl
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import json, random, sys, io, collections, time
import numpy as np
from PIL import Image
from multiprocessing import Pool
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

ROOT = _release_path('AAAI/MedForge++/medforge90k_full')
OUT = _release_path('CIRCE/v3_audit/labels_pairdiff.jsonl')
PER_STRATUM = int(sys.argv[1]) if len(sys.argv) > 1 else 36


def gray(im):
    return np.asarray(im.convert('L'), dtype=np.float32)


def jpeg_q_estimate(im):
    """Rough IJG quality estimate from luma quant table."""
    try:
        q = im.quantization
    except Exception:
        return None
    if not q:
        return None
    t = np.array(q[0], dtype=np.float64)
    # standard IJG luminance table
    std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
                    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)
    # PIL quantization is in zigzag order in older versions / natural in newer; use sum ratio (order-invariant)
    s = t.sum() / std.sum() * 100.0
    if s <= 100:
        qual = (200 - s) / 2
    else:
        qual = 5000 / s
    return float(round(qual, 1))


def lap(a):
    return ndi.laplace(a)


def thr_masks(D, M, near32):
    """Thresholded difference-map candidates vs bbox M."""
    Db = ndi.gaussian_filter(D, 2.0)
    out = {}
    area_box = M.sum()
    for name, T in [('t8', 8.0), ('t16', 16.0), ('t24', 24.0)]:
        B = Db > T
        # remove tiny specks: keep components >= 64 px
        lab, n = ndi.label(B)
        if n:
            sizes = np.bincount(lab.ravel(), minlength=n + 1)
            good = sizes >= 64; good[0] = False
            keep = good[lab]
        else:
            keep = B
        a = keep.sum()
        inter = (keep & M).sum()
        union = (keep | M).sum()
        rec = {'area_frac': float(a / D.size),
               'iou_bbox': float(inter / union) if union else None,
               'prec_in_bbox': float(inter / a) if a else None,
               'bbox_cov': float(inter / area_box) if area_box else None,
               'area_ratio_vs_bbox': float(a / area_box) if area_box else None}
        # tight box around the in-bbox part of the diff mask (largest comp inside dilated bbox)
        if a:
            ys, xs = np.where(keep & near32 if area_box else keep)
            if len(ys):
                tb = np.zeros_like(M)
                tb[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
                i2 = (tb & M).sum(); u2 = (tb | M).sum()
                rec['tightbox_iou_bbox'] = float(i2 / u2) if u2 else None
                rec['tightbox_area_ratio'] = float(tb.sum() / area_box) if area_box else None
        out[name] = rec
    # adaptive: outside-mask noise statistics
    out_vals = Db[~M] if (~M).any() else Db.ravel()
    T = float(max(8.0, np.percentile(out_vals, 99.5)))
    B = Db > T
    a = B.sum(); inter = (B & M).sum(); union = (B | M).sum()
    out['adapt'] = {'T': T, 'area_frac': float(a / D.size),
                    'iou_bbox': float(inter / union) if union else None,
                    'prec_in_bbox': float(inter / a) if a else None,
                    'bbox_cov': float(inter / area_box) if area_box else None,
                    'area_ratio_vs_bbox': float(a / area_box) if area_box else None}
    return out


def analyse(x):
    r = {k: x[k] for k in ('key', 'model', 'dataset', 'disease', 'type', 'real_image_path', 'mask_path')}
    try:
        fp = os.path.join(ROOT, x['image_path'])
        sp = os.path.join(ROOT, x['real_image_path'])
        mp = os.path.join(ROOT, x['mask_path']) if x['mask_path'] else None
        r['src_exists'] = os.path.exists(sp)
        r['mask_exists'] = bool(mp) and os.path.exists(mp)
        if not r['src_exists']:
            return r
        Fi = Image.open(fp); Si = Image.open(sp)
        r['f_size'] = Fi.size; r['s_size'] = Si.size
        r['f_q'] = jpeg_q_estimate(Fi); r['s_q'] = jpeg_q_estimate(Si)
        F = gray(Fi)
        Sg = Si.convert('L')
        r['size_match'] = Si.size == Fi.size
        if Si.size != Fi.size:
            Sg = Sg.resize(Fi.size, Image.BILINEAR)
        S = np.asarray(Sg, dtype=np.float32)
        if r['mask_exists']:
            Mi = Image.open(mp)
            r['m_size'] = Mi.size
            Ma = np.asarray(Mi.convert('L'))
            r['m_unique_n'] = int(len(np.unique(Ma)))
            if Mi.size != Fi.size:
                Ma = np.asarray(Mi.convert('L').resize(Fi.size, Image.NEAREST))
            M = Ma > 127
        else:
            M = np.zeros_like(F, dtype=bool)
        r['area_frac'] = float(M.mean())
        D = np.abs(F - S)
        E = D ** 2
        Et = E.sum()
        r['energy_in'] = float(E[M].sum() / Et) if Et > 0 else None
        r['l1_in'] = float(D[M].sum() / D.sum()) if D.sum() > 0 else None
        r['mad_in'] = float(D[M].mean()) if M.any() else None
        r['mad_out'] = float(D[~M].mean()) if (~M).any() else None
        r['med_out'] = float(np.median(D[~M])) if (~M).any() else None
        r['p95_out'] = float(np.percentile(D[~M], 95)) if (~M).any() else None
        r['frac_out_gt4'] = float((D[~M] > 4).mean()) if (~M).any() else None
        r['frac_out_gt16'] = float((D[~M] > 16).mean()) if (~M).any() else None
        r['frac_out_exact0'] = float((D[~M] == 0).mean()) if (~M).any() else None
        # outside-mask correlation (structure preserved?)
        if (~M).any():
            a = F[~M]; b = S[~M]
            a0 = a - a.mean(); b0 = b - b.mean()
            den = float(np.sqrt((a0 * a0).sum() * (b0 * b0).sum()))
            r['corr_out'] = float((a0 * b0).sum() / den) if den > 0 else None
            r['mean_shift_out'] = float((a - b).mean())
        # high-frequency residual outside mask (VAE fingerprint)
        LF = lap(F); LS = lap(S)
        if (~M).any():
            r['hf_out'] = float(np.abs(LF - LS)[~M].mean())
            r['hf_src_out'] = float(np.abs(LS)[~M].mean())
            r['hf_f_out'] = float(np.abs(LF)[~M].mean())
        # JPEG noise floor: re-encode S with F's quant tables (what a pure local paste-back + same save would give)
        try:
            buf = io.BytesIO()
            Sg.save(buf, format='JPEG', qtables=Fi.quantization, subsampling=0)
            buf.seek(0)
            S2 = np.asarray(Image.open(buf).convert('L'), dtype=np.float32)
            Dn = np.abs(S2 - S)
            r['noise_floor_out'] = float(Dn[~M].mean()) if (~M).any() else None
        except Exception as e:
            r['noise_floor_out'] = None
        # global geometric shift (phase correlation on 256px downsample)
        try:
            fs = np.asarray(Image.fromarray(F.astype(np.uint8)).resize((256, 256), Image.BILINEAR), dtype=np.float32)
            ss = np.asarray(Image.fromarray(S.astype(np.uint8)).resize((256, 256), Image.BILINEAR), dtype=np.float32)
            sh, err, _ = phase_cross_correlation(ss, fs, upsample_factor=4)
            r['shift_px1024'] = [float(sh[0] * 4), float(sh[1] * 4)]
        except Exception as e:
            r['shift_px1024'] = None
        if M.any():
            dist = ndi.distance_transform_cdt(~M, metric='chessboard')
            near32 = dist <= 32
            r['thr'] = thr_masks(D, M, near32)
            ring = near32 & ~M
            far = dist > 64
            r['mad_ring32'] = float(D[ring].mean()) if ring.any() else None
            r['mad_far64'] = float(D[far].mean()) if far.any() else None
    except Exception as e:
        r['error'] = repr(e)
    return r


def main():
    d = json.load(open(os.path.join(ROOT, 'dataset_mapping.json')))
    fk = [x for x in d['items'] if not x['image_path'].startswith('real/')]
    strata = collections.defaultdict(list)
    for x in fk:
        strata[(x['model'], x['type'], x['dataset'])].append(x)
    rng = random.Random(20261011)
    sample = []
    for k in sorted(strata):
        L = [x for x in strata[k] if os.path.exists(os.path.join(ROOT, x['real_image_path']))]
        sample += rng.sample(L, min(PER_STRATUM, len(L)))
    print('strata', len(strata), 'sample', len(sample), flush=True)
    with Pool(int(os.environ.get("NPROC","14"))) as p, open(OUT, 'w') as f:
        for i, r in enumerate(p.imap_unordered(analyse, sample, chunksize=4)):
            f.write(json.dumps(r) + '\n')
            if i % 200 == 0:
                print(i, flush=True)
    print('done', OUT)


if __name__ == '__main__':
    main()
