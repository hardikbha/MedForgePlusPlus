"""MF++ addition: implant locality (kidney FLUX, colon FLUX) + per-subset global
sharpness / format shortcut. Read-only. Writes labels_mfpp_implant.json"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, glob, random, collections
import numpy as np
import cv2
from PIL import Image
from scipy.stats import rankdata
from multiprocessing import Pool

NA = _release_path('AAAI/MedForge++/naitik_annotated')
KID_INJ = _release_path('AAAI/kidney/flux_injected')
KID_SRC = _release_path('AAAI/kidney/normal')
COL_INJ = _release_path('AAAI/MedForge++/colon_injections')
KVA = _release_path('AAAI/colon/kvasir_extracted/Kvasir-SEG/images')
OUT = _release_path('CIRCE/v3_audit/labels_mfpp_implant.json')
rng = random.Random(5)
NS = int(os.environ.get('NS', '150'))


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def sharp(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    g = g.astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    # fraction of spectral energy above 0.25 cycles/px (half-Nyquist) -> 512-upsampled images lose this
    F = np.abs(np.fft.rfft2(g - g.mean())) ** 2
    h, w = g.shape
    fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.rfftfreq(w)[None, :]
    rr = np.sqrt(fy ** 2 + fx ** 2)
    hf = float(F[rr > 0.25].sum() / F.sum())
    return {'lap_mean': float(lap.mean()), 'hf_frac_gt_quarter': hf, 'shape': g.shape}


def implant_pair(img, src, box):
    F = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    S0 = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if F is None or S0 is None:
        return None
    S = cv2.resize(S0, (F.shape[1], F.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    F = F.astype(np.float32)
    M = np.zeros_like(F, bool)
    x1, y1, x2, y2 = box
    M[max(0, y1):y2, max(0, x1):x2] = True
    D = np.abs(F - S)
    E = D ** 2
    return {'area_frac': float(M.mean()), 'energy_in': float(E[M].sum() / E.sum()), 'mad_in': float(D[M].mean()),
            'mad_out': float(D[~M].mean()), 'frac_out_gt4': float((D[~M] > 4).mean()),
            'src_size': list(S0.shape)}


def summarise(rs):
    out = {'n': len(rs)}
    for k in ['area_frac', 'energy_in', 'mad_in', 'mad_out', 'frac_out_gt4']:
        v = np.array([r[k] for r in rs])
        out[k] = {'median': float(np.median(v)), 'mean': float(v.mean())}
    return out


def main():
    res = {}
    # kidney implants
    kid = []
    for f in sorted(glob.glob(f'{KID_INJ}/manifest_*.jsonl')):
        kid += [json.loads(l) for l in open(f)]
    seen = {}
    for r in kid:
        seen[r['images'][0]] = r
    kid = list(seen.values())
    res['kidney_inject_manifest_unique'] = len(kid)
    ks = rng.sample(kid, min(NS, len(kid)))
    kr = []
    for r in ks:
        stem = os.path.basename(r['images'][0]).replace('_tumorinj.jpg', '').replace('_', ' ')
        src = f'{KID_SRC}/{stem}.jpg'
        p = implant_pair(r['images'][0], src, r['box'])
        if p:
            kr.append(p)
    res['kidney_inject_vs_source'] = summarise(kr)
    res['kidney_src_sizes'] = collections.Counter(tuple(r['src_size']) for r in kr).most_common(5)
    # colon implants
    col = [json.loads(l) for l in open(f'{COL_INJ}/annotations.jsonl')]
    cs = rng.sample(col, min(NS, len(col)))
    cr = []
    for r in cs:
        img = r['images'][0]
        stem = os.path.basename(img).replace('_polypinj.jpg', '')
        p = implant_pair(img, f'{KVA}/{stem}.jpg', r['box'])
        if p:
            cr.append(p)
    res['colon_inject_vs_source'] = summarise(cr)
    res['colon_src_sizes'] = collections.Counter(tuple(r['src_size']) for r in cr).most_common(5)
    # box statistics for implant sets (boxes from json, 1024 canvas)
    for nm, L in [('kidney_inject', kid), ('colon_inject', col)]:
        a = np.array([(b[2] - b[0]) * (b[3] - b[1]) / 1024.0 ** 2 for b in (r['box'] for r in L)])
        oob = sum(1 for r in L if min(r['box']) < 0 or max(r['box']) > 1024)
        res[f'{nm}_box_stats'] = {'n': len(L), 'area_pct_p10': float(100 * np.percentile(a, 10)), 'area_pct_median': float(100 * np.median(a)),
                                  'area_pct_p90': float(100 * np.percentile(a, 90)), 'gt50pct': float(np.mean(a > 0.5)),
                                  'lt0.5pct': float(np.mean(a < 0.005)), 'out_of_bounds': oob}
    q = collections.Counter(r.get('quality') for r in col)
    res['colon_quality_field'] = dict(q)

    # per-subset sharpness / format
    subsets = {
        'kidney_normal': sorted(glob.glob(f'{NA}/kidney_normal/images/*')),
        'kidney_real_tumor': sorted(glob.glob(f'{NA}/kidney_real_tumor/images/*')),
        'kidney_removal': sorted(glob.glob(f'{NA}/kidney_removal/images/*')),
        'kidney_inject': sorted(glob.glob(f'{KID_INJ}/images/*')),
        'polyp_normal': sorted(glob.glob(f'{NA}/polyp_normal/images/*')),
        'polyp_real': sorted(glob.glob(f'{NA}/polyp_real/images/*')),
        'polyp_removal': sorted(glob.glob(f'{NA}/polyp_removal/images/*')),
        'colon_inject': sorted(glob.glob(f'{COL_INJ}/images/*')),
    }
    sh = {}
    for k, L in subsets.items():
        smp = rng.sample(L, min(NS, len(L)))
        with Pool(12) as pool:
            vals = pool.map(sharp, smp, chunksize=4)
        vals = [v for v in vals if v]
        ext = collections.Counter(os.path.splitext(p)[1].lower() for p in L)
        sh[k] = vals
        res.setdefault('subset_stats', {})[k] = {
            'n_files': len(L), 'ext': dict(ext), 'n_sampled': len(vals),
            'lap_mean_median': float(np.median([v['lap_mean'] for v in vals])),
            'hf_frac_median': float(np.median([v['hf_frac_gt_quarter'] for v in vals]))}
    pairs = [('kidney_inject', 'kidney_normal'), ('kidney_inject', 'kidney_real_tumor'), ('kidney_removal', 'kidney_real_tumor'),
             ('kidney_removal', 'kidney_normal'), ('colon_inject', 'polyp_real'), ('colon_inject', 'polyp_normal'),
             ('polyp_removal', 'polyp_real'), ('polyp_real', 'polyp_normal')]
    res['single_scalar_auc'] = {}
    for a, b in pairs:
        for feat in ['lap_mean', 'hf_frac_gt_quarter']:
            A = [v[feat] for v in sh[a]]; B = [v[feat] for v in sh[b]]
            u = auc(A, B)
            res['single_scalar_auc'][f'{a}_vs_{b}:{feat}'] = round(max(u, 1 - u), 4)
    json.dump(res, open(OUT, 'w'), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
