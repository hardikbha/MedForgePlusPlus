"""Part 3: kidney removal deep-dive (vs polyp removal). Read-only on data.
Outputs labels_kidney.json + kidney_removal_montage.png"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, random, collections, re
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from multiprocessing import Pool

NA = _release_path('AAAI/MedForge++/naitik_annotated')
OUTD = _release_path('CIRCE/v3_audit')


def load_map(sub):
    fs = [f for f in os.listdir(f'{NA}/{sub}') if f.startswith('mapping_') and f != 'mapping_0_2.json']
    return json.load(open(f'{NA}/{sub}/{fs[0]}'))


def g(p):
    return np.asarray(Image.open(p).convert('L'), dtype=np.float32)


def hf_ratio(img, M, ring):
    L = np.abs(ndi.laplace(img))
    a = L[M].mean() if M.any() else np.nan
    b = L[ring].mean() if ring.any() else np.nan
    return float(a / b) if b > 0 else None


def one(args):
    kind, x, src = args
    r = {'kind': kind, 'key': x['key'], 'json_box': x.get('box')}
    try:
        F = g(x['image_path'])
        r['size'] = F.shape
        if not os.path.exists(src):
            r['src_missing'] = True
            return r
        S = g(src)
        if S.shape != F.shape:
            r['shape_mismatch'] = [S.shape, F.shape]
            return r
        M = np.asarray(Image.open(x['mask_path']).convert('L')) > 127 if x.get('mask_path') and os.path.exists(x['mask_path']) else np.zeros_like(F, bool)
        r['mask_area_frac'] = float(M.mean())
        r['mask_px'] = int(M.sum())
        D = np.abs(F - S)
        r['changed_px'] = int((D > 0).sum())
        r['changed_px_gt8'] = int((D > 8).sum())
        r['changed_frac'] = float((D > 0).mean())
        Et = (D ** 2).sum()
        r['energy_in'] = float((D[M] ** 2).sum() / Et) if Et > 0 else None
        r['mad_in'] = float(D[M].mean()) if M.any() else None
        r['max_in'] = float(D[M].max()) if M.any() else None
        r['mad_out'] = float(D[~M].mean())
        r['frac_out_exact0'] = float((D[~M] == 0).mean())
        # changed pixels outside the box (dilated by 4 px)
        Md = ndi.binary_dilation(M, iterations=4) if M.any() else M
        r['changed_outside_box4'] = int(((D > 2) & ~Md).sum())
        # bounding box of actual change
        ys, xs = np.where(D > 8)
        if len(ys):
            r['change_bbox'] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        if M.any():
            dist = ndi.distance_transform_cdt(~M, metric='chessboard')
            ring = (dist > 0) & (dist <= 8)
            r['hf_ratio_forged'] = hf_ratio(F, M, ring)
            r['hf_ratio_source'] = hf_ratio(S, M, ring)
            # mean intensity contrast inside vs ring (lesion conspicuity)
            r['contrast_forged'] = float(F[M].mean() - F[ring].mean())
            r['contrast_source'] = float(S[M].mean() - S[ring].mean())
            r['std_in_forged'] = float(F[M].std()); r['std_in_source'] = float(S[M].std())
            r['std_ring_forged'] = float(F[ring].std()); r['std_ring_source'] = float(S[ring].std())
    except Exception as e:
        r['error'] = repr(e)
    return r


def auc(pos, neg):
    pos = np.asarray([p for p in pos if p is not None and np.isfinite(p)])
    neg = np.asarray([p for p in neg if p is not None and np.isfinite(p)])
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    # tie handling via scipy
    from scipy.stats import rankdata
    ranks = rankdata(allv)
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    kr = load_map('kidney_removal'); kt = {x['key']: x for x in load_map('kidney_real_tumor')}
    pr = load_map('polyp_removal'); pt = {x['key']: x for x in load_map('polyp_real')}
    jobs = []
    for x in kr:
        jobs.append(('kidney_removal', x, kt[x['key']]['image_path'] if x['key'] in kt else f'{NA}/kidney_real_tumor/images/{x["key"]}'))
    for x in pr:
        jobs.append(('polyp_removal', x, pt[x['key']]['image_path'] if x['key'] in pt else f'{NA}/polyp_real/images/{x["key"]}'))
    if int(os.environ.get('NPROC', '1')) <= 1:
        res = [one(j) for j in jobs]
    else:
        with Pool(int(os.environ['NPROC'])) as p:
            res = p.map(one, jobs, chunksize=16)
    json.dump(res, open(f'{OUTD}/labels_kidney_pairs.json', 'w'))
    summ = {}
    for kind in ['kidney_removal', 'polyp_removal']:
        R = [r for r in res if r['kind'] == kind and 'mad_in' in r and r.get('mad_in') is not None]
        summ[kind] = {'n_pairs_analysed': len(R),
                      'n_src_missing': sum(1 for r in res if r['kind'] == kind and r.get('src_missing')),
                      'n_errors': sum(1 for r in res if r['kind'] == kind and r.get('error'))}
        for f in ['mask_area_frac', 'mask_px', 'changed_px', 'changed_px_gt8', 'energy_in', 'mad_in', 'max_in', 'mad_out',
                  'frac_out_exact0', 'changed_outside_box4', 'hf_ratio_forged', 'hf_ratio_source', 'contrast_forged',
                  'contrast_source']:
            v = np.array([r[f] for r in R if r.get(f) is not None], dtype=float)
            if len(v):
                summ[kind][f] = {'median': float(np.median(v)), 'mean': float(v.mean()), 'p10': float(np.percentile(v, 10)),
                                 'p90': float(np.percentile(v, 90))}
        summ[kind]['frac_identical_to_source'] = float(np.mean([r['changed_px'] == 0 for r in R])) if R else None
        summ[kind]['frac_mad_in_lt5'] = float(np.mean([r['mad_in'] < 5 for r in R]))
        summ[kind]['frac_mask_lt_0.5pct'] = float(np.mean([r['mask_area_frac'] < 0.005 for r in R]))
        # oracle-location separability: HF-ratio and contrast, removal vs its own source at the same box
        summ[kind]['auc_hfratio_removal_lower_than_source'] = auc([-r['hf_ratio_forged'] for r in R if r.get('hf_ratio_forged') is not None],
                                                               [-r['hf_ratio_source'] for r in R if r.get('hf_ratio_source') is not None])
        summ[kind]['auc_abs_contrast_source_vs_removal'] = auc([abs(r['contrast_source']) for r in R if r.get('contrast_source') is not None],
                                                            [abs(r['contrast_forged']) for r in R if r.get('contrast_forged') is not None])
        summ[kind]['auc_std_in_source_vs_removal'] = auc([r['std_in_source'] for r in R if r.get('std_in_source') is not None],
                                                      [r['std_in_forged'] for r in R if r.get('std_in_forged') is not None])
    # case-id overlap kidney normal vs tumor
    case = lambda k: re.match(r'(case_\d+)', k).group(1) if re.match(r'(case_\d+)', k) else None
    kn = [x['key'] for x in load_map('kidney_normal')]
    cn = set(map(case, kn)); ct = set(map(case, kt.keys()))
    summ['kidney_case_overlap_normal_vs_tumor'] = {'normal_cases': len(cn), 'tumor_cases': len(ct), 'overlap': len(cn & ct)}
    json.dump(summ, open(f'{OUTD}/labels_kidney.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))

    # ---------- montage ----------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = random.Random(11)
    Rk = [r for r in res if r['kind'] == 'kidney_removal' and r.get('mad_in') is not None]
    Rp = [r for r in res if r['kind'] == 'polyp_removal' and r.get('mad_in') is not None]
    pick = rng.sample(Rk, 12) + rng.sample(Rp, 6)
    xmap = {(j[0], j[1]['key']): j for j in jobs}
    fig, axes = plt.subplots(len(pick), 5, figsize=(15, 3.0 * len(pick)))
    for i, r in enumerate(pick):
        kind, x, src = xmap[(r['kind'], r['key'])]
        F = g(x['image_path']); S = g(src)
        M = np.asarray(Image.open(x['mask_path']).convert('L')) > 127
        D = np.abs(F - S)
        ys, xs = np.where(M)
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        half = max(48, int(1.5 * max(ys.max() - ys.min(), xs.max() - xs.min())))
        y0, y1 = max(0, cy - half), min(F.shape[0], cy + half); x0, x1 = max(0, cx - half), min(F.shape[1], cx + half)
        Fc = np.asarray(Image.open(x['image_path']).convert('RGB'))
        Sc = np.asarray(Image.open(src).convert('RGB'))
        ov = Fc.copy(); edge = M ^ ndi.binary_erosion(M, iterations=3)
        ov[edge] = [255, 0, 0]
        axes[i, 0].imshow(ov); axes[i, 0].set_title(f'{kind}: forged + GT box\n{r["key"][:28]}', fontsize=8)
        axes[i, 1].imshow(Sc[y0:y1, x0:x1]); axes[i, 1].set_title('source crop (real tumor/polyp)', fontsize=8)
        axes[i, 2].imshow(Fc[y0:y1, x0:x1]); axes[i, 2].set_title('forged crop (removal)', fontsize=8)
        axes[i, 3].imshow(M[y0:y1, x0:x1], cmap='gray'); axes[i, 3].set_title(f'mask crop ({100*M.mean():.3f}% of frame)', fontsize=8)
        axes[i, 4].imshow(D[y0:y1, x0:x1], cmap='magma', vmin=0, vmax=max(10, float(D.max()))); axes[i, 4].set_title(f'|forged-source| crop, mad_in={r["mad_in"]:.1f}', fontsize=8)
        for a in axes[i]:
            a.axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUTD}/kidney_removal_montage.png', dpi=70)
    print('montage saved')


if __name__ == '__main__':
    main()
