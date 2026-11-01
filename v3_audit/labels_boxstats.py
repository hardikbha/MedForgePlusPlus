"""Box/mask statistics for all MedForge-90K forgery masks + MedForge++ addition.
Writes ~/CIRCE/v3_audit/labels_boxstats.jsonl (one record per mask)."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, glob, sys
import numpy as np
from PIL import Image
from multiprocessing import Pool
from scipy import ndimage as ndi

ROOT = _release_path('AAAI/MedForge++/medforge90k_full')
PP = _release_path('AAAI/MedForge++')
OUT = _release_path('CIRCE/v3_audit/labels_boxstats.jsonl')


def stat_mask(args):
    meta, mp, imgp = args
    r = dict(meta)
    r['mask_path'] = mp
    if mp is None or not os.path.exists(mp):
        r['missing'] = True
        return r
    try:
        M = Image.open(mp)
        r['m_size'] = M.size
        a = np.asarray(M.convert('L'))
        u = np.unique(a)
        r['n_unique'] = int(len(u))
        B = a > 127
        r['area_frac'] = float(B.mean())
        r['empty'] = bool(not B.any())
        if B.any():
            lab, n = ndi.label(B)
            r['n_comp'] = int(n)
            ys, xs = np.where(B)
            r['bbox'] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            bw = (xs.max() - xs.min() + 1); bh = (ys.max() - ys.min() + 1)
            r['bbox_area_frac'] = float(bw * bh / B.size)
            r['fill_of_bbox'] = float(B.sum() / (bw * bh))  # 1.0 => pure rectangle
        if imgp and os.path.exists(imgp):
            try:
                r['img_size'] = Image.open(imgp).size
            except Exception:
                pass
    except Exception as e:
        r['error'] = repr(e)
    return r


def main():
    jobs = []
    d = json.load(open(os.path.join(ROOT, 'dataset_mapping.json')))
    for x in d['items']:
        if x['image_path'].startswith('real/'):
            continue
        meta = {'src': 'MF90K', 'model': x['model'], 'organ': x['dataset'], 'type': x['type'], 'key': x['key']}
        jobs.append((meta, os.path.join(ROOT, x['mask_path']) if x['mask_path'] else None,
                     None))
    # naitik annotated (mask files)
    for sub in ['kidney_real_tumor', 'kidney_removal', 'polyp_real', 'polyp_removal']:
        for mf in glob.glob(f'{PP}/naitik_annotated/{sub}/mapping_*.json'):
            if mf.endswith('mapping_0_2.json'):
                continue
            for x in json.load(open(mf)):
                meta = {'src': 'MF++', 'model': 'naitik', 'organ': sub.split('_')[0], 'type': sub, 'key': x['key'],
                        'json_box': x.get('box')}
                jobs.append((meta, x.get('mask_path'), x.get('image_path')))
    print('jobs', len(jobs), flush=True)
    with Pool(int(os.environ.get("NPROC","14"))) as p, open(OUT, 'w') as f:
        for i, r in enumerate(p.imap_unordered(stat_mask, jobs, chunksize=32)):
            f.write(json.dumps(r) + '\n')
            if i % 5000 == 0:
                print(i, flush=True)
    print('done')


if __name__ == '__main__':
    main()
