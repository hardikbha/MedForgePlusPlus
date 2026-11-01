"""Is the forgery local? Compare each fake with its real source inside vs outside the GT bbox mask."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import sys, io, json, random, collections, numpy as np, cv2
from PIL import Image
from multiprocessing import Pool
cv2.setNumThreads(1)
MF = _release_path('AAAI/MedForge++/medforge90k_full/')
W = _release_path('CIRCE/v3_audit/shortcut_work/')
def gray(p, size=None):
    im = Image.open(p).convert('L')
    if size and im.size != size: im = im.resize(size, Image.BICUBIC)
    return np.asarray(im).astype(np.float32)
def work(x):
    try:
        f = gray(MF + x['image_path']); r = gray(MF + x['real_image_path'], (f.shape[1], f.shape[0]))
        m = np.asarray(Image.open(MF + x['mask_path']).convert('L').resize((f.shape[1], f.shape[0]), Image.NEAREST)) > 127
        dil = cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0   # exclude a 15px halo around the box
        d = np.abs(f - r); out = ~dil
        # compression-only reference: source re-encoded at q75
        b = io.BytesIO(); Image.fromarray(r.astype(np.uint8)).save(b, 'JPEG', quality=75); rc = np.asarray(Image.open(b)).astype(np.float32)
        return dict(organ=x['dataset'], gen=x['model'], type=x['type'], mask_frac=float(m.mean()),
                    in_mad=float(d[m].mean()) if m.any() else np.nan, out_mad=float(d[out].mean()) if out.any() else np.nan,
                    out_frac_gt8=float((d[out] > 8).mean()) if out.any() else np.nan,
                    out_corr=float(np.corrcoef(f[out], r[out])[0, 1]) if out.any() else np.nan,
                    ref_q75_mad=float(np.abs(rc - r)[out].mean()) if out.any() else np.nan)
    except Exception as e:
        return dict(err=repr(e)[:150], path=x['image_path'])
if __name__ == '__main__':
    items = json.load(open(MF + 'dataset_mapping.json'))['items']
    by = collections.defaultdict(list)
    for x in items:
        if not x['image_path'].startswith('real/') and x.get('mask_path'): by[(x['dataset'], x['model'], x['type'])].append(x)
    rng = random.Random(3); pick = []
    for k, v in sorted(by.items()): pick += rng.sample(v, min(40, len(v)))
    with Pool(int(sys.argv[1])) as p: out = p.map(work, pick, chunksize=8)
    json.dump(out, open(W + 'locality.json', 'w'))
    import pandas as pd
    d = pd.DataFrame([o for o in out if 'err' not in o]); print('errors', sum('err' in o for o in out))
    pd.set_option('display.width', 250)
    t = d.groupby(['gen']).agg(n=('organ', 'size'), mask_frac=('mask_frac', 'median'), in_mad=('in_mad', 'median'), out_mad=('out_mad', 'median'),
                               out_frac_gt8=('out_frac_gt8', 'median'), out_corr=('out_corr', 'median'), ref_q75_mad=('ref_q75_mad', 'median')).round(3)
    print(t.to_string()); t.to_csv(W + 'res_locality_by_gen.csv')
    t2 = d.groupby(['organ', 'type']).agg(n=('gen', 'size'), in_mad=('in_mad', 'median'), out_mad=('out_mad', 'median'), out_frac_gt8=('out_frac_gt8', 'median'),
                                          ref_q75_mad=('ref_q75_mad', 'median')).round(3)
    print(t2.to_string()); t2.to_csv(W + 'res_locality_by_organ_type.csv')
