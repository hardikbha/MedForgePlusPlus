"""Locality check for the MF++ addition: paired images (same slice / same Kvasir frame)."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os, json, numpy as np, cv2, pandas as pd
from PIL import Image
NA = _release_path('AAAI/MedForge++/naitik_annotated/')
CI = _release_path('AAAI/MedForge++/colon_injections/images/')
W = _release_path('CIRCE/v3_audit/shortcut_work/')
def g(p): return np.asarray(Image.open(p).convert('L')).astype(np.float32)
def box_mask(shape, box):
    m = np.zeros(shape, bool)
    if box: x1, y1, x2, y2 = map(int, box); m[max(0, y1):y2, max(0, x1):x2] = True
    return m
rows = []
def pairs(sub_a, sub_b, mapping_sub, n=150, suffix_b=None, ann=None):
    mp = json.load(open(NA + mapping_sub + '/' + [f for f in os.listdir(NA + mapping_sub) if f.startswith('mapping_0_1000')][0]))
    boxes = {x['key']: x.get('box') for x in mp}
    keys = sorted(boxes)[:n] if ann is None else sorted(ann)[:n]
    for k in keys:
        pa = NA + sub_a + '/images/' + k
        pb = (NA + sub_b + '/images/' + k) if suffix_b is None else (CI + k.replace('.png', suffix_b))
        if not (os.path.exists(pa) and os.path.exists(pb)): continue
        a, b = g(pa), g(pb)
        if a.shape != b.shape: b = cv2.resize(b, (a.shape[1], a.shape[0]))
        box = boxes.get(k) if ann is None else ann[k]
        m = box_mask(a.shape, box); dil = cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
        d = np.abs(a - b); out = ~dil
        rows.append(dict(pair=f'{sub_a} vs {sub_b}', key=k, box_frac=float(m.mean()), in_mad=float(d[m].mean()) if m.any() else np.nan,
                         out_mad=float(d[out].mean()), out_frac_gt8=float((d[out] > 8).mean()), identical=bool((d == 0).all())))
pairs('kidney_real_tumor', 'kidney_removal', 'kidney_removal')
pairs('polyp_real', 'polyp_removal', 'polyp_removal')
ann = {}
for l in open(_release_path('AAAI/MedForge++/colon_injections/annotations.jsonl')):
    r = json.loads(l); stem = os.path.basename(r['images'][0]).replace('_polypinj.jpg', ''); ann[stem + '.png'] = r['box']
pairs('polyp_real', 'colon_inject', 'polyp_real', suffix_b='_polypinj.jpg', ann=ann)
d = pd.DataFrame(rows); d.to_csv(W + 'res_locality_addition_rows.csv', index=False)
t = d.groupby('pair').agg(n=('key', 'size'), box_frac=('box_frac', 'median'), in_mad=('in_mad', 'median'), out_mad=('out_mad', 'median'),
                          out_frac_gt8=('out_frac_gt8', 'median'), identical=('identical', 'mean')).round(3)
print(t.to_string()); t.to_csv(W + 'res_locality_addition.csv')
