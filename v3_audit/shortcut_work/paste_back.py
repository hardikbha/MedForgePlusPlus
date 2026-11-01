"""Simulate the recommended fix: paste the generated region back into the real source (pixels outside the
GT box are taken from the real image), then encode real and fake identically. In memory only."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import sys, io, json, random, collections, numpy as np, cv2
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, _release_path('CIRCE/v3_audit/shortcut_work'))
import extract
MF = _release_path('AAAI/MedForge++/medforge90k_full/')
W = _release_path('CIRCE/v3_audit/shortcut_work/')
GRAY = {'brain-mri', 'chest-xray'}
def enc(arr, organ):
    im = Image.fromarray(arr).convert('L' if organ in GRAY else 'RGB')
    b = io.BytesIO(); im.save(b, 'JPEG', quality=90, subsampling=0); return b.getvalue()
def feats(b):
    rgb = np.asarray(Image.open(io.BytesIO(b)).convert('RGB'))
    so, g = extract.gstats(rgb, 'n_'); so['n_block'] = extract.blockiness(g); so['file_size'] = len(b)
    so['thumb'] = [round(float(v), 2) for v in cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).ravel()]
    return so
def work(x):
    try:
        f = np.asarray(Image.open(MF + x['image_path']).convert('RGB'))
        r = np.asarray(Image.open(MF + x['real_image_path']).convert('RGB').resize((f.shape[1], f.shape[0]), Image.BICUBIC))
        m = np.asarray(Image.open(MF + x['mask_path']).convert('L').resize((f.shape[1], f.shape[0]), Image.NEAREST)) > 127
        m3 = m[..., None]
        pasted = np.where(m3, f, r).astype(np.uint8)
        # also: generator output kept but converted to gray first for gray organs (colour cast removed) - control
        out = []
        for variant, arr in [('real_source', r), ('fake_full_frame', f), ('fake_pasted_back', pasted)]:
            o = feats(enc(arr, x['dataset'])); o.update(variant=variant, organ=x['dataset'], gen=x['model'], type=x['type'],
                                                       group=x['real_image_path'], fake=x['image_path'], mask_frac=float(m.mean()))
            out.append(o)
        return out
    except Exception as e:
        return [{'err': repr(e)[:150]}]
if __name__ == '__main__':
    items = json.load(open(MF + 'dataset_mapping.json'))['items']
    by = collections.defaultdict(list)
    for x in items:
        if not x['image_path'].startswith('real/') and x.get('mask_path'): by[(x['dataset'], x['model'], x['type'])].append(x)
    rng = random.Random(11); pick = []
    for k, v in sorted(by.items()): pick += rng.sample(v, min(30, len(v)))
    with Pool(int(sys.argv[1])) as p, open(W + 'features_pasteback.jsonl', 'w') as fo:
        for outs in p.imap_unordered(work, pick, chunksize=8):
            for o in outs: fo.write(json.dumps(o) + '\n')
    print('done', len(pick))
