"""Test whether simple preprocessing removes the non-forensic shortcut.
Images are re-encoded IN MEMORY only (nothing written to disk except feature rows)."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import sys, io, json, random, numpy as np, cv2
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, _release_path('CIRCE/v3_audit/shortcut_work'))
import extract
W = _release_path('CIRCE/v3_audit/shortcut_work/')
GRAY_ORGANS = {'brain-mri', 'chest-xray', 'kidney'}

def recipe(im, organ, name):
    if name == 'R1_jpeg90':                       # re-encode only
        im = im.convert('RGB'); size = im.size
    elif name == 'R2_gray_jpeg90':                # + force true grayscale for gray modalities
        im = im.convert('L') if organ in GRAY_ORGANS else im.convert('RGB')
    elif name == 'R3_gray_resize_jpeg90':         # + non-integer downscale (breaks 8x8 grid / prior JPEG traces)
        im = im.convert('L') if organ in GRAY_ORGANS else im.convert('RGB')
        im = im.resize((720, 720), Image.BICUBIC)
    elif name == 'R4_gray_resize_png':            # lossless output instead of JPEG
        im = im.convert('L') if organ in GRAY_ORGANS else im.convert('RGB')
        im = im.resize((720, 720), Image.BICUBIC)
        b = io.BytesIO(); im.save(b, format='PNG'); return b.getvalue()
    b = io.BytesIO(); im.save(b, format='JPEG', quality=90, subsampling=0 if name != 'R1_jpeg90' else 2); return b.getvalue()

def feats(b):
    im = Image.open(io.BytesIO(b)); o = {'file_size': len(b), 'mode': im.mode, 'width': im.size[0], 'height': im.size[1]}
    if im.format == 'JPEG':
        o['jpeg_q'], o['jpeg_q_resid'] = extract.est_q(im.quantization[0])
    rgb = np.asarray(im.convert('RGB'))
    so, g = extract.gstats(rgb, 'n_'); o.update(so); o['n_block'] = extract.blockiness(g)
    o['thumb'] = [round(float(v), 2) for v in cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).ravel()]
    return o

def work(rec):
    out = []
    try:
        im0 = Image.open(rec['path']); im0.load()
        for name in ['R1_jpeg90', 'R2_gray_jpeg90', 'R3_gray_resize_jpeg90', 'R4_gray_resize_png']:
            o = feats(recipe(im0.copy(), rec['organ'], name)); o.update(path=rec['path'], recipe=name); out.append(o)
    except Exception as e:
        out.append({'path': rec['path'], 'err': repr(e)[:200]})
    return out

if __name__ == '__main__':
    recs = [json.loads(l) for l in open(W + 'sample.jsonl')]
    rng = random.Random(7); pick = []
    import collections
    by = collections.defaultdict(list)
    for r in recs:
        if r['corpus'] == 'MF90K' and 'strat' in r['tags']:
            by[(r['organ'], 'real' if r['label'] == 0 else r['gen'])].append(r)
        elif r['corpus'] == 'MF++add':
            by[(r['organ'], r['subsrc'])].append(r)
    for k, v in sorted(by.items()):
        n = 600 if k[1] == 'real' else (60 if k[0] in ('brain-mri', 'chest-xray', 'fundus') else 300)
        pick += rng.sample(v, min(n, len(v)))
    print('n images', len(pick), flush=True)
    with Pool(int(sys.argv[1])) as pool, open(W + 'features_norm.jsonl', 'w') as f:
        for i, outs in enumerate(pool.imap_unordered(work, pick, chunksize=8)):
            for o in outs: f.write(json.dumps(o) + '\n')
            if i % 1000 == 0: print(i, flush=True)
    print('done')
