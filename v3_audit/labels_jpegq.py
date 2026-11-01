"""JPEG quantisation-table census over ALL MedForge-90K images (header only, no decode)
+ MF++ addition files. Writes labels_jpegq.json"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os, json, collections, glob, hashlib
from multiprocessing import Pool
from PIL import Image

ROOT = _release_path('AAAI/MedForge++/medforge90k_full')


def qinfo(p):
    try:
        im = Image.open(p)
        fmt = im.format
        q = getattr(im, 'quantization', None)
        if q:
            luma = tuple(q[0])
            s = sum(luma)
            key = hashlib.md5(repr(sorted(q.items())).encode()).hexdigest()[:8]
            # IJG quality estimate from sum of luma table vs standard (sum 3472)
            sc = s / 3472 * 100
            qual = (200 - sc) / 2 if sc <= 100 else 5000 / sc
            sub = None
            try:
                from PIL import JpegImagePlugin
                sub = JpegImagePlugin.get_sampling(im)
            except Exception:
                pass
            return fmt, round(qual), key, sub
        return fmt, None, None, None
    except Exception as e:
        return 'ERR', None, None, None


def job(x):
    return (x[0], x[1]) + qinfo(x[2])


def main():
    d = json.load(open(os.path.join(ROOT, 'dataset_mapping.json')))
    jobs = []
    for it in d['items']:
        lab = 'real' if it['type'] == 'real' else it['model']
        jobs.append((lab, it['dataset'], os.path.join(ROOT, it['image_path'])))
    PP = _release_path('AAAI/MedForge++')
    for sub in ['kidney_normal', 'kidney_real_tumor', 'kidney_removal', 'polyp_normal', 'polyp_real', 'polyp_removal']:
        for p in glob.glob(f'{PP}/naitik_annotated/{sub}/images/*'):
            jobs.append((sub, 'mfpp', p))
    for p in glob.glob(f'{PP}/colon_injections/images/*'):
        jobs.append(('colon_inject', 'mfpp', p))
    for p in glob.glob(_release_path('AAAI/kidney/flux_injected/images/*')):
        jobs.append(('kidney_inject', 'mfpp', p))
    with Pool(int(os.environ.get('NPROC', '8'))) as pool:
        res = pool.map(job, jobs, chunksize=256)
    c = collections.Counter((r[0], r[2], r[3], r[5]) for r in res)
    out = collections.defaultdict(dict)
    for (lab, fmt, q, sub), n in c.items():
        out[lab][f'{fmt}|q{q}|sub{sub}'] = n
    bykey = collections.Counter((r[0], r[4]) for r in res)
    tables = collections.defaultdict(dict)
    for (lab, k), n in bykey.items():
        tables[lab][str(k)] = n
    json.dump({'by_label_format_quality': out, 'by_label_qtable_hash': tables}, open(_release_path('CIRCE/v3_audit/labels_jpegq.json'), 'w'), indent=1)
    for lab in sorted(out):
        print(lab, dict(sorted(out[lab].items(), key=lambda kv: -kv[1])))
    # organ x real breakdown
    rc = collections.Counter((r[1], r[3]) for r in res if r[0] == 'real')
    print('real by organ/quality', sorted(rc.items()))


if __name__ == '__main__':
    main()
