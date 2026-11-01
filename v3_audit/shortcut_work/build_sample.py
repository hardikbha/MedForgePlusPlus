"""Build the stratified sample list for the shortcut audit (read-only on data)."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, re, random, collections
R = _release_path('AAAI/MedForge++')
MF = R + '/medforge90k_full/'
OUT = _release_path('CIRCE/v3_audit/shortcut_work/sample.jsonl')
rng = random.Random(0)
items = json.load(open(MF + 'dataset_mapping.json'))['items']
srcs = set(x['real_image_path'] for x in items if not x['image_path'].startswith('real/'))

def subsrc(p):
    org = p.split('/')[1]; b = os.path.basename(p)
    if org == 'fundus':
        return 'eyepacs' if 'eyepacs' in b else ('odir' if re.match(r'^\d+_(left|right)', b) else 'fundus_other')
    if org == 'brain-mri':
        return 'kaggle_bt' if b[:3] in ('Tr-', 'Te-') else ('YG_metastases' if b.startswith('YG_') else 'brain_other')
    return 'mimic' if b.startswith('s') else 'cxr_other'

recs = {}
def add(path, **kw):
    if path in recs:
        recs[path]['tags'] = sorted(set(recs[path]['tags']) | set(kw.pop('tags', [])))
        return
    kw['path'] = path; recs[path] = kw

def mf_rec(x, tag):
    real = x['image_path'].startswith('real/')
    src = x['image_path'] if real else x['real_image_path']
    add(MF + x['image_path'], corpus='MF90K', organ=x['dataset'], label=0 if real else 1,
        cls='real' if real else x['type'], gen='real' if real else x['model'],
        group='MF:' + src, subsrc=subsrc(src), is_fake_source=(x['image_path'] in srcs) if real else None,
        tags=[tag])

# 1) stratified MF90K sample
by = collections.defaultdict(list)
for x in items:
    real = x['image_path'].startswith('real/')
    by[(x['dataset'], 'real' if real else x['type'], 'real' if real else x['model'])].append(x)
for k, v in sorted(by.items()):
    n = 2400 if k[1] == 'real' else 200
    for x in rng.sample(v, min(n, len(v))):
        mf_rec(x, 'strat')
# extra: ALL reals that were used as fake sources in a 600/organ draw (source-matched control)
src_items = [x for x in items if x['image_path'] in srcs]
bo = collections.defaultdict(list)
for x in src_items: bo[x['dataset']].append(x)
for o, v in bo.items():
    for x in rng.sample(v, min(800, len(v))): mf_rec(x, 'srcreal')

# 2) protocol files
key2item = {MF + x['image_path']: x for x in items}
def load_paths(fn):
    out = []
    for line in open(R + '/' + fn):
        r = json.loads(line); p = (r.get('images') or [r.get('image')])[0]; out.append(p)
    return out
test = load_paths('mfp_v2_test_shared.jsonl')
for p in test: mf_rec(key2item[p], 'test')
for fn, tag in [('mfp_train_crossforgery.jsonl', 'tr_cf'), ('mfp_v2_train_crossforgery_strict.jsonl', 'tr_cfs'),
                ('mfp_train_crossmodel.jsonl', 'tr_cm'), ('mfp_train_indomain.jsonl', 'tr_id')]:
    ps = load_paths(fn); r2 = random.Random({'tr_cf': 11, 'tr_cfs': 12, 'tr_cm': 13, 'tr_id': 14}[tag])
    for p in r2.sample(ps, 5000): mf_rec(key2item[p], tag)

# 3) MedForge++ addition (all) + provenance references
NA = R + '/naitik_annotated/'
def case_group(sub, fn):
    if sub.startswith('kidney'):
        m = re.match(r'(case_\d+)_slice_(\d+)', fn); return 'KiTS:' + m.group(1)
    if sub in ('polyp_real', 'polyp_removal'): return 'KV:' + os.path.splitext(fn)[0]
    return 'PN:' + fn
for sub in ['kidney_normal', 'kidney_real_tumor', 'kidney_removal', 'polyp_normal', 'polyp_real', 'polyp_removal']:
    d = NA + sub + '/images/'
    for fn in sorted(os.listdir(d)):
        organ = 'kidney' if sub.startswith('kidney') else 'colon'
        cls = 'remove' if sub.endswith('removal') else 'real'
        add(d + fn, corpus='MF++add', organ=organ, label=1 if cls != 'real' else 0, cls=cls, gen='lama' if cls == 'remove' else 'real',
            group=case_group(sub, fn), subsrc=sub, is_fake_source=None, tags=['add'])
for fn in sorted(os.listdir(R + '/colon_injections/images')):
    stem = fn.replace('_polypinj.jpg', '')
    add(R + '/colon_injections/images/' + fn, corpus='MF++add', organ='colon', label=1, cls='edit', gen='flux.1-kontext',
        group='KV:' + stem, subsrc='colon_inject', is_fake_source=None, tags=['add'])
KI = _release_path('AAAI/kidney/flux_injected/images/')
for fn in sorted(os.listdir(KI)):
    add(KI + fn, corpus='MF++add', organ='kidney', label=1, cls='edit', gen='flux.1-kontext',
        group='KN:' + fn.replace('_tumorinj.jpg', ''), subsrc='kidney_inject', is_fake_source=None, tags=['add'])
# references (not part of the 7,800): source pools of the implants
KN = _release_path('AAAI/kidney/normal/')
for fn in sorted(os.listdir(KN)):
    add(KN + fn, corpus='ref', organ='kidney', label=0, cls='real', gen='real', group='KN:' + fn.replace(' ', '_').replace('.jpg', ''),
        subsrc='ref_kidney_inject_source', is_fake_source=True, tags=['ref'])
KV = _release_path('AAAI/colon/kvasir_extracted/Kvasir-SEG/images/')
for fn in sorted(os.listdir(KV)):
    add(KV + fn, corpus='ref', organ='colon', label=0, cls='real', gen='real', group='KV:' + os.path.splitext(fn)[0],
        subsrc='ref_kvasir_original', is_fake_source=True, tags=['ref'])

with open(OUT, 'w') as f:
    for r in recs.values(): f.write(json.dumps(r) + '\n')
c = collections.Counter((r['corpus'], r['organ'], r['cls']) for r in recs.values())
for k, v in sorted(c.items()): print(k, v)
print('total', len(recs))
ct = collections.Counter(t for r in recs.values() for t in r['tags']); print(ct)
