
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, _release_path('CIRCE/v3_audit/shortcut_work'))
from common import *
d = load(); d = d[d['ok'] == 1].reset_index(drop=True)
meta_cols = ['path', 'corpus', 'organ', 'label', 'cls', 'gen', 'group', 'subsrc', 'md5']
rows = [json.loads(l) for l in open(W + 'features_norm.jsonl')]
n = pd.DataFrame([r for r in rows if 'err' not in r]); print('norm rows', len(n), 'errors', sum('err' in r for r in rows))
n['log_n_lapvar'] = np.log10(n['n_lapvar'] + 1e-3); n['log_fsize'] = np.log10(n['file_size'])
n['bpp'] = n['file_size'] / (n['width'] * n['height']); n['is_L'] = (n['mode'] == 'L').astype(float)
n = n.merge(d[meta_cols], on='path', how='inner')
paths = set(n.path)
r0 = d[d.path.isin(paths)].copy(); r0['recipe'] = 'R0_as_is'; r0['is_L'] = (r0['mode'] == 'L').astype(float)
allr = pd.concat([r0, n], ignore_index=True)
FILE = ['log_fsize', 'bpp', 'jpeg_q', 'is_L']
FEATS = {'file(size,q,mode)': FILE, 'global(ext)': GX, 'file+global': FILE + GX}
def design2(sub, cols): return sub[cols].astype(float).values
COMPS = []
for o in ['brain-mri', 'chest-xray', 'fundus']:
    COMPS.append((f'MF90K {o}: real vs fake', lambda s, o=o: s[(s.corpus == 'MF90K') & (s.organ == o)], None))
COMPS += [('ADD kidney: real vs implant', lambda s: s[s.subsrc.isin(['kidney_normal', 'kidney_real_tumor', 'kidney_inject'])], None),
          ('ADD kidney: real_tumor vs removal', lambda s: s[s.subsrc.isin(['kidney_real_tumor', 'kidney_removal'])], None),
          ('ADD colon: polyp_real vs implant', lambda s: s[s.subsrc.isin(['polyp_real', 'colon_inject'])], None),
          ('ADD colon: polyp_real vs removal', lambda s: s[s.subsrc.isin(['polyp_real', 'polyp_removal'])], None),
          ('ADD colon: polyp_real vs polyp_normal (both real; label=normal)', lambda s: s[s.subsrc.isin(['polyp_real', 'polyp_normal'])], 'polyp_normal')]
res = []
for rec in ['R0_as_is', 'R1_jpeg90', 'R2_gray_jpeg90', 'R3_gray_resize_jpeg90', 'R4_gray_resize_png']:
    s = allr[allr.recipe == rec]
    for name, f, poslab in COMPS:
        sub = f(s).copy()
        if poslab: sub['label'] = (sub.subsrc == poslab).astype(int)
        sub = sub.drop_duplicates('path').reset_index(drop=True)
        for fn, cols in FEATS.items():
            y = sub.label.values.astype(int); g = sub.group.values; sc = np.zeros(len(sub))
            skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
            for tr, te in skf.split(np.zeros(len(y)), y, g):
                m = models()['GB'](); X = design2(sub, cols); m.fit(X[tr], y[tr]); sc[te] = m.predict_proba(X[te])[:, 1]
            res.append(dict(recipe=rec, comp=name, feat=fn, n=len(sub), auc=round(auc(y, sc), 4)))
            print(res[-1], flush=True)
        # thumbnail probe
        y = sub.label.values.astype(int); g = sub.group.values; sc = np.zeros(len(sub)); X = np.stack(sub['thumb'].values).astype(float)
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g):
            m = models()['LR'](); m.fit(X[tr], y[tr]); sc[te] = m.predict_proba(X[te])[:, 1]
        res.append(dict(recipe=rec, comp=name, feat='thumb16 (LR)', n=len(sub), auc=round(auc(y, sc), 4))); print(res[-1], flush=True)
pd.DataFrame(res).to_csv(W + 'res_norm.csv', index=False)
print('DONE')
