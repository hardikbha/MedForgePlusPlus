
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, _release_path('CIRCE/v3_audit/shortcut_work'))
from common import *
rows = [json.loads(l) for l in open(W + 'features_pasteback.jsonl')]
d = pd.DataFrame([r for r in rows if 'err' not in r]); print('rows', len(d), 'errors', sum('err' in r for r in rows))
d['log_n_lapvar'] = np.log10(d['n_lapvar'] + 1e-3); d['log_fsize'] = np.log10(d['file_size'])
FE = {'global(ext)+filesize': GX + ['log_fsize'], 'thumb16': None}
res = []
for organ in ['brain-mri', 'chest-xray', 'fundus', 'ALL']:
    so = d if organ == 'ALL' else d[d.organ == organ]
    for fv in ['fake_full_frame', 'fake_pasted_back']:
        reals = so[so.variant == 'real_source'].drop_duplicates('group')
        sub = pd.concat([reals.assign(label=0), so[so.variant == fv].assign(label=1)], ignore_index=True)
        for fn, cols in FE.items():
            X = np.stack(sub['thumb'].values).astype(float) if cols is None else sub[cols].astype(float).values
            y = sub.label.values; g = sub.group.values; sc = np.zeros(len(sub))
            for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, g):
                m = models()['GB'](); m.fit(X[tr], y[tr]); sc[te] = m.predict_proba(X[te])[:, 1]
            r = dict(organ=organ, fake_variant=fv, feat=fn, n_real=int((y == 0).sum()), n_fake=int(y.sum()), auc=round(auc(y, sc), 4))
            res.append(r); print(r, flush=True)
pd.DataFrame(res).to_csv(W + 'res_pasteback.csv', index=False)
print(d.groupby('variant')['mask_frac'].median())
