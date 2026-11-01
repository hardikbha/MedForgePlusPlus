
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import sys, numpy as np, pandas as pd
sys.path.insert(0, _release_path('CIRCE/v3_audit/shortcut_work'))
from common import *
from sklearn.metrics import f1_score
d = load(); d = d[d['ok'] == 1].reset_index(drop=True)
test = d[d['tags'].apply(lambda t: 'test' in t)].reset_index(drop=True)
COLOR = {'n_sat', 'n_chroma'}; JPEG = {'n_block'}
SETS = {'gext (16 feats)': GX, 'gext minus colour (sat,chroma)': [f for f in GX if f not in COLOR],
        'gext minus colour minus blockiness': [f for f in GX if f not in COLOR | JPEG],
        'only colour (n_sat,n_chroma)': ['n_sat', 'n_chroma'], 'only n_block': ['n_block'],
        'gbasic (5 feats: mean,std,lapvar,fft_hf,sat)': GB}
res = []
# 1) one-line rule on the test set: fake iff any pixel has R!=G or G!=B (gray organs only)
for o in ['brain-mri', 'chest-xray']:
    t = test[test.organ == o]; p = (t.n_chroma > 0).astype(int); y = t.label
    res.append(dict(regime='RULE (no training): fake iff n_chroma>0', feats=o, test_acc=round((p == y).mean() * 100, 2),
                    acc_real=round((p[y == 0] == 0).mean() * 100, 2), acc_edit=round((p[t.cls == 'edit'] == 1).mean() * 100, 2),
                    acc_remove=round((p[t.cls == 'remove'] == 1).mean() * 100, 2), test_auc=round(auc(y, t.n_chroma), 4)))
    print(res[-1], flush=True)
for tag, name in [('tr_cfs', 'Cross-Forgery strict'), ('tr_cm', 'Cross-Model'), ('tr_id', 'In-Domain')]:
    tr = d[d['tags'].apply(lambda t: tag in t) & ~d.path.isin(set(test.path))].reset_index(drop=True)
    for sn, cols in SETS.items():
        m = models()['GB'](); m.fit(tr[cols].astype(float).values, tr.label.values)
        s = m.predict_proba(test[cols].astype(float).values)[:, 1]; p = (s >= 0.5).astype(int); y = test.label.values
        row = dict(regime=name, feats=sn, test_auc=round(auc(y, s), 4), test_acc=round((p == y).mean() * 100, 2),
                   test_macroF1=round(f1_score(y, p, average='macro') * 100, 2))
        for c in ['real', 'edit', 'remove']:
            mk = (test.cls == c).values; row['acc_' + c] = round((p[mk] == y[mk]).mean() * 100, 2)
        for o in ['brain-mri', 'chest-xray', 'fundus']:
            mk = (test.organ == o).values; row['acc_' + o] = round((p[mk] == y[mk]).mean() * 100, 2)
        res.append(row); print(row, flush=True)
pd.DataFrame(res).to_csv(W + 'res_protocol_ablate.csv', index=False)
print('DONE')
