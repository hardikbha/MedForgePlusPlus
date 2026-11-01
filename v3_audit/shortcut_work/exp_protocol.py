
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, _release_path('CIRCE/v3_audit/shortcut_work'))
from common import *
from sklearn.metrics import f1_score
d = load(); d = d[d['ok'] == 1].reset_index(drop=True)
test = d[d['tags'].apply(lambda t: 'test' in t)].reset_index(drop=True)
CM_HELD = {'gemini', 'gpt', 'stable-diffusion-3.5-medium', 'stable-diffusion-xl-1.0-inpainting-0.1'}
res = []
for tag, name in [('tr_cf', 'Cross-Forgery (mfp_train_crossforgery)'), ('tr_cfs', 'Cross-Forgery strict (mfp_v2_train_crossforgery_strict)'),
                  ('tr_cm', 'Cross-Model (mfp_train_crossmodel)'), ('tr_id', 'In-Domain (mfp_train_indomain)')]:
    tr = d[d['tags'].apply(lambda t: tag in t)].reset_index(drop=True)
    overlap = len(set(tr.path) & set(test.path))
    tr = tr[~tr.path.isin(set(test.path))].reset_index(drop=True)
    for fg in ['meta', 'meta_size', 'gext', 'g512', 'thumb', 'gext+thumb']:
        for m in ['LR', 'GB']:
            s = fit_apply(tr, test, fg, m)
            y = test.label.values; p = (s >= 0.5).astype(int)
            row = dict(regime=name, feat=fg, model=m, n_train=len(tr), train_test_path_overlap=overlap,
                       test_auc=round(auc(y, s), 4), test_acc=round((p == y).mean() * 100, 2),
                       test_macroF1=round(f1_score(y, p, average='macro') * 100, 2))
            for c in ['real', 'edit', 'remove']:
                mk = (test.cls == c).values; row[f'acc_{c}'] = round((p[mk] == y[mk]).mean() * 100, 2)
            for c in ['edit', 'remove']:
                mk = ((test.cls == 'real') | (test.cls == c)).values; row[f'auc_real_vs_{c}'] = round(auc(y[mk], s[mk]), 4)
            mk = ((test.cls == 'real') | test.gen.isin(CM_HELD)).values; row['auc_real_vs_CMheldout_gens'] = round(auc(y[mk], s[mk]), 4)
            for o in ['brain-mri', 'chest-xray', 'fundus']:
                mk = (test.organ == o).values; row[f'auc_{o}'] = round(auc(y[mk], s[mk]), 4)
            res.append(row); print(row, flush=True)
pd.DataFrame(res).to_csv(W + 'res_protocol.csv', index=False)
print('DONE')
