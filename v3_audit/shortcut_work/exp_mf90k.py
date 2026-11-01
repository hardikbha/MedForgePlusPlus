
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, _release_path('CIRCE/v3_audit/shortcut_work'))
from common import *
d = load(); d = d[d['ok'] == 1].reset_index(drop=True)
d['is_jpeg'] = (d['format'] == 'JPEG').astype(float)
GROUPS = ['meta', 'meta_noext', 'meta_size', 'meta_jpeg', 'gbasic', 'gext', 'g512', 'thumb']
strat = d[d['tags'].apply(lambda t: 'strat' in t) & (d.corpus == 'MF90K')].reset_index(drop=True)
res = []; slices = []
def run(name, sub, groups=GROUPS, mods=('LR', 'GB'), do_slices=True):
    for fg in groups:
        for m in mods:
            sc, aucs = oof(sub, fg, m)
            a = auc(sub['label'], sc)
            res.append(dict(exp=name, feat=fg, model=m, n=len(sub), n_fake=int(sub.label.sum()), auc=round(a, 4),
                            fold_mean=round(np.mean(aucs), 4), fold_std=round(np.std(aucs), 4)))
            print(res[-1], flush=True)
            if do_slices and m == 'GB':
                rl = sub['label'] == 0
                for key in ['gen', 'cls']:
                    for v in sorted(sub.loc[~rl, key].unique()):
                        msk = rl | (sub[key] == v)
                        slices.append(dict(exp=name, feat=fg, slice=f'{key}={v}', n_fake=int((sub[key] == v).sum()),
                                           auc=round(auc(sub.loc[msk, 'label'], sc[msk.values]), 4)))
for organ in ['brain-mri', 'chest-xray', 'fundus']:
    run(f'{organ}: real vs fake', strat[strat.organ == organ].reset_index(drop=True))
run('ALL organs pooled: real vs fake', strat, groups=['meta', 'gext', 'g512', 'thumb'], mods=('GB',))
# source-matched control: reals restricted to images that were used as sources of fakes
for organ in ['brain-mri', 'chest-xray', 'fundus']:
    sub = d[(d.corpus == 'MF90K') & (d.organ == organ) & (
        ((d.label == 1) & d['tags'].apply(lambda t: 'strat' in t)) |
        ((d.label == 0) & (d['is_fake_source'] == True)))].reset_index(drop=True)
    run(f'{organ}: SOURCE-MATCHED real vs fake', sub, groups=['meta', 'meta_size', 'gext', 'g512', 'thumb'], mods=('GB',))
# real sub-source vs fakes: is the never-edited sub-population trivially 'real'?
for organ in ['brain-mri', 'fundus']:
    sub = strat[strat.organ == organ].copy()
    sub['label'] = ((sub.label == 0) & (sub['is_fake_source'] != True)).astype(int)  # 1 = real image of a sub-source never used for fakes
    # compare never-source reals vs fakes only
    sub = sub[(sub.cls != 'real') | (sub['is_fake_source'] != True)].reset_index(drop=True)
    sub['label'] = 1 - sub['label']  # fake=1
    run(f'{organ}: fakes vs NEVER-SOURCE reals', sub, groups=['meta', 'gext', 'thumb'], mods=('GB',), do_slices=False)
# dedicated per-generator and per-type models
for organ in ['brain-mri', 'chest-xray', 'fundus']:
    so = strat[strat.organ == organ]
    for g in sorted(so.loc[so.label == 1, 'gen'].unique()):
        sub = so[(so.label == 0) | (so.gen == g)].reset_index(drop=True)
        run(f'{organ}: real vs {g} (dedicated)', sub, groups=['meta', 'meta_size', 'gext', 'g512'], mods=('GB',), do_slices=False)
    for t in ['edit', 'remove']:
        sub = so[(so.label == 0) | (so.cls == t)].reset_index(drop=True)
        run(f'{organ}: real vs {t} (dedicated)', sub, groups=['meta', 'gext', 'g512', 'thumb'], mods=('GB',), do_slices=False)
pd.DataFrame(res).to_csv(W + 'res_mf90k.csv', index=False)
pd.DataFrame(slices).to_csv(W + 'res_mf90k_slices.csv', index=False)
print('DONE')
