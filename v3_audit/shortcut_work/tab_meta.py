
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import sys, numpy as np, pandas as pd
sys.path.insert(0, _release_path('CIRCE/v3_audit/shortcut_work'))
from common import load, W, to_md
d = load()
print('rows', len(d), 'ok', int(d['ok'].sum()), 'failed', int((d['ok'] != 1).sum()))
if (d['ok'] != 1).any(): print(d.loc[d['ok'] != 1, ['path', 'err']].head(10).to_string())
d = d[d['ok'] == 1].copy()

def top(s, k=3):
    vc = s.astype(str).value_counts(normalize=True)
    return '; '.join(f'{i} {v*100:.0f}%' for i, v in vc.head(k).items())

def summarize(g):
    jq = g['jpeg_q'].dropna()
    return pd.Series({
        'n': len(g),
        'ext': top(g['ext'], 2),
        'mode': top(g['mode'], 2),
        'resolution (top3)': top(g['res'], 3),
        'n_res': g['res'].nunique(),
        'fsize_med_KB': round(g['file_size'].median() / 1024, 1),
        'bpp_med': round(g['bpp'].median(), 3),
        'jpegQ_med': (round(jq.median(), 0) if len(jq) else np.nan),
        'jpegQ (top2)': (top(jq.astype(int), 2) if len(jq) else '-'),
        'IJG_exact%': (round(100 * (g['jpeg_q_resid'] == 0).mean(), 0) if len(jq) else np.nan),
        'subsamp': top(g['subsampling'].map({-2: 'n/a', -1: 'gray/1ch', 0: '4:4:4', 1: '4:2:2', 2: '4:2:0'}), 2),
        'exif%': round(100 * g['has_exif'].mean(), 1),
        'icc%': round(100 * g['has_icc'].mean(), 1),
        'prog%': round(100 * g['progressive'].mean(), 1),
        'jfif%': round(100 * g['has_jfif'].mean(), 1),
        'n_qt_hash': g['qt_hash'].nunique(),
    })

strat = d[d['tags'].apply(lambda t: 'strat' in t)]
pd.set_option('display.width', 400); pd.set_option('display.max_colwidth', 80); pd.set_option('display.max_columns', 30)
out = []
t1 = strat.groupby(['organ', 'cls']).apply(summarize).reset_index()
t2 = strat[strat.cls != 'real'].groupby(['organ', 'gen']).apply(summarize).reset_index()
t3 = strat[strat.cls == 'real'].groupby(['organ', 'subsrc']).apply(summarize).reset_index()
add = d[d['corpus'].isin(['MF++add', 'ref'])]
t4 = add.groupby(['corpus', 'organ', 'subsrc']).apply(summarize).reset_index()
gen_all = strat[strat.cls != 'real'].groupby(['gen']).apply(summarize).reset_index()
for name, t in [('MF90K by organ x class (stratified sample)', t1), ('MF90K fakes by organ x generator', t2),
                ('MF90K fakes by generator (all organs)', gen_all),
                ('MF90K reals by organ x sub-source', t3), ('MF++ addition + provenance references', t4)]:
    print('\n### ' + name); print(t.to_string(index=False))
    t.to_csv(W + 'tab_' + name.split(' (')[0].replace(' ', '_').replace('+', 'plus').replace('/', '') + '.csv', index=False)
    out.append('### ' + name + '\n\n' + to_md(t) + '\n')
open(W + 'tables_meta.md', 'w').write('\n'.join(out))
# exact duplicates
dup = d.groupby('md5').filter(lambda g: len(g) > 1)
print('\nexact-duplicate md5 groups in sample:', dup['md5'].nunique(), 'rows', len(dup))
x = dup.groupby('md5').agg(n=('path', 'size'), subsrcs=('subsrc', lambda s: ','.join(sorted(set(s)))), labels=('label', lambda s: ','.join(map(str, sorted(set(s)))))).reset_index()
print(x.groupby(['subsrcs', 'labels']).agg(groups=('md5', 'size'), rows=('n', 'sum')).to_string())
