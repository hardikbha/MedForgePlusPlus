
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
W = _release_path('CIRCE/v3_audit/shortcut_work/')

def load():
    s = pd.DataFrame([json.loads(l) for l in open(W + 'sample.jsonl')])
    rows = []
    for l in open(os.environ.get('FEAT', W + 'features.jsonl')):
        try: rows.append(json.loads(l))
        except Exception: pass
    f = pd.DataFrame(rows)
    d = s.merge(f, on='path', how='inner' if os.environ.get('FEAT') else 'left')
    d['npix'] = d['width'] * d['height']
    d['log_npix'] = np.log10(d['npix'])
    d['aspect'] = d['width'] / d['height']
    d['log_fsize'] = np.log10(d['file_size'])
    d['bpp'] = d['file_size'] / d['npix']
    for c in ['n_lapvar', 'r_lapvar']: d['log_' + c] = np.log10(d[c] + 1e-3)
    d['res'] = d['width'].astype('Int64').astype(str) + 'x' + d['height'].astype('Int64').astype(str)
    return d

CAT_META = ['ext', 'format', 'mode', 'subsampling', 'info_keys', 'qt_hash']
NUM_META = ['width', 'height', 'log_npix', 'aspect', 'log_fsize', 'bpp', 'jpeg_q', 'jpeg_q_resid', 'jpeg_qc', 'jpeg_qc_resid',
            'n_qtables', 'progressive', 'has_exif', 'has_icc', 'has_jfif', 'has_adobe', 'has_comment', 'dpi_x', 'png_bitdepth16']
GB = ['n_mean', 'n_std', 'log_n_lapvar', 'n_fft_hf', 'n_sat']
GX = GB + ['n_fft_mid', 'n_fft_low', 'n_chroma', 'n_noise', 'n_entropy', 'n_dark_frac', 'n_bright_frac', 'n_zero_frac',
           'n_max255_frac', 'n_nlevels', 'n_block']
G512 = ['r_mean', 'r_std', 'log_r_lapvar', 'r_fft_hf', 'r_fft_mid', 'r_fft_low', 'r_sat', 'r_chroma', 'r_noise', 'r_entropy',
        'r_dark_frac', 'r_bright_frac', 'r_zero_frac', 'r_max255_frac', 'r_nlevels']

def design(d, group, train_idx=None):
    """Return feature matrix for a named feature group. Categorical one-hot vocab fixed on train rows."""
    if group in ('meta', 'meta_noext', 'meta_size', 'meta_jpeg'):
        if group == 'meta': cats, nums = CAT_META, NUM_META
        elif group == 'meta_noext': cats, nums = ['mode', 'subsampling', 'qt_hash'], NUM_META   # no extension / container
        elif group == 'meta_size': cats, nums = [], ['width', 'height', 'log_npix', 'aspect']
        else: cats, nums = ['subsampling', 'qt_hash'], ['jpeg_q', 'jpeg_q_resid', 'jpeg_qc', 'jpeg_qc_resid', 'n_qtables', 'log_fsize', 'bpp']
        X = d[nums].astype(float).copy()
        ref = d if train_idx is None else d.iloc[train_idx]
        for c in cats:
            vc = ref[c].astype(str).value_counts()
            for v in vc.index[:30]:
                X[c + '=' + v] = (d[c].astype(str) == v).astype(float)
        return X.values
    if group == 'gbasic': return d[GB].astype(float).values
    if group == 'gext': return d[GX].astype(float).values
    if group == 'g512': return d[G512].astype(float).values
    if group == 'thumb': return np.stack(d['thumb'].values).astype(float)
    if group == 'gext+thumb': return np.hstack([d[GX].astype(float).values, np.stack(d['thumb'].values).astype(float)])
    raise ValueError(group)

def models():
    return {'LR': lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
            'GB': lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)}

def oof(d, group, model='GB', k=5, seed=0):
    """Out-of-fold scores with group-aware stratified k-fold. Returns scores and per-fold AUCs."""
    y = d['label'].values.astype(int); g = d['group'].values
    sc = np.full(len(d), np.nan); aucs = []
    k = min(k, max(2, int(min(np.bincount(y)))))
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(y)), y, g):
        X = design(d, group, tr)
        m = models()[model](); m.fit(X[tr], y[tr])
        sc[te] = m.predict_proba(X[te])[:, 1]
        if len(set(y[te])) == 2: aucs.append(roc_auc_score(y[te], sc[te]))
    return sc, aucs

def auc(y, s):
    y = np.asarray(y); 
    if len(set(y)) < 2: return np.nan
    return roc_auc_score(y, s)

def fit_apply(dtr, dte, group, model='GB'):
    both = pd.concat([dtr, dte], ignore_index=True)
    X = design(both, group, np.arange(len(dtr)))
    m = models()[model](); m.fit(X[:len(dtr)], dtr['label'].values.astype(int))
    return m.predict_proba(X[len(dtr):])[:, 1]

def single_feature_aucs(d, feats):
    out = []
    y = d['label'].values
    for f in feats:
        x = d[f].astype(float).values
        ok = ~np.isnan(x)
        if ok.sum() < 10 or len(set(y[ok])) < 2 or np.nanstd(x) == 0: continue
        a = roc_auc_score(y[ok], x[ok]); out.append((f, max(a, 1 - a), '+' if a >= 0.5 else '-'))
    return sorted(out, key=lambda t: -t[1])

def to_md(t):
    cols = list(t.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for _, r in t.iterrows():
        lines.append('| ' + ' | '.join('' if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + ' |')
    return '\n'.join(lines)
