
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, _release_path('CIRCE/v3_audit/shortcut_work'))
from common import *
d = load(); d = d[d['ok'] == 1].reset_index(drop=True)
d['is_jpeg'] = (d['format'] == 'JPEG').astype(float)
d['is_png'] = (d['format'] == 'PNG').astype(float)
A = d[d.corpus.isin(['MF++add', 'ref'])].reset_index(drop=True)
GROUPS = ['meta', 'meta_noext', 'meta_size', 'gbasic', 'gext', 'g512', 'thumb']
SINGLE = ['is_jpeg', 'width', 'height', 'log_fsize', 'bpp', 'jpeg_q', 'has_exif', 'has_icc', 'n_mean', 'n_std', 'log_n_lapvar',
          'n_fft_hf', 'n_fft_mid', 'n_sat', 'n_chroma', 'n_noise', 'n_entropy', 'n_dark_frac', 'n_zero_frac', 'n_nlevels', 'n_block',
          'r_mean', 'r_std', 'log_r_lapvar', 'r_fft_hf', 'r_noise', 'r_sat']
COMP = [  # (name, positive subsrcs (label 1), negative subsrcs (label 0))
    ('ADD-all: real(3800) vs fake(4000) [paper MF++ task]', ['kidney_removal', 'polyp_removal', 'colon_inject', 'kidney_inject'], ['kidney_normal', 'kidney_real_tumor', 'polyp_normal', 'polyp_real']),
    ('kidney: real(normal+tumor) vs removal', ['kidney_removal'], ['kidney_normal', 'kidney_real_tumor']),
    ('kidney: real_tumor vs removal (same slices)', ['kidney_removal'], ['kidney_real_tumor']),
    ('kidney: normal vs removal', ['kidney_removal'], ['kidney_normal']),
    ('kidney: normal vs real_tumor (both real)', ['kidney_real_tumor'], ['kidney_normal']),
    ('kidney: real(normal+tumor) vs implant', ['kidney_inject'], ['kidney_normal', 'kidney_real_tumor']),
    ('kidney: normal vs implant', ['kidney_inject'], ['kidney_normal']),
    ('kidney: implant vs its OWN source pool (ref, not in MF++)', ['kidney_inject'], ['ref_kidney_inject_source']),
    ('kidney: KiTS normal vs implant-source pool (both real)', ['ref_kidney_inject_source'], ['kidney_normal']),
    ('colon: polyp_real vs polyp_normal (both real)', ['polyp_normal'], ['polyp_real']),
    ('colon: real(real+normal) vs removal', ['polyp_removal'], ['polyp_real', 'polyp_normal']),
    ('colon: polyp_real vs removal (same frames)', ['polyp_removal'], ['polyp_real']),
    ('colon: polyp_normal vs removal', ['polyp_removal'], ['polyp_normal']),
    ('colon: real(real+normal) vs implant', ['colon_inject'], ['polyp_real', 'polyp_normal']),
    ('colon: polyp_real vs implant (same Kvasir frames)', ['colon_inject'], ['polyp_real']),
    ('colon: Kvasir original vs polyp_real (provenance)', ['polyp_real'], ['ref_kvasir_original']),
    ('colon: Kvasir original vs implant', ['colon_inject'], ['ref_kvasir_original']),
]
res = []; singles = []
for name, pos, neg in COMP:
    sub = A[A.subsrc.isin(pos + neg)].copy()
    sub['label'] = sub.subsrc.isin(pos).astype(int)
    # dedup exact byte duplicates (polyp_normal has ~29.5% dups): keep one per md5 within the comparison
    nb = len(sub); sub = sub.drop_duplicates('md5').reset_index(drop=True)
    for fg in GROUPS:
        for m in ['LR', 'GB']:
            sc, aucs = oof(sub, fg, m)
            r = dict(comp=name, feat=fg, model=m, n_pos=int(sub.label.sum()), n_neg=int((1 - sub.label).sum()), dropped_dups=nb - len(sub),
                     auc=round(auc(sub.label, sc), 4), fold_mean=round(np.mean(aucs), 4), fold_std=round(np.std(aucs), 4))
            if name.startswith('ADD-all') and m == 'GB':
                for s_ in pos:
                    mk = (sub.label == 0) | (sub.subsrc == s_); r['auc_slice_' + s_] = round(auc(sub.label[mk], sc[mk.values]), 4)
            res.append(r); print(r, flush=True)
    for f, a, sgn in single_feature_aucs(sub, SINGLE)[:6]:
        singles.append(dict(comp=name, feature=f, auc=round(a, 4), direction=sgn))
pd.DataFrame(res).to_csv(W + 'res_addition.csv', index=False)
pd.DataFrame(singles).to_csv(W + 'res_addition_singlefeat.csv', index=False)
print('DONE')
