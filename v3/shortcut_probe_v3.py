"""
shortcut_probe_v3.py -- re-run the v3 shortcut audit on MF-Clean (before/after).            (v3, 2026-10-11)

README
------
Question: after the MF-Clean rebuild (paste-back composites, one JPEG encoder, gray organs as single-channel L,
group-disjoint splits), can non-forensic shortcuts still separate real from fake?

Probes (all CPU, no GPU; features and classifiers are REUSED from the audit,
~/CIRCE/v3_audit/shortcut_work/{extract.py,common.py}, so the numbers are directly comparable to shortcut.md):
  (1) one-line colour rule "fake iff any pixel has R!=G or G!=B" per organ on test_core.
  (2) metadata-only (`meta`) and 16-feature global-statistic (`gext`, = common.GX) HistGradientBoosting trained
      on a 5,000-image sample of MF-Clean train for each regime (ID, CF, CM, CFxCM; exclusions applied to train
      only), tested on test_core: acc, balanced acc, macro-F1, per-class recall, AUC, AUC slices, twin 2AFC.
      Extra feature sets: thumb (16x16 gray), gext+thumb, g512, gext without colour.
      Within-organ 5-fold StratifiedGroupKFold AUC (common.oof) on test_core and on the ID train sample.
  (3) BEFORE: exactly the same probes on the ORIGINAL MedForge-90K files of exactly the same rows (row["orig"]):
      same train sample, same test_core, same groups; only the pixels/encoding differ.
  (4) twin-difference sanity: 300 random fakes, |fake - twin| inside the box, in the feather ring, and outside
      the composited region (alpha == 0, alpha from build_mfclean.feather_alpha); also restricted to 8x8 JPEG
      blocks that do not touch alpha > 0 (should be exactly 0). Before-column: |orig fake - orig source|.

Outputs: ~/CIRCE/v3/reports/shortcut_mfclean.md and CSVs in ~/CIRCE/v3/reports/shortcut_mfclean/
         (feature cache: ~/CIRCE/v3/cache/shortcut_probe/features.jsonl, keyed by path + size + mtime).
         If ~/CIRCE/v3/reports/shortcut_mfclean/notes.md exists it is embedded verbatim as the "Reading" section.

Run (CPU, nice, <=16 processes; analysis env, never the default GraalPy python3):
  NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8 nice -n 19 ~/.conda/envs/aiface/bin/python \
      ~/CIRCE/v3/shortcut_probe_v3.py --nproc 14 [--wait_min 90]
  Dev run against a synthetic sample: --mfclean <dir> --out_dir <dir> --out_md <file> --cache <file>
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import argparse, collections, datetime, hashlib, json, random, sys, time, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from PIL import Image
warnings.filterwarnings('ignore')

AUDIT = _release_path('CIRCE/v3_audit/shortcut_work')
V3 = _release_path('CIRCE/v3')
sys.path.insert(0, AUDIT)
sys.path.insert(0, V3)
import extract                                    # noqa: E402  audit per-image features (read-only reuse)
import common as C                                # noqa: E402  audit design()/models()/oof()/fit_apply()
import build_mfclean as B                         # noqa: E402  MF / GRAY / CM_HOLDOUT / feather_alpha / DIL / FEATHER
from sklearn.metrics import f1_score, balanced_accuracy_score   # noqa: E402

ORGANS = ['brain-mri', 'chest-xray', 'fundus']
CM = set(B.CM_HOLDOUT)    # gemini, gpt, stable-diffusion-3.5-medium, stable-diffusion-xl-1.0-inpainting-0.1
REGIMES = {
    'ID': lambda r: True,
    'CF': lambda r: r['type'] != 'edit',
    'CM': lambda r: r['gen'] not in CM,
    'CFxCM': lambda r: r['type'] != 'edit' and r['gen'] not in CM,
}
GX_NOCOL = [f for f in C.GX if f not in ('n_sat', 'n_chroma')]
FEATS_TRANSFER = ['meta', 'gext', 'gext_nocolour', 'thumb', 'gext+thumb', 'g512']
FEATS_CV = [('meta', 'GB'), ('gext', 'GB'), ('gext', 'LR'), ('g512', 'GB'), ('thumb', 'GB'), ('thumb', 'LR'),
            ('gext+thumb', 'GB')]


# ----------------------------------------------------------------------------------------------- utilities
def log(*a):
    print(f'[{datetime.datetime.now():%H:%M:%S}]', *a, flush=True)


def md5_file(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def design(d, group, tr=None):
    if group == 'gext_nocolour':
        return d[GX_NOCOL].astype(float).values
    return C.design(d, group, tr)


def fit_apply(dtr, dte, group, model='GB'):
    both = pd.concat([dtr, dte], ignore_index=True)
    X = design(both, group, np.arange(len(dtr)))
    m = C.models()[model]()
    m.fit(X[:len(dtr)], dtr['label'].values.astype(int))
    return m.predict_proba(X[len(dtr):])[:, 1]


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def pct(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(100 * float(x), 2)


# ----------------------------------------------------------------------------------------------- features
def feat_work(p):
    o = extract.work({'path': p})
    try:
        o['mtime'] = os.path.getmtime(p)
    except OSError:
        o['mtime'] = None
    return o


def get_features(paths, cache, nproc):
    """Audit features (extract.work) for every path; cached by path, invalidated by file size or mtime."""
    have = {}
    if os.path.exists(cache):
        for l in open(cache):
            try:
                o = json.loads(l)
            except Exception:
                continue
            have[o['path']] = o
    todo = []
    for p in sorted(set(paths)):
        o = have.get(p)
        try:
            st = os.stat(p)
        except OSError:
            todo.append(p); continue
        if o is None or o.get('ok') != 1 or o.get('file_size') != st.st_size or o.get('mtime') != st.st_mtime:
            todo.append(p)
    log(f'features: {len(set(paths))} paths, {len(set(paths)) - len(todo)} cached, {len(todo)} to compute')
    if todo:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with Pool(nproc) as pool, open(cache, 'a') as f:
            for i, o in enumerate(pool.imap_unordered(feat_work, todo, chunksize=8)):
                f.write(json.dumps(o) + '\n')
                have[o['path']] = o
                if (i + 1) % 2000 == 0:
                    f.flush(); log(f'  features {i + 1}/{len(todo)}')
    return have


def frame(rows, version, feats):
    """rows -> DataFrame for version 'clean' (row['image']) or 'orig' (row['orig']) with audit-derived columns."""
    key = 'image' if version == 'clean' else 'orig'
    recs = []
    for i, r in enumerate(rows):
        o = feats.get(r[key])
        if o is None or o.get('ok') != 1:
            continue
        x = {k: v for k, v in o.items()}
        x.update(rid=i, label=int(r['type'] != 'real'), cls=r['type'], gen=r['gen'], organ=r['organ'],
                 group=r['group'], image=r['image'], twin=r.get('twin'))
        recs.append(x)
    d = pd.DataFrame(recs)
    for c in C.NUM_META + C.CAT_META:
        if c not in d.columns:
            d[c] = np.nan
    d['npix'] = d['width'] * d['height']
    d['log_npix'] = np.log10(d['npix'])
    d['aspect'] = d['width'] / d['height']
    d['log_fsize'] = np.log10(d['file_size'])
    d['bpp'] = d['file_size'] / d['npix']
    for c in ['n_lapvar', 'r_lapvar']:
        d['log_' + c] = np.log10(d[c] + 1e-3)
    return d.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- metrics
def metrics(te, s, thr=0.5):
    y = te['label'].values.astype(int)
    p = (s >= thr).astype(int)
    o = dict(n_test=len(te), test_frac_fake=r4(y.mean()), majority_acc=pct(max(y.mean(), 1 - y.mean())),
             test_auc=r4(C.auc(y, s)), test_acc=pct((p == y).mean()), test_bal_acc=pct(balanced_accuracy_score(y, p)),
             test_macroF1=pct(f1_score(y, p, average='macro')))
    for c in ['real', 'edit', 'remove']:
        mk = (te['cls'] == c).values
        o['recall_' + c] = pct((p[mk] == y[mk]).mean()) if mk.any() else None
    rl = (te['cls'] == 'real').values
    for c in ['edit', 'remove']:
        mk = rl | (te['cls'] == c).values
        o['auc_real_vs_' + c] = r4(C.auc(y[mk], s[mk]))
    mk = rl | te['gen'].isin(CM).values
    o['auc_real_vs_CMheld'] = r4(C.auc(y[mk], s[mk]))
    for org in ORGANS:
        mk = (te['organ'] == org).values
        o['auc_' + org] = r4(C.auc(y[mk], s[mk])) if mk.any() else None
        o['acc_' + org] = pct((p[mk] == y[mk]).mean()) if mk.any() else None
    # twin 2AFC: for each test fake whose twin real is also in the test frame, P(score_fake > score_twin)
    idx = {im: i for i, im in enumerate(te['image'].values)}
    w = []
    for i in np.where(y == 1)[0]:
        j = idx.get(te['twin'].values[i])
        if j is not None:
            w.append(1.0 if s[i] > s[j] else (0.5 if s[i] == s[j] else 0.0))
    o['twin_pairs'] = len(w)
    o['twin_2afc'] = pct(np.mean(w)) if w else None
    return o


def slices(te, s, tag):
    y = te['label'].values.astype(int)
    rl = (te['cls'] == 'real').values
    out = []
    for key in ['gen', 'cls']:
        for v in sorted(te.loc[~rl, key].unique()):
            mk = rl | (te[key] == v).values
            out.append(dict(tag, slice=f'{key}={v}', n_fake=int(((te[key] == v).values & ~rl).sum()),
                            auc=r4(C.auc(y[mk], s[mk]))))
    for org in ORGANS:
        so = (te['organ'] == org).values
        for v in sorted(te.loc[~rl & so, 'gen'].unique()):
            mk = so & (rl | (te['gen'] == v).values)
            out.append(dict(tag, slice=f'{org}|gen={v}', n_fake=int((mk & ~rl).sum()), auc=r4(C.auc(y[mk], s[mk]))))
    return out


# ----------------------------------------------------------------------------------------------- twin diff
MAPPING = {}


def _init_mapping(m):
    global MAPPING
    MAPPING = m


def twin_work(r):
    try:
        F = np.asarray(Image.open(r['image']))
        T = np.asarray(Image.open(r['twin']))
        if F.shape != T.shape:
            return dict(image=r['image'], ok=0, why=f'shape {F.shape} vs {T.shape}')
        H, W = F.shape[:2]
        bx = r['box']
        box = [int(round(bx[0] * W / 1024)), int(round(bx[1] * H / 1024)), int(round(bx[2] * W / 1024)),
               int(round(bx[3] * H / 1024))]
        al = B.feather_alpha((H, W), box, B.DIL, B.FEATHER)[..., 0]
        inside = np.zeros((H, W), bool)
        inside[box[1]:box[3] + 1, box[0]:box[2] + 1] = True
        ring = (al > 0) & ~inside
        out = al == 0
        hb, wb = (H + 7) // 8, (W + 7) // 8
        touch = np.zeros((hb * 8, wb * 8), bool)
        touch[:H, :W] = al > 0
        blk = touch.reshape(hb, 8, wb, 8).any(axis=(1, 3))
        out_blk = ~np.repeat(np.repeat(blk, 8, 0), 8, 1)[:H, :W]
        D = np.abs(F.astype(np.int16) - T.astype(np.int16)).astype(np.float32)
        Dm = D.mean(-1) if D.ndim == 3 else D             # per-pixel mean over channels
        Dx = D.max(-1) if D.ndim == 3 else D
        ys, xs = np.where(al > 0)
        o = dict(image=r['image'], twin=r['twin'], organ=r['organ'], gen=r['gen'], cls=r['type'], ok=1, mode=Image.open(r['image']).mode,
                 H=H, W=W, box_frac=r4(inside.mean()), alpha_support_px=int(max(box[0] - xs.min(), xs.max() - box[2],
                                                                                 box[1] - ys.min(), ys.max() - box[3])),
                 frac_out=r4(out.mean()), frac_out_blocks=r4(out_blk.mean()),
                 mad_in=r4(Dm[inside].mean()), mad_ring=r4(Dm[ring].mean()) if ring.any() else None,
                 mad_out=float(Dm[out].mean()) if out.any() else None,
                 max_out=float(Dx[out].max()) if out.any() else None,
                 frac_out_nonzero=float((Dx[out] > 0).mean()) if out.any() else None,
                 frac_out_gt2=float((Dx[out] > 2).mean()) if out.any() else None,
                 mad_out_blocks=float(Dm[out_blk].mean()) if out_blk.any() else None,
                 max_out_blocks=float(Dx[out_blk].max()) if out_blk.any() else None)
        # is the twin the fake's exact authentic source? (it can be a verified near-duplicate representative)
        rel = r['orig'][len(B.MF):] if r['orig'].startswith(B.MF) else None
        src = MAPPING.get(rel)
        twin_stem = r['twin'].split('/img/', 1)[-1].rsplit('.', 1)[0]
        o['src_rel'] = src
        o['twin_is_exact_source'] = int(src is not None and os.path.splitext(src)[0] == twin_stem)
        # BEFORE: original full-frame fake vs its original source, same regions
        if src is not None and os.path.exists(B.MF + src):
            Fo = np.asarray(Image.open(r['orig']).convert('RGB'), dtype=np.int16)
            So_im = Image.open(B.MF + src).convert('RGB')
            if So_im.size != (Fo.shape[1], Fo.shape[0]):
                So_im = So_im.resize((Fo.shape[1], Fo.shape[0]), Image.BICUBIC)
            So = np.asarray(So_im, dtype=np.int16)
            if Fo.shape[:2] == (H, W):
                Do = np.abs(Fo - So).mean(-1)
                o['orig_mad_in'] = r4(Do[inside].mean())
                o['orig_mad_out'] = float(Do[out].mean())
                o['orig_frac_out_gt8'] = float((np.abs(Fo - So).max(-1)[out] > 8).mean())
            # re-encode noise of the cleaning itself: clean twin vs original source, converted the same way
            Sc = np.asarray(So_im.convert('L' if r['organ'] in B.GRAY else 'RGB'), dtype=np.int16)
            if Sc.shape == T.shape:
                o['reencode_mad_twin_vs_src'] = float(np.abs(T.astype(np.int16) - Sc).mean())
        return o
    except Exception as e:  # noqa
        return dict(image=r.get('image'), ok=0, why=repr(e)[:200])


# ----------------------------------------------------------------------------------------------- report
PCT_COL = __import__('re').compile(r'(acc|recall|F1|2AFC|2afc|majority)', __import__('re').I)


def md_table(df, cols=None, rename=None):
    if cols:
        df = df[cols]
    if rename:
        df = df.rename(columns=rename)
    lines = ['| ' + ' | '.join(map(str, df.columns)) + ' |', '|' + '---|' * len(df.columns)]
    for _, r in df.iterrows():
        cells = []
        for c, v in zip(df.columns, r.values):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append('–')
            elif isinstance(v, (float, np.floating)):
                if PCT_COL.search(str(c)):
                    cells.append(f'{v:.2f}')               # percentages
                elif abs(v) < 0.01 and v != 0:
                    cells.append(f'{v:.2e}')
                else:
                    cells.append(f'{v:.4f}' if abs(v) < 10 else f'{v:.2f}')
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def ba(df, idx_cols, val, versions=('orig', 'clean')):
    """before/after pivot: one row per idx, columns val_before / val_after."""
    p = df.pivot_table(index=idx_cols, columns='version', values=val, aggfunc='first')
    p = p.reindex(columns=list(versions))
    p.columns = [f'{val} {"BEFORE" if v == "orig" else "AFTER"}' for v in p.columns]
    return p.reset_index()


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mfclean', default=_release_path('AAAI/MedForge++/mfclean'))
    ap.add_argument('--out_dir', default=V3 + '/reports/shortcut_mfclean')
    ap.add_argument('--out_md', default=V3 + '/reports/shortcut_mfclean.md')
    ap.add_argument('--cache', default=V3 + '/cache/shortcut_probe/features.jsonl')
    ap.add_argument('--n_train', type=int, default=5000)
    ap.add_argument('--n_twin', type=int, default=300)
    ap.add_argument('--nproc', type=int, default=14)
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--wait_min', type=float, default=0, help='poll for stats.json every 120 s up to this many minutes')
    args = ap.parse_args()
    assert args.nproc <= 16, 'max 16 processes on the shared node'
    t0 = time.time()
    stats_p = os.path.join(args.mfclean, 'stats.json')
    waited = 0
    while not os.path.exists(stats_p) and waited < args.wait_min * 60:
        log(f'waiting for {stats_p} ({waited / 60:.0f}/{args.wait_min:.0f} min)')
        time.sleep(120); waited += 120
    if not os.path.exists(stats_p):
        sys.exit(f'{stats_p} not found: MF-Clean build not finished')
    os.makedirs(args.out_dir, exist_ok=True)
    stats = json.load(open(stats_p))
    train = read_jsonl(os.path.join(args.mfclean, 'train_all.jsonl'))
    test = read_jsonl(os.path.join(args.mfclean, 'test_core.jsonl'))
    sel_p = os.path.join(args.mfclean, 'select_core.jsonl')
    select = read_jsonl(sel_p) if os.path.exists(sel_p) else []
    inputs = {n: md5_file(os.path.join(args.mfclean, n)) for n in ('train_all.jsonl', 'test_core.jsonl', 'stats.json')}
    log(f'train_all {len(train)}  test_core {len(test)}  select_core {len(select)}')
    assert all(r['split'] == 'train' for r in train) and all(r['split'] == 'test' for r in test)
    tr_groups, te_groups = {r['group'] for r in train}, {r['group'] for r in test}
    group_overlap = len(tr_groups & te_groups)
    tr_orig, te_orig = {r['orig'] for r in train}, {r['orig'] for r in test}
    orig_overlap = len(tr_orig & te_orig)

    # ---------- regime samples (nested: one permutation, first n eligible rows per regime -> maximal overlap)
    rng = random.Random(args.seed)
    perm = list(range(len(train)))
    rng.shuffle(perm)
    samples = {}
    for reg, ok in REGIMES.items():
        samples[reg] = [train[i] for i in perm if ok(train[i])][:args.n_train]
    comp = []
    for reg, rs in list(samples.items()) + [('test_core', test)]:
        c = collections.Counter(r['type'] for r in rs)
        g = collections.Counter(r['gen'] for r in rs if r['type'] != 'real')
        comp.append(dict(set=reg, n=len(rs), real=c['real'], edit=c['edit'], remove=c['remove'],
                         frac_fake=r4(1 - c['real'] / max(1, len(rs))), n_gens=len(g),
                         n_CMheld_fakes=sum(v for k, v in g.items() if k in CM),
                         **{f'n_{o}': sum(r['organ'] == o for r in rs) for o in ORGANS}))
    comp = pd.DataFrame(comp)
    comp.to_csv(f'{args.out_dir}/composition.csv', index=False)
    print(comp.to_string())
    for reg in ('CF', 'CFxCM'):
        assert not any(r['type'] == 'edit' for r in samples[reg])
    for reg in ('CM', 'CFxCM'):
        assert not any(r['gen'] in CM for r in samples[reg])
    union_rows = {id(r): r for rs in samples.values() for r in rs}
    union_rows = list(union_rows.values())
    test_twin_in_core = sum(1 for r in test if r['type'] != 'real' and r.get('twin') in {x['image'] for x in test})
    test_fakes = sum(1 for r in test if r['type'] != 'real')

    # ---------- features (clean + orig) for union of train samples + test_core
    paths = [r['image'] for r in union_rows + test] + [r['orig'] for r in union_rows + test]
    feats = get_features(paths, args.cache, args.nproc)
    nbad = sum(1 for p in set(paths) if feats.get(p, {}).get('ok') != 1)
    log(f'feature failures: {nbad}')
    TE = {v: frame(test, v, feats) for v in ('orig', 'clean')}
    TR = {(reg, v): frame(rs, v, feats) for reg, rs in samples.items() for v in ('orig', 'clean')}

    # ---------- (1) colour rule on test_core
    rule = []
    for v in ('orig', 'clean'):
        d = TE[v]
        for org in ORGANS + ['ALL gray (brain+chest)']:
            t = d[d.organ.isin(['brain-mri', 'chest-xray'])] if org.startswith('ALL') else d[d.organ == org]
            if not len(t):
                continue
            y = t.label.values
            p = (t.n_chroma.values > 0).astype(int)
            row = dict(version=v, organ=org, n=len(t), n_real=int((y == 0).sum()), n_fake=int(y.sum()),
                       rule_acc=pct((p == y).mean()), rule_bal_acc=pct(balanced_accuracy_score(y, p)),
                       majority_acc=pct(max(y.mean(), 1 - y.mean())),
                       recall_real=pct((p[y == 0] == 0).mean()) if (y == 0).any() else None)
            for c in ('edit', 'remove'):
                mk = (t.cls == c).values
                row['recall_' + c] = pct((p[mk] == 1).mean()) if mk.any() else None
            row['frac_real_any_chroma'] = r4((t.n_chroma.values[y == 0] > 0).mean()) if (y == 0).any() else None
            row['frac_fake_any_chroma'] = r4((t.n_chroma.values[y == 1] > 0).mean()) if (y == 1).any() else None
            row['chroma_auc'] = r4(C.auc(y, t.n_chroma.values))
            row['mode_counts'] = json.dumps(dict(collections.Counter(t['mode'])))
            rule.append(row)
    rule = pd.DataFrame(rule)
    rule.to_csv(f'{args.out_dir}/colour_rule.csv', index=False)
    print(rule.to_string())

    # ---------- (2)+(3) transfer: train sample -> test_core, before (orig) and after (clean)
    res, sl, scores = [], [], []
    for v in ('orig', 'clean'):
        te = TE[v]
        for reg in REGIMES:
            tr = TR[(reg, v)]
            for fg in FEATS_TRANSFER:
                s = fit_apply(tr, te, fg, 'GB')
                tag = dict(version=v, regime=reg, feat=fg, model='GB')
                row = dict(tag, n_train=len(tr), train_frac_fake=r4(tr.label.mean()), **metrics(te, s))
                res.append(row)
                sl += slices(te, s, tag)
                scores.append(pd.DataFrame(dict(version=v, regime=reg, feat=fg, image=te['image'].values,
                                                cls=te['cls'].values, gen=te['gen'].values, organ=te['organ'].values,
                                                score=np.round(s, 5))))
                log(f"transfer {v:5s} {reg:5s} {fg:13s} auc={row['test_auc']} acc={row['test_acc']} "
                    f"bal={row['test_bal_acc']} F1={row['test_macroF1']} 2afc={row['twin_2afc']}")
    res = pd.DataFrame(res)
    res.to_csv(f'{args.out_dir}/transfer.csv', index=False)
    pd.DataFrame(sl).to_csv(f'{args.out_dir}/transfer_slices.csv', index=False)
    pd.concat(scores).to_csv(f'{args.out_dir}/transfer_scores.csv.gz', index=False, compression='gzip')

    # ---------- within-organ 5-fold group CV (audit sec. 3), on test_core and on the ID train sample
    cv = []
    for pool_name, getd in [('test_core', lambda v: TE[v]), ('train_ID_sample', lambda v: TR[('ID', v)])]:
        for v in ('orig', 'clean'):
            d0 = getd(v)
            for org in ORGANS + ['ALL']:
                sub = d0 if org == 'ALL' else d0[d0.organ == org]
                sub = sub.reset_index(drop=True)
                if sub.label.nunique() < 2 or min(np.bincount(sub.label)) < 2:
                    continue
                for fg, mdl in FEATS_CV:
                    if org == 'ALL' and mdl == 'LR':
                        continue
                    if fg == 'gext_nocolour':
                        continue
                    sc, aucs = C.oof(sub, fg, mdl)
                    cv.append(dict(pool=pool_name, version=v, organ=org, feat=fg, model=mdl, n=len(sub),
                                   n_fake=int(sub.label.sum()), auc=r4(C.auc(sub.label, sc)),
                                   fold_mean=r4(np.mean(aucs)), fold_std=r4(np.std(aucs))))
                log(f'cv {pool_name} {v} {org} done')
    cv = pd.DataFrame(cv)
    cv.to_csv(f'{args.out_dir}/cv_within_organ.csv', index=False)

    # ---------- residual-prior probe: among train reals, can features tell a forgery SOURCE (twin) from a real that
    # was never edited? (a 'looks like a source' prior could be learned as 'fake'; test_core reals are all twins)
    twins_tr = {r['twin'] for r in train if r['type'] != 'real' and r.get('twin')}
    prior = []
    for v in ('orig', 'clean'):
        d0 = TR[('ID', v)]
        rl = d0[d0.cls == 'real'].copy()
        rl['label'] = rl['image'].isin(twins_tr).astype(int)
        for org in ORGANS + ['ALL']:
            sub = (rl if org == 'ALL' else rl[rl.organ == org]).reset_index(drop=True)
            if sub.label.nunique() < 2 or min(np.bincount(sub.label)) < 5:
                prior.append(dict(version=v, organ=org, n=len(sub), n_source=int(sub.label.sum()), feat='-', auc=None))
                continue
            for fg in ['meta', 'gext', 'thumb']:
                sc, aucs = C.oof(sub, fg, 'GB')
                prior.append(dict(version=v, organ=org, n=len(sub), n_source=int(sub.label.sum()), feat=fg,
                                  auc=r4(C.auc(sub.label, sc)), fold_std=r4(np.std(aucs))))
    prior = pd.DataFrame(prior)
    prior.to_csv(f'{args.out_dir}/source_prior_cv.csv', index=False)
    print(prior.to_string())

    # ---------- (4) twin-difference sanity on n_twin random fakes
    pool_fakes = [r for r in train + select + test if r['type'] != 'real' and r.get('twin') and r.get('box')]
    trng = random.Random(args.seed + 1)
    tw_rows = trng.sample(pool_fakes, min(args.n_twin, len(pool_fakes)))
    items = json.load(open(B.MF + 'dataset_mapping.json'))['items']
    mapping = {it['image_path']: it.get('real_image_path') for it in items if not it['image_path'].startswith('real/')}
    with Pool(min(args.nproc, 12), initializer=_init_mapping, initargs=(mapping,)) as pool:
        tw = list(pool.imap(twin_work, tw_rows, chunksize=4))
    tw = pd.DataFrame(tw)
    tw.to_csv(f'{args.out_dir}/twin_diff_per_image.csv', index=False)
    twok = tw[tw.ok == 1].copy()
    summ = []
    for name, sub in [('all', twok), ('twin == exact source', twok[twok.twin_is_exact_source == 1]),
                      ('twin = near-dup representative', twok[twok.twin_is_exact_source == 0])] +\
                     [(f'organ={o}', twok[twok.organ == o]) for o in ORGANS] +\
                     [(f'type={c}', twok[twok.cls == c]) for c in ('edit', 'remove')]:
        if not len(sub):
            continue
        summ.append(dict(subset=name, n=len(sub),
                         mad_in_mean=r4(sub.mad_in.mean()), mad_ring_mean=r4(sub.mad_ring.mean()),
                         mad_out_mean=float(sub.mad_out.mean()), mad_out_median=float(sub.mad_out.median()),
                         mad_out_max=float(sub.mad_out.max()), max_abs_out_max=float(sub.max_out.max()),
                         frac_out_nonzero_mean=float(sub.frac_out_nonzero.mean()),
                         frac_out_gt2_mean=float(sub.frac_out_gt2.mean()),
                         mad_out_blocks_mean=float(sub.mad_out_blocks.mean()),
                         max_abs_out_blocks_max=float(sub.max_out_blocks.max()),
                         n_images_out_blocks_bitexact=int((sub.max_out_blocks == 0).sum()),
                         orig_mad_in_mean=r4(sub.orig_mad_in.mean()) if 'orig_mad_in' in sub else None,
                         orig_mad_out_mean=r4(sub.orig_mad_out.mean()) if 'orig_mad_out' in sub else None,
                         orig_mad_out_median=r4(sub.orig_mad_out.median()) if 'orig_mad_out' in sub else None,
                         orig_frac_out_gt8_mean=r4(sub.orig_frac_out_gt8.mean()) if 'orig_frac_out_gt8' in sub else None,
                         reencode_mad_mean=r4(sub.reencode_mad_twin_vs_src.mean()) if 'reencode_mad_twin_vs_src' in sub else None,
                         alpha_support_px_max=int(sub.alpha_support_px.max()),
                         box_frac_median=r4(sub.box_frac.median())))
    summ = pd.DataFrame(summ)
    summ.to_csv(f'{args.out_dir}/twin_diff_summary.csv', index=False)
    print(summ.to_string())

    # ---------- report
    write_report(args, stats, inputs, comp, rule, res, pd.DataFrame(sl), cv, tw, summ, prior,
                 dict(group_overlap=group_overlap, orig_overlap=orig_overlap, test_twin_in_core=test_twin_in_core,
                      test_fakes=test_fakes, feat_fail=nbad, n_feat_paths=len(set(paths)),
                      minutes=(time.time() - t0) / 60))
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {args.out_md}')


def write_report(args, stats, inputs, comp, rule, res, sl, cv, tw, summ, prior, misc):
    L = []
    A = L.append
    A('# Shortcut probe on MF-Clean: before/after (v3)\n')
    A(f'Generated by `~/CIRCE/v3/shortcut_probe_v3.py` on {datetime.datetime.now():%Y-%m-%d %H:%M}, CPU only '
      f'({misc["minutes"]:.1f} min). Inputs: `{args.mfclean}` (md5 train_all {inputs["train_all.jsonl"][:10]}, '
      f'test_core {inputs["test_core.jsonl"][:10]}, stats {inputs["stats.json"][:10]}). CSVs: `{args.out_dir}/`.\n')
    A('**BEFORE** = the original MedForge-90K files (`orig` field) of exactly the same rows (same train sample, same '
      'test_core, same group-disjoint split). **AFTER** = the MF-Clean files (`image` field: paste-back composite, '
      'gray organs single-channel L, JPEG q90 4:4:4). Features and classifiers are the audit\'s '
      '(`~/CIRCE/v3_audit/shortcut_work/extract.py`, `common.py`: `meta` = file metadata, `gext` = the 16 global '
      'statistics, `thumb` = 16x16 gray thumbnail; sklearn HistGradientBoosting, 200 iters).\n')
    notes = os.path.join(args.out_dir, 'notes.md')
    if os.path.exists(notes):
        A('## Reading\n')
        A(open(notes).read().strip() + '\n')
    # headline
    A('## 1. Headline: shortcut models trained on 5,000 train images, tested on test_core\n')
    h = res[res.feat.isin(['meta', 'gext'])].copy()
    hb = h.pivot_table(index=['regime', 'feat'], columns='version',
                       values=['test_acc', 'test_bal_acc', 'test_macroF1', 'test_auc', 'twin_2afc'], aggfunc='first')
    rows = []
    for (reg, fg), r in hb.iterrows():
        rows.append({'regime': reg, 'features': fg,
                     'acc BEFORE': r[('test_acc', 'orig')], 'acc AFTER': r[('test_acc', 'clean')],
                     'bal-acc BEFORE': r[('test_bal_acc', 'orig')], 'bal-acc AFTER': r[('test_bal_acc', 'clean')],
                     'macro-F1 BEFORE': r[('test_macroF1', 'orig')], 'macro-F1 AFTER': r[('test_macroF1', 'clean')],
                     'AUC BEFORE': r[('test_auc', 'orig')], 'AUC AFTER': r[('test_auc', 'clean')],
                     'twin-2AFC BEFORE': r[('twin_2afc', 'orig')], 'twin-2AFC AFTER': r[('twin_2afc', 'clean')]})
    order = {k: i for i, k in enumerate(REGIMES)}
    hdf = pd.DataFrame(rows)
    hdf['o'] = hdf.regime.map(order)
    hdf = hdf.sort_values(['feat' if 'feat' in hdf else 'features', 'o']).drop(columns='o')
    A(md_table(hdf))
    tc = comp[comp.set == 'test_core'].iloc[0]
    A(f'\ntest_core: n={tc.n} ({tc.real} real / {tc.edit} edit / {tc.remove} remove); always-"fake" accuracy '
      f'{100 * max(tc.frac_fake, 1 - tc.frac_fake):.2f}%, balanced accuracy of any constant predictor 50%. '
      f'twin-2AFC = P(score(fake) > score(its own twin real)) over the {misc["test_twin_in_core"]} of '
      f'{misc["test_fakes"]} test fakes whose twin is in test_core (50% = no signal).\n')
    ref_p = os.path.join(AUDIT, 'res_protocol.csv')
    if os.path.exists(ref_p):
        ref = pd.read_csv(ref_p)
        ref = ref[(ref.model == 'GB') & ref.feat.isin(['meta', 'gext'])]
        A('Reference, OLD protocol (audit `res_protocol.csv`: 5,000 images of each old regime file, tested on '
          '`mfp_v2_test_shared.jsonl`, n=2,994; not group-disjoint, original files):\n')
        A(md_table(ref[['regime', 'feat', 'test_acc', 'test_macroF1', 'test_auc', 'acc_real', 'acc_edit', 'acc_remove',
                        'auc_real_vs_CMheldout_gens']]))
        A('')
    A('### 1.1 Per-class recall and AUC slices (GB)\n')
    cols = ['version', 'regime', 'feat', 'n_train', 'train_frac_fake', 'test_acc', 'test_macroF1', 'test_auc',
            'recall_real', 'recall_edit', 'recall_remove', 'auc_real_vs_edit', 'auc_real_vs_remove',
            'auc_real_vs_CMheld', 'auc_brain-mri', 'auc_chest-xray', 'auc_fundus', 'twin_2afc']
    r2 = res.copy()
    r2['o'] = r2.regime.map(order)
    r2['f'] = r2.feat.map({k: i for i, k in enumerate(FEATS_TRANSFER)})
    r2 = r2.sort_values(['f', 'o', 'version'], ascending=[True, True, False])
    A(md_table(r2, cols))
    A('\nAll metrics at threshold 0.5 except AUC/2AFC. `auc_real_vs_CMheld` = reals vs fakes from the 4 CM held-out '
      'generators (in CM/CFxCM these generators were never seen in training). In CF/CFxCM no implant ("edit") was '
      'seen in training, so `recall_edit` is the cross-forgery number.\n')
    # colour rule
    A('## 2. One-line colour rule "fake iff any pixel has R!=G or G!=B" (no training), test_core\n')
    A(md_table(rule, ['version', 'organ', 'n', 'n_real', 'n_fake', 'rule_acc', 'rule_bal_acc', 'majority_acc',
                      'recall_real', 'recall_edit', 'recall_remove', 'frac_real_any_chroma', 'frac_fake_any_chroma',
                      'chroma_auc', 'mode_counts']))
    A('\nFundus is a colour modality (every real has chroma), so the rule is not meaningful there; it is listed for '
      'completeness.\n')
    # CV
    A('## 3. Within-organ 5-fold group CV AUC (StratifiedGroupKFold on MF-Clean `group`, audit sec. 3 protocol)\n')
    if len(cv):
        c2 = cv.copy()
        c2['fm'] = c2.feat + ' ' + c2.model
        p = c2.pivot_table(index=['pool', 'organ', 'fm'], columns='version', values='auc', aggfunc='first')
        p = p.reindex(columns=['orig', 'clean']).reset_index()
        p.columns = ['pool', 'organ', 'features', 'AUC BEFORE', 'AUC AFTER']
        nn = c2[c2.version == 'clean'].groupby(['pool', 'organ']).agg(n=('n', 'first'), n_fake=('n_fake', 'first')).reset_index()
        p = p.merge(nn, on=['pool', 'organ'], how='left')
        fs = cv[['pool', 'version', 'organ', 'feat', 'model', 'fold_std']]
        fs = fs[fs.version == 'clean'].assign(features=lambda x: x.feat + ' ' + x.model)[['pool', 'organ', 'features', 'fold_std']]
        p = p.merge(fs.rename(columns={'fold_std': 'fold std AFTER'}), on=['pool', 'organ', 'features'], how='left')
        A(md_table(p))
    if len(prior):
        A('\n### 3.1 Residual source-selection prior (train reals only)\n')
        A('Among the reals of the ID train sample: label 1 = the real is the twin (authentic source) of a train fake, '
          '0 = a real of the same organ/disease family that was never edited. 5-fold group CV AUC (GB). A high AUC '
          'would mean a classifier can learn "looks like a forgery source", a non-forensic prior; on test_core it '
          'cannot help because every test real is itself a twin.\n')
        p = prior.pivot_table(index=['organ', 'feat'], columns='version', values='auc', aggfunc='first')
        p = p.reindex(columns=['orig', 'clean']).reset_index()
        p.columns = ['organ', 'features', 'AUC BEFORE', 'AUC AFTER']
        nn = prior[prior.version == 'clean'].groupby('organ').agg(n=('n', 'first'), n_source=('n_source', 'first')).reset_index()
        A(md_table(p.merge(nn, on='organ', how='left')))
    # slices
    A('\n## 4. AUC by generator and forgery type (GB, ID regime, test_core)\n')
    s2 = sl[(sl.regime == 'ID') & sl.feat.isin(['meta', 'gext', 'thumb']) & ~sl.slice.str.contains(r'\|')]
    if len(s2):
        p = s2.pivot_table(index='slice', columns=['feat', 'version'], values='auc', aggfunc='first')
        p.columns = [f'{f} {"BEFORE" if v == "orig" else "AFTER"}' for f, v in p.columns]
        p = p[[c for f in ['meta', 'gext', 'thumb'] for c in (f + ' BEFORE', f + ' AFTER') if c in p.columns]]
        nf = s2[(s2.version == 'clean') & (s2.feat == 'gext')].set_index('slice').n_fake
        p.insert(0, 'n_fake', nf)
        A(md_table(p.reset_index()))
    s3 = sl[(sl.regime == 'ID') & (sl.feat == 'gext') & sl.slice.str.contains(r'\|')]
    if len(s3):
        A('\nPer organ x generator (gext GB, ID): see `transfer_slices.csv`; range of AUC over organ x generator cells: '
          + ', '.join(f'{"BEFORE" if v == "orig" else "AFTER"} {g.auc.min():.3f}-{g.auc.max():.3f} (median {g.auc.median():.3f})'
                      for v, g in s3.groupby('version', sort=False)) + '.\n')
    # twin diff
    A('## 5. Twin-difference sanity: is a fake identical to its twin real outside the box?\n')
    A(f'{int((tw.ok == 1).sum())} of {len(tw)} random fakes (train + select + test) checked; failures: '
      f'{int((tw.ok != 1).sum())}. Values are mean absolute differences in grey levels (RGB: mean over channels). '
      '"outside" = pixels where the paste-back alpha (box dilated by '
      f'{B.DIL} px, Gaussian feather sigma {B.FEATHER}) is exactly 0; "outside, clear blocks" = 8x8 JPEG blocks that '
      'do not touch alpha > 0 (JPEG 4:4:4 codes each block independently, so these should be bit-identical). '
      'BEFORE = original full-frame fake vs its original source on the same outside region (audit sec. 5 found '
      '3.1-8.2 median); re-encode = clean twin vs original source (the cleaning\'s own JPEG q90 noise).\n')
    A(md_table(summ))
    # composition and integrity
    A('\n## 6. Samples, composition and integrity checks\n')
    A(md_table(comp))
    A(f'\n- Train/test group overlap: {misc["group_overlap"]} groups; train/test original-file overlap: '
      f'{misc["orig_overlap"]} files.')
    A(f'- Regime samples are nested (one seeded permutation of train_all, first {args.n_train} eligible rows per '
      'regime). CF/CFxCM samples contain 0 edit fakes and CM/CFxCM samples 0 fakes of '
      f'{sorted(CM)} (asserted in code).')
    A(f'- Feature extraction: {misc["n_feat_paths"]} distinct files, {misc["feat_fail"]} failures.')
    A(f'- MF-Clean stats.json: `{json.dumps({k: v for k, v in stats.items() if k != "drops"})[:900]}`')
    A(f'- Drops: `{json.dumps(stats.get("drops", {}))}`\n')
    A('## 7. Files\n')
    for f in sorted(os.listdir(args.out_dir)):
        if not f.startswith('.'):
            A(f'- `{args.out_dir}/{f}`')
    A('\nReproduce: `NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8 nice -n 19 ~/.conda/envs/aiface/bin/python '
      '~/CIRCE/v3/shortcut_probe_v3.py --nproc 14`\n')
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    tmp = args.out_md + '.tmp'
    with open(tmp, 'w') as f:
        f.write('\n'.join(L) + '\n')
    os.replace(tmp, args.out_md)


if __name__ == '__main__':
    main()
