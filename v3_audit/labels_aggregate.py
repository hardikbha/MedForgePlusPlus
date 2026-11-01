"""Aggregate pairdiff + boxstats into tables (markdown snippets printed to stdout and saved)."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, collections, sys
import numpy as np

D = _release_path('CIRCE/v3_audit/')
out_lines = []


def P(s=''):
    print(s); out_lines.append(s)


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else float('nan')


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float('nan')


short = {'flux.1-dev': 'flux-dev', 'gemini': 'gemini', 'gpt': 'gpt', 'qwen': 'qwen', 'seeddream': 'seedream',
         'stable-diffusion-3.5-large': 'sd3.5-L', 'stable-diffusion-3.5-medium': 'sd3.5-M', 'stable-diffusion-inpainting': 'sd-inpaint',
         'stable-diffusion-xl-1.0-inpainting-0.1': 'sdxl-inpaint', 'step1x-edit': 'step1x'}

# ---------------- pairdiff ----------------
R = [json.loads(l) for l in open(D + 'labels_pairdiff.jsonl')]
ok = [r for r in R if r.get('energy_in') is not None and r.get('mask_exists')]
P(f'pairdiff records: {len(R)}; analysable: {len(ok)}; errors: {sum(1 for r in R if r.get("error"))}; '
  f'src missing: {sum(1 for r in R if not r.get("src_exists", True))}; mask missing: {sum(1 for r in R if r.get("src_exists") and not r.get("mask_exists"))}')
P(f'size match forged/source: {np.mean([r["size_match"] for r in ok]):.4f}')
fq = collections.Counter((short[r['model']], r.get('f_q')) for r in ok)
P('forged JPEG quality estimate per generator: ' + str(sorted(fq.items())))
P('source JPEG quality: ' + str(collections.Counter(r.get('s_q') for r in ok).most_common(5)))

# label sanity: edit source should be normal, remove source should be the matching disease folder
def src_cat(r):
    return r['real_image_path'].split('/')[2]
ed = [r for r in ok if r['type'] == 'edit']; rm = [r for r in ok if r['type'] == 'remove']
P(f'edit sources in a /normal/ folder: {np.mean([src_cat(r)=="normal" for r in ed]):.4f} (n={len(ed)})')
P(f'remove sources in a /normal/ folder: {np.mean([src_cat(r)=="normal" for r in rm]):.4f} (n={len(rm)})')
def norm(s):
    return s.lower().replace('-', '_')
P(f'remove source folder == disease label: {np.mean([norm(src_cat(r))==norm(r["disease"]) for r in rm]):.4f}')
mm = collections.Counter((r['dataset'], src_cat(r), r['disease']) for r in rm if norm(src_cat(r)) != norm(r['disease']))
P('remove source/disease mismatches (top): ' + str(mm.most_common(8)))
mme = collections.Counter((r['dataset'], src_cat(r)) for r in ed if src_cat(r) != 'normal')
P('edit sources NOT normal (top): ' + str(mme.most_common(8)))

P()
P('| generator | type | n | box area % (med) | energy in box (med) | L1 in box (med) | mean |d| in box | mean |d| outside | JPEG floor outside | frac out |d|>4 | HF residual out | corr out | |shift|>=1px | verdict |')
P('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
gen_rows = {}
for m in sorted(short):
    for t in ['edit', 'remove']:
        S = [r for r in ok if r['model'] == m and r['type'] == t]
        if not S:
            continue
        e = med([r['energy_in'] for r in S]); l1 = med([r['l1_in'] for r in S])
        mo = mean([r['mad_out'] for r in S]); nf = mean([r['noise_floor_out'] for r in S])
        fo4 = mean([r['frac_out_gt4'] for r in S]); hf = mean([r['hf_out'] for r in S])
        co = med([r.get('corr_out') for r in S])
        sh = np.mean([r['shift_px1024'] is not None and (abs(r['shift_px1024'][0]) >= 1 or abs(r['shift_px1024'][1]) >= 1) for r in S])
        mi = mean([r['mad_in'] for r in S])
        verdict = 'LOCAL' if (mo < 2.0 and fo4 < 0.10) else ('GLOBAL re-encode' if fo4 > 0.25 else 'mixed/near-global')
        gen_rows[(m, t)] = dict(n=len(S), energy_in=e, mad_out=mo, frac_out_gt4=fo4, verdict=verdict)
        P(f'| {short[m]} | {t} | {len(S)} | {100*med([r["area_frac"] for r in S]):.2f} | {e:.3f} | {l1:.3f} | {mi:.1f} | {mo:.2f} | {nf:.2f} | {fo4:.3f} | {hf:.2f} | {co:.4f} | {sh:.2f} | {verdict} |')

P()
P('Per generator (both types pooled), organ breakdown of mean |d| outside:')
P('| generator | brain | chest | fundus |')
P('|---|---|---|---|')
for m in sorted(short):
    row = []
    for o in ['brain-mri', 'chest-xray', 'fundus']:
        S = [r for r in ok if r['model'] == m and r['dataset'] == o]
        row.append(f'{mean([r["mad_out"] for r in S]):.2f} (E_in {med([r["energy_in"] for r in S]):.2f})')
    P(f'| {short[m]} | ' + ' | '.join(row) + ' |')

# global
P()
P(f'ALL: median energy-in-box {med([r["energy_in"] for r in ok]):.3f}; median box area {100*med([r["area_frac"] for r in ok]):.2f}%; '
  f'frac samples with energy_in<0.5: {np.mean([r["energy_in"]<0.5 for r in ok]):.3f}; frac with energy_in<0.25: {np.mean([r["energy_in"]<0.25 for r in ok]):.3f}')
P(f'ALL: mean |d| ring(0-32px outside box) {mean([r.get("mad_ring32") for r in ok]):.2f} vs far(>64px) {mean([r.get("mad_far64") for r in ok]):.2f}')
loc = [r for r in ok if gen_rows[(r['model'], r['type'])]['verdict'] == 'LOCAL']
glo = [r for r in ok if gen_rows[(r['model'], r['type'])]['verdict'] != 'LOCAL']
P(f'samples from LOCAL (gen,type) cells: {len(loc)} ; from global/mixed cells: {len(glo)} ({len(glo)/len(ok):.3f})')

# ---------------- thresholded diff-map target ----------------
P()
P('Thresholded difference-map target vs bbox (Gaussian sigma=2 on |forged-source|, comps>=64px):')
P('| subset | thr | n | diffmask area % (med) | IoU(diffmask,bbox) med | precision in bbox med | bbox coverage med | area ratio diff/bbox med | tight-box IoU vs bbox med | tight-box area/bbox med |')
P('|---|---|---|---|---|---|---|---|---|---|')
for name, S in [('LOCAL cells', loc), ('GLOBAL/mixed cells', glo), ('ALL', ok)]:
    for thr in ['t8', 't16', 't24', 'adapt']:
        T = [r['thr'][thr] for r in S if r.get('thr')]
        P(f'| {name} | {thr} | {len(T)} | {100*med([t["area_frac"] for t in T]):.2f} | {med([t["iou_bbox"] for t in T]):.3f} | '
          f'{med([t["prec_in_bbox"] for t in T]):.3f} | {med([t["bbox_cov"] for t in T]):.3f} | {med([t["area_ratio_vs_bbox"] for t in T]):.3f} | '
          f'{med([t.get("tightbox_iou_bbox") for t in T]):.3f} | {med([t.get("tightbox_area_ratio") for t in T]):.3f} |')
P()
P('t16 per generator x type (LOCAL-relevant): IoU(diffmask,bbox) / bbox coverage / tight-box IoU / tight-box area ratio')
P('| generator | edit | remove |')
P('|---|---|---|')
for m in sorted(short):
    row = []
    for t in ['edit', 'remove']:
        T = [r['thr']['t16'] for r in ok if r['model'] == m and r['type'] == t and r.get('thr')]
        row.append(f'{med([x["iou_bbox"] for x in T]):.2f} / {med([x["bbox_cov"] for x in T]):.2f} / {med([x.get("tightbox_iou_bbox") for x in T]):.2f} / {med([x.get("tightbox_area_ratio") for x in T]):.2f}')
    P(f'| {short[m]} | ' + ' | '.join(row) + ' |')
# fraction where edit is essentially invisible (mad_in small)
P()
P(f'fraction of forgeries with mean |d| inside box < 5 gray levels (edit barely changed pixels): {np.mean([r["mad_in"]<5 for r in ok]):.3f}')
for t in ['edit', 'remove']:
    P(f'  {t}: {np.mean([r["mad_in"]<5 for r in ok if r["type"]==t]):.3f}')
P(f'fraction where mean |d| inside < mean |d| outside (box misses the edit): {np.mean([r["mad_in"] < r["mad_out"] for r in ok]):.3f}')
bad = collections.Counter(short[r['model']] + '/' + r['type'] for r in ok if r['mad_in'] < r['mad_out'])
P('  by gen/type: ' + str(bad.most_common(10)))

open(D + 'labels_aggregate_pairdiff.md', 'w').write('\n'.join(out_lines))
out_lines.clear()

# ---------------- boxstats ----------------
B = [json.loads(l) for l in open(D + 'labels_boxstats.jsonl')]
P(f'boxstats records: {len(B)}')
for src in ['MF90K', 'MF++']:
    S = [b for b in B if b['src'] == src]
    P(f'{src}: n={len(S)} missing={sum(1 for b in S if b.get("missing"))} errors={sum(1 for b in S if b.get("error"))} '
      f'empty={sum(1 for b in S if b.get("empty"))} non-binary(>2 unique)={sum(1 for b in S if b.get("n_unique",0)>2)} '
      f'multi-component={sum(1 for b in S if b.get("n_comp",1)>1)} non-rectangular(fill<0.98)={sum(1 for b in S if b.get("fill_of_bbox",1)<0.98)}')
    P(f'  mask sizes: {collections.Counter(tuple(b["m_size"]) for b in S if b.get("m_size")).most_common(4)}')
P()
P('| set | organ | type | n | area % p10 | median | p90 | max | >50% frame | >25% | <0.5% (tiny) | <0.1% | multi-comp |')
P('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
groups = collections.defaultdict(list)
for b in B:
    if b.get('area_frac') is None:
        continue
    groups[(b['src'], b['organ'], b['type'])].append(b)
for k in sorted(groups):
    S = groups[k]; a = np.array([b['area_frac'] for b in S])
    P(f'| {k[0]} | {k[1]} | {k[2]} | {len(S)} | {100*np.percentile(a,10):.2f} | {100*np.median(a):.2f} | {100*np.percentile(a,90):.2f} | {100*a.max():.1f} | '
      f'{np.mean(a>0.5):.4f} | {np.mean(a>0.25):.4f} | {np.mean(a<0.005):.4f} | {np.mean(a<0.001):.4f} | {np.mean([b.get("n_comp",1)>1 for b in S]):.4f} |')
P()
P('MF90K per generator: median area % / >25% / <0.5% / multi-comp')
for m in sorted(short):
    S = [b for b in B if b['src'] == 'MF90K' and b['model'] == m and b.get('area_frac') is not None]
    a = np.array([b['area_frac'] for b in S])
    P(f'  {short[m]}: n={len(S)} med={100*np.median(a):.2f}% >25%={np.mean(a>0.25):.4f} <0.5%={np.mean(a<0.005):.4f} multi={np.mean([b.get("n_comp",1)>1 for b in S]):.4f}')
# MF++ json box vs mask
S = [b for b in B if b['src'] == 'MF++' and b.get('bbox') and b.get('json_box')]
def iou(a, c):
    ix = max(0, min(a[2], c[2]) - max(a[0], c[0])); iy = max(0, min(a[3], c[3]) - max(a[1], c[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (c[2] - c[0]) * (c[3] - c[1]) - i
    return i / u if u > 0 else 0
ious = [iou(b['bbox'], b['json_box']) for b in S]
P(f'MF++ mapping-json box vs mask bbox IoU: n={len(ious)} median={np.median(ious):.3f} frac<0.5={np.mean(np.array(ious)<0.5):.4f}')
for ty in ['kidney_real_tumor', 'kidney_removal', 'polyp_real', 'polyp_removal']:
    T = [b for b in B if b['src'] == 'MF++' and b['type'] == ty and b.get('bbox')]
    px = np.array([(b['bbox'][2]-b['bbox'][0])*(b['bbox'][3]-b['bbox'][1]) for b in T])
    side = np.array([min(b['bbox'][2]-b['bbox'][0], b['bbox'][3]-b['bbox'][1]) for b in T])
    P(f'  {ty}: n={len(T)} box px median={np.median(px):.0f} p10={np.percentile(px,10):.0f} min={px.min()} ; short side median={np.median(side):.0f}px, <16px={np.mean(side<16):.3f}, <32px={np.mean(side<32):.3f}; fill<0.98={np.mean([b.get("fill_of_bbox",1)<0.98 for b in T]):.3f}')
open(D + 'labels_aggregate_boxes.md', 'w').write('\n'.join(out_lines))
