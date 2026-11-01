
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import pandas as pd, re, os
from collections import Counter, defaultdict
df=pd.read_csv(_release_path('CIRCE/v3_audit/data/protocol_rows.tsv'),sep='\t',dtype=str,keep_default_na=False)
pp=df[df.organ.str.contains(':')].copy()
pp['subset']=pp.organ.str.split(':').str[1]
def ids(p):
    b=os.path.basename(p)
    m=re.match(r'(case_\d{5})_slice_(\d+)',b)
    if m: return 'kits:'+m.group(1), 'kits:'+m.group(1)+'_s'+m.group(2)
    m=re.match(r'(?:unresolved_)?(c[jk][0-9a-z]{20,})',b)
    if m: return 'kvasir:'+m.group(1),'kvasir:'+m.group(1)
    m=re.match(r'Normal-_?\((\d+)\)',b.replace(' ','_'))
    if m: return 'kagglekid:'+m.group(1),'kagglekid:'+m.group(1)
    m=re.match(r'train_normal_ ?\((\d+)\)',b)
    if m: return 'colonnormal:'+m.group(1),'colonnormal:'+m.group(1)
    return 'other:'+b,'other:'+b
pp['case'],pp['slice']=zip(*pp.path.map(ids))
print('id namespace per subset:'); print(pp.drop_duplicates('path').groupby('subset').case.agg(lambda s: dict(Counter(x.split(':')[0] for x in s))).to_string())
ev=pp[pp.file=='circe_eval_split_v2.jsonl']
for tr in ['circe_sft_v3.jsonl','circe_sft_v2.jsonl','circe_sft_full.jsonl','pfplus_replay_removal_focus_sft.jsonl','pfplus_replay_v2_balanced_sft.jsonl']:
    T=pp[pp.file==tr]; Tp=set(T.path); Tc=set(T.case); Ts=set(T.slice)
    print(f'\n[{tr} (uniq {len(Tp)}) vs circe_eval_split_v2 (800)] exact-path overlap = {sum(p in Tp for p in ev.path)}')
    r=ev.assign(path_in=ev.path.isin(Tp), slice_in=ev.slice.isin(Ts), case_in=ev.case.isin(Tc))
    print(r.groupby('subset')[['path_in','slice_in','case_in']].sum().astype(int).assign(n=r.groupby('subset').size()).to_string())
# within the MF++ addition: case sharing between subsets
u=pp.drop_duplicates('path')
k=u[u.case.str.startswith('kits')]
print('\nKiTS: unique cases per subset', k.groupby('subset').case.nunique().to_dict(), 'slices', k.groupby('subset').slice.nunique().to_dict())
cs={s:set(g.case) for s,g in k.groupby('subset')}
ss={s:set(g.slice) for s,g in k.groupby('subset')}
for a in cs:
    for b in cs:
        if a<b: print(f'  cases shared {a}&{b}: {len(cs[a]&cs[b])}; identical slices shared: {len(ss[a]&ss[b])}')
print('  total distinct KiTS cases in addition:', len(set(k.case)))
kv=u[u.case.str.startswith('kvasir')]
cs={s:set(g.case) for s,g in kv.groupby('subset')}
print('Kvasir: unique stems per subset',{s:len(v) for s,v in cs.items()})
for a in cs:
    for b in cs:
        if a<b: print(f'  stems shared {a}&{b}: {len(cs[a]&cs[b])}')
print('  total distinct Kvasir stems:', len(set(kv.case)))
# kidney_inject sources: are they present as reals anywhere?
ki=u[u.subset=='kidney_inject']
print('kidney_inject: id namespace', Counter(x.split(':')[0] for x in ki.case), '; distinct source ids', ki.case.nunique())
# eval split case-level leakage when train = sft_v3 minus eval paths (what a 'clean' exclusion by path would give)
T=pp[(pp.file=='circe_sft_v3.jsonl') & (~pp.path.isin(set(ev.path)))]
Tc=set(T.case)
r=ev.assign(case_in=ev.case.isin(Tc))
print('\nIf eval PATHS were removed from circe_sft_v3 (as pfplus replay does): eval rows whose case/stem still in train:')
print(r.groupby('subset').case_in.agg(['sum','size']).to_string())
# group-wise: for each eval image in kidney/polyp, siblings (same slice) in train under other subsets
print('\nSame-slice sibling (other subset) in sft_v3-minus-eval:')
Ts=defaultdict(set)
for _,x in T.iterrows(): Ts[x.slice].add(x.subset)
r=ev.assign(sib=ev.apply(lambda x: len(Ts.get(x.slice,set())-{x.subset})>0,axis=1))
print(r.groupby('subset').sib.agg(['sum','size']).to_string())
