
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import re, os, numpy as np, pandas as pd
from collections import Counter
A=_release_path('CIRCE/v3_audit/')
h=pd.read_csv(A+'data/image_hashes.tsv',sep='\t',dtype={'md5':str,'dhash':str,'err':str},keep_default_na=False)
h=h[h.group!='mf90k'].reset_index(drop=True)
th=np.load(A+'data/thumbs32.npy').astype(np.int16)
LUT=np.array([bin(i).count('1') for i in range(1<<16)],dtype=np.uint8)
D=np.array([np.uint64(int(s,16)) for s in h.dhash],dtype=np.uint64); I=h.idx.astype(int).values
def key(p):
    b=os.path.basename(p)
    m=re.match(r'(case_\d{5})_slice_(\d+)',b)
    if m: return 'kits:'+m.group(1), int(m.group(2))
    m=re.match(r'(?:unresolved_)?(c[jk][0-9a-z]{20,})',b)
    if m: return 'kvasir:'+m.group(1), None
    m=re.match(r'Normal-[ _]\((\d+)\)',b)
    if m: return 'kagglekid:'+m.group(1), None
    return 'other:'+h.md5[h.path==p].iloc[0], None
K=[key(p) for p in h.path]
cnt=Counter(); kind=Counter()
pairs=[]
for s in range(0,len(D),512):
    x=np.bitwise_xor(D[s:s+512,None],D[None,:]); H=LUT[x.view(np.uint16).reshape(x.shape[0],x.shape[1],4)].sum(axis=2)
    a,b=np.nonzero(H<=4)
    for i,j in zip(s+a,b):
        if i>=j: continue
        if np.abs(th[I[i]]-th[I[j]]).mean()>3: continue
        gi,gj=sorted([h.group[i],h.group[j]])
        ki,kj=K[i],K[j]
        if ki[0]==kj[0]:
            k='same-slice/frame' if (ki[1]==kj[1]) else 'same-case-other-slice'
        else: k='different-case'
        cnt[(gi,gj,k)]+=1; pairs.append((h.path[i],h.path[j],k))
print('Verified (dHash<=4 & thumb-MAE<=3) near-dup pairs inside the MF++ addition:')
for k,v in sorted(cnt.items()): print('  ',k,v)
# eval split leakage if eval paths are removed from circe_sft_v3
pr=pd.read_csv(A+'data/protocol_rows.tsv',sep='\t',dtype=str,keep_default_na=False)
ev=pr[pr.file=='circe_eval_split_v2.jsonl']; evs=set(ev.path)
tr=set(pr.path[pr.file=='circe_sft_v3.jsonl'])-evs
sub=dict(zip(ev.path,ev.path.str.extract(r'(naitik_annotated/[^/]+|colon_injections|flux_injected)')[0]))
nb=Counter(); kinds=Counter()
hit={}
for a,b,k in pairs:
    for e,t in [(a,b),(b,a)]:
        if e in evs and t in tr:
            hit.setdefault(e,set()).add(k)
for e,ks in hit.items():
    nb[sub[e]]+=1
    for k in ks: kinds[(sub[e],k)]+=1
print('\nIf eval paths were simply removed from circe_sft_v3: eval images with a verified near-dup still in train, by subset:',dict(nb),'total',len(hit))
print('   by relation:',dict(kinds))

# effective number of distinct images per subset (connected components of verified near-dup graph within subset)
par={}
def f(x):
    par.setdefault(x,x)
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
g=dict(zip(h.path,h.group))
for a,b,k in pairs:
    if g[a]==g[b]: par[f(a)]=f(b)
print('\nEffective distinct images per subset (verified near-dup components within the subset):')
for grp,sub_ in h.groupby('group'):
    comps={f(p) for p in sub_.path}
    print(f'   {grp:28s} files={len(sub_):5d} components={len(comps)}')
