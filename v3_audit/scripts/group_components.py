
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
# Feasibility of a leak-free grouped split for MF90K: connected components over
# (forgery -> its source scan) + (same patient: fundus both eyes / Yale patient) + (verified duplicate reals).
import json, os, re, numpy as np, pandas as pd
from collections import Counter, defaultdict
A=_release_path('CIRCE/v3_audit/'); MF=_release_path('AAAI/MedForge++/medforge90k_full/')
items=json.load(open(MF+'dataset_mapping.json'))['items']
par={}
def f(x):
    par.setdefault(x,x)
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
def u(a,b): par[f(a)]=f(b)
ondisk=set()
for it in items:
    p=it['image_path']
    if not os.path.exists(MF+p): continue
    ondisk.add(p); f(p)
    if it['type']!='real': u(p,it['real_image_path'])   # source link (source may be missing on disk: still a group key)
    b=os.path.splitext(os.path.basename(p if it['type']=='real' else it['real_image_path']))[0]
    org=it['dataset']
    m=re.match(r'^(\d+)_(left|right)(_eyepacs)?$',b) if org=='fundus' else None
    if m: u(p,'PAT:fundus'+('E' if m.group(3) else 'O')+m.group(1))
    m=re.match(r'^(YG_[A-Z0-9]+)_',b) if org=='brain-mri' else None
    if m: u(p,'PAT:'+m.group(1))
# verified duplicate reals (dHash<=4 & thumb MAE<=3) and md5 duplicates
h=pd.read_csv(A+'data/image_hashes.tsv',sep='\t',dtype={'md5':str,'dhash':str,'err':str},keep_default_na=False)
h=h[(h.group=='mf90k')].copy(); h['rel']=h.path.str.replace(MF,'',regex=False)
for m_,g in h.groupby('md5'):
    if len(g)>1:
        L=list(g.rel)
        for x in L[1:]: u(L[0],x)
th=np.load(A+'data/thumbs32.npy').astype(np.int16)
LUT=np.array([bin(i).count('1') for i in range(1<<16)],dtype=np.uint8)
r=h[h.rel.str.startswith('real/')]
for org in ['brain-mri','chest-xray','fundus']:
    s=r[r.rel.str.startswith('real/'+org)]
    L=list(s.rel); I=s.idx.astype(int).values; D=np.array([np.uint64(int(x,16)) for x in s.dhash],dtype=np.uint64)
    for st in range(0,len(L),512):
        x=np.bitwise_xor(D[st:st+512,None],D[None,:]); H=LUT[x.view(np.uint16).reshape(x.shape[0],x.shape[1],4)].sum(axis=2)
        a,b=np.nonzero(H<=4)
        for i,j in zip(st+a,b):
            if i<j and np.abs(th[I[i]]-th[I[j]]).mean()<=3: u(L[i],L[j])
comp=defaultdict(list)
for p in ondisk: comp[f(p)].append(p)
sizes=np.array([len(v) for v in comp.values()])
info={it['image_path']:it for it in items}
print('on-disk MF90K images',len(ondisk),'-> leak-free groups (connected components):',len(comp))
print('group size: median',int(np.median(sizes)),'p90',int(np.percentile(sizes,90)),'max',sizes.max(),'| singletons',int((sizes==1).sum()))
big=sorted(comp.values(),key=len,reverse=True)[:5]
for v in big: print('  largest group',len(v),Counter(info[x]['dataset'] for x in v),Counter(info[x]['type'] for x in v))
# groups containing forgeries vs pure-real groups
hasf=[v for v in comp.values() if any(info[x]['type']!='real' for x in v)]
print('groups containing >=1 forgery:',len(hasf),'covering',sum(len(v) for v in hasf),'images; pure-real groups:',len(comp)-len(hasf),'covering',sum(len(v) for v in comp.values() if not any(info[x]['type']!='real' for x in v)))
for org in ['brain-mri','chest-xray','fundus']:
    vs=[v for v in hasf if info[v[0]]['dataset']==org]
    print(f'  {org}: forgery-bearing groups {len(vs)}, images {sum(len(v) for v in vs)}, forgeries {sum(1 for v in vs for x in v if info[x]["type"]!="real")}')
