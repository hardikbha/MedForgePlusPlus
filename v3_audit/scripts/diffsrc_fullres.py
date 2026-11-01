
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
# Full-resolution (256x256 gray) check of 'different-source' verified near-dup pairs (train vs eval).
import json, os, re, numpy as np, pandas as pd
from PIL import Image
from collections import Counter
A=_release_path('CIRCE/v3_audit/'); MF=_release_path('AAAI/MedForge++/medforge90k_full/')
h=pd.read_csv(A+'data/image_hashes.tsv',sep='\t',dtype={'md5':str,'dhash':str,'err':str},keep_default_na=False)
dh={p:np.uint64(int(s,16)) for p,s in zip(h.path,h.dhash)}; idx=dict(zip(h.path,h.idx.astype(int))); md5=dict(zip(h.path,h.md5))
th=np.load(A+'data/thumbs32.npy').astype(np.int16)
LUT=np.array([bin(i).count('1') for i in range(1<<16)],dtype=np.uint8)
def ham(a,B):
    x=np.bitwise_xor(a[:,None],B[None,:]); return LUT[x.view(np.uint16).reshape(x.shape[0],x.shape[1],4)].sum(axis=2,dtype=np.uint8)
info={}
for it in json.load(open(MF+'dataset_mapping.json'))['items']:
    info[MF+it['image_path']]=dict(cls=it['type'],src=MF+(it.get('real_image_path') or it['image_path']),organ=it['dataset'])
pr=pd.read_csv(A+'data/protocol_rows.tsv',sep='\t',dtype=str,keep_default_na=False)
files={fn:sorted(set(g.path)) for fn,g in pr.groupby('file')}
def g256(p): return np.asarray(Image.open(p).convert('L').resize((256,256),Image.BILINEAR),dtype=np.float32)
def run(ev,tr):
    E=[e for e in files[ev] if e in info]; T=[t for t in files[tr] if t in info]
    Td=np.array([dh[t] for t in T],dtype=np.uint64); Tth=th[[idx[t] for t in T]]
    same,diff=[],[]
    for s in range(0,len(E),256):
        Ed=np.array([dh[e] for e in E[s:s+256]],dtype=np.uint64); H=ham(Ed,Td)
        for k in range(H.shape[0]):
            e=E[s+k]; js=np.nonzero(H[k]<=4)[0]
            if not len(js): continue
            mm=np.abs(Tth[js]-th[idx[e]]).mean(axis=(1,2))
            for j,m in zip(js,mm):
                if m<=3 and T[j]!=e:
                    (same if info[T[j]]['src']==info[e]['src'] else diff).append((e,T[j]))
    rng=np.random.default_rng(0)
    ss=[same[i] for i in rng.choice(len(same),min(300,len(same)),replace=False)]
    m_same=np.array([np.abs(g256(a)-g256(b)).mean() for a,b in ss])
    # one pair per eval image for diff
    seen=set(); dd=[]
    for a,b in diff:
        if a not in seen: seen.add(a); dd.append((a,b))
    m_diff=np.array([np.abs(g256(a)-g256(b)).mean() for a,b in dd])
    byte=sum(md5[a]==md5[b] for a,b in dd)
    print(f'[{ev} vs {tr}] same-source verified pairs={len(same)} (sampled {len(ss)}): 256px MAE median {np.median(m_same):.2f} p90 {np.percentile(m_same,90):.2f}')
    print(f'   different-source eval images={len(dd)}: 256px MAE median {np.median(m_diff):.2f} p90 {np.percentile(m_diff,90):.2f}; <= same-source p90: {(m_diff<=np.percentile(m_same,90)).sum()}; byte-identical {byte}')
    c=Counter((info[a]['organ'],info[a]['cls'],info[b]['cls']) for (a,b),m in zip(dd,m_diff) if m<=np.percentile(m_same,90))
    print('   confirmed diff-source dups by (organ, eval cls, train cls):',dict(c))
    for (a,b),m in sorted(zip(dd,m_diff),key=lambda t:t[1])[:4]: print(f'     m256={m:.2f} {a.replace(MF,"")} ~ {b.replace(MF,"")}')
run('mfp_v2_test_shared.jsonl','mfp_train_indomain.jsonl')
run('mfp_v2_test_shared.jsonl','mfp_train_crossforgery.jsonl')
run('v3_select/mfp_v3_select_clean.jsonl','mfp_train_indomain.jsonl')
