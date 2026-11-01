
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json,re,os,collections
R=_release_path('AAAI/MedForge++')
pat=re.compile(r'(/home/sachin\.chaudhary/AAAI/[^"\s]+?\.(?:png|jpe?g))',re.I)
def paths(f):
    s=[]
    for l in open(f,errors="ignore"): s+= [p for p in pat.findall(l) if ('naitik_annotated' in p or 'colon_injections' in p or 'kidney/flux' in p or 'kvasir' in p.lower())]
    return s
def group(p):
    b=os.path.basename(p)
    m=re.match(r'(case_\d+)_slice',b)
    if m: return 'kits:'+m.group(1)
    b=b.replace('unresolved_','').replace('_polypinj.jpg','')
    b=os.path.splitext(b)[0]
    if b.startswith('c') and len(b)>20: return 'kvasir:'+b
    return 'other:'+b
ev=[json.loads(l)["image"] for l in open(f"{R}/circe_eval_split_v2.jsonl")]
evg=collections.Counter(group(p) for p in ev)
evp=set(ev)
for tf in ["circe_sft_v3.jsonl","circe_sft_v2.jsonl","circe_sft_full.jsonl"]:
    tp=paths(f"{R}/{tf}"); tps=set(tp); tg=set(group(p) for p in tp)
    sub=collections.Counter()
    for e in json.loads('['+','.join(open(f"{R}/circe_eval_split_v2.jsonl").read().split('\n')[:-1])+']'):
        if group(e["image"]) in tg: sub[e["subset"]]+=1
    print(f"{tf}: MF++ image refs {len(tps)}; eval images also in train (exact path): {len(evp&tps)}; eval items whose KiTS case / Kvasir frame group appears in train: {sum(group(p) in tg for p in ev)}/800 ->", dict(sub))
