
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
# Do sibling forgeries (same source scan, same type, same disease, different generator) share the GT box?
import json, re, numpy as np
from collections import defaultdict
MF=_release_path('AAAI/MedForge++/medforge90k_full/')
D=_release_path('AAAI/MedForge++/')
info={}
for it in json.load(open(MF+'dataset_mapping.json'))['items']:
    if it['type']!='real': info[MF+it['image_path']]=(it['real_image_path'],it['type'],it['disease'],it['model'])
BX=re.compile(r'x1="(\d+)" y1="(\d+)" x2="(\d+)" y2="(\d+)"')
def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1]))
    I=ix*iy; U=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-I
    return I/U if U>0 else 0
for tr in ['mfp_train_indomain.jsonl','mfp_train_crossmodel.jsonl']:
    sig=defaultdict(list)
    for l in open(D+tr):
        r=json.loads(l); p=r['images'][0]
        if p not in info: continue
        m=BX.findall(r['messages'][-1]['content'])
        if m: sig[info[p][:3]].append(tuple(int(v) for v in m[0]))
    for ev in ['mfp_v2_test_shared.jsonl','v3_select/mfp_v3_select_clean.jsonl']:
        best=[]; nb=0; n=0
        for l in open(D+ev):
            r=json.loads(l); p=r['image']
            if p not in info or not r.get('box'): continue
            n+=1
            s=sig.get(info[p][:3])
            if not s: continue
            nb+=1; best.append(max(iou(r['box'],b) for b in s))
        best=np.array(best)
        print(f'{ev} vs {tr}: test forgeries with box={n}; with same-edit sibling(s) in train={nb}; max sibling-box IoU: median {np.median(best):.3f}, mean {best.mean():.3f}, frac>=0.9 {np.mean(best>=0.9):.3f}, frac>=0.5 {np.mean(best>=0.5):.3f}')
