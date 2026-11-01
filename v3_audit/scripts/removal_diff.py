
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
# Paired check: how much does each removal image differ from its same-source real (inside vs outside the box)?
import json, numpy as np, os, re
from PIL import Image
R=_release_path('AAAI/MedForge++/naitik_annotated/')
out=[]
for org,rem,real in [('polyp','polyp_removal','polyp_real'),('kidney','kidney_removal','kidney_real_tumor')]:
    m=json.load(open(R+rem+'/mapping_0_1000.json'))
    res=[]
    for x in m:
        b=os.path.basename(x['image_path']); stem=re.sub('^unresolved_','',b)
        rp=R+real+'/images/'+stem
        if not os.path.exists(rp): res.append((b,None,None,None)); continue
        a=np.asarray(Image.open(x['image_path']).convert('L'),dtype=np.float32)
        r=np.asarray(Image.open(rp).convert('L'),dtype=np.float32)
        if a.shape!=r.shape: res.append((b,'shape',a.shape,r.shape)); continue
        x1,y1,x2,y2=[int(v) for v in x['box']]; H,W=a.shape
        sx,sy=W/1024,H/1024
        msk=np.zeros_like(a,bool); msk[int(y1*sy):int(y2*sy),int(x1*sx):int(x2*sx)]=True
        d=np.abs(a-r)
        res.append((b,float(d[msk].mean()) if msk.any() else -1,float(d[~msk].mean()),float(msk.mean())))
    ok=[t for t in res if isinstance(t[1],float)]
    unres=[t for t in ok if t[0].startswith('unresolved_')]; resl=[t for t in ok if not t[0].startswith('unresolved_')]
    def s(L):
        if not L: return 'n=0'
        i=np.array([t[1] for t in L]); o=np.array([t[2] for t in L]); f=np.array([t[3] for t in L])
        return f'n={len(L)} in-box |diff| median {np.median(i):.2f} (p10 {np.percentile(i,10):.2f}); out-box median {np.median(o):.2f}; in-box<1.0 gray level: {(i<1.0).sum()}; box area frac median {np.median(f):.3f}'
    print(org,'paired with same-source real:',len(ok),'of',len(m),'| missing pair',sum(t[1] is None for t in res),'| shape mismatch',sum(t[1]=='shape' for t in res))
    print('   resolved  :',s(resl)); print('   unresolved:',s(unres))
