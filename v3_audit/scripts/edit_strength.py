
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
# Sampled check of MF90K forgeries vs their authentic source: is the edit real, and is it confined to the mask?
import json, os, random, numpy as np
from PIL import Image
from multiprocessing import Pool
from collections import defaultdict
MF=_release_path('AAAI/MedForge++/medforge90k_full/')
items=[it for it in json.load(open(MF+'dataset_mapping.json'))['items'] if it['type']!='real']
random.seed(0)
by=defaultdict(list)
for it in items: by[(it['model'],it['type'])].append(it)
samp=[]
for k,v in sorted(by.items()):
    random.shuffle(v); samp+=v[:60]
def work(it):
    f=MF+it['image_path']; s=MF+it['real_image_path']; m=MF+it['mask_path']
    if not (os.path.exists(f) and os.path.exists(s) and os.path.exists(m)): return None
    a=Image.open(f).convert('L'); b=Image.open(s).convert('L'); k=Image.open(m).convert('L')
    sz=a.size
    b2=b.resize(sz,Image.BICUBIC) if b.size!=sz else b
    k2=k.resize(sz,Image.NEAREST) if k.size!=sz else k
    A=np.asarray(a,np.float32); B=np.asarray(b2,np.float32); K=np.asarray(k2)>127
    d=np.abs(A-B)
    return dict(gen=it['model'],type=it['type'],organ=it['dataset'],fsize=a.size,ssize=b.size,
                din=float(d[K].mean()) if K.any() else -1, dout=float(d[~K].mean()), mfrac=float(K.mean()))
with Pool(8) as p: R=[r for r in p.map(work,samp,chunksize=8) if r]
import pandas as pd
df=pd.DataFrame(R)
df['size_match']=df.fsize==df.ssize
df['ratio']=df.din/(df.dout+1e-6)
print('sampled forgeries',len(df),'(60 per generator x type)')
g=df.groupby(['gen','type']).agg(n=('din','size'),in_mask=('din','median'),out_mask=('dout','median'),ratio=('ratio','median'),
    noop_in=('din',lambda s:int((s<1.0).sum())),size_mismatch=('size_match',lambda s:int((~s).sum())),mask_frac=('mfrac','median'))
print(g.round(3).to_string())
print('\nforgeries whose in-mask mean |diff| < 1 gray level (effectively unedited):',int((df.din<1).sum()),'of',len(df))
print('forgeries whose out-of-mask diff >= in-mask diff (edit not localised to mask):',int((df.dout>=df.din).sum()),'of',len(df))
print('forgery size != source size:',int((~df.size_match).sum()),'of',len(df), df[~df.size_match].groupby('gen').size().to_dict())
