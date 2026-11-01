
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os
from collections import Counter, defaultdict
import pandas as pd
MF=_release_path('AAAI/MedForge++/medforge90k_full/')
items=json.load(open(MF+'dataset_mapping.json'))['items']
m=pd.DataFrame(items)
m['exists']=m.image_path.map(lambda p: os.path.exists(MF+p))
m['mask_exists']=m.mask_path.map(lambda p: os.path.exists(MF+p) if isinstance(p,str) else None)
disk=set()
for top in ['real','deepfake']:
    for root,ds,fs in os.walk(MF+top):
        rel=os.path.relpath(root,MF)
        if '/gt' in '/'+rel+'/' and rel.split('/')[0]=='deepfake' and len(rel.split('/'))>2 and rel.split('/')[2]=='gt': continue
        for f in fs:
            if os.path.splitext(f)[1].lower() in ('.jpg','.jpeg','.png'):
                disk.add(os.path.join(rel,f))
mp=set(m.image_path)
print('mapping items',len(m),'unique image_path',len(mp),'| on-disk image files (excl gt)',len(disk))
print('mapping items missing on disk',int((~m.exists).sum()),'| on-disk files not in mapping',len(disk-mp))
print('missing by organ x type:',m[~m.exists].groupby(['dataset','type']).size().to_dict())
print('missing by disease:',m[~m.exists].groupby(['dataset','disease']).size().to_dict())
fg=m[m.type!='real']
print('forgeries with mask file missing:',int((fg.mask_exists==False).sum()))
# forgeries whose SOURCE real is missing on disk
src_missing=fg.real_image_path.map(lambda p: not os.path.exists(MF+p))
print('forgeries whose source real is missing on disk:',int(src_missing.sum()))
t=m.assign(present=m.exists.astype(int)).pivot_table(index=['dataset','type'],columns='model',values='present',aggfunc='sum',fill_value=0)
t['TOTAL']=t.sum(1)
print('\nON-DISK images per organ x class x generator (mapping-listed & present):'); print(t.to_string())
t2=m.pivot_table(index=['dataset','type'],columns='model',values='key',aggfunc='count',fill_value=0); t2['TOTAL']=t2.sum(1)
print('\nMAPPING-listed per organ x class x generator:'); print(t2.to_string())
print('\nclass totals on disk:',m[m.exists].type.value_counts().to_dict(),'mapping:',m.type.value_counts().to_dict())
print('organ totals on disk:',m[m.exists].dataset.value_counts().to_dict())
print('diseases (real) on disk:',m[m.exists&(m.type=="real")].groupby(['dataset','disease']).size().to_dict())
m[~m.exists][['image_path','dataset','type','disease']].to_csv(_release_path('CIRCE/v3_audit/out/mf90k_missing_files.csv'),index=False)
