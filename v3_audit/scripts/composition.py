
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import pandas as pd, os, sys
from collections import Counter
pd.set_option('display.width',250); pd.set_option('display.max_rows',500); pd.set_option('display.max_columns',50)
df=pd.read_csv(_release_path('CIRCE/v3_audit/data/protocol_rows.tsv'),sep='\t',dtype=str,keep_default_na=False)
df['organ0']=df.organ.str.split(':').str[0]
out=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.append(s)
# existence
up=df.path.unique()
ex={p:os.path.exists(p) for p in up}
df['exists']=df.path.map(ex)
P('unique paths across all files',len(up),'missing on disk',sum(not v for v in ex.values()))
for fn,g in df.groupby('file',sort=False):
    n=len(g); u=g.path.nunique()
    P(f'\n=== {fn}: rows={n} unique_paths={u} dup_rows={n-u} missing_on_disk={int((~g.exists).sum())}')
    P('  class:',dict(Counter(g.cls)))
    P('  organ x class:',g.groupby(['organ0','cls']).size().to_dict())
    gg=g[g.cls!='real']
    P('  generator:',dict(sorted(Counter(gg.gen).items())))
    if (g.organ.str.contains(':')).any():
        P('  MF++ subset:',dict(Counter(g.organ[g.organ.str.contains(':')])))
    # label consistency
    if (g.verdict_text!='').any():
        exp=g.cls.map(lambda c:'real' if c=='real' else 'deepfake')
        bad=(g.verdict_text!=exp)
        P(f'  verdict-vs-path mismatches: {int(bad.sum())} (NONE={int((g.verdict_text=="NONE").sum())})', dict(Counter(zip(g.cls[bad],g.verdict_text[bad]))) if bad.any() else '')
    if (g.label_field!='').any():
        m={'real':'real','edit':'edit','remove':'remove','inject':'edit'}
        lf=g.label_field.map(lambda x:m.get(x,x))
        bad=lf!=g.cls
        P(f'  type-field-vs-path mismatches: {int(bad.sum())}', dict(Counter(zip(g.cls[bad],g.label_field[bad]))) if bad.any() else '')
open(_release_path('CIRCE/v3_audit/out/composition.txt'),'w').write('\n'.join(out))
