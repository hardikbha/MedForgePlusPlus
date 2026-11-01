
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
# Dataset-origin confound test: can a linear model on 32x32 gray thumbnails (no format/high-freq info) separate subsets?
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
A=_release_path('CIRCE/v3_audit/')
h=pd.read_csv(A+'data/image_hashes.tsv',sep='\t',dtype=str,keep_default_na=False)
th=np.load(A+'data/thumbs32.npy').reshape(-1,1024).astype(np.float32)/255.
g={k:h.idx[h.group==k].astype(int).values for k in h.group.unique()}
def auc(a,b,name):
    X=np.r_[th[g[a]],th[g[b]]]; y=np.r_[np.ones(len(g[a])),np.zeros(len(g[b]))]
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=2000))
    p=cross_val_predict(clf,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method='predict_proba')[:,1]
    print(f'{name:70s} n={len(y)}  5-fold CV AUC = {roc_auc_score(y,p):.3f}')
pairs=[('pp_polyp_real','pp_polyp_normal','POSITIVE CONTROL polyp_real (Kvasir) vs polyp_normal (other colon set), both REAL'),
('src_kidney_normal_kaggle','pp_kidney_normal','Kaggle kidney normals (implant sources, REAL) vs KiTS kidney_normal (REAL)'),
('pp_kidney_inject','pp_kidney_normal','kidney_inject (FLUX on Kaggle) vs kidney_normal (KiTS)'),
('pp_kidney_inject','pp_kidney_real_tumor','kidney_inject vs kidney_real_tumor (KiTS)'),
('pp_kidney_inject','src_kidney_normal_kaggle','kidney_inject vs its own Kaggle sources (true implant signal only)'),
('pp_kidney_removal','pp_kidney_real_tumor','kidney_removal vs kidney_real_tumor (same slices)'),
('pp_kidney_removal','pp_kidney_normal','kidney_removal vs kidney_normal'),
('pp_colon_inject','pp_polyp_real','colon_inject vs polyp_real (same frames)'),
('pp_polyp_removal','pp_polyp_real','polyp_removal vs polyp_real (same frames)'),
('pp_polyp_removal','pp_polyp_normal','polyp_removal vs polyp_normal')]
for a,b,n in pairs: auc(a,b,n)
