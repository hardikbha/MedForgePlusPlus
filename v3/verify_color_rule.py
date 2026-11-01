import json, sys, numpy as np, collections
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
def chk(p):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.int16)
    return bool((a[...,0] != a[...,1]).any() or (a[...,1] != a[...,2]).any())
def organ(p):
    for k in ('chest','brain','fundus'):
        if k in p: return k
rows = [json.loads(l) for l in open(sys.argv[1])]
with ProcessPoolExecutor(12) as ex:
    col = list(ex.map(chk, [r['image'] for r in rows], chunksize=16))
st = collections.defaultdict(lambda: [0, 0])
for r, c in zip(rows, col):
    gt_fake = r['type'] != 'real'
    o = organ(r['image'])
    st[o][0] += (c == gt_fake); st[o][1] += 1
    st['ALL'][0] += (c == gt_fake); st['ALL'][1] += 1
for k, (a, n) in sorted(st.items()): print(f'{sys.argv[1].split("/")[-1]} {k:7s} color-rule acc {a}/{n} = {a/n:.4f}')
