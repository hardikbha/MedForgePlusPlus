"""Does having same-source siblings in training change test accuracy? (CPU, uses existing gen files)"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json,re,collections,sys,math
sys.path.insert(0,_release_path('CIRCE/circe'))
from circe_reward import parse_output
R=_release_path('AAAI/MedForge++')
items=json.load(open(f"{R}/medforge90k_full/dataset_mapping.json"))["items"]; keys={it["image_path"]:it for it in items}
pat=re.compile(r'medforge90k_full/((?:real|deepfake)/[^"\s\\]+?\.(?:jpe?g|png))',re.I)
def refs(f):
    s=set()
    for l in open(f,errors="ignore"): s.update(pat.findall(l))
    return s&set(keys)
src=lambda k: k if keys[k]["type"]=="real" else keys[k]["real_image_path"]
TR={"CF":["mfp_train_crossforgery.jsonl","mfp_v2_replay_crossforgery_strict.jsonl"],
    "CM":["mfp_train_crossmodel.jsonl","mfp_crossmodel_replay.jsonl"],
    "ID":["mfp_train_indomain.jsonl","mfp_indomain_replay.jsonl"]}
GENS=[("CF","v3_select/gen_test_cf_s1replay.jsonl"),("CF","v3_select/gen_sel_cf_s1replay.jsonl"),("CF","v3_select/gen_sel_cf_s2grpo.jsonl"),
      ("CM","v3_select/gen_sel_cm_s1replay.jsonl"),("ID","v3_select/gen_sel_id_s1replay.jsonl")]
cache={}
for reg,g in GENS:
    if reg not in cache:
        tr=set().union(*[refs(f"{R}/{f}") for f in TR[reg]]); cache[reg]=(tr,collections.Counter(src(k) for k in tr))
    tr,trsrc=cache[reg]
    grp=collections.defaultdict(lambda:[0,0])
    for l in open(f"{R}/{g}"):
        r=json.loads(l); m=pat.search(r["image"]); 
        if not m or m.group(1) not in keys: continue
        k=m.group(1); it=keys[k]
        p=parse_output(r["output"]); v=p.get("verdict") if isinstance(p,dict) else None
        gt="real" if it["type"]=="real" else "deepfake"
        if it["type"]=="real": tag="real:"+("is-source-of-train-fakes" if trsrc[k]>0 else "no-train-derivatives")
        else: tag=it["type"]+":"+("source-or-sibling-in-train" if trsrc[src(k)]>0 else "no-source-in-train")
        grp[tag][0]+= (v==gt); grp[tag][1]+=1
    print(f"[{reg}] {g}")
    for t,(c,n) in sorted(grp.items()):
        p=c/n; ci=1.96*math.sqrt(p*(1-p)/n) if n else 0
        print(f"    {t:45s} acc={p*100:6.2f} +/-{ci*100:4.1f}  (n={n})")
