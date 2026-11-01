
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json,re,collections
ROOT=_release_path('AAAI/MedForge++')
items=json.load(open(f"{ROOT}/medforge90k_full/dataset_mapping.json"))["items"]; keys={it["image_path"]:it for it in items}
pat=re.compile(r'(?:medforge90k_full|MedForge-90K)/((?:real|deepfake)/[^"\s\\]+?\.(?:jpe?g|png))',re.I)
def refs(f):
    s=set()
    for line in open(f,errors="ignore"): s.update(pat.findall(line))
    return s&set(keys)
src=lambda k: k if keys[k]["type"]=="real" else keys[k]["real_image_path"]
for name,trf in [("CF",["mfp_train_crossforgery.jsonl","mfp_v2_replay_crossforgery_strict.jsonl"]),("CM",["mfp_train_crossmodel.jsonl","mfp_crossmodel_replay.jsonl"]),("ID",["mfp_train_indomain.jsonl","mfp_indomain_replay.jsonl"])]:
    tr=set().union(*[refs(f"{ROOT}/{f}") for f in trf])
    trsrc=collections.Counter(src(k) for k in tr)
    for tname,tf in [("shared_test",f"{ROOT}/mfp_v2_test_shared.jsonl"),("v3_select",f"{ROOT}/v3_select/mfp_v3_select_clean.jsonl")]:
        t=refs(tf)
        fakes=[k for k in t if keys[k]["type"]!="real"]; reals=[k for k in t if keys[k]["type"]=="real"]
        fs=[trsrc[src(k)] for k in fakes]
        print(f"[{name}] {tname}: fakes {len(fakes)}, with same-source sibling/real in {name}-train: {sum(x>0 for x in fs)} (median siblings {sorted(fs)[len(fs)//2]}); reals {len(reals)} that are a source of {name}-train fakes: {sum(trsrc[k]>0 for k in reals)}; real source img itself in train: {sum(src(k) in tr for k in fakes)}")
