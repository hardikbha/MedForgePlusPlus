"""Stats of the strict CF RL pool + which 400 prompts GRPO actually consumed. READ-ONLY."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, re, os, collections, sys
ROOT = _release_path('AAAI/MedForge++/')
RL = ROOT + "mfp_v2_rl_crossforgery_strict.jsonl"
mp = json.load(open(ROOT + "medforge90k_full/dataset_mapping.json"))["items"]
by_img = {}
for x in mp:
    by_img.setdefault(ROOT + "medforge90k_full/" + x["image_path"], x)
BOX = re.compile(r'x1="(\d+)" y1="(\d+)" x2="(\d+)" y2="(\d+)"')
rows = []
for l in open(RL):
    r = json.loads(l)
    img = r["images"][0]
    asst = " ".join(m["content"] for m in r["messages"] if m["role"] == "assistant")
    gt_v = "deepfake" if "deepfake" in asst else "real"   # identical rule to train_grpo._rows_from_file
    bm = BOX.search(asst)
    m = by_img.get(img)
    rows.append(dict(img=img, gt_v=gt_v, box=bm is not None, type=(m or {}).get("type", "?"),
                     model=(m or {}).get("model", "?"), organ=(m or {}).get("dataset", "?"),
                     src=(m or {}).get("real_image_path")))
def summ(rs, tag):
    c = collections.Counter((r["gt_v"], r["type"]) for r in rs)
    pathreal = sum(r["img"].split("medforge90k_full/")[-1].startswith("real/") for r in rs)
    mism = sum((r["gt_v"] == "real") != r["img"].split("medforge90k_full/")[-1].startswith("real/") for r in rs)
    box_fake = sum(r["box"] for r in rs if r["gt_v"] == "deepfake"); nf = sum(r["gt_v"] == "deepfake" for r in rs)
    box_real = sum(r["box"] for r in rs if r["gt_v"] == "real")
    with_src = [r for r in rs if r["gt_v"] == "deepfake" and r["src"]]
    src_exist = sum(os.path.exists(ROOT + "medforge90k_full/" + r["src"]) for r in with_src)
    print(f"[{tag}] n={len(rs)} (gt_v,type)={dict(c)} path_real={pathreal} label/path_mismatch={mism} "
          f"fakes_with_box={box_fake}/{nf} reals_with_box={box_real} fakes_with_src={len(with_src)} src_exists={src_exist}")
    print(f"[{tag}] organs={dict(collections.Counter(r['organ'] for r in rs))}")
    print(f"[{tag}] models={dict(collections.Counter(r['model'] for r in rs))}")
summ(rows, "rl_strict_all")
summ(rows[:6400], "rl_strict_first6400(--limit)")
if len(sys.argv) > 1:
    used = [int(x) for x in open(sys.argv[1]).read().split()]
    summ([rows[i] for i in used], "rl_strict_used_by_400_steps")
    # how many of the used reals are sources of some fake in the pool -> pairing feasibility
srcs = collections.Counter(r["src"] for r in rows if r["gt_v"] == "deepfake" and r["src"])
pool_reals = set(r["img"].split("medforge90k_full/")[-1] for r in rows if r["gt_v"] == "real")
print(f"[pairing] distinct sources referenced by pool fakes={len(srcs)}; of these present as RL-pool reals={len(set(srcs)&pool_reals)}; "
      f"max fakes per source={max(srcs.values())}; mean={sum(srcs.values())/len(srcs):.2f}")
# global (MF90K mapping): families
fam = collections.Counter(x["real_image_path"] for x in mp if x["type"] in ("edit", "remove") and x.get("real_image_path"))
print(f"[pairing-MF90K] fakes with real_image_path={sum(fam.values())}; distinct sources={len(fam)}; sources with >=2 fakes={sum(v>=2 for v in fam.values())}")
