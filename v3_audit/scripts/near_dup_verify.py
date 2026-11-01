"""Calibrate the 64-bit dHash (Hamming<=4) near-duplicate criterion and recompute train->eval
near-duplicate leakage with a pixel-level verification (32x32 gray thumbnail MAE). READ-ONLY on data."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, re
from collections import Counter, defaultdict
import numpy as np, pandas as pd

A = _release_path('CIRCE/v3_audit/')
MF = _release_path('AAAI/MedForge++/medforge90k_full/')
out = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


h = pd.read_csv(A + "data/image_hashes.tsv", sep="\t", dtype={"md5": str, "dhash": str, "err": str}, keep_default_na=False)
h["dh"] = h.dhash.map(lambda s: np.uint64(int(s, 16)))
th = np.load(A + "data/thumbs32.npy").astype(np.int16)
idx = dict(zip(h.path, h.idx.astype(int)))
dh = dict(zip(h.path, h.dh))
md5 = dict(zip(h.path, h.md5))
LUT = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def ham(a, B):
    x = np.bitwise_xor(a[:, None], B[None, :])
    return LUT[x.view(np.uint16).reshape(x.shape[0], x.shape[1], 4)].sum(axis=2, dtype=np.uint8)


def hd(p, q):
    return int(bin(int(dh[p]) ^ int(dh[q])).count("1"))


def mae(p, q):
    return float(np.abs(th[idx[p]] - th[idx[q]]).mean())


items = json.load(open(MF + "dataset_mapping.json"))["items"]
info = {}
for it in items:
    info[MF + it["image_path"]] = dict(cls=it["type"], src=MF + (it.get("real_image_path") or it["image_path"]), organ=it["dataset"])


def pp_src(p):
    b = os.path.basename(p)
    m = re.match(r"(case_\d{5}_slice_\d+)", b)
    if m: return "kits:" + m.group(1)
    m = re.match(r"(?:unresolved_)?(c[jk][0-9a-z]{20,})", b)
    if m: return "kvasir:" + m.group(1)
    m = re.match(r"Normal-[ _]\((\d+)\)", b)
    return ("kagglekid:" + m.group(1)) if m else ("other:" + b)


for p in h.path:
    if p not in info:
        sub = "colon" if ("polyp" in p or "colon" in p) else "kidney"
        cls = "remove" if "removal" in p else ("edit" if ("colon_injections" in p or "flux_injected" in p) else "real")
        info[p] = dict(cls=cls, src=pp_src(p), organ=sub)

P("## Calibration of the 64-bit dHash<=4 criterion")
# (1) forgery vs its own authentic source (known near-identical except a local edit)
fs = [p for p in h.path if info[p]["cls"] != "real" and info[p]["src"] in idx and "medforge90k_full" in p]
d1 = np.array([hd(p, info[p]["src"]) for p in fs]); m1 = np.array([mae(p, info[p]["src"]) for p in fs])
P(f"  MF90K forgery vs OWN source (n={len(fs)}): dHash dist median {np.median(d1):.0f}, frac<=4 {np.mean(d1<=4):.3f}; thumb-MAE median {np.median(m1):.2f}, p90 {np.percentile(m1,90):.2f}, frac MAE<=3 {np.mean(m1<=3):.3f}")
pp = [p for p in h.path if info[p]["cls"] != "real" and "medforge90k_full" not in p and "/kidney/normal/" not in p]
for name, sel in [("kidney_removal vs kidney_real_tumor (same slice)", lambda p: "kidney_removal" in p),
                  ("polyp_removal vs polyp_real (same frame)", lambda p: "polyp_removal" in p),
                  ("colon_inject vs polyp_real (same frame)", lambda p: "colon_injections" in p),
                  ("kidney_inject vs Kaggle source normal", lambda p: "flux_injected" in p)]:
    pairs = []
    for p in pp:
        if not sel(p): continue
        b = os.path.basename(p)
        if "kidney_removal" in p: q = p.replace("kidney_removal", "kidney_real_tumor")
        elif "polyp_removal" in p: q = os.path.dirname(p).replace("polyp_removal", "polyp_real") + "/" + re.sub("^unresolved_", "", b)
        elif "colon_injections" in p: q = _release_path('AAAI/MedForge++/naitik_annotated/polyp_real/images/') + b.replace("_polypinj.jpg", ".png")
        else: q = _release_path('AAAI/kidney/normal/') + b.replace("_tumorinj.jpg", ".jpg").replace("Normal-_", "Normal- ")
        if q in idx: pairs.append((p, q))
    d = np.array([hd(p, q) for p, q in pairs]); m = np.array([mae(p, q) for p, q in pairs])
    if len(pairs):
        P(f"  {name} (n={len(pairs)}): dHash median {np.median(d):.0f}, frac<=4 {np.mean(d<=4):.3f}; thumb-MAE median {np.median(m):.2f}, p90 {np.percentile(m,90):.2f}, frac MAE<=3 {np.mean(m<=3):.3f}")
# (2) md5-identical pairs (true duplicates) -> MAE 0 by construction
# (3) cross-organ pairs with dHash<=4 (certain false positives)
kid = [p for p in h.path if "kidney_normal/images" in p]
col = [p for p in h.path if "polyp_real/images" in p] + [p for p in h.path if "medforge90k_full/real/chest-xray" in p][:3000]
Hm = ham(np.array([dh[p] for p in kid], dtype=np.uint64), np.array([dh[p] for p in col], dtype=np.uint64))
ii, jj = np.nonzero(Hm <= 4)
cm = np.array([mae(kid[i], col[j]) for i, j in zip(ii, jj)])
P(f"  cross-modality pairs with dHash<=4 (kidney CT vs colonoscopy/chest X-ray; certain false positives): {len(cm)} of {Hm.size} pairs "
  f"({len(cm)/Hm.size*100:.3f}%); thumb-MAE median {np.median(cm):.2f}, min {cm.min():.2f}, frac MAE<=3 {np.mean(cm<=3):.4f}")
# (4) within-organ real-real dHash<=4 pairs: MAE histogram (bimodality)
reals = [p for p in h.path if info[p]["cls"] == "real" and "medforge90k_full" in p]
by_org = defaultdict(list)
for p in reals: by_org[info[p]["organ"]].append(p)
allm = []
for org, L in by_org.items():
    D = np.array([dh[p] for p in L], dtype=np.uint64)
    ms = []
    for s in range(0, len(L), 512):
        H2 = ham(D[s:s+512], D)
        a, b = np.nonzero(H2 <= 4)
        k = (s + a) < b
        for x, y in zip(a[k], b[k]):
            ms.append(float(np.abs(th[idx[L[s+x]]] - th[idx[L[y]]]).mean()))
    ms = np.array(ms); allm.append(ms)
    hist = np.histogram(ms, bins=[0, 1, 2, 3, 4, 6, 8, 12, 20, 255])[0]
    P(f"  {org} real-real dHash<=4 pairs={len(ms)}; thumb-MAE histogram bins[0,1,2,3,4,6,8,12,20,255]: {hist.tolist()}")
TAU = 3.0
P(f"  -> verified near-duplicate := dHash<=4 AND 32x32 thumb-MAE <= {TAU} (gray levels); chosen at the valley of the real-real MAE histogram; "
  f"cross-modality false-positive survival at this tau = {np.mean(cm<=TAU)*100:.3f}%")

# ---------------- verified train -> eval near-dups ----------------
pr = pd.read_csv(A + "data/protocol_rows.tsv", sep="\t", dtype=str, keep_default_na=False)
files = {fn: sorted(set(g.path)) for fn, g in pr.groupby("file")}
TRAIN = ["mfp_train_indomain.jsonl", "mfp_train_crossmodel.jsonl", "mfp_train_crossforgery.jsonl",
         "mfp_v2_replay_crossforgery_strict.jsonl", "mfp_replay_implant_focus_sft.jsonl", "mfp_v2_rl_crossforgery_strict.jsonl",
         "mf90k_train.jsonl", "circe_sft_v2.jsonl", "circe_sft_v3.jsonl", "circe_sft_full.jsonl",
         "pfplus_replay_removal_focus_sft.jsonl", "pfplus_replay_v2_balanced_sft.jsonl"]
EVAL = ["mfp_v2_test_shared.jsonl", "v3_select/mfp_v3_select_clean.jsonl", "circe_eval_split_v2.jsonl"]
P(f"\n## Verified near-duplicates (dHash<=4 & thumb-MAE<={TAU}) between training files and eval files (excluding identical path)")
rows = []
for ev in EVAL:
    E = files[ev]
    Ed = np.array([dh[p] for p in E], dtype=np.uint64)
    for tr in TRAIN:
        T = files[tr]
        if not T: continue
        Td = np.array([dh[p] for p in T], dtype=np.uint64)
        Tth = th[[idx[p] for p in T]]
        hit = Counter(); hit_same = Counter(); hit_diff = Counter(); conf = Counter(); exact = Counter(); tot = Counter(info[e]["cls"] for e in E)
        diffex = []
        for s in range(0, len(E), 256):
            H2 = ham(Ed[s:s+256], Td)
            for k in range(H2.shape[0]):
                e = E[s+k]
                js = np.nonzero(H2[k] <= 4)[0]
                if len(js) == 0: continue
                mm = np.abs(Tth[js] - th[idx[e]]).mean(axis=(1, 2))
                js = [j for j, m in zip(js, mm) if m <= TAU and T[j] != e]
                if not js: continue
                c = info[e]["cls"]
                hit[c] += 1
                srcs = {info[T[j]]["src"] for j in js}
                if info[e]["src"] in srcs: hit_same[c] += 1
                if srcs - {info[e]["src"]}:
                    hit_diff[c] += 1
                    if len(diffex) < 4:
                        j = [j for j in js if info[T[j]]["src"] != info[e]["src"]][0]
                        diffex.append((e, T[j]))
                if any((info[T[j]]["cls"] == "real") != (c == "real") for j in js): conf[c] += 1
                if any(md5[T[j]] == md5[e] for j in js): exact[c] += 1
        n = sum(hit.values())
        P(f"  [{ev} vs {tr}] verified near-dup: {n}/{len(E)} ({n/len(E)*100:.1f}%) by class {dict(hit)} | same-source {sum(hit_same.values())} {dict(hit_same)} | "
          f"DIFFERENT source {sum(hit_diff.values())} {dict(hit_diff)} | opposite-label match {sum(conf.values())} {dict(conf)} | byte-identical {sum(exact.values())}")
        for a, b in diffex:
            P(f"       diff-src e.g. {a.replace(MF,'').replace('${MEDFORGE_WORKSPACE}/AAAI/','')}  ~  {b.replace(MF,'').replace('${MEDFORGE_WORKSPACE}/AAAI/','')}  (d={hd(a,b)}, mae={mae(a,b):.2f})")
        rows.append(dict(eval=ev, train=tr, n=len(E), verified=n, same_src=sum(hit_same.values()), diff_src=sum(hit_diff.values()),
                         opposite_label=sum(conf.values()), byte_identical=sum(exact.values())))
pd.DataFrame(rows).to_csv(A + "out/near_dup_verified.csv", index=False)

# verified real-real duplicate clusters in MF90K
P("\n## Verified duplicate REAL scans inside MF90K (different files, dHash<=4 & MAE<=3)")
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
npairs = Counter()
for org, L in by_org.items():
    D = np.array([dh[p] for p in L], dtype=np.uint64)
    for s in range(0, len(L), 512):
        H2 = ham(D[s:s+512], D)
        a, b = np.nonzero(H2 <= 4)
        k = (s + a) < b
        for x, y in zip(a[k], b[k]):
            p, q = L[s+x], L[y]
            if np.abs(th[idx[p]] - th[idx[q]]).mean() <= TAU:
                parent[find(p)] = find(q); npairs[org] += 1
cl = defaultdict(list)
for x in list(parent): cl[find(x)].append(x)
P(f"  verified pairs by organ {dict(npairs)}; clusters={len(cl)} covering {sum(len(v) for v in cl.values())} real files "
  f"(redundant files={sum(len(v)-1 for v in cl.values())}); by organ {dict(Counter(info[v[0]]['organ'] for v in cl.values()))}")
P(f"  clusters that are byte-identical (single md5): {sum(1 for v in cl.values() if len({md5[x] for x in v})==1)}")
cldis = Counter()
for v in cl.values():
    dz = {re.sub(r'^real/[^/]+/([^/]+)/.*$', r'\1', x.replace(MF, '')) for x in v}
    cldis["multi-disease-folder" if len(dz) > 1 else "same-folder"] += 1
P(f"  clusters spanning >1 disease folder (same scan, different label folder): {cldis['multi-disease-folder']}; same folder: {cldis['same-folder']}")
open(A + "out/near_dup_verified.txt", "w").write("\n".join(out))
