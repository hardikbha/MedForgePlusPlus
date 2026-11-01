"""Exact (md5) and near-duplicate (64-bit dHash, Hamming<=4) analysis. READ-ONLY on data."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, re, sys
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

A = _release_path('CIRCE/v3_audit/')
MF = _release_path('AAAI/MedForge++/medforge90k_full/')
TH = 4
out = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)


h = pd.read_csv(A + "data/image_hashes.tsv", sep="\t", dtype={"md5": str, "dhash": str, "err": str}, keep_default_na=False)
P(f"hashed images: {len(h)}; decode errors: {(h.err != '').sum()}")
if (h.err != "").any():
    P(h[h.err != ""][["group", "path", "err"]].head(20).to_string())
h = h[h.err == ""].copy()
h["dh"] = h.dhash.map(lambda s: np.uint64(int(s, 16)))
thumbs = np.load(A + "data/thumbs32.npy")
path2i = dict(zip(h.path, h.idx.astype(int)))
path2md5 = dict(zip(h.path, h.md5))
path2dh = dict(zip(h.path, h.dh))

# ---------- metadata per image ----------
items = json.load(open(MF + "dataset_mapping.json"))["items"]
info = {}
for it in items:
    p = MF + it["image_path"]
    info[p] = dict(cls=it["type"], src=MF + (it.get("real_image_path") or it["image_path"]), gen=it["model"], organ=it["dataset"])


def pp_meta(p):
    b = os.path.basename(p)
    if "naitik_annotated" in p:
        sub = re.search(r"naitik_annotated/([^/]+)/", p).group(1)
    elif "colon_injections" in p:
        sub = "colon_inject"
    elif "flux_injected" in p:
        sub = "kidney_inject"
    elif "/kidney/normal/" in p:
        sub = "src_kidney_normal_kaggle"
    else:
        sub = "?"
    m = re.match(r"(case_\d{5}_slice_\d+)", b)
    if m:
        src = "kits:" + m.group(1)
    else:
        m = re.match(r"(?:unresolved_)?(c[jk][0-9a-z]{20,})", b)
        if m:
            src = "kvasir:" + m.group(1)
        else:
            m = re.match(r"Normal-[ _]\((\d+)\)", b)
            src = ("kagglekid:" + m.group(1)) if m else ("other:" + b)
    cls = "remove" if sub.endswith("removal") else ("edit" if sub.endswith("inject") and not sub.startswith("src") else "real")
    organ = "kidney" if ("kidney" in sub) else "colon"
    return dict(cls=cls, src=src, gen=sub, organ=organ)


for p in h.path:
    if p not in info:
        info[p] = pp_meta(p)
h["cls"] = h.path.map(lambda p: info[p]["cls"])
h["src"] = h.path.map(lambda p: info[p]["src"])
h["gen"] = h.path.map(lambda p: info[p]["gen"])
h["organ"] = h.path.map(lambda p: info[p]["organ"])
h["binlab"] = np.where(h.cls == "real", "real", "fake")

# ---------- census from hash table ----------
P("\n## A0. Image properties per group (from hashing pass)")
g = h.groupby(["group", "organ", "cls"]).agg(n=("path", "size"), fmt=("fmt", lambda s: dict(Counter(s))), size=("w", lambda s: None))
for (grp, org, cls), sub in h.groupby(["group", "organ", "cls"]):
    sizes = Counter(zip(sub.w, sub.h)).most_common(3)
    P(f"  {grp:28s} {org:10s} {cls:6s} n={len(sub):6d} fmt={dict(Counter(sub.fmt))} mode={dict(Counter(sub['mode']))} top_sizes={sizes} "
      f"median_bytes={int(sub.bytes.median())} mean_gray_median={sub['mean'].median():.1f}")

# ---------- A. exact duplicates ----------
P("\n## A. Exact duplicates by md5 (content hash)")
grp = h.groupby("md5")
dupg = {m: g for m, g in grp if len(g) > 1}
P(f"  md5 groups with >1 file: {len(dupg)}; redundant files: {sum(len(g)-1 for g in dupg.values())}")
cat = Counter()
cat_files = Counter()
conflict_examples = []
for m, g in dupg.items():
    groups = set(g.group)
    labs = set(g.binlab)
    if groups == {"mf90k"}:
        clss = tuple(sorted(Counter(g.cls).items()))
        gens = set(g.gen)
        key = "mf90k:" + ("+".join(sorted(set(g.cls)))) + (":multi-gen" if len(gens - {"real"}) > 1 else "")
    else:
        key = "+".join(sorted(groups))
    cat[key] += 1
    cat_files[key] += len(g) - 1
    if len(labs) > 1 and len(conflict_examples) < 15:
        conflict_examples.append(list(g.path))
for k, v in cat.most_common():
    P(f"    {k}: groups={v} redundant_files={cat_files[k]}")
P(f"  groups whose members carry DIFFERENT binary labels (identical bytes labelled real AND fake): {sum(1 for g in dupg.values() if g.binlab.nunique() > 1)}")
for ex in conflict_examples[:10]:
    P("     e.g. " + " | ".join(ex))
# real-real md5 dups in MF90K (distinct files, same content)
rr = [g for g in dupg.values() if set(g.group) == {"mf90k"} and set(g.cls) == {"real"}]
P(f"  MF90K real-real identical-content groups: {len(rr)} (files {sum(len(g) for g in rr)}); by organ {dict(Counter(g.organ.iloc[0] for g in rr))}")
for g in rr[:5]:
    P("     e.g. " + " | ".join(p.replace(MF, "") for p in g.path))
# MF++ per subset
for sub in sorted(h.group.unique()):
    if sub == "mf90k":
        continue
    s = h[h.group == sub]
    c = s.md5.value_counts()
    P(f"  {sub}: files={len(s)} unique_md5={s.md5.nunique()} dup_groups={(c > 1).sum()} redundant={int((c[c > 1] - 1).sum())}")
# fake whose bytes equal its own source real
same_as_src = [(p) for p in h.path[h.cls != "real"] if info[p]["src"] in path2md5 and path2md5[info[p]["src"]] == path2md5[p]]
P(f"  forgeries byte-identical to their own source real: {len(same_as_src)}")

# ---------- protocol files ----------
pr = pd.read_csv(A + "data/protocol_rows.tsv", sep="\t", dtype=str, keep_default_na=False)
files = {fn: sorted(set(g.path)) for fn, g in pr.groupby("file")}
TRAIN = ["mfp_train_indomain.jsonl", "mfp_train_crossmodel.jsonl", "mfp_train_crossforgery.jsonl",
         "mfp_v2_replay_crossforgery_strict.jsonl", "mfp_replay_implant_focus_sft.jsonl", "mfp_crossmodel_replay.jsonl", "mfp_indomain_replay.jsonl",
         "mfp_v2_rl_crossforgery_strict.jsonl", "mfp_v2_rl_crossmodel.jsonl", "mfp_v2_rl_indomain.jsonl",
         "raft_crossforgery.jsonl", "raft_crossmodel.jsonl", "mf90k_train.jsonl",
         "circe_sft_v3.jsonl", "circe_sft_full.jsonl", "pfplus_replay_removal_focus_sft.jsonl", "pfplus_replay_v2_balanced_sft.jsonl"]
EVAL = ["mfp_v2_test_shared.jsonl", "v3_select/mfp_v3_select_clean.jsonl", "circe_eval_split_v2.jsonl"]

# popcount LUT
LUT = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def ham(a, B):
    """a: (k,) uint64, B: (n,) uint64 -> (k,n) hamming"""
    x = np.bitwise_xor(a[:, None], B[None, :])
    v = x.view(np.uint16).reshape(x.shape[0], x.shape[1], 4)
    return LUT[v].sum(axis=2, dtype=np.uint8)


P("\n## B. Exact-content (md5) overlap between training files and eval files (different path, identical bytes)")
for ev in EVAL:
    E = [p for p in files[ev] if p in path2md5]
    for tr in TRAIN:
        T = [p for p in files[tr] if p in path2md5]
        Tm = defaultdict(list)
        for p in T:
            Tm[path2md5[p]].append(p)
        Tp = set(T)
        hits = [e for e in E if path2md5[e] in Tm and any(t != e for t in Tm[path2md5[e]])]
        exact = sum(e in Tp for e in E)
        if hits or exact:
            P(f"  {ev} vs {tr}: identical PATH={exact}; identical BYTES under another path={len(hits)}")
P("  (pairs not listed: 0 path overlap and 0 md5 overlap)")

P(f"\n## C. Near-duplicates: 64-bit dHash (9x8 gray LANCZOS), Hamming <= {TH}")
P("   For each eval image: does ANY training image lie within Hamming<=4?  'same-src' = the matched train image derives from the same authentic scan/case;")
P("   'label-conflict' = the nearest matched train image has the opposite binary label (real vs fake). Verification: median 32x32 thumbnail MAE of matched pairs.")
rows = []
for ev in EVAL:
    E = [p for p in files[ev] if p in path2dh]
    Edh = np.array([path2dh[p] for p in E], dtype=np.uint64)
    for tr in TRAIN:
        T = [p for p in files[tr] if p in path2dh]
        if not T:
            continue
        # skip MF90K-vs-MF++ pairs that cannot match by construction? keep all; cheap enough
        Tdh = np.array([path2dh[p] for p in T], dtype=np.uint64)
        nn_d = np.full(len(E), 99, np.int16); nn_j = np.zeros(len(E), np.int64)
        any_same, any_diff, conflict, n_hit, maes = 0, 0, 0, 0, []
        diff_src_examples = []
        for s in range(0, len(E), 256):
            Hm = ham(Edh[s:s + 256], Tdh)
            for k in range(Hm.shape[0]):
                e = E[s + k]
                js = np.nonzero(Hm[k] <= TH)[0]
                js = [j for j in js if T[j] != e]
                if not js:
                    continue
                n_hit += 1
                srcs = [info[T[j]]["src"] for j in js]
                same = any(sc == info[e]["src"] for sc in srcs)
                diff = any(sc != info[e]["src"] for sc in srcs)
                any_same += same
                any_diff += diff
                jbest = min(js, key=lambda j: Hm[k][j])
                if (info[T[jbest]]["cls"] == "real") != (info[e]["cls"] == "real"):
                    conflict += 1
                maes.append(float(np.abs(thumbs[path2i[e]].astype(int) - thumbs[path2i[T[jbest]]].astype(int)).mean()))
                if diff and len(diff_src_examples) < 3:
                    jd = [j for j in js if info[T[j]]["src"] != info[e]["src"]][0]
                    diff_src_examples.append((e, T[jd], int(Hm[k][jd])))
        P(f"  [{ev} (n={len(E)}) vs {tr} (n={len(T)})] eval imgs with a near-dup in train: {n_hit} ({n_hit/len(E)*100:.1f}%); "
          f"same-src: {any_same}; from a DIFFERENT source scan: {any_diff}; nearest-match label-conflict: {conflict}; "
          f"thumb-MAE median {np.median(maes) if maes else float('nan'):.2f}")
        for ex in diff_src_examples:
            P(f"       diff-src e.g. d={ex[2]}: {ex[0].replace(MF,'')}  ~  {ex[1].replace(MF,'')}")
        rows.append(dict(eval=ev, train=tr, n_eval=len(E), n_train=len(T), hit=n_hit, same_src=any_same, diff_src=any_diff, label_conflict=conflict))
pd.DataFrame(rows).to_csv(A + "out/near_dup_train_eval.csv", index=False)

# per-class breakdown for the main regimes on the shared test
P("\n## C2. Near-dup hits by eval class (shared test & clean selection) for the three base regimes")
for ev in EVAL[:2]:
    E = [p for p in files[ev] if p in path2dh]
    Edh = np.array([path2dh[p] for p in E], dtype=np.uint64)
    for tr in ["mfp_train_indomain.jsonl", "mfp_train_crossmodel.jsonl", "mfp_train_crossforgery.jsonl"]:
        T = [p for p in files[tr] if p in path2dh]
        Tdh = np.array([path2dh[p] for p in T], dtype=np.uint64)
        cnt = Counter(); tot = Counter(info[e]["cls"] for e in E); cnt_diff = Counter(); cnt_lab = Counter()
        for s in range(0, len(E), 256):
            Hm = ham(Edh[s:s + 256], Tdh)
            for k in range(Hm.shape[0]):
                e = E[s + k]
                js = [j for j in np.nonzero(Hm[k] <= TH)[0] if T[j] != e]
                if js:
                    cnt[info[e]["cls"]] += 1
                    if any(info[T[j]]["src"] != info[e]["src"] for j in js):
                        cnt_diff[info[e]["cls"]] += 1
                    labs = Counter("real" if info[T[j]]["cls"] == "real" else "fake" for j in js)
                    cnt_lab[(info[e]["cls"], tuple(sorted(labs)))] += 1
        P(f"  {ev} vs {tr}: hits by class {dict(cnt)} of {dict(tot)}; diff-source hits {dict(cnt_diff)}; matched-train label sets {dict(cnt_lab)}")

# ---------- D. within-MF90K near-dup clusters among REAL images of different files ----------
P("\n## D. MF90K REAL images: near-duplicate pairs between DIFFERENT real files (Hamming<=4), within organ")
reals = h[(h.group == "mf90k") & (h.cls == "real")].reset_index(drop=True)
pairs = []
for org, sub in reals.groupby("organ"):
    P_ = list(sub.path); D_ = np.array(list(sub.dh), dtype=np.uint64)
    n = 0
    for s in range(0, len(P_), 512):
        Hm = ham(D_[s:s + 512], D_)
        ii, jj = np.nonzero(Hm <= TH)
        for a, b in zip(ii, jj):
            a2 = s + a
            if a2 < b:
                pairs.append((P_[a2], P_[b], int(Hm[a, b])))
                n += 1
    P(f"  {org}: reals={len(P_)} near-dup pairs={n}")
# clusters
parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for a, b, d in pairs:
    parent[find(a)] = find(b)
cl = defaultdict(list)
for x in list(parent):
    cl[find(x)].append(x)
P(f"  real near-dup clusters: {len(cl)} covering {sum(len(v) for v in cl.values())} real files; exact-md5 pairs among them: {sum(1 for a,b,d in pairs if path2md5[a]==path2md5[b])}")
P(f"  pair distance histogram: {dict(sorted(Counter(d for a,b,d in pairs).items()))}")
ex = sorted(pairs, key=lambda t: t[2])[:6] + [t for t in pairs if t[2] == TH][:4]
for a, b, d in ex:
    P(f"     d={d} mae={np.abs(thumbs[path2i[a]].astype(int)-thumbs[path2i[b]].astype(int)).mean():.2f}: {a.replace(MF,'')} ~ {b.replace(MF,'')}")
# thumbnail MAE distribution of real-real pairs
maes = [np.abs(thumbs[path2i[a]].astype(int) - thumbs[path2i[b]].astype(int)).mean() for a, b, d in pairs]
if maes:
    P(f"  real-real pair thumb-MAE: median {np.median(maes):.2f}, p90 {np.percentile(maes,90):.2f}; pairs with MAE<2 (visually same image): {sum(m<2 for m in maes)}")
# how many test reals have a different-file real near-dup in each base training file
rset = defaultdict(set)
for a, b, d in pairs:
    rset[a].add(b); rset[b].add(a)
for ev in EVAL[:2]:
    Er = [p for p in files[ev] if info.get(p, {}).get("cls") == "real" and p in path2dh]
    for tr in ["mfp_train_indomain.jsonl", "mfp_train_crossforgery.jsonl"]:
        T = set(files[tr])
        hit = [e for e in Er if rset.get(e, set()) & T]
        P(f"  {ev}: test REALS with a different real file near-dup in {tr}: {len(hit)}/{len(Er)}")

# ---------- E. MF++ addition near-dups across subsets / within ----------
P("\n## E. MF++ addition: near-duplicates within/between subsets (Hamming<=4)")
pp = h[h.group != "mf90k"].reset_index(drop=True)
P_ = list(pp.path); D_ = np.array(list(pp.dh), dtype=np.uint64)
cnt = Counter(); cnt_same = Counter()
for s in range(0, len(P_), 512):
    Hm = ham(D_[s:s + 512], D_)
    ii, jj = np.nonzero(Hm <= TH)
    for a, b in zip(ii, jj):
        a2 = s + a
        if a2 < b:
            ga, gb = pp.group[a2], pp.group[b]
            key = tuple(sorted([ga, gb]))
            cnt[key] += 1
            if info[P_[a2]]["src"] == info[P_[b]]["src"]:
                cnt_same[key] += 1
for k, v in sorted(cnt.items()):
    P(f"  {k[0]} ~ {k[1]}: pairs={v} (same source id: {cnt_same[k]})")

open(A + "out/near_dup.txt", "w").write("\n".join(out))
