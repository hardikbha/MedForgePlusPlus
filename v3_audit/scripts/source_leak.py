"""Source-level and patient-level leakage for MF90K protocol files (READ-ONLY on data)."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, re, random
from collections import Counter, defaultdict
import pandas as pd

MF = _release_path('AAAI/MedForge++/medforge90k_full/')
D = _release_path('AAAI/MedForge++/')
PROBE = _release_path('AAAI/MedForge/dataset_probe/')
out = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)


items = json.load(open(MF + "dataset_mapping.json"))["items"]
info = {}
for it in items:
    rel = it["image_path"]
    src = it.get("real_image_path") or rel
    info[rel] = dict(src=src, type=it["type"], gen=it["model"], organ=it["dataset"], disease=it["disease"],
                     mask=it.get("mask_path"))
# source group members
group = defaultdict(list)
for rel, d in info.items():
    group[d["src"]].append(rel)


def patient_key(rel):
    """Patient / acquisition key for a REAL image path (or None if indeterminable)."""
    b = os.path.splitext(os.path.basename(rel))[0]
    org = rel.split("/")[1]
    if org == "fundus":
        m = re.match(r"^(\d+)_(left|right)(_eyepacs)?$", b)
        if m:
            return ("fundus-eyepacs" if m.group(3) else "fundus-odir") + ":" + m.group(1)
        m = re.match(r"^(\d+)-[LR]-(\d+)$", b)
        if m:
            return "fundus-LR:" + m.group(1)
        return None
    if org == "brain-mri":
        m = re.match(r"^(YG_[A-Z0-9]+)_", b)
        if m:
            return "brain-yale:" + m.group(1)
        return None
    if org == "chest-xray":
        return None  # MIMIC study id only (sNNNNNNNN); no patient id in file name
    return None


def exam_key(rel):
    b = os.path.splitext(os.path.basename(rel))[0]
    m = re.match(r"^(YG_[A-Z0-9]+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_", b)
    return m.group(1) if m else None


def rel_of(p):
    i = p.find("medforge90k_full/")
    return p[i + len("medforge90k_full/"):] if i >= 0 else None


df = pd.read_csv(_release_path('CIRCE/v3_audit/data/protocol_rows.tsv'), sep="\t", dtype=str,
                 keep_default_na=False)
df["rel"] = df.path.map(rel_of)
files = {fn: set(g.rel.dropna()) for fn, g in df.groupby("file")}

# ---------------- 1. reproduce the protocol split ----------------
P("## 1. Reproducing build_medforge_protocol.py split with the local mapping (md5 eb4f6116...)")
it2 = json.load(open(MF + "dataset_mapping.json"))["items"]
random.seed(42)
random.shuffle(it2)
n = len(it2); n_sft = n * 5 // 9; n_gspo = n // 9
sft_items = it2[:n_sft]; gspo_items = it2[n_sft:n_sft + n_gspo]; test_items = it2[n_sft + n_gspo:]
by_type = {"real": [], "edit": [], "remove": []}
for it in test_items:
    by_type[it.get("type", "real")].append(it)
test_sel = []
for t, lst in by_type.items():
    random.shuffle(lst)
    test_sel.extend(lst[:1000])
exists = lambda rel: os.path.exists(MF + rel)
rep_test = {it["image_path"] for it in test_sel if exists(it["image_path"])}
rep_sft = {it["image_path"] for it in sft_items if exists(it["image_path"])}
P(f"  total={n} sft={len(sft_items)} gspo={len(gspo_items)} test_pool={len(test_items)}")
P(f"  reproduced test (existing files) = {len(rep_test)}; equal to mfp_test_shared: {rep_test == files['mfp_test_shared.jsonl']}")
P(f"  reproduced sft (existing files) = {len(rep_sft)}; equal to mfp_train_indomain: {rep_sft == files['mfp_train_indomain.jsonl']}")
P(f"  sft items whose file is missing on disk = {sum(not exists(it['image_path']) for it in sft_items)}")
gspo_set = {it["image_path"] for it in gspo_items}
testpool_set = {it["image_path"] for it in test_items}
sel = files["v3_select/mfp_v3_select_clean.jsonl"]
P(f"  v3_select clean (3000): from gspo pool={len(sel & gspo_set)}, from test pool={len(sel & testpool_set)}, from sft pool={len(sel & rep_sft)}")

# subset relations
P("\n## 2. Subset relations between training files")
base = {"ID": "mfp_train_indomain.jsonl", "CM": "mfp_train_crossmodel.jsonl", "CF": "mfp_train_crossforgery.jsonl"}
derived = {"ID": ["mfp_v2_train_indomain.jsonl", "mfp_v2_val_indomain.jsonl", "mfp_v2_rl_indomain.jsonl", "mfp_indomain_replay.jsonl", "raft_crossmodel.jsonl"],
           "CM": ["mfp_v2_train_crossmodel.jsonl", "mfp_v2_val_crossmodel.jsonl", "mfp_v2_rl_crossmodel.jsonl", "mfp_crossmodel_replay.jsonl", "raft_crossmodel.jsonl", "raft_crossmodel_cs.jsonl"],
           "CF": ["mfp_v2_train_crossforgery_strict.jsonl", "mfp_v2_val_crossforgery_strict.jsonl", "mfp_v2_rl_crossforgery_strict.jsonl", "mfp_v2_replay_crossforgery_strict.jsonl", "mfp_replay_implant_focus_sft.jsonl", "raft_crossforgery.jsonl", "raft_crossforgery_cs.jsonl", "raft_cf_cs.jsonl"]}
for r, fs in derived.items():
    b = files[base[r]]
    for f in fs:
        s = files[f]
        P(f"  {r}: {f} ({len(s)} uniq) subset of {base[r]}: {s <= b}  (outside: {len(s - b)})")
P(f"  v2 train ∪ v2 val == base? ID {files['mfp_v2_train_indomain.jsonl'] | files['mfp_v2_val_indomain.jsonl'] == files[base['ID']]}, "
  f"CM {files['mfp_v2_train_crossmodel.jsonl'] | files['mfp_v2_val_crossmodel.jsonl'] == files[base['CM']]}, "
  f"CF {files['mfp_v2_train_crossforgery_strict.jsonl'] | files['mfp_v2_val_crossforgery_strict.jsonl'] == files[base['CF']]}")
P(f"  v2 rl == v2 train? ID {files['mfp_v2_rl_indomain.jsonl'] == files['mfp_v2_train_indomain.jsonl']} CM {files['mfp_v2_rl_crossmodel.jsonl'] == files['mfp_v2_train_crossmodel.jsonl']} CF {files['mfp_v2_rl_crossforgery_strict.jsonl'] == files['mfp_v2_train_crossforgery_strict.jsonl']}")

# regime training unions
REG = {
    "ID": ["mfp_train_indomain.jsonl", "mfp_indomain_replay.jsonl", "mfp_v2_rl_indomain.jsonl"],
    "CM": ["mfp_train_crossmodel.jsonl", "mfp_crossmodel_replay.jsonl", "mfp_v2_rl_crossmodel.jsonl"],
    "CF": ["mfp_train_crossforgery.jsonl", "mfp_v2_replay_crossforgery_strict.jsonl", "mfp_v2_rl_crossforgery_strict.jsonl"],
    "CF+leakyReplay": ["mfp_train_crossforgery.jsonl", "mfp_replay_implant_focus_sft.jsonl"],
    "mf90k_train(old)": ["mf90k_train.jsonl", "mf90k_replay.jsonl"],
}
EVAL = ["mfp_v2_test_shared.jsonl", "v3_select/mfp_v3_select_clean.jsonl", "mf90k_test_indomain.jsonl", "mf90k_test_crossmodel.jsonl", "medforge90k_eval/eval.jsonl"]

P("\n## 3. Source-level leakage (MF90K): an eval image and a training image derived from the same authentic scan")
P("   definitions: src(x)=real_image_path for a forgery, x itself for a real.")
P("   A = eval rows whose exact path is in train; B = eval FORGERIES whose authentic source scan is itself a TRAIN real;")
P("   C = eval forgeries whose source scan appears in train via ANY image (source real or sibling forgery);")
P("   C' = eval forgeries with a sibling forgery in train that has the SAME source, SAME type and SAME disease (= same edit by another generator; same mask region)")
P("   E = eval REALS that are the authentic source of >=1 training forgery; F = eval rows of any class whose source group has any train member")
rows_csv = []
for ev in EVAL:
    E = sorted(files[ev])
    for rg, fl in REG.items():
        T = set().union(*[files[f] for f in fl])
        Tsrc = Counter(info[t]["src"] for t in T)
        Treal = {t for t in T if info[t]["type"] == "real"}
        Tsig = {(info[t]["src"], info[t]["type"], info[t]["disease"]) for t in T if info[t]["type"] != "real"}
        forg = [e for e in E if info[e]["type"] != "real"]
        reals = [e for e in E if info[e]["type"] == "real"]
        A = sum(e in T for e in E)
        B = sum(info[e]["src"] in Treal for e in forg)
        C = sum(info[e]["src"] in Tsrc for e in forg)
        Cp = sum((info[e]["src"], info[e]["type"], info[e]["disease"]) in Tsig for e in forg)
        Ereal = sum(Tsrc.get(e, 0) - (1 if e in T else 0) > 0 for e in reals)
        F = sum(info[e]["src"] in Tsrc for e in E)
        # by type
        bt = {}
        for t in ["edit", "remove"]:
            ft = [e for e in forg if info[e]["type"] == t]
            bt[t] = (len(ft), sum(info[e]["src"] in Tsrc for e in ft),
                     sum((info[e]["src"], info[e]["type"], info[e]["disease"]) in Tsig for e in ft),
                     sum(info[e]["src"] in Treal for e in ft))
        P(f"  [{ev} vs {rg}] n={len(E)} (forged {len(forg)}, real {len(reals)}) A={A} B={B} ({B/max(1,len(forg))*100:.1f}%) "
          f"C={C} ({C/max(1,len(forg))*100:.1f}%) C'={Cp} ({Cp/max(1,len(forg))*100:.1f}%) E={Ereal}/{len(reals)} F={F}/{len(E)} ({F/len(E)*100:.1f}%)")
        P(f"      per type (n, C, C', B): edit={bt['edit']} remove={bt['remove']}")
        rows_csv.append(dict(eval=ev, regime=rg, n=len(E), forged=len(forg), real=len(reals), A=A, B=B, C=C, Cp=Cp, E=Ereal, F=F,
                             edit_n=bt['edit'][0], edit_C=bt['edit'][1], edit_Cp=bt['edit'][2], edit_B=bt['edit'][3],
                             remove_n=bt['remove'][0], remove_C=bt['remove'][1], remove_Cp=bt['remove'][2], remove_B=bt['remove'][3]))
pd.DataFrame(rows_csv).to_csv(_release_path('CIRCE/v3_audit/out/source_leak.csv'), index=False)

# CM-specific: held-out-generator test forgeries whose same edit exists in CM train from a seen generator
P("\n## 4. Cross-Model: test forgeries from HELD-OUT generators whose identical edit (same source, type, disease) is in CM training via a SEEN generator")
CMH = {"gemini", "gpt", "stable-diffusion-3.5-medium", "stable-diffusion-xl-1.0-inpainting-0.1"}
for ev in ["mfp_v2_test_shared.jsonl", "v3_select/mfp_v3_select_clean.jsonl"]:
    T = files["mfp_train_crossmodel.jsonl"]
    Tsig = {(info[t]["src"], info[t]["type"], info[t]["disease"]) for t in T if info[t]["type"] != "real"}
    Tsrc = {info[t]["src"] for t in T}
    ho = [e for e in files[ev] if info[e]["gen"] in CMH]
    P(f"  {ev}: held-out-gen forgeries={len(ho)}; same-edit sibling in CM train={sum((info[e]['src'],info[e]['type'],info[e]['disease']) in Tsig for e in ho)}; source group in CM train={sum(info[e]['src'] in Tsrc for e in ho)}")
P("\n## 5. Cross-Forgery: test IMPLANTS (edit) and their relation to CF training (which has no implants)")
for ev in ["mfp_v2_test_shared.jsonl", "v3_select/mfp_v3_select_clean.jsonl"]:
    T = files["mfp_train_crossforgery.jsonl"]
    Treal = {t for t in T if info[t]["type"] == "real"}
    ed = [e for e in files[ev] if info[e]["type"] == "edit"]
    P(f"  {ev}: implants={len(ed)}; whose untouched source scan is a CF-train REAL (labelled real) = {sum(info[e]['src'] in Treal for e in ed)}")
    T2 = files["mfp_replay_implant_focus_sft.jsonl"]
    T2sig = {(info[t]["src"], info[t]["type"], info[t]["disease"]) for t in T2 if info[t]["type"] != "real"}
    T2src = {info[t]["src"] for t in T2 if info[t]["type"] == "edit"}
    P(f"    vs LEAKY implant-focus replay: same-edit sibling present = {sum((info[e]['src'],info[e]['type'],info[e]['disease']) in T2sig for e in ed)}; same source implanted (any disease) = {sum(info[e]['src'] in T2src for e in ed)}")

# ---------------- 6. patient-level ----------------
P("\n## 6. Patient-level leakage (beyond same-source): eval rows whose source scan's PATIENT has a different scan in train")
P("   patient keys: fundus ODIR/EyePACS '<id>_left|right' -> id (both eyes), fundus '<id>-L|R-<n>' -> id, brain Yale-mets 'YG_<pid>_<date>_<seq>' -> pid;")
P("   chest (MIMIC 's<study>') and Kaggle brain 'Tr-/Te-xx_NNNN' carry no patient id -> not determinable")
pk = {rel: patient_key(info[rel]["src"]) for rel in info}
cov = Counter((info[r]["organ"], pk[r] is not None) for r in info if info[r]["type"] == "real")
P("   real images with a determinable patient key (organ, has_key): " + str(dict(cov)))
for ev in ["mfp_v2_test_shared.jsonl", "v3_select/mfp_v3_select_clean.jsonl"]:
    for rg in ["ID", "CM", "CF"]:
        T = set().union(*[files[f] for f in REG[rg]])
        Tpat = defaultdict(set)
        for t in T:
            if pk[t]:
                Tpat[pk[t]].add(info[t]["src"])
        E = files[ev]
        withkey = [e for e in E if pk[e]]
        hit = [e for e in withkey if pk[e] in Tpat and (Tpat[pk[e]] - {info[e]["src"]})]
        byorg = Counter(info[e]["organ"] for e in hit)
        P(f"  [{ev} vs {rg}] rows with patient key={len(withkey)}; patient has a DIFFERENT scan in train={len(hit)} ({len(hit)/max(1,len(withkey))*100:.1f}%) by organ {dict(byorg)}")
# Yale exam-level (same exam, different sequence)
ek = {rel: exam_key(info[rel]["src"]) for rel in info}
for ev in ["mfp_v2_test_shared.jsonl", "v3_select/mfp_v3_select_clean.jsonl"]:
    T = files["mfp_train_crossforgery.jsonl"]
    Tex = defaultdict(set)
    for t in T:
        if ek[t]:
            Tex[ek[t]].add(info[t]["src"])
    E = [e for e in files[ev] if ek[e]]
    hit = [e for e in E if ek[e] in Tex and (Tex[ek[e]] - {info[e]["src"]})]
    P(f"  Yale brain-mets SAME EXAM (other sequence) in CF train: {ev}: {len(hit)}/{len(E)}")

# ---------------- 7. official MedForge split ----------------
P("\n## 7. Context: the OFFICIAL MedForge-90K split (dataset_probe sft/grpo/test lists)")
osft = set(l.strip() for l in open(PROBE + "sft_imgs.txt") if l.strip())
ogrpo = set(l.strip() for l in open(PROBE + "grpo_imgs.txt") if l.strip())
otest = set(l.strip() for l in open(PROBE + "test_imgs.txt") if l.strip())
P(f"  official sft={len(osft)} grpo={len(ogrpo)} test={len(otest)}; sft∩test={len(osft & otest)} grpo∩test={len(ogrpo & otest)}")
Osrc = Counter(info[t]["src"] for t in osft | ogrpo if t in info)
Osig = {(info[t]["src"], info[t]["type"], info[t]["disease"]) for t in osft | ogrpo if t in info and info[t]["type"] != "real"}
of = [e for e in otest if e in info and info[e]["type"] != "real"]
P(f"  official test forgeries={len(of)}; source group in official train(sft+grpo)={sum(info[e]['src'] in Osrc for e in of)} "
  f"({sum(info[e]['src'] in Osrc for e in of)/len(of)*100:.1f}%); same-edit sibling in train={sum((info[e]['src'],info[e]['type'],info[e]['disease']) in Osig for e in of)}")
otr = [e for e in otest if e in info and info[e]["type"] == "real"]
P(f"  official test reals={len(otr)}; that are source of a train forgery={sum(Osrc.get(e,0)>0 for e in otr)}")
Osrc_test = Counter(info[e]["src"] for e in otest if e in info)
P(f"  official split by source group?  source groups spanning train and test = {len(set(Osrc) & set(Osrc_test))} of {len(set(Osrc_test))} test source groups")
for ev in ["mfp_v2_test_shared.jsonl", "v3_select/mfp_v3_select_clean.jsonl"]:
    E = files[ev]
    P(f"  our {ev}: in official test={len(E & otest)}, in official sft={len(E & osft)}, in official grpo={len(E & ogrpo)}")
for f in ["mfp_train_indomain.jsonl", "mfp_train_crossforgery.jsonl", "mfp_train_crossmodel.jsonl"]:
    P(f"  our {f}: rows in official TEST={len(files[f] & otest)} of {len(files[f])}")

# ---------------- 8. sources concentration ----------------
P("\n## 8. How many authentic scans underlie the forged half")
srcs = Counter(d["src"] for d in info.values() if d["type"] != "real")
P(f"  forged images={sum(srcs.values())}, distinct source scans={len(srcs)}; by organ: " +
  str(dict(Counter(info[s]['organ'] for s in srcs))))
P(f"  by type: edit sources={len({d['src'] for d in info.values() if d['type']=='edit'})} remove sources={len({d['src'] for d in info.values() if d['type']=='remove'})}")
P(f"  reals that are never a source = {sum(1 for r,d in info.items() if d['type']=='real' and r not in srcs)} of {sum(1 for d in info.values() if d['type']=='real')}")
for ev in ["mfp_v2_test_shared.jsonl", "v3_select/mfp_v3_select_clean.jsonl"]:
    E = files[ev]
    reals = [e for e in E if info[e]["type"] == "real"]
    P(f"  {ev}: reals that are a forgery source anywhere = {sum(e in srcs for e in reals)}/{len(reals)}")

open(_release_path('CIRCE/v3_audit/out/source_leak.txt'), "w").write("\n".join(out))
