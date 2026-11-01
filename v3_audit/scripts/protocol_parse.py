"""Parse every protocol / training / eval jsonl into a normalized table (READ-ONLY on data).
Output: ~/CIRCE/v3_audit/data/protocol_rows.tsv  (file, row_idx, path, cls, organ, gen, label_field, verdict_text)
  cls   : path-derived class: real | edit | remove (MF90K) or the MF++ subset-derived class
  organ : chest-xray | brain-mri | fundus | kidney | colon
  gen   : generator directory (MF90K) / 'lama' / 'flux-kontext' / 'real'
  label_field : value of an explicit 'type' field if the file has one, else ''
  verdict_text: final verdict parsed from the assistant turn (SFT-format files), else ''
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, re, glob, sys
from collections import Counter

D = _release_path('AAAI/MedForge++')
OUT = _release_path('CIRCE/v3_audit/data/protocol_rows.tsv')
HF = _release_path('.cache/huggingface/hub/datasets--RichardChenZH--MedForge-90K/snapshots/7bbf54f870584b37d6594593da76b2fc78fe4c5b/')
MF = D + "/medforge90k_full/"

FILES = [
    # base SFT (train+val), strict v2 train/val/rl
    "mfp_train_indomain.jsonl", "mfp_train_crossmodel.jsonl", "mfp_train_crossforgery.jsonl",
    "mfp_v2_train_indomain.jsonl", "mfp_v2_train_crossmodel.jsonl", "mfp_v2_train_crossforgery_strict.jsonl",
    "mfp_v2_val_indomain.jsonl", "mfp_v2_val_crossmodel.jsonl", "mfp_v2_val_crossforgery_strict.jsonl",
    "mfp_v2_val_indomain_eval.jsonl", "mfp_v2_val_crossmodel_eval.jsonl", "mfp_v2_val_crossforgery_strict_eval.jsonl",
    "mfp_v2_rl_indomain.jsonl", "mfp_v2_rl_crossmodel.jsonl", "mfp_v2_rl_crossforgery_strict.jsonl",
    # replay
    "mfp_replay_implant_focus_sft.jsonl", "mfp_v2_replay_crossforgery_strict.jsonl",
    "mfp_crossmodel_replay.jsonl", "mfp_indomain_replay.jsonl",
    # RAFT / older
    "raft_cf_a.jsonl", "raft_cf_b.jsonl", "raft_cf_cs_a.jsonl", "raft_cf_cs_b.jsonl", "raft_cf_cs.jsonl",
    "raft_crossforgery_cs.jsonl", "raft_crossforgery.jsonl", "raft_crossmodel_cs.jsonl", "raft_crossmodel.jsonl",
    "mf90k_train.jsonl", "mf90k_replay.jsonl", "mf90k_train_eval600.jsonl",
    "mf90k_test_indomain.jsonl", "mf90k_test_crossmodel.jsonl", "mf90k_test_crossmodel_fixed.jsonl",
    "mf90k_test_indomain_f.jsonl", "mf90k_test_crossmodel_f.jsonl",
    # MF++ addition
    "circe_sft_v3.jsonl", "circe_sft_v2.jsonl", "circe_sft_full.jsonl",
    "pfplus_replay_removal_focus_sft.jsonl", "pfplus_replay_v2_balanced_sft.jsonl",
    "mix8.jsonl", "genmix.jsonl",
    # eval
    "mfp_v2_test_shared.jsonl", "mfp_test_shared.jsonl", "v3_select/mfp_v3_select_clean.jsonl",
    "circe_eval_split_v2.jsonl", "circe_eval_split.jsonl", "aggregate_test_medforgepp.jsonl",
    "medforge90k_eval/eval.jsonl",
]

VERD = re.compile(r"This image is (?:a |an )?(deepfake|real)", re.I)


def norm_path(p):
    p = p.replace(HF, MF)
    return os.path.normpath(p)


def classify(p):
    """Return (cls, organ, gen) from the path."""
    parts = p.split("/")
    if "medforge90k_full" in parts:
        i = parts.index("medforge90k_full")
        top = parts[i + 1]
        if top == "real":
            return "real", parts[i + 2], "real"
        if top == "deepfake":
            gen = parts[i + 2]
            seg = parts[i + 3]
            m = re.search(r"^(.*)-(edit|remove)$", seg)
            if m:
                return m.group(2), m.group(1), gen
            return "?", "?", gen
    m = re.search(r"naitik_annotated/([^/]+)/", p)
    if m:
        sub = m.group(1)
        organ = "kidney" if sub.startswith("kidney") else "colon"
        cls = "remove" if sub.endswith("removal") else "real"
        return cls, organ + ":" + sub, ("lama" if cls == "remove" else "real")
    if "colon_injections" in p:
        return "edit", "colon:colon_inject", "flux-kontext"
    if "flux_injected" in p:
        return "edit", "kidney:kidney_inject", "flux-kontext"
    return "?", "?", "?"


def main():
    n_all = 0
    with open(OUT, "w") as fo:
        fo.write("file\trow\tpath\tcls\torgan\tgen\tlabel_field\tverdict_text\n")
        for fn in FILES:
            fp = os.path.join(D, fn)
            if not os.path.exists(fp):
                print("[missing]", fn)
                continue
            n = 0
            for i, line in enumerate(open(fp)):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "images" in r:
                    p = r["images"][0]
                else:
                    p = r["image"]
                p = norm_path(p)
                cls, organ, gen = classify(p)
                lab = str(r.get("type", "")) if "type" in r else ""
                vt = ""
                if "messages" in r:
                    a = [m for m in r["messages"] if m.get("role") == "assistant"]
                    if a:
                        c = a[-1]["content"]
                        hits = VERD.findall(c)
                        vt = hits[-1].lower() if hits else "NONE"
                fo.write(f"{fn}\t{i}\t{p}\t{cls}\t{organ}\t{gen}\t{lab}\t{vt}\n")
                n += 1
            n_all += n
            print(f"{fn}\t{n}", flush=True)
    print("total rows", n_all)


if __name__ == "__main__":
    main()
