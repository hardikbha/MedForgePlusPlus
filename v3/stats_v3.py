"""
stats_v3.py -- CPU scorer for saved generation files (v3 MF-Clean evals AND old eval_medforge.py files).

README
------
Usage
  # one file (JSON + markdown)
  python stats_v3.py --gen gen_A.jsonl --out_json A.json --out_md A.md [--fields test_core_fields.jsonl]
  # two files on the same images: everything above for both + exact McNemar + paired group-bootstrap CIs
  python stats_v3.py --gen gen_A.jsonl --gen2 gen_B.jsonl --out_json AB.json --out_md AB.md
  # batch re-score (one summary row per file, optional paired rows)
  python stats_v3.py --rescore 'dir1/gen_*.jsonl' 'dir2/gen_*.jsonl' --pair A.jsonl,B.jsonl --out_md T.md --out_json T.json
  # unit tests of every metric on synthetic data with known answers
  python stats_v3.py --self_test
  Env: ~/.conda/envs/aiface/bin/python (NOT the default python3 = GraalPy). numpy only; scipy is used only by
  --self_test to cross-check the exact McNemar p-value if it is installed.

Input rows (one JSON per line). Both layouts are accepted:
  v3   (eval_v3.py):        image, type in real|edit|remove, gen, box, organ, group, split, disease, twin, orig, output
  old  (eval_medforge.py):  image, gt_verdict, gt_box, type, subset, output
  For old rows the missing organ / generator / group / twin are resolved from the image path:
    MedForge-90K paths (incl. v3_counterfactual norm/pasteback copies and mfclean/img copies) -> the table built by
    legacy_meta_v3.py (group ids identical to MF-Clean's: forgery+source scan+patient+verified duplicates);
    MF++ paths (naitik_annotated/<subset>/...) -> rules (kidney: KiTS case id; polyp/colon: Kvasir stem; twins =
    kidney_real_tumor->kidney_removal, polyp_real->polyp_removal, polyp_real->colon_inject by stem).
  The model output is re-parsed here (single parse path): prompts_v3.parse_v3 first; if it finds no v3 evidence
  fields, circe_reward.parse_output (CIRCE-v2 tags, then legacy "This image is a deepfake/real image" and the
  MedForge-Reasoner x1="" box syntax).

Conventions (stated once, used everywhere)
  * gt verdict: deepfake iff type != real. Classes: real / implant (edit, inject) / removal (remove).
  * An UNPARSED verdict is a third output ("invalid"): it is wrong for every GT class and is a positive for neither
    class. acc / balanced acc / macro-F1 / recalls / twin use this convention. `acc_legacy` re-scores with
    eval_medforge.py's old rule (unparsed -> "real") so old metrics_*.json numbers can be reproduced.
  * macro-F1 = unweighted mean of F1(real) and F1(deepfake), binary verdict.
  * balanced acc = (recall_real + recall_deepfake) / 2   (binary; a constant-verdict predictor scores exactly 0.5).
    bal_acc_typebal = (recall_real + mean_t recall_t) / 2 over the fake types t present (implant, removal):
    also 0.5 for any constant predictor, but implant and removal weigh equally regardless of their counts.
  * per-class recall: fraction of that class given the correct binary verdict (real -> "real", fakes -> "deepfake").
  * mIoU / IoU@0.5 over fakes that have a GT box; a missing predicted box scores IoU 0; IoU@0.5 uses IoU >= 0.5.
    Primary mIoU uses the parsed box whatever the verdict (old eval_medforge convention); mIoU_gated scores 0
    when the verdict is not "deepfake".
  * twin consistency: over pairs (fake, its authentic-source twin real) with BOTH images in the file, the
    fraction where the fake is called deepfake AND the twin is called real. A constant predictor scores 0.
  * rationale accuracy per evidence field (type/location/size/intensity): exact match of the parsed v3 field
    against GT from --fields (rows {image, type|location|size|intensity} or {image, fields:{...}}); without a
    fields file, type is derived from the class and location/size from the GT box (prompts_v3.location_of /
    size_of, reals = "none"); intensity needs a fields file. A missing/unparsed field counts as wrong.
  * 95% CIs: percentile bootstrap, 1,000 resamples, resampling GROUPS with replacement (all images of a drawn group
    enter with its multiplicity; a twin pair belongs to its fake's group). Seed 2027.
  * paired comparison (same images; key = image path, or --pair_key orig to pair counterfactual copies by their
    source path): exact two-sided McNemar on discordant pairs (b = only A correct, c = only B correct), and the
    paired group-bootstrap CI of B - A (same resampled groups for both files).
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import argparse, collections, glob, json, math, os, re, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, _release_path('CIRCE/circe'))
from prompts_v3 import parse_v3, location_of, size_of  # noqa: E402
from circe_reward import parse_output, iou  # noqa: E402

MF = _release_path('AAAI/MedForge++/medforge90k_full/')
META_PATH = os.path.join(HERE, "cache", "mf90k_legacy_meta.json")
CF_TEST_DIR = _release_path('AAAI/MedForge++/v3_counterfactual/test')
FIELDS = ("type", "location", "size", "intensity")
CLS_OF = {"real": "real", "edit": "implant", "inject": "implant", "injection": "implant", "implant": "implant",
          "remove": "removal", "removal": "removal"}
CLASSES = ("real", "implant", "removal", "fake")   # "fake" = deepfake of unknown type
B_DEFAULT, SEED = 1000, 2027


# ----------------------------------------------------------------------------------------------- parsing
def parse_any(text):
    """-> dict(verdict 'deepfake'|'real'|None, box|None, fields dict, well_formed bool, format)."""
    text = text or ""
    p = parse_v3(text)
    if p["verdict"] is not None and any(v is not None for v in p["fields"].values()):
        return {**p, "format": "v3"}
    lp = parse_output(text)
    fmt = lp.get("format", "legacy") if lp["verdict"] is not None else "none"
    return {"verdict": lp["verdict"], "box": lp["box"], "fields": {k: None for k in FIELDS},
            "well_formed": lp["verdict"] is not None, "format": fmt}


# ----------------------------------------------------------------------------------------------- legacy metadata
class Resolver:
    """organ / generator / group / source for rows that do not carry them (old eval_medforge files)."""

    def __init__(self):
        self._meta = None
        self._noext = None
        self._cf = None
        self.fallback = collections.Counter()

    @property
    def meta(self):
        if self._meta is None:
            if not os.path.exists(META_PATH):
                raise SystemExit(f"missing {META_PATH}: run legacy_meta_v3.py first")
            self._meta = json.load(open(META_PATH))
            self._noext = {os.path.splitext(k)[0]: k for k in self._meta}
        return self._meta

    def cf_orig(self, image):
        if self._cf is None:
            self._cf = {}
            for fn in glob.glob(CF_TEST_DIR + "/*.jsonl"):
                for l in open(fn):
                    r = json.loads(l)
                    if r.get("orig"):
                        self._cf[r["image"]] = r["orig"]
        return self._cf.get(image)

    def mf_rel(self, row):
        """MedForge-90K relative path of the image's original, or None."""
        for cand in (row.get("orig"), row["image"]):
            if cand and "/medforge90k_full/" in cand:
                return cand.split("/medforge90k_full/", 1)[1]
        img = row["image"]
        if "/v3_counterfactual/" in img:
            o = self.cf_orig(img)
            if o:
                return o.split("/medforge90k_full/", 1)[1]
            stem = os.path.splitext(os.path.basename(img))[0].replace("__", "/")   # build_counterfactual_test naming
            self.meta
            return self._noext.get(stem)
        if "/mfclean/img/" in img:
            self.meta
            return self._noext.get(os.path.splitext(img.split("/mfclean/img/", 1)[1])[0])
        return None

    @staticmethod
    def mfpp_subset(row):
        s = row.get("subset")
        if s:
            return s
        m = re.search(r"/naitik_annotated/([^/]+)/", row["image"])
        return m.group(1) if m else None

    @staticmethod
    def mfpp_stem(path):
        return re.sub(r"_(polypinj|tumorinj)$", "", os.path.splitext(os.path.basename(path))[0])

    def resolve(self, row):
        """-> dict(organ, gen, group, src_key, dup, key_orig)."""
        rel = self.mf_rel(row)
        if rel is not None and rel in self.meta:
            g, dup, organ, gen, typ, src = self.meta[rel]
            return {"organ": organ, "gen": gen, "group": g, "src_key": src, "dup": dup, "key_orig": rel}
        sub = self.mfpp_subset(row)
        if sub:
            stem = self.mfpp_stem(row["image"])
            organ = "kidney-ct" if sub.startswith("kidney") else ("colonoscopy" if sub.startswith(("polyp", "colon"))
                                                                  else sub)
            m = re.match(r"(case_\d+)", stem)
            if m:
                group = "kits:" + m.group(1)
            elif sub.startswith(("polyp_real", "polyp_removal", "colon_inject")):
                group = "kvasir:" + stem
            else:
                group = f"{sub}:{stem}"
            src = None
            if sub == "kidney_removal":
                src = "kidney_real_tumor/" + stem
            elif sub in ("polyp_removal", "colon_inject"):
                src = "polyp_real/" + stem
            typ = row.get("type", "real")
            return {"organ": organ, "gen": ("real" if typ == "real" else "mf++:" + sub), "group": group,
                    "src_key": src, "dup": None, "key_orig": f"{sub}/{stem}"}
        self.fallback["unresolved"] += 1
        return {"organ": "unknown", "gen": "real" if row.get("type") == "real" else "unknown",
                "group": "img:" + row["image"], "src_key": None, "dup": None, "key_orig": row["image"]}


RESOLVER = Resolver()


# ----------------------------------------------------------------------------------------------- loading
def load_fields(path):
    """GT evidence fields -> {image: {field: value}} (also keyed by path without extension)."""
    out = {}
    if not path:
        return out
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        f = r.get("fields") or r.get("gt_fields") or ({k: r[k] for k in FIELDS if k in r} if "location" in r else {})
        f = {k: (str(v).lower() if v is not None else None) for k, v in f.items() if k in FIELDS}
        for key in (r.get("image"), r.get("orig")):
            if key:
                out[key] = f
                out[os.path.splitext(key)[0]] = f
    return out


def derive_fields(cls, gt_box):
    if cls == "real":
        return {"type": "none", "location": "none", "size": "none"}
    d = {"type": cls} if cls in ("implant", "removal") else {}
    if gt_box:
        d["location"], d["size"] = location_of(gt_box), size_of(gt_box)
    return d


def normalize(row, fields_gt=None):
    typ = row.get("type")
    if typ in CLS_OF:
        cls = CLS_OF[typ]
    elif row.get("gt_verdict") == "real":
        cls = "real"
    else:
        cls = "fake"
    gt_v = "real" if cls == "real" else "deepfake"
    gt_box = row.get("gt_box", row.get("box")) if cls != "real" else None
    p = parse_any(row.get("output", ""))
    has_v3_meta = "group" in row and "organ" in row
    if has_v3_meta:
        organ, gen, group = row["organ"], row.get("gen") or ("real" if cls == "real" else "unknown"), row["group"]
        res = None
        key_orig = row.get("orig") or row["image"]
        if "/medforge90k_full/" in key_orig:
            key_orig = key_orig.split("/medforge90k_full/", 1)[1]
    else:
        res = RESOLVER.resolve(row)
        organ, gen, group, key_orig = res["organ"], res["gen"], res["group"], res["key_orig"]
    gtf = None
    if fields_gt:
        for key in (row["image"], os.path.splitext(row["image"])[0], row.get("orig")):
            if key and key in fields_gt:
                gtf, src = fields_gt[key], "fields_file"
                break
    if gtf is None:   # GT fields carried by the row itself (eval_v3 run on *_fields.jsonl copies "fields")
        for k in ("gt_fields", "fields"):
            if isinstance(row.get(k), dict):
                gtf, src = {f: (str(v).lower() if v is not None else None) for f, v in row[k].items() if f in FIELDS}, "row"
                break
    if gtf is None:
        gtf, src = derive_fields(cls, gt_box), "derived"
    return {"image": row["image"], "cls": cls, "gt_v": gt_v, "gt_box": gt_box, "organ": organ, "gen": gen,
            "group": group, "key_orig": key_orig, "twin_explicit": row.get("twin") if has_v3_meta else None,
            "has_v3_meta": has_v3_meta, "res": res, "pred_v": p["verdict"], "pred_box": p["box"],
            "pred_fields": p["fields"], "well_formed": p["well_formed"], "format": p["format"],
            "gt_fields": gtf, "fields_src": src, "truncated": row.get("truncated")}


def attach_twins(recs):
    """Sets rec['twin_idx'] (index of its twin real inside this file) for fakes; None if the twin is absent."""
    by_img = {r["image"]: i for i, r in enumerate(recs)}
    real_by_key, real_by_dup = {}, {}
    for i, r in enumerate(recs):
        if r["cls"] != "real":
            continue
        real_by_key.setdefault(r["key_orig"], i)
        if r["res"] and r["res"].get("dup"):
            real_by_dup.setdefault(r["res"]["dup"], i)
    for r in recs:
        r["twin_idx"] = None
        if r["cls"] == "real":
            continue
        if r["has_v3_meta"]:
            t = r["twin_explicit"]
            if t and t in by_img and recs[by_img[t]]["cls"] == "real":
                r["twin_idx"] = by_img[t]
            continue
        res = r["res"]
        if not res or not res.get("src_key"):
            continue
        i = real_by_key.get(res["src_key"])
        if i is None and res.get("dup"):
            i = real_by_dup.get(res["dup"])
        r["twin_idx"] = i
    return recs


def load_gen(path, fields_gt=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    seen, recs, ndup = set(), [], 0
    for row in rows:
        if row["image"] in seen:
            ndup += 1
            continue
        seen.add(row["image"])
        recs.append(normalize(row, fields_gt))
    if ndup:
        print(f"[warn] {path}: {ndup} duplicate image rows ignored (first kept)", file=sys.stderr)
    return attach_twins(recs)


# ----------------------------------------------------------------------------------------------- sufficient stats
COLS = ["n", "correct", "correct_legacy", "parsed", "well_formed",
        "n_real", "tp_real", "fp_real", "n_fake", "tp_fake", "fp_fake",
        "n_implant", "c_implant", "n_removal", "c_removal", "n_fakeu", "c_fakeu",
        "n_boxed", "iou_sum", "iou_gated_sum", "hit50", "n_pairs", "pair_ok", "pair_fake", "pair_twin",
        "pred_fake", "box_on_pred_fake", "box_on_pred_real", "consistent", "n_v3",
        "fp_real_legacy", "tp_real_legacy", "tp_fake_legacy", "fp_fake_legacy"]
for _k in FIELDS:
    COLS += [f"fn_{_k}", f"fc_{_k}", f"fnf_{_k}", f"fcf_{_k}"]
CI = {c: i for i, c in enumerate(COLS)}


def per_image_matrix(recs):
    """(N, K) per-image contributions; twin-pair columns are credited to the fake's row."""
    N = len(recs)
    X = np.zeros((N, len(COLS)), dtype=np.float64)
    for i, r in enumerate(recs):
        if r.get("_shadow"):      # twin real pulled into a subset only so that its fake's pair can be scored
            continue
        x = X[i]
        pv, gv = r["pred_v"], r["gt_v"]
        pv_leg = pv or "real"
        x[CI["n"]] = 1
        x[CI["correct"]] = pv == gv
        x[CI["correct_legacy"]] = pv_leg == gv
        x[CI["parsed"]] = pv is not None
        x[CI["well_formed"]] = bool(r["well_formed"])
        x[CI["pred_fake"]] = pv == "deepfake"
        if r["pred_box"] is not None:
            x[CI["box_on_pred_fake"]] = pv == "deepfake"
            x[CI["box_on_pred_real"]] = pv == "real"
        if gv == "real":
            x[CI["n_real"]] = 1
            x[CI["tp_real"]] = pv == "real"
            x[CI["fp_fake"]] = pv == "deepfake"
            x[CI["tp_real_legacy"]] = pv_leg == "real"
            x[CI["fp_fake_legacy"]] = pv_leg == "deepfake"
        else:
            x[CI["n_fake"]] = 1
            x[CI["tp_fake"]] = pv == "deepfake"
            x[CI["fp_real"]] = pv == "real"
            x[CI["tp_fake_legacy"]] = pv_leg == "deepfake"
            x[CI["fp_real_legacy"]] = pv_leg == "real"
            key = {"implant": "implant", "removal": "removal"}.get(r["cls"], "fakeu")
            x[CI["n_" + key]] = 1
            x[CI["c_" + key]] = pv == "deepfake"
            if r["gt_box"]:
                v = iou(r["pred_box"], r["gt_box"]) if r["pred_box"] else 0.0
                x[CI["n_boxed"]] = 1
                x[CI["iou_sum"]] = v
                x[CI["iou_gated_sum"]] = v if pv == "deepfake" else 0.0
                x[CI["hit50"]] = v >= 0.5
            t = r.get("twin_idx")
            if t is not None:
                tv = recs[t]["pred_v"]
                x[CI["n_pairs"]] = 1
                x[CI["pair_fake"]] = pv == "deepfake"
                x[CI["pair_twin"]] = tv == "real"
                x[CI["pair_ok"]] = pv == "deepfake" and tv == "real"
        if r["format"] == "v3":
            x[CI["n_v3"]] = 1
            f = r["pred_fields"]
            say_fake = (f.get("type") not in (None, "none")) or r["pred_box"] is not None
            x[CI["consistent"]] = (pv == "deepfake") == say_fake
        for k in FIELDS:
            g = (r["gt_fields"] or {}).get(k)
            if g is None:
                continue
            ok = (r["pred_fields"] or {}).get(k) == g
            x[CI["fn_" + k]] = 1
            x[CI["fc_" + k]] = ok
            if gv == "deepfake":
                x[CI["fnf_" + k]] = 1
                x[CI["fcf_" + k]] = ok
    return X


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def metrics_from_sums(S):
    """S: (..., K) column sums -> dict of metric arrays (shape S.shape[:-1])."""
    c = lambda name: S[..., CI[name]]  # noqa: E731
    out = {}
    out["acc"] = _div(c("correct"), c("n"))
    out["acc_legacy"] = _div(c("correct_legacy"), c("n"))
    rr, rf = _div(c("tp_real"), c("n_real")), _div(c("tp_fake"), c("n_fake"))
    out["rec_real"], out["rec_fake"] = rr, rf
    out["bal_acc"] = (rr + rf) / 2
    for t, key in (("implant", "implant"), ("removal", "removal"), ("fake_untyped", "fakeu")):
        out["rec_" + t] = _div(c("c_" + key), c("n_" + key))
    typ = np.stack([out["rec_implant"], out["rec_removal"], out["rec_fake_untyped"]], axis=-1)
    with np.errstate(invalid="ignore"):
        cnt = np.sum(~np.isnan(typ), axis=-1)
        mt = np.where(cnt > 0, np.nansum(typ, axis=-1) / np.maximum(cnt, 1), np.nan)
    out["bal_acc_typebal"] = (rr + mt) / 2

    def f1(tp, fp, n_pos):
        fn = n_pos - tp
        return _div(2 * tp, 2 * tp + fp + fn)
    f1r = f1(c("tp_real"), c("fp_real"), c("n_real"))
    f1f = f1(c("tp_fake"), c("fp_fake"), c("n_fake"))
    out["f1_real"], out["f1_fake"] = f1r, f1f
    out["f1_macro"] = (f1r + f1f) / 2
    out["f1_macro_legacy"] = (f1(c("tp_real_legacy"), c("fp_real_legacy"), c("n_real"))
                              + f1(c("tp_fake_legacy"), c("fp_fake_legacy"), c("n_fake"))) / 2
    out["mIoU"] = _div(c("iou_sum"), c("n_boxed"))
    out["mIoU_gated"] = _div(c("iou_gated_sum"), c("n_boxed"))
    out["IoU@0.5"] = _div(c("hit50"), c("n_boxed"))
    out["twin"] = _div(c("pair_ok"), c("n_pairs"))
    out["twin_fake_flagged"] = _div(c("pair_fake"), c("n_pairs"))
    out["twin_real_cleared"] = _div(c("pair_twin"), c("n_pairs"))
    out["parse_rate"] = _div(c("parsed"), c("n"))
    out["well_formed_rate"] = _div(c("well_formed"), c("n"))
    out["box_rate_pred_fake"] = _div(c("box_on_pred_fake"), c("pred_fake"))
    out["box_on_pred_real_rate"] = _div(c("box_on_pred_real"), c("n") - c("pred_fake") - (c("n") - c("parsed")))
    out["v3_format_rate"] = _div(c("n_v3"), c("n"))
    out["evidence_verdict_consistent"] = _div(c("consistent"), c("n_v3"))
    for k in FIELDS:
        out[f"field_{k}"] = _div(c("fc_" + k), c("fn_" + k))
        out[f"field_{k}_fakes"] = _div(c("fcf_" + k), c("fnf_" + k))
    return out


CI_METRICS = ["acc", "bal_acc", "bal_acc_typebal", "f1_macro", "rec_real", "rec_fake", "rec_implant", "rec_removal",
              "mIoU", "mIoU_gated", "IoU@0.5", "twin", "acc_legacy"] + [f"field_{k}" for k in FIELDS]


def group_sums(X, groups):
    gid = {g: i for i, g in enumerate(sorted(set(groups)))}
    idx = np.array([gid[g] for g in groups], dtype=np.int64)
    S = np.zeros((len(gid), X.shape[1]))
    np.add.at(S, idx, X)
    return S, idx


def boot_weights(G, B=B_DEFAULT, seed=SEED):
    rng = np.random.default_rng(seed)
    W = np.empty((B, G), dtype=np.float64)
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, G, size=G), minlength=G)
    return W


def _pct(a):
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return [None, None]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def _f(v):
    v = float(v)
    return None if math.isnan(v) else v


def summarize(recs, B=B_DEFAULT, seed=SEED, X=None):
    """Point estimates for every metric + group-bootstrap 95% CIs for CI_METRICS."""
    if not recs:
        return {"n": 0}
    if X is None:
        X = per_image_matrix(recs)
    keep = np.array([not r.get("_shadow") for r in recs])
    S, _ = group_sums(X[keep], [r["group"] for r in recs if not r.get("_shadow")])
    point = {k: _f(v) for k, v in metrics_from_sums(S.sum(0)).items()}
    W = boot_weights(S.shape[0], B, seed)
    T = W @ S
    bm = metrics_from_sums(T)
    ci = {k: _pct(bm[k]) for k in CI_METRICS}
    tot = S.sum(0)
    counts = {c: int(tot[CI[c]]) for c in ("n", "n_real", "n_fake", "n_implant", "n_removal", "n_fakeu", "n_boxed",
                                           "n_pairs", "parsed", "n_v3")}
    counts["groups"] = int(S.shape[0])
    return {"counts": counts, "point": point, "ci95": ci}


# ----------------------------------------------------------------------------------------------- comparisons
def mcnemar_exact(b, c):
    """Two-sided exact McNemar (binomial on discordant pairs, p = min(1, 2 P[X <= min(b,c)]), X~Bin(b+c, .5))."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2 ** n)


def paired(recsA, recsB, key="image", B=B_DEFAULT, seed=SEED):
    kA = {r[key]: i for i, r in enumerate(recsA)}
    kB = {r[key]: i for i, r in enumerate(recsB)}
    common = [k for k in kA if k in kB]
    if not common:
        return {"n_common": 0}
    XA_all, XB_all = per_image_matrix(recsA), per_image_matrix(recsB)
    ia = np.array([kA[k] for k in common]); ib = np.array([kB[k] for k in common])
    XA, XB = XA_all[ia], XB_all[ib]
    okA, okB = XA[:, CI["correct"]].astype(bool), XB[:, CI["correct"]].astype(bool)
    b, c = int((okA & ~okB).sum()), int((~okA & okB).sum())
    groups = [recsA[kA[k]]["group"] for k in common]
    SA, _ = group_sums(XA, groups)
    SB, _ = group_sums(XB, groups)
    W = boot_weights(SA.shape[0], B, seed)
    mA, mB = metrics_from_sums(W @ SA), metrics_from_sums(W @ SB)
    pA, pB = metrics_from_sums(SA.sum(0)), metrics_from_sums(SB.sum(0))
    diff = {}
    for m in ("acc", "bal_acc", "f1_macro", "rec_real", "rec_fake", "mIoU", "IoU@0.5", "twin"):
        d = _f(pB[m] - pA[m])
        diff[m] = {"A": _f(pA[m]), "B": _f(pB[m]), "B_minus_A": d, "ci95": _pct(mB[m] - mA[m])}
    gA = collections.Counter(r["gt_v"] for r in (recsA[kA[k]] for k in common))
    mismatch = sum(recsA[kA[k]]["gt_v"] != recsB[kB[k]]["gt_v"] for k in common)
    return {"n_common": len(common), "n_only_A": len(kA) - len(common), "n_only_B": len(kB) - len(common),
            "groups": int(SA.shape[0]), "gt_mismatch": mismatch, "gt_counts": dict(gA),
            "mcnemar": {"b_A_only_correct": b, "c_B_only_correct": c, "p_exact": mcnemar_exact(b, c)},
            "diff": diff}


# ----------------------------------------------------------------------------------------------- full report
def breakdown(recs, attr, B, seed, only=None):
    out = {}
    vals = sorted({r[attr] for r in recs if only is None or only(r)})
    for v in vals:
        sub = [r for r in recs if r[attr] == v]
        # twin pairs need both members present: keep the twin reals of the subset's fakes
        out[v] = summarize(remap_subset(recs, sub), B, seed)
    return out


def remap_subset(recs, sub):
    """Copy a subset so twin indices point inside it (twin reals outside the subset are pulled in only for pairs)."""
    idx_of = {id(r): i for i, r in enumerate(recs)}
    new = [dict(r) for r in sub]
    pos = {idx_of[id(r)]: j for j, r in enumerate(sub)}
    for j, r in enumerate(sub):
        t = r.get("twin_idx")
        new[j]["twin_idx"] = pos.get(t) if t is not None else None
        if t is not None and new[j]["twin_idx"] is None:
            # twin real lives outside this subset (e.g. per-generator table): score the pair via a shadow copy
            # that is excluded from every non-pair statistic.
            shadow = dict(recs[t]); shadow["_shadow"] = True
            new.append(shadow); new[j]["twin_idx"] = len(new) - 1
    return new


def full_report(recs, B=B_DEFAULT, seed=SEED, name=""):
    rep = {"file": name, "overall": summarize(recs, B, seed)}
    rep["per_organ"] = breakdown(recs, "organ", B, seed)
    rep["per_generator"] = breakdown(recs, "gen", B, seed)
    rep["per_class"] = breakdown(recs, "cls", B, seed)
    rep["formats"] = dict(collections.Counter(r["format"] for r in recs))
    rep["fields_source"] = dict(collections.Counter(r["fields_src"] for r in recs))
    tr = [r["truncated"] for r in recs if r.get("truncated") is not None]
    rep["truncated_rate"] = (sum(map(bool, tr)) / len(tr)) if tr else None
    rep["resolver_fallback"] = dict(RESOLVER.fallback)
    return rep


# ----------------------------------------------------------------------------------------------- markdown
def pct(v, d=2):
    return "--" if v is None else f"{100 * v:.{d}f}"


def pci(s, m, d=2):
    v = s["point"].get(m)
    ci = s["ci95"].get(m, [None, None])
    if v is None:
        return "--"
    if ci[0] is None:
        return pct(v, d)
    return f"{pct(v, d)} [{pct(ci[0], 1)}, {pct(ci[1], 1)}]"


def md_report(rep):
    o = rep["overall"]; P = o["point"]; C = o["counts"]
    L = [f"### {rep['file']}", "",
         f"n={C['n']} (real {C['n_real']}, implant {C['n_implant']}, removal {C['n_removal']}"
         + (f", untyped fake {C['n_fakeu']}" if C['n_fakeu'] else "") + f"), groups={C['groups']}, "
         f"twin pairs={C['n_pairs']}, formats={rep['formats']}", "",
         "| metric | value [95% group-bootstrap CI] |", "|---|---|"]
    for m, lab in (("acc", "accuracy"), ("bal_acc", "balanced acc (binary)"), ("bal_acc_typebal", "balanced acc (type-balanced)"),
                   ("f1_macro", "macro-F1 (binary)"), ("rec_real", "recall real"), ("rec_implant", "recall implant"),
                   ("rec_removal", "recall removal"), ("mIoU", "mIoU (fakes)"), ("mIoU_gated", "mIoU gated by verdict"),
                   ("IoU@0.5", "IoU@0.5 (fakes)"), ("twin", "twin consistency"), ("acc_legacy", "acc, legacy (unparsed->real)")):
        L.append(f"| {lab} | {pci(o, m)} |")
    L.append(f"| twin: fake flagged / twin cleared | {pct(P['twin_fake_flagged'])} / {pct(P['twin_real_cleared'])} |")
    L.append(f"| parse rate / well-formed / v3-format | {pct(P['parse_rate'])} / {pct(P['well_formed_rate'])} / {pct(P['v3_format_rate'])} |")
    L.append(f"| box given when 'deepfake' / box given when 'real' | {pct(P['box_rate_pred_fake'])} / {pct(P['box_on_pred_real_rate'])} |")
    L.append(f"| evidence-verdict consistency (v3 rows) | {pct(P['evidence_verdict_consistent'])} |")
    for k in FIELDS:
        L.append(f"| rationale {k}: all / fakes | {pci(o, 'field_' + k)} / {pct(P['field_' + k + '_fakes'])} |")
    L.append(f"\nfield GT source: {rep['fields_source']}")
    for title, key in (("organ", "per_organ"), ("generator", "per_generator")):
        L += ["", f"| {title} | n | acc | bal acc | rec real | rec implant | rec removal | mIoU | IoU@0.5 | twin (pairs) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for v, s in rep[key].items():
            if not s.get("counts"):
                continue
            p = s["point"]
            L.append(f"| {v} | {s['counts']['n']} | {pci(s, 'acc', 1)} | {pci(s, 'bal_acc', 1)} | {pct(p['rec_real'], 1)} | "
                     f"{pct(p['rec_implant'], 1)} | {pct(p['rec_removal'], 1)} | {pct(p['mIoU'], 1)} | {pct(p['IoU@0.5'], 1)} | "
                     f"{pct(p['twin'], 1)} ({s['counts']['n_pairs']}) |")
    return "\n".join(L)


def md_paired(pr, nameA, nameB):
    if not pr.get("n_common"):
        return f"\nno common images between {nameA} and {nameB}\n"
    m = pr["mcnemar"]
    L = [f"#### paired: A={nameA}  vs  B={nameB}", "",
         f"common images {pr['n_common']} (only A {pr['n_only_A']}, only B {pr['n_only_B']}), groups {pr['groups']}, "
         f"GT mismatches {pr['gt_mismatch']}. Exact McNemar: b(A only correct)={m['b_A_only_correct']}, "
         f"c(B only correct)={m['c_B_only_correct']}, p={m['p_exact']:.4g}", "",
         "| metric | A | B | B-A [95% paired group-bootstrap CI] |", "|---|---|---|---|"]
    for k, d in pr["diff"].items():
        ci = d["ci95"]
        s = "--" if d["B_minus_A"] is None else f"{100 * d['B_minus_A']:+.2f}" + (
            f" [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]" if ci[0] is not None else "")
        L.append(f"| {k} | {pct(d['A'])} | {pct(d['B'])} | {s} |")
    return "\n".join(L)


def md_batch(rows):
    L = ["| file | n | groups | acc [95% CI] | bal acc [95% CI] | macro-F1 | rec real | rec implant | rec removal | "
         "mIoU [95% CI] | IoU@0.5 | twin (pairs) | parse | acc legacy |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows:
        p, c = s["point"], s["counts"]
        L.append(f"| {name} | {c['n']} | {c['groups']} | {pci(s, 'acc')} | {pci(s, 'bal_acc')} | {pct(p['f1_macro'])} | "
                 f"{pct(p['rec_real'], 1)} | {pct(p['rec_implant'], 1)} | {pct(p['rec_removal'], 1)} | {pci(s, 'mIoU')} | "
                 f"{pct(p['IoU@0.5'])} | {pct(p['twin'], 1)} ({c['n_pairs']}) | {pct(p['parse_rate'])} | "
                 f"{pct(p['acc_legacy'])} |")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------- self test
def _mk(image, typ, out, group, organ="brain-mri", gen=None, box=None, twin=None):
    return {"image": image, "type": typ, "gen": gen or ("real" if typ == "real" else "g1"), "box": box,
            "organ": organ, "group": group, "split": "test", "twin": twin, "output": out}


def _v3(verdict, box=None, ftype="none", intensity="none"):
    from prompts_v3 import format_target
    return format_target(verdict, box, ftype, intensity)


def _recs(rows, fields=None):
    return attach_twins([normalize(r, fields) for r in rows])


def self_test():
    import random
    eps = 1e-9
    ok = lambda a, b: abs(a - b) < 1e-9  # noqa: E731
    # ---- synthetic twin-structured set: 4 groups x (1 real + implant + removal); boxes known
    rows, truth = [], []
    for g in range(4):
        real = f"/x/g{g}_real.jpg"
        rows.append(_mk(real, "real", "", f"G{g}"))
        rows.append(_mk(f"/x/g{g}_imp.jpg", "edit", "", f"G{g}", box=[100, 100, 300, 300], twin=real, gen="ga"))
        rows.append(_mk(f"/x/g{g}_rem.jpg", "remove", "", f"G{g}", box=[600, 600, 700, 800], twin=real, gen="gb"))

    def with_preds(fn):
        out = []
        for r in rows:
            r2 = dict(r); r2["output"] = fn(r); out.append(r2)
        return _recs(out)

    # 1) perfect predictor (v3 format, correct evidence)
    perfect = with_preds(lambda r: _v3("real") if r["type"] == "real" else
                         _v3("deepfake", r["box"], "implant" if r["type"] == "edit" else "removal", "brighter"))
    s = summarize(perfect, B=200)["point"]
    for m in ("acc", "bal_acc", "bal_acc_typebal", "f1_macro", "rec_real", "rec_implant", "rec_removal", "mIoU",
              "IoU@0.5", "twin", "parse_rate", "well_formed_rate", "field_type", "field_location", "field_size",
              "evidence_verdict_consistent"):
        assert ok(s[m], 1.0), (m, s[m])
    assert s["field_intensity"] is None, "intensity has no derived GT without a fields file"
    # 2) constant 'real' predictor: bal acc 0.5, twin 0, macro-F1 = nr/(2nr+nf) = 4/(8+8) = 0.25
    cr = with_preds(lambda r: _v3("real"))
    s = summarize(cr, B=200)["point"]
    assert ok(s["bal_acc"], 0.5) and ok(s["bal_acc_typebal"], 0.5) and ok(s["twin"], 0.0), s
    assert ok(s["acc"], 4 / 12) and ok(s["f1_macro"], 0.25) and ok(s["rec_real"], 1) and ok(s["rec_implant"], 0)
    assert ok(s["mIoU"], 0.0) and ok(s["twin_real_cleared"], 1.0) and ok(s["twin_fake_flagged"], 0.0)
    # 3) constant 'deepfake' predictor (box elsewhere): bal acc 0.5, twin 0, F1_fake = 2*8/(16+4) = 0.8 -> macro 0.4
    cf = with_preds(lambda r: _v3("deepfake", [0, 0, 50, 50], "implant"))
    s = summarize(cf, B=200)["point"]
    assert ok(s["bal_acc"], 0.5) and ok(s["twin"], 0.0) and ok(s["f1_macro"], 0.4) and ok(s["acc"], 8 / 12), s
    assert ok(s["mIoU"], 0.0) and ok(s["IoU@0.5"], 0.0)
    # 4) unparsable output: invalid for every class; legacy rule maps it to 'real'
    un = with_preds(lambda r: "I am not sure.")
    s = summarize(un, B=200)["point"]
    assert ok(s["acc"], 0) and ok(s["bal_acc"], 0) and ok(s["parse_rate"], 0) and ok(s["twin"], 0)
    assert ok(s["acc_legacy"], 4 / 12) and ok(s["f1_macro"], 0.0) and ok(s["f1_macro_legacy"], 0.25)
    # 5) hand-computed confusion: reals 4 (3 right, 1 called fake), implants 4 (2 right), removals 4 (4 right)
    def f5(r):
        i = int(r["image"].split("/g")[1][0])
        if r["type"] == "real":
            return _v3("deepfake", [0, 0, 9, 9], "implant") if i == 0 else _v3("real")
        if r["type"] == "edit":
            return _v3("deepfake", [100, 100, 300, 500], "implant") if i < 2 else _v3("real")   # IoU 0.5
        return _v3("deepfake", [650, 600, 750, 800], "removal")                              # IoU 1/3
    h = with_preds(f5)
    s = summarize(h, B=200)["point"]
    # TP_real 3, FP_real (fake called real) 2, FN_real 1 -> F1_real = 6/9; TP_fake 6, FP_fake 1, FN_fake 2 -> 12/15
    assert ok(s["acc"], 9 / 12) and ok(s["rec_real"], 0.75) and ok(s["rec_implant"], 0.5) and ok(s["rec_removal"], 1.0)
    assert ok(s["bal_acc"], (0.75 + 6 / 8) / 2) and ok(s["bal_acc_typebal"], (0.75 + 0.75) / 2)
    assert ok(s["f1_macro"], (6 / 9 + 12 / 15) / 2), s["f1_macro"]
    # implants i<2: IoU 0.5 (hit); i>=2: verdict real, no box -> 0; removals: IoU 1/3 (miss)
    assert ok(s["mIoU"], (2 * 0.5 + 4 / 3) / 8) and ok(s["IoU@0.5"], 2 / 8), (s["mIoU"], s["IoU@0.5"])
    # twins: G0 real flagged -> both its pairs fail; G1: both ok; G2,G3: removal ok, implant missed -> 1+2+... = 4/8
    assert ok(s["twin"], (0 + 2 + 1 + 1) / 8), s["twin"]
    # 6) twin pairs only count when the twin is in the file
    part = _recs([r for r in rows if r["image"] != "/x/g0_real.jpg"])
    assert summarize(part, B=50)["counts"]["n_pairs"] == 6
    # 7) rationale accuracy against a fields file (intensity GT given): 3 of 4 intensity right on fakes
    fields_gt = {}
    for r in rows:
        if r["type"] != "real":
            fields_gt[r["image"]] = {"type": "implant" if r["type"] == "edit" else "removal",
                                     "location": location_of(r["box"]), "size": size_of(r["box"]),
                                     "intensity": "brighter" if r["type"] == "edit" else "flattened"}
        else:
            fields_gt[r["image"]] = {k: "none" for k in FIELDS}
    out = []
    for r in rows:
        r2 = dict(r)
        if r["type"] == "real":
            r2["output"] = _v3("real")
        else:
            r2["output"] = _v3("deepfake", r["box"], "implant" if r["type"] == "edit" else "removal", "brighter")
        out.append(r2)
    fr = attach_twins([normalize(r, fields_gt) for r in out])
    s = summarize(fr, B=50)["point"]
    assert ok(s["field_intensity_fakes"], 0.5) and ok(s["field_intensity"], (4 + 4) / 12) and ok(s["field_type"], 1)
    # 8) parsing: v3, legacy CIRCE, MedForge-Reasoner, garbage
    p = parse_any(_v3("deepfake", [1, 2, 30, 40], "removal", "flattened"))
    assert p["format"] == "v3" and p["verdict"] == "deepfake" and p["box"] == [1, 2, 30, 40] and p["well_formed"]
    p = parse_any("<think> x1=\"10\" y1=\"20\" x2=\"5\" y2=\"60\" </think>\n\nThis image is a deepfake.<|im_end|>")
    assert p["format"] == "legacy" and p["verdict"] == "deepfake" and p["box"] == [5, 20, 10, 60]
    p = parse_any("To analyze whether this image is a deepfake: ... </think>\n\nThis image is a real image.")
    assert p["verdict"] == "real"
    assert parse_any("blah")["verdict"] is None and parse_any("")["format"] == "none"
    # 9) IoU known values
    assert ok(iou([0, 0, 10, 10], [0, 0, 10, 20]), 0.5) and ok(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
    # 10) exact McNemar: matches scipy.stats.binomtest when available, and known values
    assert ok(mcnemar_exact(0, 0), 1.0) and ok(mcnemar_exact(15, 15), 1.0) and ok(mcnemar_exact(0, 5), 0.0625)
    assert abs(mcnemar_exact(3, 11) - 0.057373046875) < 1e-12
    try:
        from scipy.stats import binomtest
        rng = random.Random(0)
        for _ in range(200):
            b, c = rng.randint(0, 60), rng.randint(0, 60)
            if b + c:
                assert abs(mcnemar_exact(b, c) - binomtest(b, b + c, 0.5).pvalue) < 1e-9, (b, c)
        scipy_note = "cross-checked vs scipy.binomtest on 200 random (b,c)"
    except ImportError:
        scipy_note = "scipy not installed: known values only"
    # 11) the bootstrap resamples GROUPS: 2 groups x 50 images, one all-right, one all-wrong -> CI = [0, 1];
    #     an image-level bootstrap would give roughly [0.40, 0.60]
    rows2 = []
    for g in range(2):
        for j in range(50):
            rows2.append(_mk(f"/y/{g}_{j}.jpg", "real", _v3("real") if g == 0 else _v3("deepfake", [0, 0, 5, 5]), f"H{g}"))
    sm = summarize(_recs(rows2), B=1000)
    assert ok(sm["point"]["acc"], 0.5) and sm["ci95"]["acc"] == [0.0, 1.0], sm["ci95"]["acc"]
    # 12) bootstrap CI width for 2000 singleton groups with p=0.8 ~ 2*1.96*sqrt(.16/2000) = 0.0351
    rng = random.Random(1)
    rows3 = [_mk(f"/z/{j}.jpg", "real", _v3("real") if rng.random() < 0.8 else _v3("deepfake", [0, 0, 5, 5]), f"S{j}")
             for j in range(2000)]
    sm = summarize(_recs(rows3), B=1000)
    w = sm["ci95"]["acc"][1] - sm["ci95"]["acc"][0]
    pnt = sm["point"]["acc"]
    expect = 2 * 1.96 * math.sqrt(pnt * (1 - pnt) / 2000)
    assert abs(w - expect) / expect < 0.15 and sm["ci95"]["acc"][0] < pnt < sm["ci95"]["acc"][1], (w, expect)
    # 13) paired comparison: identical files -> b=c=0, p=1, diff CI [0,0]; B fixes the 2 implants missed in h
    pr = paired(h, h)
    assert pr["mcnemar"]["p_exact"] == 1.0 and pr["diff"]["acc"]["ci95"] == [0.0, 0.0]
    pr = paired(h, perfect)
    assert pr["mcnemar"]["b_A_only_correct"] == 0 and pr["mcnemar"]["c_B_only_correct"] == 3
    assert ok(pr["diff"]["acc"]["B_minus_A"], 3 / 12) and ok(pr["mcnemar"]["p_exact"], 0.25)
    assert pr["diff"]["acc"]["ci95"][0] >= 0 and pr["diff"]["acc"]["ci95"][1] <= 0.5
    # 14) per-generator table scores twin pairs although the twin real is not a member of that generator
    rep = full_report(perfect, B=50, name="perfect")
    assert ok(rep["per_generator"]["ga"]["point"]["twin"], 1.0) and rep["per_generator"]["ga"]["counts"]["n_pairs"] == 4
    assert rep["per_generator"]["ga"]["counts"]["n"] == 4, "shadow twin reals must not count as images"
    cr_rep = full_report(cr, B=50, name="const-real")
    assert ok(cr_rep["per_generator"]["gb"]["point"]["twin"], 0.0)
    # 15) old-format rows resolve organ / generator / group / twin from the path (needs the legacy table)
    legacy_note = "skipped (no legacy table)"
    if os.path.exists(META_PATH):
        meta = RESOLVER.meta
        fake_rel = next(k for k, v in meta.items() if v[5] and v[5] in meta and v[4] == "remove")
        src_rel = meta[fake_rel][5]
        old = [{"image": MF + fake_rel, "gt_verdict": "deepfake", "gt_box": [10, 10, 50, 50], "type": "remove",
                "subset": None, "output": "This image is a deepfake."},
               {"image": MF + src_rel, "gt_verdict": "real", "gt_box": None, "type": "real", "subset": None,
                "output": "This image is a real image."}]
        rr = attach_twins([normalize(r) for r in old])
        assert rr[0]["group"] == rr[1]["group"] and rr[0]["twin_idx"] == 1 and rr[0]["gen"] == meta[fake_rel][3]
        assert rr[0]["organ"] == meta[fake_rel][2] and ok(summarize(rr, B=20)["point"]["twin"], 1.0)
        mpp = [{"image": "/a/naitik_annotated/kidney_removal/images/case_00532_slice_084.png", "type": "remove",
                "gt_verdict": "deepfake", "gt_box": [1, 1, 5, 5], "subset": "kidney_removal", "output": "This image is a deepfake."},
               {"image": "/a/naitik_annotated/kidney_real_tumor/images/case_00532_slice_084.png", "type": "real",
                "gt_verdict": "real", "gt_box": None, "subset": "kidney_real_tumor", "output": "This image is a deepfake."}]
        rr = attach_twins([normalize(r) for r in mpp])
        assert rr[0]["twin_idx"] == 1 and rr[0]["group"] == rr[1]["group"] == "kits:case_00532"
        assert ok(summarize(rr, B=20)["point"]["twin"], 0.0)
        legacy_note = "ok"
    print("[self_test] PASS: perfect / constant-real (bal acc 0.5, twin 0, F1 0.25) / constant-deepfake "
          "(bal acc 0.5, twin 0, F1 0.4) / unparsed / hand-computed confusion+IoU+twin / twin presence / "
          f"rationale fields / parsers / IoU / McNemar ({scipy_note}) / group bootstrap (2-group CI [0,1]; "
          f"width {w:.4f} vs analytic {expect:.4f}) / paired diff / per-generator twin / legacy resolver: {legacy_note}")


# ----------------------------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--gen", help="generation file A")
    ap.add_argument("--gen2", help="generation file B (paired comparison on common images)")
    ap.add_argument("--fields", help="GT evidence-field jsonl (*_fields.jsonl)")
    ap.add_argument("--rescore", nargs="+", help="glob(s) of generation files: one summary row each")
    ap.add_argument("--pair", action="append", default=[], help="A,B paths for paired rows in --rescore mode")
    ap.add_argument("--pair_key", default="image", choices=["image", "key_orig"])
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out_json")
    ap.add_argument("--out_md")
    ap.add_argument("--full", action="store_true", help="--rescore: also write per-file full reports into --out_json")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    fields_gt = load_fields(a.fields)
    md, js = [], {}
    if a.gen:
        A = load_gen(a.gen, fields_gt)
        repA = full_report(A, a.B, a.seed, os.path.basename(a.gen))
        js["A"] = repA
        md.append(md_report(repA))
        if a.gen2:
            Bf = load_gen(a.gen2, fields_gt)
            repB = full_report(Bf, a.B, a.seed, os.path.basename(a.gen2))
            js["B"] = repB
            md.append(md_report(repB))
            js["paired"] = paired(A, Bf, a.pair_key, a.B, a.seed)
            md.append(md_paired(js["paired"], os.path.basename(a.gen), os.path.basename(a.gen2)))
    elif a.rescore:
        files = sorted({f for g in a.rescore for f in glob.glob(os.path.expanduser(g))})
        rows, cache = [], {}
        js["files"] = {}
        for f in files:
            try:
                recs = load_gen(f, fields_gt)
            except Exception as e:  # noqa
                print(f"[skip] {f}: {e!r}", file=sys.stderr)
                continue
            if not recs:
                print(f"[skip] {f}: empty", file=sys.stderr)
                continue
            cache[f] = recs
            s = full_report(recs, a.B, a.seed, os.path.basename(f)) if a.full else summarize(recs, a.B, a.seed)
            js["files"][f] = s
            rows.append((os.path.basename(f)[:-6], s["overall"] if a.full else s))
            print(f"[scored] {f} n={len(recs)}", file=sys.stderr)
        md.append(md_batch(rows))
        js["paired"] = {}
        for pa in a.pair:
            fa, fb = [os.path.expanduser(x) for x in pa.split(",")]
            if fa not in cache or fb not in cache:
                for f in (fa, fb):
                    if f not in cache and os.path.exists(f):
                        cache[f] = load_gen(f, fields_gt)
            if fa in cache and fb in cache:
                pr = paired(cache[fa], cache[fb], a.pair_key, a.B, a.seed)
                js["paired"][pa] = pr
                md.append(md_paired(pr, os.path.basename(fa), os.path.basename(fb)))
    else:
        ap.error("pass --self_test, --gen or --rescore")
    text = "\n\n".join(md)
    print(text)
    if a.out_md:
        with open(a.out_md, "w") as fh:
            fh.write(text + "\n")
    if a.out_json:
        with open(a.out_json, "w") as fh:
            json.dump(js, fh, indent=1)


if __name__ == "__main__":
    main()
