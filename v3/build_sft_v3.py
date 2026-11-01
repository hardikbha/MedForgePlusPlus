"""
build_sft_v3.py -- MF-Clean manifests -> v3 SFT / RL / eval-field files (data component of the v3 rebuild).

README
------
Input  (from build_mfclean.py, --mfclean):  train_all.jsonl, select_core.jsonl, test_core.jsonl
         rows {image,type in real|edit|remove,gen,box(1024 frame, fakes only),organ,group,split,disease,twin,orig}
Output (--out, default ~/AAAI/MedForge++/mfclean/v3data/):
  sft_{id,cf,cm,cfcm}.jsonl   {"images":[path],"messages":[system SYS_V3, user "<image>"+USER_V3,
                               assistant format_target(...)]}   (3-message records; train_sft.py's collator
                               unmasks every assistant span, see check_sft_collator_v3.py)
  rl_{id,cf,cm,cfcm}.jsonl    twin-pair rows {"kind":"pair","fake":{image,box,type,gen,organ,disease,fields,
                               fields_src},"real":{image,organ},"group","src_rank"}  +  singleton real rows
                               {"kind":"single","fake":null,"real":{...},"group"} (<= ~20% of rows)
  select_core_fields.jsonl, test_core_fields.jsonl   eval rows + "fields", "fields_src", "target"
  evidence_stats.jsonl        raw per-fake pixel statistics behind every evidence field (all splits)
  build_info.json             T, counts, md5 of every output
Report : ~/CIRCE/v3/reports/data_v3.md (auto section between AUTO markers; text outside is preserved)

Design (all deterministic, no RNG state):
  * SFT/RL split of TRAIN groups: sha1("sftrl-v1:"+group) % 6 == 0 -> RL (1/6), else SFT (5/6). A group (and
    hence a source scan, its twin real and every sibling forgery) is never in both pools.
  * Regimes filter the train split only: ID all; CF no "edit" fakes; CM no fakes from CM_HOLDOUT; CFxCM both.
  * SFT: all reals of SFT groups; fakes capped at --cap (4) per group (--cap_unit group) or per source scan
    (--cap_unit twin), chosen by a greedy spread: least-used generator first, then least-used source scan,
    then sha1 tie-break.
  * RL: one row per (eligible fake, its twin real) when the twin is a real of the same RL group; fakes without
    a usable twin are not used in RL; singleton reals = RL-group reals that are no pair's twin in that regime,
    capped at floor(0.25 * n_pairs) (<= 20% of rows), sha1-ranked.
  * Evidence fields from pixels of the re-encoded images (gray = PIL "L"; fundus RGB -> luma):
      type      edit->implant, remove->removal
      location  prompts_v3.location_of(GT box), size prompts_v3.size_of(GT box)
      intensity implant : d = mean(fake_in) - mean(twin_in); d>+T brighter, d<-T darker, else the sign of
                          (mean(fake_in) - mean(fake_ring)).
                removal : flattened if |in-ring| contrast drops from twin to fake; else brighter/darker by the
                          sign of mean(fake_in) - mean(twin_in).
                no twin : fake-only (implant: sign(in-ring); removal: flattened if |in-ring| < T else sign),
                          fields_src="fake_only".
      ring = pixels at Chebyshev distance 8..16 px from the GT box (native pixels). T is chosen on TRAIN implants:
      the smallest grid value such that for every T' >= T, sign(d) agrees with the polarity of the strongest
      change (signed mean of the top-10% |fake-twin| in-box pixels) on >= --agree (0.95) of implants with
      |d| > T'. --T overrides.
Run    : NUMPY_MADVISE_HUGEPAGE=0 nice -n 19 ~/.conda/envs/aiface/bin/python build_sft_v3.py --workers 15
         (CPU only; --selftest_only re-runs the checks on existing outputs)
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import argparse, collections, hashlib, io, json, math, os, random, sys
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from v3.split_protocol import audit_directory, require_pass

V3 = _release_path('CIRCE/v3')
CIRCE = _release_path('CIRCE/circe')
MF = _release_path('AAAI/MedForge++/medforge90k_full/')
sys.path.insert(0, V3)
from prompts_v3 import SYS_V3, USER_V3, format_target, parse_v3, location_of, size_of  # noqa: E402

CM_HOLDOUT = {"gemini", "gpt", "stable-diffusion-3.5-medium", "stable-diffusion-xl-1.0-inpainting-0.1"}
REGIMES = {"id": (False, False), "cf": (True, False), "cm": (False, True), "cfcm": (True, True)}
TYPE_WORD = {"edit": "implant", "remove": "removal"}
RING = (8, 16)          # spec ring (px from the GT box edge); partially inside the paste-back feather (DIL 8, sigma 3)
RING_CLEAN = (18, 26)   # robustness ring entirely outside the feather (alpha < 1e-3); reported, not used for labels
FAR = (40, 64)          # twin sanity ring: paste-back + identical encoder => fake == twin here (expected MAD 0)
GRAY, Q = {"brain-mri", "chest-xray"}, 90   # must match build_mfclean.GRAY / Q (asserted at runtime)
NONE_FIELDS = {"type": "none", "location": "none", "size": "none", "intensity": "none"}
AUTO_BEGIN, AUTO_END = "<!-- AUTO-BEGIN (build_sft_v3.py) -->", "<!-- AUTO-END -->"


def h(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def pool_of(group):
    return "rl" if h("sftrl-v1:" + group) % 6 == 0 else "sft"


def eligible(r, regime):
    no_edit, no_cm = REGIMES[regime]
    if r["type"] == "real":
        return True
    if no_edit and r["type"] == "edit":
        return False
    if no_cm and r["gen"] in CM_HOLDOUT:
        return False
    return True


# ----------------------------------------------------------------------------------------------- pixel stats
def _gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def _px_box(box, W, H):
    x1, y1, x2, y2 = box
    if (W, H) != (1024, 1024):
        x1, x2 = int(round(x1 * W / 1024)), int(round(x2 * W / 1024))
        y1, y2 = int(round(y1 * H / 1024)), int(round(y2 * H / 1024))
    x1, y1 = max(0, min(W - 1, x1)), max(0, min(H - 1, y1))
    x2, y2 = max(x1, min(W - 1, x2)), max(y1, min(H - 1, y2))
    return x1, y1, x2, y2          # inclusive pixel indices (box_from_mask convention)


def _rect_sum(S, x1, y1, x2, y2):  # S = 2-D integral image with a zero row/col; inclusive rect
    return S[y2 + 1, x2 + 1] - S[y1, x2 + 1] - S[y2 + 1, x1] + S[y1, x1]


def _ring_mean(S, b, r1, r2, W, H):
    x1, y1, x2, y2 = b
    def ex(d):
        return max(0, x1 - d), max(0, y1 - d), min(W - 1, x2 + d), min(H - 1, y2 + d)
    o, i = ex(r2), ex(r1 - 1)
    n = (o[2] - o[0] + 1) * (o[3] - o[1] + 1) - (i[2] - i[0] + 1) * (i[3] - i[1] + 1)
    if n <= 0:
        return None
    return float((_rect_sum(S, *o) - _rect_sum(S, *i)) / n)


def _integral(a):
    S = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    S[1:, 1:] = a.astype(np.float64).cumsum(0).cumsum(1)
    return S


def reencode_like_mfclean(src, organ):
    """In-memory replica of build_mfclean.save_jpeg applied to an authentic source (decode RGB -> uint8 -> L for
    gray organs / RGB -> JPEG q90 4:4:4) -> decoded PIL image. Used to rebuild a twin MF-Clean did not materialise."""
    a = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    im = im.convert("L") if organ in GRAY else im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=Q, subsampling=0)
    buf.seek(0)
    return Image.open(buf)


def box_stats(job):
    """Raw statistics for one fake. Never raises.
    job = (image, twin_file_or_None, box1024, virtual_src_or_None, organ, check_twin_or_None)
      twin_file   : materialised twin used for the statistics (only when it IS the composite's source)
      virtual_src : authentic source re-encoded in memory with reencode_like_mfclean (twin missing or near-dup)
      check_twin  : manifest twin compared in the FAR ring only (-> far_mad_manifest; 0 == pixel-exact twin)"""
    image, twin, box, vsrc, organ, chk = job
    try:
        f = _gray(image)
        H, W = f.shape
        b = _px_box(box, W, H)
        x1, y1, x2, y2 = b
        Sf = _integral(f)
        out = {"image": image, "W": W, "H": H, "n_in": (x2 - x1 + 1) * (y2 - y1 + 1),
               "fin": float(f[y1:y2 + 1, x1:x2 + 1].mean()),
               "fring": _ring_mean(Sf, b, *RING, W, H), "fring_c": _ring_mean(Sf, b, *RING_CLEAN, W, H)}
        if twin or vsrc:
            t = (Image.open(twin) if twin else reencode_like_mfclean(vsrc, organ)).convert("L")
            out["twin_virtual"] = not twin
            out["twin_resized"] = t.size != (W, H)
            if out["twin_resized"]:
                t = t.resize((W, H), Image.BICUBIC)
            t = np.asarray(t, dtype=np.float32)
            St = _integral(t)
            d = f[y1:y2 + 1, x1:x2 + 1] - t[y1:y2 + 1, x1:x2 + 1]
            ad = np.abs(d).ravel()
            k = max(1, int(0.1 * ad.size))
            top = np.argpartition(ad, -k)[-k:]
            Sd = _integral(np.abs(f - t))
            out.update({"tin": float(t[y1:y2 + 1, x1:x2 + 1].mean()),
                        "tring": _ring_mean(St, b, *RING, W, H), "tring_c": _ring_mean(St, b, *RING_CLEAN, W, H),
                        "pol": float(d.ravel()[top].mean()), "mad_in": float(ad.mean()),
                        "far_mad": _ring_mean(Sd, b, *FAR, W, H)})
        if chk:
            c = Image.open(chk).convert("L")
            if c.size != (W, H):
                c = c.resize((W, H), Image.BICUBIC)
            out["far_mad_manifest"] = _ring_mean(_integral(np.abs(f - np.asarray(c, dtype=np.float32))), b, *FAR, W, H)
        elif twin:
            out["far_mad_manifest"] = out["far_mad"]
        out["ok"] = True
        return out
    except Exception as e:  # noqa
        return {"image": image, "ok": False, "why": repr(e)[:200]}


def replica_check(job):
    """(materialised_twin, twin_original, fake_mapping_source, organ) -> (twin_orig_reproduces, fake_src_reproduces)."""
    twin, torig, fsrc, organ = job
    a = np.asarray(Image.open(twin).convert("L"))
    out = []
    for src in (torig, fsrc):
        if not src:
            out.append(False)
            continue
        b = np.asarray(reencode_like_mfclean(src, organ).convert("L"))
        out.append(a.shape == b.shape and bool((a == b).all()))
    return tuple(out)


def intensity_of(st, typ, T, clean_ring=False):
    """-> (label, rule). rule in {mean, ring, contrast, fake_only}."""
    fr = st["fring_c"] if clean_ring else st["fring"]
    if "tin" not in st:                                        # no twin: fake-only statistics
        c = st["fin"] - fr
        if typ == "edit":
            return ("brighter" if c > 0 else "darker"), "fake_only"
        return ("flattened" if abs(c) < T else ("brighter" if c > 0 else "darker")), "fake_only"
    tr = st["tring_c"] if clean_ring else st["tring"]
    dm = st["fin"] - st["tin"]
    if typ == "edit":
        if dm > T:
            return "brighter", "mean"
        if dm < -T:
            return "darker", "mean"
        return ("brighter" if st["fin"] - fr > 0 else "darker"), "ring"
    if abs(st["fin"] - fr) < abs(st["tin"] - tr):
        return "flattened", "contrast"
    return ("brighter" if dm > 0 else "darker"), "mean"


def choose_T(stats_edit, agree, grid=None, min_n=200):
    """Smallest T s.t. for all T' >= T (with >= min_n implants above T'), sign(d) == sign(pol) on >= agree."""
    grid = grid if grid is not None else [round(0.25 * i, 2) for i in range(0, 121)]
    d = np.array([s["fin"] - s["tin"] for s in stats_edit])
    p = np.array([s["pol"] for s in stats_edit])
    curve = []
    for T in grid:
        m = np.abs(d) > T
        n = int(m.sum())
        a = float((np.sign(d[m]) == np.sign(p[m])).mean()) if n else float("nan")
        curve.append((T, n, a))
    valid = [c for c in curve if c[1] >= min_n]
    chosen = None
    for i, (T, n, a) in enumerate(valid):
        if all(c[2] >= agree for c in valid[i:]):
            chosen = T
            break
    if chosen is None:  # never stable at the target: take the most-agreeing T (documented in the report)
        chosen = max(valid, key=lambda c: c[2])[0] if valid else 2.0
    return chosen, curve, d, p


# ---------------------------------------------------------------------------------------------- selection
def spread_rank(fakes, salt):
    """Greedy order: least-used generator, then least-used (type, disease), then least-used source scan (twin),
    then sha1 -> the first k picks of a group cover distinct generators / lesion types / scans where possible."""
    left = sorted(fakes, key=lambda r: h(salt + r["image"]))
    ug, ud, ut, out = collections.Counter(), collections.Counter(), collections.Counter(), []
    while left:
        j = min(range(len(left)), key=lambda i: (ug[left[i]["gen"]], ud[(left[i]["type"], left[i].get("disease"))],
                                                 ut[left[i]["twin"] or left[i]["image"]], i))
        r = left.pop(j)
        ug[r["gen"]] += 1
        ud[(r["type"], r.get("disease"))] += 1
        ut[r["twin"] or r["image"]] += 1
        out.append(r)
    return out


def capped(fakes, cap, cap_edit, salt):
    """Spread-ranked fakes of one unit (group or twin), at most `cap` removals and `cap_edit` implants, and at
    most max(cap, cap_edit) in total."""
    out, n = [], collections.Counter()
    for r in spread_rank(fakes, salt):
        lim = cap_edit if r["type"] == "edit" else cap
        if n[r["type"]] < lim and len(out) < max(cap, cap_edit):
            out.append(r)
            n[r["type"]] += 1
    return out


def sft_record(r):
    if r["type"] == "real":
        tgt = format_target("real")
    else:
        tgt = format_target("deepfake", r["box"], r["fields"]["type"], r["fields"]["intensity"])
    return {"images": [r["image"]],
            "messages": [{"role": "system", "content": SYS_V3},
                         {"role": "user", "content": "<image>" + USER_V3},
                         {"role": "assistant", "content": tgt}]}


def fake_payload(r):
    return {"image": r["image"], "box": r["box"], "type": r["type"], "gen": r["gen"], "organ": r["organ"],
            "disease": r.get("disease", ""), "fields": r["fields"], "fields_src": r["fields_src"]}


# ------------------------------------------------------------------------------------------------- io utils
def read_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(p, rows):
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, p)


def md5_file(p):
    m = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            m.update(chunk)
    return m.hexdigest()


def recover_box(r, mapping):
    """Fallback for fakes whose manifest box is null (build_mfclean.work returns row.get('box_px') = None when an
    image already existed on disk). Same arithmetic as build_mfclean: mask bbox, rescale to image, then 1024 frame."""
    rel = r.get("orig", "").split("/medforge90k_full/", 1)[-1]
    it = mapping.get(rel)
    if not it:
        return None
    m = np.asarray(Image.open(MF + it["mask_path"]).convert("L")) > 127
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    W, H = Image.open(r["image"]).size
    if m.shape != (H, W):
        sy, sx = H / m.shape[0], W / m.shape[1]
        box = [int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)]
    return [int(box[0] * 1024 / W), int(box[1] * 1024 / H), int(box[2] * 1024 / W), int(box[3] * 1024 / H)]


# --------------------------------------------------------------------------------------------- reporting
def table(counter_rows, row_key, col_key, rows=None, cols=None):
    """counter over (row,col) -> markdown table with totals."""
    rows = rows or sorted({k[0] for k in counter_rows})
    cols = cols or sorted({k[1] for k in counter_rows})
    out = ["| " + row_key + " \\ " + col_key + " | " + " | ".join(cols) + " | total |",
           "|---" * (len(cols) + 2) + "|"]
    for r in rows:
        vals = [counter_rows.get((r, c), 0) for c in cols]
        out.append(f"| {r} | " + " | ".join(str(v) for v in vals) + f" | {sum(vals)} |")
    tot = [sum(counter_rows.get((r, c), 0) for r in rows) for c in cols]
    out.append("| **total** | " + " | ".join(str(v) for v in tot) + f" | {sum(tot)} |")
    return "\n".join(out)


GEN_SHORT = {"flux.1-dev": "flux", "gemini": "gemini", "gpt": "gpt", "qwen": "qwen", "seeddream": "seedream",
             "stable-diffusion-3.5-large": "sd35L", "stable-diffusion-3.5-medium": "sd35M*",
             "stable-diffusion-inpainting": "sdinp", "stable-diffusion-xl-1.0-inpainting-0.1": "sdxl*",
             "step1x-edit": "step1x"}


def gs(g):
    return GEN_SHORT.get(g, g) if g not in ("gemini", "gpt") else g + "*"


# ------------------------------------------------------------------------------------------------ selftest
def selftest(out, mfclean, log):
    """Re-read every output file and assert the invariants. Returns list of (check, result) lines."""
    sys.path.insert(0, CIRCE)
    from circe_reward import parse_output
    res = []

    def ok(name, cond, detail=""):
        res.append((name, "PASS" if cond else "FAIL", detail))
        log(f"[selftest] {'PASS' if cond else 'FAIL'}  {name}  {detail}")
        return cond

    train = read_jsonl(f"{mfclean}/train_all.jsonl")
    # Validate before indexing by image (a dict would silently hide duplicates).
    audit_directory(mfclean)
    by_img = {r["image"]: r for r in train}
    for name in ("select_core", "test_core"):
        for r in read_jsonl(f"{mfclean}/{name}.jsonl"):
            by_img[r["image"]] = r
    ev_int = {e["image"]: e["intensity"] for e in read_jsonl(f"{out}/evidence_stats.jsonl")}
    sets_img, sets_grp = collections.defaultdict(set), collections.defaultdict(set)
    n_targets = n_bad = 0
    bad_examples = []

    def check_target(tgt, verdict, box, fields):
        nonlocal n_targets, n_bad
        n_targets += 1
        a, b = parse_v3(tgt), parse_output(tgt)
        good = (a["well_formed"] and a["verdict"] == verdict and a["box"] == box and a["fields"] == fields
                and b["verdict"] == verdict and b["box"] == box and b.get("format") == "v2")
        if not good:
            n_bad += 1
            if len(bad_examples) < 3:
                bad_examples.append((tgt[:120], verdict, box, fields, a, b))

    for reg in REGIMES:
        sft = read_jsonl(f"{out}/sft_{reg}.jsonl")
        schema_ok = True
        n_edit = n_cmgen = 0
        for x in sft:
            m = x["messages"]
            schema_ok &= (len(m) == 3 and [k["role"] for k in m] == ["system", "user", "assistant"]
                          and m[0]["content"] == SYS_V3 and m[1]["content"] == "<image>" + USER_V3
                          and len(x["images"]) == 1 and isinstance(x["images"][0], str))
            img = x["images"][0]
            r = by_img[img]
            sets_img["sft"].add(img)
            sets_grp["sft"].add(r["group"])
            if r["type"] == "real":
                check_target(m[2]["content"], "real", None, NONE_FIELDS)
            else:
                b = [int(v) for v in r["box"]]
                fl = parse_v3(m[2]["content"])["fields"]
                exp = {"type": TYPE_WORD[r["type"]], "location": location_of(b), "size": size_of(b),
                       "intensity": ev_int.get(img)}
                check_target(m[2]["content"], "deepfake", b, exp)
                if fl["intensity"] not in ("brighter", "darker", "flattened"):
                    schema_ok = False
            n_edit += r["type"] == "edit"
            n_cmgen += r["gen"] in CM_HOLDOUT
        ok(f"sft_{reg}: 3-message schema, SYS_V3/USER_V3, 1 image", schema_ok, f"n={len(sft)}")
        if reg in ("cf", "cfcm"):
            ok(f"sft_{reg}: zero edit fakes", n_edit == 0, f"edit={n_edit}")
        if reg in ("cm", "cfcm"):
            ok(f"sft_{reg}: zero CM-held-out generators", n_cmgen == 0, f"heldout={n_cmgen}")
        ok(f"sft_{reg}: only SFT-pool groups of the train split",
           all(by_img[x["images"][0]]["split"] == "train" and pool_of(by_img[x["images"][0]]["group"]) == "sft"
               for x in sft))
        rl = read_jsonl(f"{out}/rl_{reg}.jsonl")
        n_edit = n_cmgen = n_single = 0
        pair_ok = True
        for x in rl:
            rr = by_img[x["real"]["image"]]
            pair_ok &= rr["type"] == "real" and rr["group"] == x["group"] and rr["split"] == "train"\
                and pool_of(x["group"]) == "rl"
            sets_img["rl"].add(x["real"]["image"])
            sets_grp["rl"].add(x["group"])
            if x["fake"] is None:
                n_single += 1
                continue
            fr = by_img[x["fake"]["image"]]
            pair_ok &= (fr["group"] == x["group"] and fr["twin"] == x["real"]["image"]
                        and fr["type"] == x["fake"]["type"] and fr["box"] == x["fake"]["box"])
            sets_img["rl"].add(x["fake"]["image"])
            n_edit += fr["type"] == "edit"
            n_cmgen += fr["gen"] in CM_HOLDOUT
            b = [int(v) for v in fr["box"]]
            exp = {"type": TYPE_WORD[fr["type"]], "location": location_of(b), "size": size_of(b),
                   "intensity": ev_int.get(fr["image"])}
            pair_ok &= x["fake"]["fields"] == exp
            check_target(format_target("deepfake", b, x["fake"]["fields"]["type"], x["fake"]["fields"]["intensity"]),
                         "deepfake", b, exp)
        npair = len(rl) - n_single
        ok(f"rl_{reg}: pairs = (fake, its twin) in one RL group; singles are RL-group reals", pair_ok,
           f"pairs={npair} singles={n_single}")
        ok(f"rl_{reg}: singleton share <= 20%", n_single <= 0.2 * max(1, len(rl)) + 1e-9,
           f"{n_single}/{len(rl)}")
        if reg in ("cf", "cfcm"):
            ok(f"rl_{reg}: zero implant (edit) fakes", n_edit == 0, f"edit={n_edit}")
        if reg in ("cm", "cfcm"):
            ok(f"rl_{reg}: zero CM-held-out generators", n_cmgen == 0, f"heldout={n_cmgen}")
    for name in ("select_core", "test_core"):
        rows = read_jsonl(f"{out}/{name}_fields.jsonl")
        base = read_jsonl(f"{mfclean}/{name}.jsonl")
        same = len(rows) == len(base) and all(
            {k: v for k, v in a.items() if k not in ("fields", "fields_src", "target")} == b
            for a, b in zip(rows, base))
        ok(f"{name}_fields: same rows as {name}.jsonl (+fields,fields_src,target)", same, f"n={len(rows)}")
        key = "select" if name.startswith("select") else "test"
        for r in rows:
            sets_img[key].add(r["image"])
            sets_grp[key].add(r["group"])
            if r["type"] == "real":
                check_target(r["target"], "real", None, NONE_FIELDS)
            else:
                b = [int(v) for v in r["box"]]
                check_target(r["target"], "deepfake", b, {"type": TYPE_WORD[r["type"]], "location": location_of(b),
                                                          "size": size_of(b), "intensity": ev_int.get(r["image"])})
    ok("every target parses with prompts_v3.parse_v3 AND circe_reward.parse_output to the GT verdict/box and "
       "fields == (GT type, location_of, size_of, intensity in evidence_stats.jsonl)",
       n_bad == 0, f"targets={n_targets} bad={n_bad} {bad_examples[:1]}")
    names = ["sft", "rl", "select", "test"]
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = names[i], names[j]
            ok(f"image paths disjoint: {a} vs {b}", not (sets_img[a] & sets_img[b]),
               f"overlap={len(sets_img[a] & sets_img[b])}")
            ok(f"groups disjoint: {a} vs {b}", not (sets_grp[a] & sets_grp[b]),
               f"overlap={len(sets_grp[a] & sets_grp[b])}")
    allimg = set().union(*sets_img.values())
    missing = [p for p in allimg if not (os.path.exists(p) and os.path.getsize(p) > 0)]
    ok("every referenced image exists and is non-empty", not missing, f"n={len(allimg)} missing={len(missing)}")
    return res


# ------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfclean", default=_release_path('AAAI/MedForge++/mfclean'))
    ap.add_argument("--out", default=None, help="default <mfclean>/v3data")
    ap.add_argument("--report", default=f"{V3}/reports/data_v3.md")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--cap_unit", choices=["group", "twin"], default="group")
    ap.add_argument("--cap_edit", type=int, default=None, help="separate cap for implant fakes (default = --cap)")
    ap.add_argument("--single_frac", type=float, default=0.25, help="max singles per pair (0.25 -> <=20%% rows)")
    ap.add_argument("--agree", type=float, default=0.95)
    ap.add_argument("--T", type=float, default=None, help="override the data-driven implant threshold")
    ap.add_argument("--selftest_only", action="store_true")
    ap.add_argument("--no_virtual_twin", action="store_true",
                    help="do not rebuild missing twins from the authentic source; use fake-only statistics instead")
    ap.add_argument("--replica_n", type=int, default=300, help="train twins used for the re-encode replica check")
    args = ap.parse_args()
    assert args.workers <= 15, "cluster rule: at most 16 processes (15 workers + main)"
    try:
        os.nice(19)
    except OSError:
        pass
    out = args.out or f"{args.mfclean}/v3data"
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    import build_mfclean as BM   # read-only import: constants must agree with the builder that made the data
    assert BM.GRAY == GRAY and BM.Q == Q and BM.CM_HOLDOUT == CM_HOLDOUT, "constants drifted from build_mfclean.py"
    if args.selftest_only:
        require_pass(selftest(out, args.mfclean, log))
        return

    # Stop before pixel processing or output writes if any declared split leaks.
    split_audit = audit_directory(args.mfclean)
    log(f"[split] PASS: {split_audit['audit']['counts']}")
    train = read_jsonl(f"{args.mfclean}/train_all.jsonl")
    evals = {n: read_jsonl(f"{args.mfclean}/{n}.jsonl") for n in ("select_core", "test_core")}
    stats_json = json.load(open(f"{args.mfclean}/stats.json")) if os.path.exists(f"{args.mfclean}/stats.json") else {}
    log(f"[in] train_all={len(train)} select_core={len(evals['select_core'])} test_core={len(evals['test_core'])}")
    assert all(r["split"] == "train" for r in train), "train_all.jsonl must be split=train only"

    # ---- box sanity / recovery
    allrows = train + evals["select_core"] + evals["test_core"]
    nobox = [r for r in allrows if r["type"] != "real" and not r.get("box")]
    n_recovered = n_unrecoverable = 0
    mp = {it["image_path"]: it for it in json.load(open(MF + "dataset_mapping.json"))["items"]}
    if nobox:
        for r in nobox:
            b = recover_box(r, mp)
            if b:
                r["box"], r["box_recovered"] = b, True
                n_recovered += 1
            else:
                n_unrecoverable += 1
    log(f"[box] fakes with null box: {len(nobox)} (recovered from mask {n_recovered}, unrecoverable "
        f"{n_unrecoverable})")
    bad_geom = [r for r in allrows if r["type"] != "real" and r.get("box") and not
                (0 <= r["box"][0] <= r["box"][2] <= 1023 and 0 <= r["box"][1] <= r["box"][3] <= 1023)]
    log(f"[box] fakes with box outside [0,1023] or inverted: {len(bad_geom)}")

    # ---- pixel statistics for every fake (train + eval)
    fk = [r for r in allrows if r["type"] != "real" and r.get("box")]

    def src_of(r):  # authentic source the composite was built from (dataset_mapping real_image_path)
        it = mp.get(r.get("orig", "").split("/medforge90k_full/", 1)[-1])
        return MF + it["real_image_path"] if it and os.path.exists(MF + it["real_image_path"]) else None

    row_of = {x["image"]: x for x in allrows}

    def job_of(r):
        tw, src = r.get("twin"), src_of(r)
        tw_is_src = bool(tw and src and row_of.get(tw, {}).get("orig") == src)
        if tw_is_src or (tw and (args.no_virtual_twin or not src)):
            return (r["image"], tw, r["box"], None, r["organ"], None)          # materialised twin
        if src and not args.no_virtual_twin:
            return (r["image"], None, r["box"], src, r["organ"], tw)           # true source, re-encoded in memory
        return (r["image"], None, r["box"], None, r["organ"], None)            # fake-only

    jobs = [job_of(r) for r in fk]
    log(f"[stats] fakes: {len(fk)}; manifest twin null: {sum(1 for r in fk if not r.get('twin'))}; manifest twin is "
        f"not the composite's source (dup-cluster representative): "
        f"{sum(1 for r in fk if r.get('twin') and row_of.get(r['twin'], {}).get('orig') != src_of(r))}; "
        f"statistics vs materialised twin {sum(1 for j in jobs if j[1])}, vs re-encoded true source "
        f"{sum(1 for j in jobs if j[3])}, fake-only {sum(1 for j in jobs if not j[1] and not j[3])}")
    # replica check: the in-memory re-encode must reproduce MF-Clean's materialised twins bit-exactly
    tw_rows = sorted((r for r in fk if r.get("twin") and r["split"] == "train"), key=lambda r: h("rep-v1:" + r["image"]))
    rep_jobs = [(r["twin"], row_of.get(r["twin"], {}).get("orig"), src_of(r), r["organ"])
                for r in tw_rows[:args.replica_n]]
    with ProcessPoolExecutor(args.workers) as ex:
        rep = list(ex.map(replica_check, rep_jobs, chunksize=4))
    rep_twin = sum(1 for x in rep if x[0]); rep_src = sum(1 for x in rep if x[1])
    log(f"[replica] re-encode(twin's own original) == materialised twin: {rep_twin}/{len(rep)}; "
        f"re-encode(fake's mapping source) == materialised twin: {rep_src}/{len(rep)}")
    log(f"[stats] computing pixel statistics for {len(jobs)} fakes on {args.workers} CPU workers ...")
    with ProcessPoolExecutor(args.workers) as ex:
        st = list(ex.map(box_stats, jobs, chunksize=32))
    fails = [s for s in st if not s["ok"]]
    log(f"[stats] failures: {len(fails)} {collections.Counter(s['why'][:60] for s in fails).most_common(3)}")
    S = {s["image"]: s for s in st if s["ok"]}

    # ---- choose T on TRAIN implants with twin
    tr_img = {r["image"] for r in train}
    edit_tw = [S[r["image"]] for r in fk if r["type"] == "edit" and r["image"] in tr_img and r["image"] in S
               and "tin" in S[r["image"]]]
    if args.T is None:
        T, curve, dvec, pvec = choose_T(edit_tw, args.agree)
        T_src = f"data (agree>={args.agree})"
    else:
        T, curve, dvec, pvec = args.T, choose_T(edit_tw, args.agree)[1], None, None
        T_src = "--T override"
    log(f"[T] implants used to choose T: {len(edit_tw)}; T = {T} ({T_src})")

    # ---- assign fields
    rule_ct = collections.Counter()
    agree_ct = collections.Counter()
    for r in allrows:
        if r["type"] == "real":
            r["fields"], r["fields_src"] = dict(NONE_FIELDS), "real"
            continue
        s = S.get(r["image"])
        if s is None:
            r["fields"] = None
            continue
        b = [int(v) for v in r["box"]]
        lab, rule = intensity_of(s, r["type"], T)
        lab_c, _ = intensity_of(s, r["type"], T, clean_ring=True)
        r["fields"] = {"type": TYPE_WORD[r["type"]], "location": location_of(b), "size": size_of(b),
                       "intensity": lab}
        r["fields_src"] = ("source_reencoded" if s.get("twin_virtual") else "twin") if "tin" in s else "fake_only"
        r["twin_exact"] = s.get("far_mad_manifest") == 0 if r.get("twin") else None
        sp = r["split"]
        rule_ct[(sp, r["type"], rule)] += 1
        agree_ct[(r["type"], lab == lab_c)] += 1
        if r["type"] == "edit" and "pol" in s:
            agree_ct[("edit_vs_pol", rule, lab == ("brighter" if s["pol"] > 0 else "darker"))] += 1
    usable = lambda r: r["type"] == "real" or r.get("fields") is not None  # noqa: E731
    write_jsonl(f"{out}/evidence_stats.jsonl",
                [dict(S[r["image"]], split=r["split"], type=r["type"], gen=r["gen"], organ=r["organ"],
                      group=r["group"], disease=r.get("disease"), intensity=r["fields"]["intensity"],
                      fields_src=r["fields_src"], twin_exact=r.get("twin_exact"))
                 for r in fk if r["image"] in S])

    # ---- eval files with fields
    for n, rows in evals.items():
        o = []
        for r in rows:
            if not usable(r):
                continue
            x = {k: v for k, v in r.items() if k not in ("fields", "fields_src", "box_recovered", "twin_exact")}
            x["fields"], x["fields_src"] = r["fields"], r["fields_src"]
            x["target"] = (format_target("real") if r["type"] == "real" else
                           format_target("deepfake", r["box"], r["fields"]["type"], r["fields"]["intensity"]))
            o.append(x)
        write_jsonl(f"{out}/{n}_fields.jsonl", o)
        log(f"[eval] {n}_fields.jsonl: {len(o)} rows (dropped {len(rows) - len(o)} without fields)")

    # ---- SFT / RL pools
    pool = {g: pool_of(g) for g in {r["group"] for r in train}}
    by_group = collections.defaultdict(list)
    for r in train:
        if usable(r):
            by_group[r["group"]].append(r)
    img2row = {r["image"]: r for r in train}
    info = {"T": T, "T_source": T_src, "cap": args.cap, "cap_unit": args.cap_unit, "files": {},
            "groups": dict(collections.Counter(pool.values())),
            "train_rows_by_pool": {f"{p}/{t}": n for (p, t), n in sorted(
                collections.Counter((pool[r["group"]], r["type"]) for r in train).items())}}
    counts = {}
    cap_edit = args.cap if args.cap_edit is None else args.cap_edit
    for reg in REGIMES:
        sft_rows, rl_rows = [], []
        for g in sorted(by_group):
            mem = by_group[g]
            reals = [r for r in mem if r["type"] == "real"]
            fakes = [r for r in mem if r["type"] != "real" and eligible(r, reg)]
            if pool[g] == "sft":
                sft_rows += reals
                if args.cap_unit == "group":
                    sft_rows += capped(fakes, args.cap, cap_edit, "pick-v1:")
                else:
                    byt = collections.defaultdict(list)
                    for r in fakes:
                        byt[r["twin"] or r["image"]].append(r)
                    for t in sorted(byt):
                        sft_rows += capped(byt[t], args.cap, cap_edit, "pick-v1:")
            else:
                byt = collections.defaultdict(list)
                for r in fakes:
                    tw = img2row.get(r["twin"]) if r.get("twin") else None
                    if tw is not None and tw["type"] == "real" and tw["group"] == g and usable(tw):
                        byt[r["twin"]].append(r)
                twins_used = set()
                for t in sorted(byt):
                    for k, r in enumerate(spread_rank(byt[t], "pair-v1:")):
                        rl_rows.append({"kind": "pair", "fake": fake_payload(r),
                                        "real": {"image": t, "organ": img2row[t]["organ"]},
                                        "group": g, "src_rank": k, "twin_exact": r.get("twin_exact")})
                        twins_used.add(t)
                for r in reals:
                    if r["image"] not in twins_used:
                        rl_rows.append({"kind": "single", "fake": None,
                                        "real": {"image": r["image"], "organ": r["organ"]}, "group": g})
        pairs = [x for x in rl_rows if x["kind"] == "pair"]
        singles = sorted((x for x in rl_rows if x["kind"] == "single"), key=lambda x: h("single-v1:" + x["real"]["image"]))
        n_single_cands = len(singles)
        singles = singles[:int(math.floor(args.single_frac * len(pairs)))]
        rl_rows = pairs + singles
        rl_rows.sort(key=lambda x: h("rlorder-v1:" + (x["fake"]["image"] if x["fake"] else x["real"]["image"])))
        rl_unpaired = sum(1 for g in by_group if pool[g] == "rl" for r in by_group[g]
                          if r["type"] != "real" and eligible(r, reg)) - len(pairs)
        sft_rows.sort(key=lambda r: h("sftorder-v1:" + r["image"]))
        write_jsonl(f"{out}/sft_{reg}.jsonl", [sft_record(r) for r in sft_rows])
        write_jsonl(f"{out}/rl_{reg}.jsonl", rl_rows)
        counts[reg] = {"sft": sft_rows, "rl_pairs": pairs, "rl_singles": singles,
                       "rl_single_cands": n_single_cands, "rl_unpaired_fakes": rl_unpaired}
        log(f"[{reg}] sft={len(sft_rows)} (real {sum(r['type'] == 'real' for r in sft_rows)}, fake "
            f"{sum(r['type'] != 'real' for r in sft_rows)}) | rl pairs={len(pairs)} singles={len(singles)} "
            f"(of {n_single_cands} candidates) unpaired fakes dropped={rl_unpaired}")

    # ---- self-tests
    st_res = selftest(out, args.mfclean, log)
    for fn in sorted(os.listdir(out)):
        if fn.endswith(".jsonl"):
            info["files"][fn] = {"md5": md5_file(f"{out}/{fn}"),
                                 "rows": sum(1 for _ in open(f"{out}/{fn}"))}
    info["selftest"] = {f"{n}": r for n, r, _ in st_res}
    json.dump(info, open(f"{out}/build_info.json", "w"), indent=1)

    # ---- report
    R = []
    R.append(f"Generated by `build_sft_v3.py` from `{args.mfclean}` -> `{out}`. "
             f"MF-Clean stats.json present: {bool(stats_json)}" + (" (SYNTHETIC dev sample)" if stats_json.get("synthetic") else ""))
    R.append("")
    R.append("### Inputs")
    R.append(f"- train_all {len(train)} rows, select_core {len(evals['select_core'])}, test_core {len(evals['test_core'])}.")
    R.append(f"- fakes with null box: {len(nobox)} (recovered from the mask {n_recovered}, unrecoverable "
             f"{n_unrecoverable}); boxes outside [0,1023]: {len(bad_geom)}; pixel-stat failures: {len(fails)}.")
    tw_missing = collections.Counter((r["split"], r["organ"], r["type"]) for r in fk if not r.get("twin"))
    R.append(f"- fakes whose manifest `twin` is null: {sum(tw_missing.values())} "
             f"{ {'/'.join(k): v for k, v in sorted(tw_missing.items())} if tw_missing else ''}.")
    R.append(f"- evidence source per fake (`fields_src`): "
             f"{dict(collections.Counter((r['split'], r['fields_src']) for r in fk if r.get('fields')))}. "
             f"`twin_reencoded` = twin rebuilt in memory from the authentic source with build_mfclean's encoder.")
    R.append(f"- replica check on {len(rep)} train twins: re-encoding the twin's own original reproduces the "
             f"materialised twin bit-exactly in {rep_twin}/{len(rep)}; re-encoding the fake's dataset_mapping source "
             f"reproduces it in {rep_src}/{len(rep)}.")
    far = np.array([s["far_mad"] for s in S.values() if s.get("far_mad") is not None])
    if far.size:
        R.append(f"- twin integrity (reference actually used for the statistics): mean |fake - reference| in the "
                 f"{FAR[0]}-{FAR[1]} px ring around the box (paste-back + same encoder => expected exactly 0): "
                 f"share exactly 0 = {(far == 0).mean():.4f}, max {far.max():.4f} (n={far.size}).")
    fm = np.array([s["far_mad_manifest"] for s in S.values() if s.get("far_mad_manifest") is not None])
    if fm.size:
        nz = fm[fm > 0]
        R.append(f"- manifest `twin` vs composite background in the same ring: exactly 0 for {(fm == 0).sum()}/{fm.size}; "
                 f"non-zero for {nz.size} (`twin_exact=false`: the manifest twin is a verified near-duplicate scan, "
                 f"not the composite's own source; their mean |d| p50 "
                 f"{(np.median(nz) if nz.size else 0):.2f}, p99 {(np.percentile(nz, 99) if nz.size else 0):.2f}). "
                 f"By split/organ: {dict(collections.Counter((r['split'], r['organ']) for r in fk if r.get('twin_exact') is False))}.")
    R.append("")
    R.append("### SFT / RL group split (train split only)")
    R.append(f"- train groups: {dict(collections.Counter(pool.values()))} (RL share "
             f"{sum(v == 'rl' for v in pool.values()) / max(1, len(pool)):.4f}; target 1/6 = 0.1667).")
    R.append(f"- train rows by pool/type: {info['train_rows_by_pool']}")
    R.append("")
    R.append("### Intensity threshold T (implants, TRAIN split, twin available)")
    if dvec is not None and len(dvec):
        q = np.percentile(dvec, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        aq = np.percentile(np.abs(dvec), [10, 25, 50, 75, 90])
        R.append(f"- n = {len(dvec)} implants. d = mean(fake_in) - mean(twin_in), grey levels: "
                 + ", ".join(f"p{p}={v:.2f}" for p, v in zip([1, 5, 10, 25, 50, 75, 90, 95, 99], q)))
        R.append(f"- |d|: " + ", ".join(f"p{p}={v:.2f}" for p, v in zip([10, 25, 50, 75, 90], aq))
                 + f"; share d>0: {(dvec > 0).mean():.3f}.")
        R.append(f"- reference polarity = sign of the mean of the top-10% |fake-twin| in-box pixels; "
                 f"overall sign(d)==sign(pol): {(np.sign(dvec) == np.sign(pvec)).mean():.4f}.")
    R.append("")
    R.append("| T | implants with abs(d) > T | share | sign(d) agrees with polarity |")
    R.append("|---|---|---|---|")
    ntot = max(1, len(edit_tw))
    for T_, n_, a_ in curve:
        if T_ in (0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30) or T_ == T:
            R.append(f"| {T_}{' **(chosen)**' if T_ == T else ''} | {n_} | {n_ / ntot:.3f} | {a_:.4f} |")
    R.append("")
    R.append(f"**T = {T}** ({T_src}).")
    R.append("")
    R.append("Rule usage by split / type (mean = in-box mean change vs twin beyond T, ring = fallback sign of in-box minus "
             "8-16 px ring on the fake, contrast = removal contrast drop -> flattened, fake_only = no twin):")
    R.append("")
    R.append(table({(f"{sp}/{t}", ru): n for (sp, t, ru), n in rule_ct.items()}, "split/type", "rule"))
    R.append("")
    ev_pol = {k[1:]: v for k, v in agree_ct.items() if k[0] == "edit_vs_pol"}
    if ev_pol:
        R.append("Implant label vs reference polarity (agreement), by rule: " + ", ".join(
            f"{ru}: {ev_pol.get((ru, True), 0)}/{ev_pol.get((ru, True), 0) + ev_pol.get((ru, False), 0)}"
            f" = {ev_pol.get((ru, True), 0) / max(1, ev_pol.get((ru, True), 0) + ev_pol.get((ru, False), 0)):.3f}"
            for ru in ("mean", "ring")))
    for t in ("edit", "remove"):
        a, b = agree_ct.get((t, True), 0), agree_ct.get((t, False), 0)
        R.append(f"- ring robustness ({t}): label with the {RING[0]}-{RING[1]} px ring == label with the clean "
                 f"{RING_CLEAN[0]}-{RING_CLEAN[1]} px ring (outside the paste-back feather) for {a}/{a + b} = "
                 f"{a / max(1, a + b):.4f}.")
    rem = [S[r["image"]] for r in fk if r["type"] == "remove" and r["image"] in S and "tin" in S[r["image"]]
           and r["split"] == "train"]
    if rem:
        drop = np.array([abs(s["tin"] - s["tring"]) - abs(s["fin"] - s["fring"]) for s in rem])
        R.append(f"- removals (train, twin): contrast drop |twin_in-ring| - |fake_in-ring| percentiles "
                 + ", ".join(f"p{p}={v:.2f}" for p, v in zip([10, 25, 50, 75, 90], np.percentile(drop, [10, 25, 50, 75, 90])))
                 + f"; share >0 (flattened): {(drop > 0).mean():.3f}; share > T: {(drop > T).mean():.3f}.")
    R.append("")
    R.append("Intensity label by generator (TRAIN fakes):")
    R.append("")
    R.append(table(collections.Counter((f"{r['type']}/{gs(r['gen'])}", r["fields"]["intensity"])
                                       for r in fk if r.get("fields") and r["split"] == "train"),
                   "type/generator", "intensity"))
    R.append("")
    R.append("### Evidence-field distributions (all fakes with fields, by split)")
    for fld in ("intensity", "size", "location"):
        R.append("")
        R.append(f"**{fld}** (organ/type x value)")
        R.append("")
        R.append(table(collections.Counter((f"{r['split']}/{r['organ']}/{r['type']}", r["fields"][fld])
                                           for r in fk if r.get("fields")), "split/organ/type", fld))
    R.append("")
    R.append("### Output files: class x organ")
    for reg in REGIMES:
        c = counts[reg]
        R.append("")
        R.append(f"**sft_{reg}.jsonl** ({len(c['sft'])} records)")
        R.append("")
        R.append(table(collections.Counter((r["organ"], r["type"]) for r in c["sft"]), "organ", "class",
                       cols=["real", "edit", "remove"]))
        R.append("")
        R.append(f"**rl_{reg}.jsonl**: {len(c['rl_pairs'])} pairs + {len(c['rl_singles'])} singleton reals "
                 f"(of {c['rl_single_cands']} candidates); eligible RL fakes without a usable twin, dropped: "
                 f"{c['rl_unpaired_fakes']}. Pairs per twin real: "
                 + (", ".join(f"{k}:{v}" for k, v in sorted(collections.Counter(collections.Counter(
                     x["real"]["image"] for x in c["rl_pairs"]).values()).items())) or "-") + " (pairs:twins).")
        R.append("")
        R.append(table(collections.Counter((x["fake"]["organ"] if x["fake"] else x["real"]["organ"],
                                            x["fake"]["type"] if x["fake"] else "single-real")
                                           for x in c["rl_pairs"] + c["rl_singles"]), "organ", "row kind",
                       cols=["edit", "remove", "single-real"]))
    R.append("")
    R.append("Non-exact twin pairs (`twin_exact=false`, manifest twin is a near-duplicate scan) per RL file: " + ", ".join(
        f"rl_{reg} {sum(1 for x in counts[reg]['rl_pairs'] if x['twin_exact'] is False)}/{len(counts[reg]['rl_pairs'])}"
        for reg in REGIMES) + ".")
    sft_g = [g for g in by_group if pool[g] == "sft"]
    eg = [g for g in sft_g if any(r["type"] == "edit" for r in by_group[g])]
    rg = [g for g in sft_g if any(r["type"] == "remove" for r in by_group[g])]
    R.append("")
    R.append(f"SFT-pool supply: {len(eg)} groups hold {sum(r['type'] == 'edit' for g in eg for r in by_group[g])} implants, "
             f"{len(rg)} groups hold {sum(r['type'] == 'remove' for g in rg for r in by_group[g])} removals. The per-group "
             f"cap therefore under-samples implants (few implant sources, ~10-110 forgeries each). Implants in sft_id "
             f"for other `--cap_edit` values (removal cap fixed at {args.cap}): " + ", ".join(
                 f"{ce}: {sum(x['type'] == 'edit' for g in sft_g for x in capped([r for r in by_group[g] if r['type'] != 'real'], args.cap, ce, 'pick-v1:'))}"
                 for ce in (4, 8, 12, 16, 24)) + ".")
    R.append("")
    R.append("### Generator tables (fakes)")
    gens = sorted({r["gen"] for r in fk})
    gcols = [gs(g) for g in gens]
    gc = collections.Counter()
    for reg in REGIMES:
        for r in counts[reg]["sft"]:
            if r["type"] != "real":
                gc[(f"sft_{reg}", gs(r["gen"]))] += 1
        for x in counts[reg]["rl_pairs"]:
            gc[(f"rl_{reg}", gs(x["fake"]["gen"]))] += 1
    for n, rows in evals.items():
        for r in rows:
            if r["type"] != "real":
                gc[(n, gs(r["gen"]))] += 1
    R.append("")
    R.append("(* = CM held-out generator)")
    R.append("")
    R.append(table(gc, "file", "generator", rows=[f"sft_{g}" for g in REGIMES] + [f"rl_{g}" for g in REGIMES]
                   + list(evals), cols=gcols))
    R.append("")
    R.append("### Eval files: class x organ")
    for n, rows in evals.items():
        R.append("")
        R.append(f"**{n}_fields.jsonl**")
        R.append("")
        R.append(table(collections.Counter((r["organ"], r["type"]) for r in rows), "organ", "class",
                       cols=["real", "edit", "remove"]))
    R.append("")
    R.append("### Self-tests")
    R.append("")
    R.append("| check | result | detail |")
    R.append("|---|---|---|")
    for n, r, d in st_res:
        R.append(f"| {n} | {r} | {str(d)[:160].replace('|', '/')} |")
    R.append("")
    R.append("### Output md5")
    R.append("")
    for fn, v in info["files"].items():
        R.append(f"- `{fn}`: {v['rows']} rows, md5 {v['md5']}")
    auto = "\n".join(R)
    old = open(args.report).read() if os.path.exists(args.report) else ""
    if AUTO_BEGIN in old and AUTO_END in old:
        new = old.split(AUTO_BEGIN)[0] + AUTO_BEGIN + "\n" + auto + "\n" + AUTO_END + old.split(AUTO_END, 1)[1]
    else:
        new = (old or "# v3 data build (build_sft_v3.py)\n\n") + "\n" + AUTO_BEGIN + "\n" + auto + "\n" + AUTO_END + "\n"
    with open(args.report + ".tmp", "w") as fh:
        fh.write(new)
    os.replace(args.report + ".tmp", args.report)
    nfail = sum(r == "FAIL" for _, r, _ in st_res)
    log(f"[done] report -> {args.report}; selftest failures: {nfail}")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
