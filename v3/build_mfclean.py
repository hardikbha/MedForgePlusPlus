"""
MF-Clean: a shortcut-controlled, source-disjoint rebuild of MedForge-90K (v3, 2026-10-11).

Fixes measured by the v3 audit (~/CIRCE/v3_audit/shortcut.md, labels.md):
  1. whole-frame re-rendering fingerprint  -> PASTE-BACK: fake = authentic source everywhere except inside the
     GT box dilated by DIL px, blended with a Gaussian-feathered alpha (sigma FEATHER px).
  2. colour cast + JPEG-quality split      -> ONE ENCODER: gray modalities forced to single-channel L; every
     image (real and fake) decoded and re-encoded once as JPEG q90 4:4:4.
  3. real sub-populations never edited     -> reals kept only from "families" (organ, disease folder, filename
     shape) that contain at least one forgery source.
  4. exact / near duplicate reals          -> union-find on md5 and dHash<=4 plus thumbnail MAE<=3; one image kept
     per duplicate cluster; all members share a group.
  5. source leakage across splits          -> GROUP = duplicate cluster of the authentic source scan (fundus:
     additionally merged by patient id left/right). Groups are split 55/15/30 (train/select/test) by a
     deterministic hash, so a forgery, its source, and every sibling forgery always share a split.
  6. twin-structured evaluation            -> select/test "core" sets keep one real per group and round-robin
     generator samples. Some fake twins are absent; the split audit reports this coverage limitation.

Outputs (all under OUT):
  img/<rel path>.jpg                     re-encoded images (reals, composites)
  manifest.jsonl                         one row per image kept, with split/group/organ/type/gen/box/twin
  train_all.jsonl, select_core.jsonl, test_core.jsonl   eval-format rows {image,type,gen,box,organ,group,twin}
  stats.json                             counts and what was dropped and why
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import argparse, collections, hashlib, json, os, re, sys
import numpy as np
from PIL import Image, ImageFilter
from concurrent.futures import ProcessPoolExecutor
from v3.split_protocol import validate_splits

MF = _release_path('AAAI/MedForge++/medforge90k_full/')
HASHES = _release_path('CIRCE/v3_audit/data/image_hashes.tsv')
GRAY = {"brain-mri", "chest-xray"}
CM_HOLDOUT = {"gemini", "gpt", "stable-diffusion-3.5-medium", "stable-diffusion-xl-1.0-inpainting-0.1"}
DIL, FEATHER, Q = 8, 3.0, 90
SPLIT = (("train", 55), ("select", 15), ("test", 30))


def fname_shape(name):
    s = os.path.splitext(name)[0]
    s = re.sub(r"[0-9a-f]{16,}", "H", s)
    s = re.sub(r"\d+", "D", s)
    return s


def family(rel):
    # rel like real/<organ>/<disease>/<file>
    parts = rel.split("/")
    return (parts[1], parts[2] if len(parts) > 3 else "", fname_shape(parts[-1]))


def known_patient_id(source, organ):
    """Only filename patient conventions already used by the grouping code."""
    stem = os.path.splitext(os.path.basename(source))[0]
    if organ == "fundus":
        match = re.match(r"^(\d+)_(left|right)(_eyepacs)?$", stem)
        if match:
            return "medforge90k:fundus:" + ("E:" if match.group(3) else "O:") + match.group(1)
    if organ == "brain-mri":
        match = re.match(r"^(YG_[A-Z0-9]+)_", stem)
        if match:
            return "medforge90k:brain:" + match.group(1)
    return None


class UF:
    def __init__(self):
        self.p = {}

    def f(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a, b):
        ra, rb = self.f(a), self.f(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def split_of(group):
    h = int(hashlib.sha1(("mfclean-v1:" + group).encode()).hexdigest(), 16) % 100
    acc = 0
    for name, pct in SPLIT:
        acc += pct
        if h < acc:
            return name
    return SPLIT[-1][0]


def feather_alpha(shape, box, dil, sigma):
    h, w = shape
    x1, y1, x2, y2 = box
    m = Image.new("L", (w, h), 0)
    m.paste(255, (max(0, x1 - dil), max(0, y1 - dil), min(w, x2 + dil + 1), min(h, y2 + dil + 1)))
    if sigma > 0:
        m = m.filter(ImageFilter.GaussianBlur(sigma))
    return np.asarray(m, dtype=np.float32)[..., None] / 255.0


def save_jpeg(arr, organ, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    im = im.convert("L") if organ in GRAY else im.convert("RGB")
    tmp = path + ".tmp"
    im.save(tmp, "JPEG", quality=Q, subsampling=0)
    os.replace(tmp, path)


def box_from_mask(mask_rel):
    m = np.asarray(Image.open(MF + mask_rel).convert("L")) > 127
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None, m.shape
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], m.shape


def work(job):
    kind, row, out = job
    try:
        dst = out + "/img/" + os.path.splitext(row["rel"])[0] + ".jpg"
        cached = os.path.exists(dst) and os.path.getsize(dst) > 0
        if cached and kind == "real":
            return {"ok": True, "image": dst}
        if kind == "real":
            a = np.asarray(Image.open(MF + row["rel"]).convert("RGB"), dtype=np.float32)
            save_jpeg(a, row["organ"], dst)
            return {"ok": True, "image": dst}
        f = np.asarray(Image.open(MF + row["rel"]).convert("RGB"), dtype=np.float32)
        r = Image.open(MF + row["src_rel"]).convert("RGB")
        if r.size != (f.shape[1], f.shape[0]):
            r = r.resize((f.shape[1], f.shape[0]), Image.BICUBIC)
        r = np.asarray(r, dtype=np.float32)
        box, mshape = box_from_mask(row["mask_rel"])
        if box is None:
            return {"ok": False, "why": "empty_mask"}
        if mshape != f.shape[:2]:  # rescale box to image size
            sy, sx = f.shape[0] / mshape[0], f.shape[1] / mshape[1]
            box = [int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)]
        al = feather_alpha(f.shape[:2], box, DIL, FEATHER)
        comp = al * f + (1 - al) * r
        # Reconstruct the annotation even on resume; input rows have no box_px.
        if not cached:
            save_jpeg(comp, row["organ"], dst)
        H, W = f.shape[:2]
        box1024 = [int(box[0] * 1024 / W), int(box[1] * 1024 / H), int(box[2] * 1024 / W), int(box[3] * 1024 / H)]
        return {"ok": True, "image": dst, "box": box1024}
    except Exception as e:  # noqa
        return {"ok": False, "why": repr(e)[:160]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=_release_path('AAAI/MedForge++/mfclean'))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--plan_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    items = json.load(open(MF + "dataset_mapping.json"))["items"]
    md5, dh = {}, {}
    for l in open(HASHES):
        p = l.rstrip("\n").split("\t")
        if p[0] == "idx" or len(p) < 5 or not p[3]:
            continue
        if "/medforge90k_full/" in p[2]:
            rel = p[2].split("/medforge90k_full/", 1)[1]
            md5[rel], dh[rel] = p[3], p[4]
    drop = collections.Counter()
    reals = {it["image_path"]: it for it in items if it["image_path"].startswith("real/")}
    fakes = [it for it in items if not it["image_path"].startswith("real/")]
    ondisk = {it["image_path"] for it in items if it["image_path"] in md5}
    drop["real_missing_on_disk"] = sum(1 for r in reals if r not in md5)
    # GROUP union-find (split unit; census logic, ~/CIRCE/v3_audit/scripts/group_components.py):
    #   forgery<->source scan, fundus patient (both eyes, EyePACS id space separate), Yale brain patient,
    #   md5 duplicates, verified near-duplicate reals (dHash<=4 and 32px-thumb MAE<=3).
    # DUP union-find (which reals are dropped as redundant): md5 + verified near-duplicates only.
    uf, dup = UF(), UF()
    for it in items:
        p_ = it["image_path"]
        if p_ not in ondisk:
            continue
        uf.f(p_)
        if not p_.startswith("real/"):
            uf.u(p_, it["real_image_path"])
        b_ = os.path.splitext(os.path.basename(p_ if p_.startswith("real/") else it["real_image_path"]))[0]
        if it["dataset"] == "fundus":
            m = re.match(r"^(\d+)_(left|right)(_eyepacs)?$", b_)
            if m:
                uf.u(p_, "PAT:fundus" + ("E" if m.group(3) else "O") + m.group(1))
        if it["dataset"] == "brain-mri":
            m = re.match(r"^(YG_[A-Z0-9]+)_", b_)
            if m:
                uf.u(p_, "PAT:" + m.group(1))
    by_md5 = collections.defaultdict(list)
    for rel in reals:
        if rel in md5:
            dup.f(rel)
            by_md5[md5[rel]].append(rel)
    for grp in by_md5.values():
        for x in grp[1:]:
            uf.u(grp[0], x); dup.u(grp[0], x)
    th = np.load(_release_path('CIRCE/v3_audit/data/thumbs32.npy')).astype(np.int16)
    idx_of = {}
    for l in open(HASHES):
        p = l.rstrip("\n").split("\t")
        if p[0] != "idx" and "/medforge90k_full/" in p[2]:
            idx_of[p[2].split("/medforge90k_full/", 1)[1]] = int(p[0])
    LUT = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)
    for org in ("brain-mri", "chest-xray", "fundus"):
        L = [r for r in reals if r in md5 and r.startswith("real/" + org)]
        I = np.array([idx_of[r] for r in L])
        D = np.array([np.uint64(int(dh[r], 16)) for r in L], dtype=np.uint64)
        for st in range(0, len(L), 512):
            x = np.bitwise_xor(D[st:st + 512, None], D[None, :])
            H = LUT[x.view(np.uint16).reshape(x.shape[0], x.shape[1], 4)].sum(axis=2)
            ia, jb = np.nonzero(H <= 4)
            for i, j in zip(st + ia, jb):
                if i < j and np.abs(th[I[i]] - th[I[j]]).mean() <= 3:
                    uf.u(L[i], L[j]); dup.u(L[i], L[j])
    # --- sources and families
    src_used = set()
    for it in fakes:
        s = it.get("real_image_path")
        if s in md5:
            src_used.add(s)
    fam_with_src = {family(s) for s in src_used}
    # keep one representative real per verified-duplicate cluster (prefer a forgery source)
    clusters = collections.defaultdict(list)
    for rel in reals:
        if rel in md5:
            clusters[dup.f(rel)].append(rel)
    keep_real = {}
    for root, mem in clusters.items():
        srcs = [m for m in mem if m in src_used]
        rep = sorted(srcs)[0] if srcs else sorted(mem)[0]
        if family(rep) not in fam_with_src:
            drop["real_family_never_edited"] += len(mem)
            continue
        keep_real[rep] = uf.f(rep)
        drop["real_dup_removed"] += len(mem) - 1
    rows = []
    for rep, root in keep_real.items():
        it = reals[rep]
        rows.append({"rel": rep, "type": "real", "gen": "real", "organ": rep.split("/")[1],
                     "group": "g" + hashlib.md5(root.encode()).hexdigest()[:12], "src_rel": rep,
                     "disease": it.get("disease", "")})
    for it in fakes:
        s = it.get("real_image_path")
        if s not in md5:
            drop["fake_source_missing"] += 1
            continue
        if it["image_path"] not in md5:
            drop["fake_missing_on_disk"] += 1
            continue
        root = uf.f(it["image_path"])
        if family(s) not in fam_with_src:  # cannot happen (s is a source) but keep the invariant explicit
            drop["fake_family"] += 1
            continue
        rows.append({"rel": it["image_path"], "type": it["type"], "gen": it["model"], "organ": it["dataset"],
                     "group": "g" + hashlib.md5(root.encode()).hexdigest()[:12], "src_rel": s,
                     "mask_rel": it["mask_path"], "disease": it.get("disease", "")})
    for r in rows:
        r["split"] = split_of(r["group"])
    # twin = the kept representative real of the fake's own source duplicate-cluster
    rep_of_dup = {dup.f(r["rel"]): r["rel"] for r in rows if r["type"] == "real"}
    for r in rows:
        r["twin_rel"] = rep_of_dup.get(dup.f(r["src_rel"])) if r["type"] != "real" else None
    # --- choose which images to materialise: all train rows; cores for select/test
    import random
    rng = random.Random(2027)
    core = collections.defaultdict(list)
    by_group = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    gen_count = collections.Counter()   # global round-robin so every generator is equally represented
    for g in sorted(by_group):
        mem = by_group[g]
        sp = mem[0]["split"]
        if sp == "train":
            continue
        real = [m for m in mem if m["type"] == "real"]
        fk = [m for m in mem if m["type"] != "real" and m["twin_rel"]]
        if not real or not fk:
            continue                       # core = twin groups only (real-only family groups are excluded)
        k = 5 if fk[0]["type"] == "edit" else 1
        gens = sorted({m["gen"] for m in fk}, key=lambda x: (gen_count[(sp, x)], rng.random()))
        pick = []
        for gname in gens[:k]:
            pick.append(rng.choice([m for m in fk if m["gen"] == gname]))
            gen_count[(sp, gname)] += 1
        core[sp] += real[:1] + pick
    mat = [r for r in rows if r["split"] == "train"] + core["select"] + core["test"]
    stats = {"drops": dict(drop), "rows_total": len(rows),
             "rows_by_split_type": {f"{s}/{t}": n for (s, t), n in
                                    collections.Counter((r["split"], r["type"]) for r in rows).items()},
             "groups_by_split": dict(collections.Counter(split_of(g) for g in by_group)),
             "core_select": dict(collections.Counter(r["type"] for r in core["select"])),
             "core_test": dict(collections.Counter(r["type"] for r in core["test"])),
             "materialise": len(mat)}
    print(json.dumps(stats, indent=1))
    if args.plan_only:
        return
    jobs = [("real" if r["type"] == "real" else "fake", r, args.out) for r in mat]
    res = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (r, o) in enumerate(zip(mat, ex.map(work, jobs, chunksize=16))):
            res[r["rel"]] = o
            if (i + 1) % 5000 == 0:
                print(f"  [{i+1}/{len(mat)}]", flush=True)
    fails = collections.Counter(o.get("why", "?") for o in res.values() if not o["ok"])
    stats["materialise_failures"] = dict(fails)
    if fails:
        raise RuntimeError(f"Materialisation failed; refusing to publish partial split manifests: {dict(fails)}")

    def evrow(r):
        o = res.get(r["rel"])
        if not o or not o["ok"]:
            return None
        twin = res.get(r["twin_rel"]) if r.get("twin_rel") else None
        return {"image": o["image"], "type": r["type"], "gen": r["gen"], "box": o.get("box"),
                "organ": r["organ"], "group": r["group"], "split": r["split"], "disease": r["disease"],
                "twin": twin["image"] if twin and twin["ok"] else None, "orig": MF + r["rel"],
                "source_id": "medforge90k:" + r["src_rel"],
                "duplicate_cluster": "medforge90k:" + dup.f(r["src_rel"]),
                "patient_id": known_patient_id(r["src_rel"], r["organ"]),
                "content_md5": md5[r["rel"]]}

    partitions = {"train": [evrow(r) for r in rows if r["split"] == "train"],
                  "select": [evrow(r) for r in core["select"]],
                  "test": [evrow(r) for r in core["test"]]}
    stats["split_audit"] = validate_splits(partitions)

    def dump(name, rs):
        out = [x for x in (evrow(r) for r in rs) if x]
        tmp = f"{args.out}/{name}.tmp"
        with open(tmp, "w") as fh:
            for x in out:
                fh.write(json.dumps(x) + "\n")
        os.replace(tmp, f"{args.out}/{name}")
        stats[f"n_{name}"] = len(out)

    dump("train_all.jsonl", [r for r in rows if r["split"] == "train"])
    dump("select_core.jsonl", core["select"])
    dump("test_core.jsonl", core["test"])
    dump("manifest.jsonl", mat)
    json.dump(stats, open(f"{args.out}/stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k.startswith("n_") or k == "materialise_failures"}, indent=1))


if __name__ == "__main__":
    main()
