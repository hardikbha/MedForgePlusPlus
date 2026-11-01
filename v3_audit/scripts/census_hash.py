"""Census + hashing pass (READ-ONLY on data). Writes only under ~/CIRCE/v3_audit/.

For every image file in the MedForge-90K tree (excluding gt/ annotation assets), the
MedForge++ addition (naitik_annotated/*/images, colon_injections/images,
~/AAAI/kidney/flux_injected/images) and the kidney FLUX source pool (~/AAAI/kidney/normal),
compute: md5(bytes), 64-bit dHash (9x8 gray LANCZOS, row-wise left>right), width, height,
PIL mode, PIL format, file size, plus a 32x32 gray thumbnail (stored in an .npy) for
verification of near-duplicate hits.
Mask files are counted (not hashed) in a separate listing.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os, sys, hashlib, io, json, time
from multiprocessing import Pool
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
OUT = _release_path('CIRCE/v3_audit/data')
os.makedirs(OUT, exist_ok=True)
MF = _release_path('AAAI/MedForge++/medforge90k_full')
PP = _release_path('AAAI/MedForge++')
KID = _release_path('AAAI/kidney')
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def enumerate_files():
    imgs, masks = [], []
    # MF90K real
    for root, dirs, files in os.walk(os.path.join(MF, "real")):
        for f in files:
            p = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXT:
                imgs.append(("mf90k", p))
    for root, dirs, files in os.walk(os.path.join(MF, "deepfake")):
        rel = os.path.relpath(root, MF).split("/")
        for f in files:
            p = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if ext not in IMG_EXT:
                continue
            if len(rel) >= 3 and rel[2] == "gt":
                masks.append(("mf90k_gt_" + (rel[3] if len(rel) > 3 else "?"), p))
            else:
                imgs.append(("mf90k", p))
    # naitik_annotated
    na = os.path.join(PP, "naitik_annotated")
    for sub in sorted(os.listdir(na)):
        d = os.path.join(na, sub)
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            rel = os.path.relpath(root, d).split("/")
            for f in files:
                p = os.path.join(root, f)
                ext = os.path.splitext(f)[1].lower()
                if ext not in IMG_EXT:
                    continue
                if rel[0] == "images":
                    imgs.append(("pp_" + sub, p))
                else:
                    masks.append(("pp_" + sub + "_" + "_".join(rel), p))
    for f in os.listdir(os.path.join(PP, "colon_injections", "images")):
        if os.path.splitext(f)[1].lower() in IMG_EXT:
            imgs.append(("pp_colon_inject", os.path.join(PP, "colon_injections", "images", f)))
    for f in os.listdir(os.path.join(KID, "flux_injected", "images")):
        if os.path.splitext(f)[1].lower() in IMG_EXT:
            imgs.append(("pp_kidney_inject", os.path.join(KID, "flux_injected", "images", f)))
    for f in os.listdir(os.path.join(KID, "normal")):
        if os.path.splitext(f)[1].lower() in IMG_EXT:
            imgs.append(("src_kidney_normal_kaggle", os.path.join(KID, "normal", f)))
    return imgs, masks


def dhash64(gray):
    a = np.asarray(gray.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def work(item):
    group, p = item
    try:
        b = open(p, "rb").read()
        md5 = hashlib.md5(b).hexdigest()
        im = Image.open(io.BytesIO(b))
        fmt = im.format
        mode = im.mode
        w, h = im.size
        g = im.convert("L")
        dh = dhash64(g)
        th = np.asarray(g.resize((32, 32), Image.LANCZOS), dtype=np.uint8)
        mean = float(np.asarray(g, dtype=np.float32).mean())
        return (group, p, md5, "%016x" % dh, w, h, mode, fmt, len(b), round(mean, 3), "", th)
    except Exception as e:  # corrupt / unreadable
        try:
            sz = os.path.getsize(p)
        except Exception:
            sz = -1
        return (group, p, "", "", -1, -1, "", "", sz, -1, repr(e)[:200], np.zeros((32, 32), np.uint8))


def main():
    t0 = time.time()
    imgs, masks = enumerate_files()
    print(f"[enum] images={len(imgs)} masks={len(masks)} ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(OUT, "mask_files.tsv"), "w") as f:
        for g, p in masks:
            f.write(f"{g}\t{p}\n")
    thumbs = np.zeros((len(imgs), 32, 32), np.uint8)
    with Pool(16) as pool, open(os.path.join(OUT, "image_hashes.tsv"), "w") as f:
        f.write("idx\tgroup\tpath\tmd5\tdhash\tw\th\tmode\tfmt\tbytes\tmean\terr\n")
        for i, r in enumerate(pool.imap(work, imgs, chunksize=64)):
            thumbs[i] = r[-1]
            f.write("\t".join([str(i)] + [str(x) for x in r[:-1]]) + "\n")
            if i % 10000 == 0:
                print(f"[hash] {i}/{len(imgs)} {time.time()-t0:.0f}s", flush=True)
    np.save(os.path.join(OUT, "thumbs32.npy"), thumbs)
    print(f"[done] {len(imgs)} images in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
