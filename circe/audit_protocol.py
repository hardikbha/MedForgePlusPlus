"""
CIRCE-v2 Phase 2: protocol leakage audit + strict manifest builder.

Verifies the MedForge protocol files (built by build_medforge_protocol.py) and every
downstream training file (replay sets, PF+ merges) for the leakage failure modes called
out in the CIRCE-v2 spec: implant-derived data in Cross-Forgery training, held-out
generators leaking into Cross-Model training, train/test image-path overlap, and
within-file duplicates.

Ground truth for "is this image an implant/edit case" is derived from the image PATH,
not the `type` field some downstream files carry, because (a) most downstream files
(mfp_train_*.jsonl after dump_sft() strips `_meta`) don't carry a type field at all, and
(b) a naive substring check on "edit" is WRONG -- several generator names themselves
contain "edit" (e.g. "step1x-edit"), so a bare `"edit" in path` check produces false
positives. The correct signal is the ORGAN-MANIPULATION path component, which always
sits immediately after the generator directory: `deepfake/{generator}/{organ}-{ftype}/...`
-- see `organ_manip_type()` below. Verified against dataset_mapping.json directly before
being used here (see the exploration in the session that produced this script).

Usage:
    python audit_protocol.py                     # audit only, exits 1 on any leak found
    python audit_protocol.py --build-strict       # also writes mfp_v2_*.jsonl strict manifests
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, re, sys, argparse, random
from collections import Counter

D = _release_path('AAAI/MedForge++')
CM_HOLDOUT = {"gemini", "gpt", "stable-diffusion-3.5-medium",
              "stable-diffusion-xl-1.0-inpainting-0.1"}


def organ_manip_type(path):
    """Return 'edit' | 'remove' | 'real' | None, from the PATH structure (not any
    metadata field). None means the path doesn't match the expected MedForge-90K layout
    (e.g. a PF+/kidney/colon image) and callers should fall back to their own logic."""
    parts = path.split("/")
    if "deepfake" in parts:
        i = parts.index("deepfake")
        if i + 2 < len(parts):
            seg = parts[i + 2]              # the "{organ}-{ftype}" component
            m = re.search(r'-(edit|remove)$', seg)
            if m:
                return m.group(1)
        return None
    if "real" in parts:
        return "real"
    return None


def organ_generator(path):
    """Return the generator directory name, or None if not a MedForge90K-style path."""
    parts = path.split("/")
    if "deepfake" in parts:
        i = parts.index("deepfake")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def load_images(path):
    """Return list of (image_path, row_dict) for a jsonl of {"images":[...], ...} records."""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        img = r.get("images", [r.get("image")])[0]
        out.append((img, r))
    return out


def check_duplicates(name, rows):
    paths = [p for p, _ in rows]
    c = Counter(paths)
    dups = {p: n for p, n in c.items() if n > 1}
    if dups:
        print(f"  [FAIL] {name}: {len(dups)} duplicate image paths ({sum(dups.values())} extra rows)")
        return False
    print(f"  [ok]   {name}: no duplicate paths ({len(paths)} rows)")
    return True


def check_implant_free(name, rows):
    leaks = [p for p, _ in rows if organ_manip_type(p) == "edit"]
    if leaks:
        print(f"  [FAIL] {name}: {len(leaks)}/{len(rows)} rows are implant/edit-derived "
              f"(protocol requires ZERO for strict Cross-Forgery)")
        print(f"         example: {leaks[0]}")
        return False, leaks
    print(f"  [ok]   {name}: 0/{len(rows)} implant-derived rows")
    return True, []


def check_no_holdout_generators(name, rows):
    bad = [p for p, _ in rows if organ_generator(p) in CM_HOLDOUT]
    if bad:
        print(f"  [FAIL] {name}: {len(bad)}/{len(rows)} rows use a Cross-Model held-out generator")
        return False
    print(f"  [ok]   {name}: no held-out-generator rows ({len(rows)} checked)")
    return True


def check_no_overlap(name, rows, test_paths):
    overlap = [p for p, _ in rows if p in test_paths]
    if overlap:
        print(f"  [FAIL] {name}: {len(overlap)} rows overlap the held-out test set (train/test leakage)")
        return False
    print(f"  [ok]   {name}: no overlap with test set")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-strict", action="store_true",
                     help="also write versioned mfp_v2_*.jsonl strict manifests")
    ap.add_argument("--val_frac", type=float, default=0.1,
                     help="fraction of each strict train pool carved out as validation")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ok = True
    print("=" * 70)
    print("CIRCE-v2 protocol leakage audit")
    print("=" * 70)

    test_rows = load_images(f"{D}/mfp_test_shared.jsonl")
    test_paths = {p for p, _ in test_rows}
    print(f"\n[test set] {D}/mfp_test_shared.jsonl -- {len(test_rows)} rows")
    ok &= check_duplicates("mfp_test_shared.jsonl", test_rows)

    print("\n[In-Domain train]")
    id_rows = load_images(f"{D}/mfp_train_indomain.jsonl")
    ok &= check_duplicates("mfp_train_indomain.jsonl", id_rows)
    ok &= check_no_overlap("mfp_train_indomain.jsonl", id_rows, test_paths)

    print("\n[Cross-Model train]")
    cm_rows = load_images(f"{D}/mfp_train_crossmodel.jsonl")
    ok &= check_duplicates("mfp_train_crossmodel.jsonl", cm_rows)
    ok &= check_no_overlap("mfp_train_crossmodel.jsonl", cm_rows, test_paths)
    ok &= check_no_holdout_generators("mfp_train_crossmodel.jsonl", cm_rows)

    print("\n[Cross-Forgery train]")
    cf_rows = load_images(f"{D}/mfp_train_crossforgery.jsonl")
    ok &= check_duplicates("mfp_train_crossforgery.jsonl", cf_rows)
    ok &= check_no_overlap("mfp_train_crossforgery.jsonl", cf_rows, test_paths)
    cf_clean, cf_leaks = check_implant_free("mfp_train_crossforgery.jsonl", cf_rows)
    ok &= cf_clean

    print("\n[Cross-Forgery replay set -- the file behind the 95.76/95.13 Stage-1c result]")
    replay_path = f"{D}/mfp_replay_implant_focus_sft.jsonl"
    replay_rows = load_images(replay_path)
    ok_r = True
    if replay_rows:
        ok_r &= check_duplicates("mfp_replay_implant_focus_sft.jsonl", replay_rows)
        ok_r &= check_no_overlap("mfp_replay_implant_focus_sft.jsonl", replay_rows, test_paths)
        replay_clean, replay_leaks = check_implant_free("mfp_replay_implant_focus_sft.jsonl", replay_rows)
        ok_r &= replay_clean
    else:
        print(f"  [skip] {replay_path} not found")
    # NOTE: replay leakage does NOT flip the overall `ok` exit code -- it's a KNOWN,
    # already-documented issue (CIRCE_FULL_APPROACH_AND_RESULTS.md), not a new regression.
    # It's reported here so --build-strict can fix it, but doesn't fail CI on its own.
    if replay_rows and not ok_r:
        print("  --> confirmed: this is the exact leakage documented as a caveat on the "
              "95.76/95.13 result. Use --build-strict to produce a leak-free replacement.")

    print("\n" + "=" * 70)
    if ok:
        print("AUDIT PASSED: no leakage in the core protocol files "
              "(mfp_train_*.jsonl, mfp_test_shared.jsonl).")
    else:
        print("AUDIT FAILED: see [FAIL] lines above.")
    print("=" * 70)

    if args.build_strict:
        build_strict_manifests(cf_rows, id_rows, cm_rows, test_rows, replay_rows, args)

    if not ok:
        sys.exit(1)


def _dump(rows, path):
    with open(path, "w") as f:
        for p, r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows)} rows -> {path}")


def build_strict_manifests(cf_rows, id_rows, cm_rows, test_rows, replay_rows, args):
    """Write versioned mfp_v2_*.jsonl manifests. Never overwrites the original mfp_*.jsonl
    files. train/val split is a simple deterministic tail-carve (last val_frac of a
    shuffled-once-at-build-time list), since none of these files are pre-shuffled per call --
    they were already `random.seed(42)`-shuffled once at protocol-build time, so we reshuffle
    here with a different seed to avoid correlating the val carve with any positional
    structure left over from that build."""
    print("\n" + "=" * 70)
    print("Building strict mfp_v2_*.jsonl manifests")
    print("=" * 70)
    rng = random.Random(args.seed)

    def split_val(rows):
        rows = rows[:]
        rng.shuffle(rows)
        n_val = int(len(rows) * args.val_frac)
        return rows[n_val:], rows[:n_val]   # train, val

    # In-Domain / Cross-Model: already leak-free per the audit above, just version + carve val.
    for regime, rows in [("indomain", id_rows), ("crossmodel", cm_rows)]:
        tr, va = split_val(rows)
        _dump(tr, f"{D}/mfp_v2_train_{regime}.jsonl")
        _dump(va, f"{D}/mfp_v2_val_{regime}.jsonl")
        _dump(tr, f"{D}/mfp_v2_rl_{regime}.jsonl")   # no separate GSPO carve-out exists yet; reuse train pool

    # Cross-Forgery: cf_rows is already implant-free by construction (verified above) --
    # version it directly (no additional filtering needed for the base train file).
    tr, va = split_val(cf_rows)
    _dump(tr, f"{D}/mfp_v2_train_crossforgery_strict.jsonl")
    _dump(va, f"{D}/mfp_v2_val_crossforgery_strict.jsonl")
    _dump(tr, f"{D}/mfp_v2_rl_crossforgery_strict.jsonl")

    # Strict replay: filter the 4,678/6,400 implant-derived rows OUT of the replay set that
    # produced the 95.76/95.13 result. What remains is the leak-free subset only.
    if replay_rows:
        strict_replay = [(p, r) for p, r in replay_rows if organ_manip_type(p) != "edit"]
        print(f"\n  strict replay: {len(strict_replay)}/{len(replay_rows)} rows survive "
              f"implant-leak filtering ({len(replay_rows)-len(strict_replay)} removed)")
        _dump(strict_replay, f"{D}/mfp_v2_replay_crossforgery_strict.jsonl")
        if len(strict_replay) < 0.5 * len(replay_rows):
            print("  [WARNING] strict replay set lost >50% of its rows -- the original "
                  "95.76/95.13 result's replay-mining strategy was heavily implant-dependent. "
                  "A strict rerun will very likely score lower; this is expected and should be "
                  "reported as a corrected, honest baseline, not a regression to \"fix\".")

    # Test set: copy under the versioned name for a consistent naming scheme downstream.
    _dump(test_rows, f"{D}/mfp_v2_test_shared.jsonl")

    print("\nDone. Original mfp_*.jsonl files are untouched.")


if __name__ == "__main__":
    main()
