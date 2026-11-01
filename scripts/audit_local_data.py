#!/usr/bin/env python3
"""Read-only checks of the original corpus and the MF-Clean split manifests.

Only aggregate results are saved; image data and individual identifiers are excluded.
The audit does not establish patient independence when patient IDs are unavailable.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def rows(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def image_path(row):
    return row.get("image") or row.get("images", [None])[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--kidney-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root
    report = {"audit_date": "2026-11-01", "scope": "read-only filesystem and manifest checks; no model training"}
    mapping = json.loads((root / "medforge90k_full/dataset_mapping.json").read_text())
    items = mapping["items"] if isinstance(mapping, dict) else mapping
    present = Counter()
    missing = Counter()
    for item in items:
        target = present if (root / "medforge90k_full" / item["image_path"]).is_file() else missing
        target[item["type"]] += 1
    report["medforge90k"] = {"mapping_items": len(items), "present_by_type": dict(present),
                             "missing_by_type": dict(missing), "present_total": sum(present.values())}
    subsets = {name: root / "naitik_annotated" / name / "images" for name in
               ["kidney_normal", "kidney_real_tumor", "kidney_removal", "polyp_normal", "polyp_real", "polyp_removal"]}
    subsets.update(colon_inject=root / "colon_injections/images", kidney_inject=args.kidney_root / "flux_injected/images")
    report["addition"] = {}
    for name, directory in subsets.items():
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        formats = Counter()
        digests = Counter()
        for path in files:
            with path.open("rb") as handle:
                header = handle.read(16)
            formats["PNG" if header.startswith(b"\x89PNG\r\n\x1a\n") else "JPEG" if header.startswith(b"\xff\xd8\xff") else "other"] += 1
            if name == "polyp_normal":
                digests[hashlib.sha256(path.read_bytes()).hexdigest()] += 1
        entry = {"files": len(files), "actual_formats": dict(formats)}
        if digests:
            entry.update(unique_byte_hashes=len(digests), redundant_byte_duplicates=sum(digests.values())-len(digests))
        report["addition"][name] = entry
    eval_paths = {image_path(row) for row in rows(root / "circe_eval_split_v2.jsonl")}
    report["legacy_eval_overlap"] = {"eval_images": len(eval_paths)}
    for name in ["circe_sft_v2.jsonl", "circe_sft_v3.jsonl"]:
        paths = {image_path(row) for row in rows(root / name)}
        report["legacy_eval_overlap"][name] = {"unique_train_images": len(paths), "eval_paths_in_train": len(eval_paths & paths)}
    groups, paths = {}, {}
    report["mfclean"] = {}
    for split in ["train_all", "select_core", "test_core"]:
        records = list(rows(root / "mfclean" / (split + ".jsonl")))
        groups[split] = {row["group"] for row in records}
        paths[split] = {row["image"] for row in records}
        report["mfclean"][split] = {"rows": len(records), "groups": len(groups[split]),
                                    "types": dict(Counter(row["type"] for row in records))}
    report["mfclean"]["pairwise_overlap"] = {}
    for first, second in [("train_all", "select_core"), ("train_all", "test_core"), ("select_core", "test_core")]:
        report["mfclean"]["pairwise_overlap"][first + "__" + second] = {
            "groups": len(groups[first] & groups[second]), "image_paths": len(paths[first] & paths[second])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
