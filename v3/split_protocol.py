"""Audit declared split identities and freeze manifest bytes; no image/GPU dependencies.

This checks supplied identities, not whether unknown patients or duplicate images
were correctly discovered. See docs/SPLIT_PROTOCOL.md for the data-side gate.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import posixpath

FILES = {"train": "train_all.jsonl", "select": "select_core.jsonl", "test": "test_core.jsonl"}
IDENTITIES = ("group", "orig", "source_id", "patient_id", "duplicate_cluster", "content_md5")


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_splits(splits):
    """Reject overlapping images/identity groups, bad declarations and invalid twins.

    Optional provenance identities must be globally namespaced by their producer.
    An identity cannot belong to two groups, even within a split: otherwise a
    later SFT/RL division could leak it. Missing twins are reported, not invented.
    """
    if set(splits) != set(FILES):
        raise ValueError("Expected exactly train, select and test partitions")
    seen, identities, coverage, counts = {}, {}, {}, {}
    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"Empty partition: {split}")
        coverage[split] = dict.fromkeys(IDENTITIES, 0)
        counts[split] = {"rows": len(rows), "groups": len({r.get('group') for r in rows}),
                         "types": dict(Counter(r.get("type") for r in rows))}
        for row in rows:
            image, group = row.get("image"), row.get("group")
            if not isinstance(image, str) or not image.strip() or not isinstance(group, str) or not group.strip():
                raise ValueError(f"Missing image/group identity in {split}")
            if row.get("split") != split:
                raise ValueError(f"Incorrect split declaration for {image}: expected {split}")
            if row.get("type") not in ("real", "edit", "remove"):
                raise ValueError(f"Unknown type for {image}")
            # Collapse lexical aliases, but do not depend on files existing locally.
            key = posixpath.normpath(image)
            if key in seen:
                raise ValueError(f"Duplicate image within/across partitions: {image}")
            seen[key] = (split, row)
            for field in IDENTITIES:
                value = row.get(field)
                if value is None or value == "":
                    continue
                if not isinstance(value, str):
                    raise ValueError(f"Identity {field} must be a string: {image}")
                coverage[split][field] += 1
                identity = (field, posixpath.normpath(value) if field == "orig" else value)
                previous = identities.setdefault(identity, (split, group))
                if previous != (split, group):
                    raise ValueError(f"Identity overlap/inconsistent grouping: {field}={value}")
    missing_twins, fakes = 0, 0
    for split, row in seen.values():
        if row["type"] == "real":
            if row.get("twin"):
                raise ValueError(f"Real image cannot declare a fake's twin: {row['image']}")
            continue
        fakes += 1
        twin = row.get("twin")
        if not twin:
            missing_twins += 1
            continue
        target = seen.get(posixpath.normpath(twin))
        if target is None:
            raise ValueError(f"Dangling declared twin: {row['image']}")
        ts, tr = target
        if ts != split or tr["group"] != row["group"] or tr["type"] != "real":
            raise ValueError(f"Twin crosses split/group or is not real: {row['image']}")
    return {"status": "PASS", "counts": counts, "identity_coverage_rows": coverage,
            "fake_rows": fakes, "fake_rows_without_twin": missing_twins,
            "scope": "Declared metadata only; raw pixels and unknown patient identities not verified"}


def audit_directory(root):
    root = Path(root)
    # Hash before and after reading to reject a manifest changing during the audit.
    before = {name: sha256(root / file) for name, file in FILES.items()}
    splits = {name: read_jsonl(root / file) for name, file in FILES.items()}
    report = validate_splits(splits)
    after = {name: sha256(root / file) for name, file in FILES.items()}
    if before != after:
        raise ValueError("Manifests changed during audit")
    return {"schema": "mfclean-split-lock-v1", "files": {
        name: {"name": file, "sha256": before[name]} for name, file in FILES.items()}, "audit": report}


def verify_lock(root, lock_path):
    expected = json.loads(Path(lock_path).read_text())
    actual = audit_directory(root)
    if actual != expected:
        raise ValueError("Split lock mismatch: manifests or audit changed; do not reuse this protocol version")
    return actual


def write_new_json(path, value):
    # Never overwrite a previously frozen protocol/selection decision.
    with open(path, "x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def require_pass(checks):
    failures = [name for name, status, _ in checks if status != "PASS"]
    if failures:
        raise ValueError("Data self-test failed: " + "; ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mfclean", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", help="Create a new lock; fails if path already exists")
    mode.add_argument("--verify-lock", help="Re-audit and compare with an existing lock")
    args = parser.parse_args()
    report = (verify_lock(args.mfclean, args.verify_lock) if args.verify_lock
              else audit_directory(args.mfclean))
    if args.freeze:
        write_new_json(args.freeze, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
