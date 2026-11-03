"""Inventory the released upstream SFT/RL/test JSONs without opening images.

Only aggregate counts and file hashes are emitted; source/patient independence
and actual image availability require a separate data-host audit.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import re

FILES = {"sft": "sft_train_dataset.json", "rl": "grpo_train_10k_train.json", "test": "test_dataset.json"}
CM_HOLDOUT = {"gemini", "gpt", "stable-diffusion-3.5-medium", "stable-diffusion-xl-1.0-inpainting-0.1"}


def inventory(root):
    report = {"scope": "Manifest inventory only; no image verification, model inference or training",
              "type_source": "Released image path; not a clinical annotation audit", "splits": {}}
    identities = {}
    for split, filename in FILES.items():
        path = Path(root) / filename
        raw = path.read_bytes()
        rows = json.loads(raw)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Expected nonempty JSON array: {filename}")
        keys, counts, generators, eligible = [], Counter(), Counter(), Counter()
        for row in rows:
            images = row.get("images", [])
            if len(images) != 1:
                raise ValueError(f"Expected one image per record: {filename}")
            image = images[0]
            if isinstance(image, dict):
                image = image["path"]
            match = re.search(r"(?:^|/)(real|deepfake)/(.+)$", image)
            if not match:
                raise ValueError(f"Unresolved release-relative image identity: {filename}")
            key = match.group(1) + "/" + match.group(2)
            keys.append(key)
            kind = "real" if key.startswith("real/") else (
                "edit" if re.search(r"-edit[/.]", key) else
                "remove" if re.search(r"-remove[/.]", key) else "unknown")
            if kind == "unknown":
                raise ValueError(f"Unresolved manipulation type: {filename}")
            generator = "real" if kind == "real" else key.split("/")[1]
            counts[kind] += 1
            generators[generator] += 1
            eligible["id"] += 1
            eligible["cf"] += kind != "edit"
            eligible["cm"] += generator not in CM_HOLDOUT
        identities[split] = set(keys)
        report["splits"][split] = {"file": filename, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "rows": len(rows), "unique_image_ids": len(identities[split]),
            "duplicate_image_ids": len(rows) - len(identities[split]), "by_type": dict(counts),
            "by_generator": dict(generators),
            "training_eligible_by_regime": dict(eligible) if split != "test" else None}
    report["cross_split_image_overlap"] = {
        f"{a}_vs_{b}": len(identities[a] & identities[b]) for a, b in itertools.combinations(FILES, 2)}
    report["limitations"] = ["No raw-image existence/hash verification", "No source/patient/near-duplicate audit",
                             "Eligibility counts are proposed filters, not evidence of prior training exclusions",
                             "Does not establish that released manifests generated the published scores"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = inventory(args.manifest_dir)
    report["dataset_revision"] = args.dataset_revision
    with open(args.out, "x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
