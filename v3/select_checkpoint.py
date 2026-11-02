"""Select a candidate on the locked selection split, never on test predictions.

Candidate JSON: [{"id": "sft-step100", "predictions": "...jsonl",
                  "artifact": "...safetensors", "config": "...json"}, ...].
Paths are relative to the candidate JSON. Pre-register candidates/budgets before
running them; this tool cannot prove historical non-exposure to test data.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v3.split_protocol import read_jsonl, sha256, verify_lock, write_new_json


def select(root, lock, candidates_path):
    from v3.stats_v3 import parse_any  # use the released scorer's verdict semantics

    verify_lock(root, lock)
    reference = {r["image"]: r for r in read_jsonl(Path(root) / "select_core.jsonl")}
    totals = {verdict: sum((r["type"] == "real") == (verdict == "real") for r in reference.values())
              for verdict in ("real", "deepfake")}
    if not all(totals.values()):
        raise ValueError("Selection balanced accuracy requires both real and fake examples")
    path = Path(candidates_path)
    plan_hash = sha256(path)
    candidates = json.loads(path.read_text())
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Expected a nonempty candidate list")
    records, ids = [], set()
    for candidate in candidates:
        name = candidate["id"]
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError("Candidate IDs must be nonempty and unique")
        ids.add(name)
        files = {key: path.parent / candidate[key] for key in ("predictions", "artifact", "config")}
        hashes = {key: sha256(file) for key, file in files.items()}
        rows = read_jsonl(files["predictions"])
        if len(rows) != len(reference) or len({r["image"] for r in rows}) != len(rows):
            raise ValueError(f"Incomplete or duplicated selection predictions: {name}")
        correct = {"real": 0, "deepfake": 0}
        for row in rows:
            truth = reference.get(row["image"])
            if truth is None or any(row.get(key) != value for key, value in truth.items()):
                raise ValueError(f"Prediction identity/GT differs from locked selection set: {name}")
            if not isinstance(row.get("output"), str):
                raise ValueError(f"Missing raw model output: {name}")
            verdict = "real" if truth["type"] == "real" else "deepfake"
            correct[verdict] += parse_any(row["output"])["verdict"] == verdict
        score = sum(Fraction(correct[v], totals[v]) for v in totals) / 2
        if hashes != {key: sha256(file) for key, file in files.items()}:
            raise ValueError("Candidate files changed during selection")
        records.append({"id": name, "balanced_accuracy": float(score),
                        "score_fraction": [score.numerator, score.denominator],
                        "correct": correct, "sha256": hashes})
    # Exact fractions avoid floating-point tie ambiguity; ID breaks exact ties.
    records.sort(key=lambda r: (-Fraction(*r["score_fraction"]), r["id"]))
    verify_lock(root, lock)
    if sha256(path) != plan_hash:
        raise ValueError("Candidate plan changed during selection")
    return {"schema": "mfclean-selection-v1", "split_lock_sha256": sha256(lock),
            "candidate_plan_sha256": plan_hash, "metric": "binary balanced accuracy",
            "tie_break": "lexicographically smallest candidate ID", "selected": records[0]["id"],
            "candidates": records, "scope": "Selection predictions only; training provenance not certified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mfclean", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = select(args.mfclean, args.lock, args.candidates)
    write_new_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
