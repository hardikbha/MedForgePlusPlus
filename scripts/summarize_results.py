#!/usr/bin/env python3
"""Regenerate the public results table from the saved aggregate metric files."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lines = ["# Saved MF-Clean test results", "", "Snapshot: 2026-11-01. Percentages; balanced accuracy = (real recall + fake recall)/2.",
         "All rows use test_core: 2,756 images in 1,008 source groups. These are existing experiments, not new training runs.",
         "Different regimes and training histories must be compared within matched conditions. See [the review](CVPR_REVIEW.md).", "",
         "| Metric file | Accuracy | Balanced accuracy [95% group CI] | Macro-F1 | Removal recall | mIoU | IoU@0.5 |", "|---|---:|---:|---:|---:|---:|---:|"]
for path in sorted((root / "results/metrics").glob("metrics_test_*.json")):
    record = json.loads(path.read_text())
    overall = record["overall"]
    point, ci = overall["point"], overall["ci95"]["bal_acc"]
    def pct(key):
        value = point.get(key)
        return f"{100*value:.2f}" if value is not None else "—"
    lines.append(f"| [{path.stem}](../results/metrics/{path.name}) | {pct('acc')} | {pct('bal_acc')} [{100*ci[0]:.2f}, {100*ci[1]:.2f}] | {pct('f1_macro')} | {pct('rec_removal')} | {pct('mIoU')} | {pct('IoU@0.5')} |")
(root / "docs/RESULTS.md").write_text("\n".join(lines) + "\n")
print(f"Wrote {len(lines)-8} result rows")
