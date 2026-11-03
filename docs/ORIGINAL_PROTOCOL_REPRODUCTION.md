# First milestone: run Bridge on the full official MedForge splits

Latest user decision, 2026-11-03: proceed directly with our Bridge method on the full official MedForge splits. Retraining MedForge-Reasoner and evaluating its released checkpoint are deferred; neither is a prerequisite for this run. This supersedes the earlier baseline-reproduction-first work order. The document filename is retained to preserve existing links.

Match the official data identities, regime exclusions and evaluation definitions while using our own method. This is a full-protocol Bridge evaluation, not a reproduction of the original MedForge training algorithm. The earlier 2,994-image evaluation and MF-Clean's 2,756-image evaluation do not establish this comparison. Keep kidney/colon additions out of this first experiment.

## Assets inspected and work actually completed

- Official code: [ACL2026-MedForge](https://github.com/richardChenzhihui/ACL2026-MedForge), inspected commit `f60505600d3de0959f4fcdc8bd1741754ed2c85f`.
- Official dataset: [MedForge-90K](https://huggingface.co/datasets/RichardChenZH/MedForge-90K/tree/7bbf54f870584b37d6594593da76b2fc78fe4c5b), pinned revision `7bbf54f870584b37d6594593da76b2fc78fe4c5b`.
- Downloaded and parsed the three complete JSON manifests, not the raw images or model weights. Results and SHA-256 hashes are in [official_manifest_inventory_20261103.json](evidence/official_manifest_inventory_20261103.json).

| Released file | Rows | Real | Implant | Removal |
|---|---:|---:|---:|---:|
| `sft_train_dataset.json` | 55,276 | 16,990 | 21,389 | 16,897 |
| `grpo_train_10k_train.json` | 10,000 | 5,000 | 3,013 | 1,987 |
| `test_dataset.json` | 30,000 | 10,000 | 10,952 | 9,048 |

Types above are derived from release-relative image paths. Each file has unique image IDs; all three pairwise image-ID overlaps are zero. This is not an audit of pixel duplicates, shared sources, patients, image availability or annotation validity. The manifests total 95,276 rows. Their counts/class proportions differ from the paper's nominal 50K/10K/30K and 1:1:1 description; preserve this discrepancy rather than silently resampling to match the prose. The release alone does not prove these exact manifests produced the published scores.

Reproduce this inventory without GPU access:

```bash
python scripts/audit_official_manifests.py \
  --manifest-dir /path/to/downloaded/official/jsons \
  --dataset-revision 7bbf54f870584b37d6594593da76b2fc78fe4c5b \
  --out /path/to/new/official-manifest-inventory.json
```

## Execute in this order on the data/GPU host

1. **Verify the complete assets.** Pin the above revisions, resolve manifest image paths to the local raw release, and verify every image/annotation. Do not silently skip missing images or regenerate a random split. Preserve the exact official sample identities. Record missing files and the source/patient audit separately; any repaired split is a distinct benchmark.
2. **Prepare and test the Bridge implementation and scorer.** Pin the environment used by our trainers. Repair/test the released evaluator or document a compatibility implementation with the same metric definitions; do not silently substitute MF-Clean metrics. Verify parsing, subgroup F1, coordinates and invalid-output handling on synthetic cases. Validate answer-label separation in inference: the reference assistant answer must not enter the model prompt. These checks do not require a MedForge-Reasoner run.
3. **Freeze Bridge's run plan and training-side validation.** Use the Qwen3-VL-8B base checkpoint with our SFT/replay/RL/interpolation pipeline. Keep the official SFT and RL populations separate. Predeclare validation carve-outs, seeds, sampling, budgets, checkpoint selection and alpha candidates; log the actual optimizer-training counts. Preserve all eligible records apart from the documented validation carve-outs, with no arbitrary subset cap. Do not initialize from a regime-unknown MedForge-Reasoner checkpoint.
4. **Train Bridge separately for ID, CF and CM.** Use each complete eligible official pool shown below. Apply exclusions to every SFT, replay and RL stage. Replay can use eligible training examples only. Select the final checkpoint/stage/alpha on validation; do not repair a weak result by selecting on test performance. Record the objective actually implemented, including any differences from our historical paper.
5. **Evaluate each selected Bridge model on all 30,000 official test examples.** Produce per-image predictions, coverage counts and Table-1-style per-category and aggregate accuracy/F1. Save localization metrics with explicit denominators. Compare against the original paper's published rows, clearly labeled as published reference results, not independently reproduced baselines. Document data-release/metric uncertainties before making a superiority claim.
6. **Evaluate Bridge for Table 2 separately.** Recover the original 100-example reasoning sample, references, prompts and judge versions if possible. If unavailable, report our result on a documented sample as a new evaluation; do not claim an exact numeric comparison. Rerunning other models is deferred rather than a prerequisite. Detection and parsing scores cannot substitute for judge scores.
7. **Then isolate gains and add data.** Evaluate our stage ablations and matched continued-SFT controls. Treat extra kidney/colon training data as a separately labeled augmentation condition, retaining the original test to measure its benefit. Clinical and clean-benchmark work remains valuable but must not silently alter this full-official-split experiment.

Filtering both official training pools by the paper's exclusions yields the following inventory (these counts are not evidence of historical training):

| Regime | Eligible SFT rows | Eligible RL rows | Evaluation |
|---|---:|---:|---|
| In-Domain | 55,276 | 10,000 | Same full 30,000 test |
| Cross-Forgery: exclude implants | 33,887 | 6,987 | Same full 30,000 test |
| Cross-Model: exclude gemini, gpt, SD3.5-medium, SDXL-inpainting | 39,591 | 8,247 | Same full 30,000 test |

## Evaluation requirements and deferred upstream-reproduction issues

- [Detection evaluator](https://github.com/richardChenzhihui/ACL2026-MedForge/blob/f60505600d3de0959f4fcdc8bd1741754ed2c85f/evaluation/detection_eval/eval.py) calls `classify_image`, `extract_bbox`, `extract_model_name`, `compute_iou` and `visualize_result` without defining/importing them. A local synthetic 8×8 image check actually failed with `NameError: name 'classify_image' is not defined`. Restore intended implementations or document/test a minimal compatibility patch; do not silently substitute a different metric.
- **Deferred, not a Bridge launch blocker:** the upstream [GSPO launcher](https://github.com/richardChenzhihui/ACL2026-MedForge/blob/f60505600d3de0959f4fcdc8bd1741754ed2c85f/medforge_reasoner/train_gspo.sh) supplies both `--beta 0.001` and `--beta 0`, and both `--vllm_mode colocate` and `--vllm_mode server`. Resolving the authors' historical configuration is necessary only if baseline reproduction is resumed. Our run must specify its own unambiguous settings.
- **Deferred:** upstream `setup_env.sh` installs an unpinned SWIFT checkout. Use and record our Bridge training environment; installing or reproducing upstream SWIFT is not required by this work order.
- Upstream SFT/RL validation ratios are `0.05`/`0.01`. These are reference settings, not an obligation to reproduce their trainer. Predeclare our validation policy, report any differences and count actual training/validation records.
- Training/inference paths must match the HF filenames explicitly; avoid inadvertently applying an adapter twice when choosing between merged weights and base-plus-adapter inference.
- The judge scripts expect `data/correct_samples.jsonl`; that file is absent from the inspected Git checkout. Exact Table 2 sample provenance remains unresolved. Judge code compares response/reference text; it is not itself a blinded clinician image review.

## Completion criteria

Report Bridge's complete eligible training exposure; full-test prediction coverage; per-regime accuracy/F1 with reconciled aggregation; checkpoint/config/environment hashes; validation selection; and any paper/code/data discrepancies. Preserve raw predictions so results can be rescored. Report Table 2 evaluation independently. Published MedForge scores remain reference values; a newly reproduced MedForge baseline is not a completion requirement.

Current status: official manifest inventory and a synthetic upstream-evaluator failure check completed. **No Bridge model inference, GPU training, raw-image verification or complete full-test evaluation has been performed here.** The node operator still needs to run the protocol; no remote connection has been established.
