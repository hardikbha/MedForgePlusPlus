# First milestone: reproduce the original MedForge benchmark

User priority, 2026-11-03: run the full released training/evaluation setup before introducing Bridge changes or the kidney/colon extension. The earlier 2,994-image evaluation and MF-Clean's 2,756-image evaluation do not establish this reproduction.

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
2. **Make the original implementation reproducible.** Pin a compatible SWIFT/PyTorch/Transformers/vLLM environment, repair and test the released evaluator, and document the training-script conflicts below. Validate answer-label separation in inference: the reference assistant answer must not enter the model prompt.
3. **Run a full-checkpoint evaluation smoke test, then all 30,000 test examples.** Use the released MedForge-Reasoner checkpoint to verify the harness. A single released checkpoint does not establish all three regime baselines unless its training regime is documented. Record checkpoint hashes, generation settings, missing/unparsed cases and per-image predictions.
4. **Reproduce the original SFT → GSPO training baseline.** Use the complete eligible official pools, the original backbone and documented recipe. Keep SFT and RL populations separate. Account for the released scripts' internal validation carve-outs. Resolve paper/code differences before calling the run exact. Train independent ID, CF and CM checkpoints, then evaluate each on the unchanged full test.
5. **Evaluate Bridge against that baseline under matched data conditions.** Keep the same splits, exclusions, scorer and test population; disclose differences in adapters, optimization and compute. Select checkpoints/alpha using training-side validation only, never the test. Do not add kidney/colon at this stage.
6. **Reproduce Table 2 separately.** Recover the original 100-example reasoning sample, references, prompts and judge versions if possible. If the exact sample is unavailable, rerun all compared models on a documented common sample and label it a new matched evaluation. Detection and parsing scores cannot substitute for judge scores.
7. **Only then test novelty.** Add replay/RL/interpolation changes one at a time. Treat extra kidney/colon training data as a separately labeled augmentation condition, retaining the original test to measure its benefit. Clinical and clean-benchmark work remains valuable but must not silently alter this reproduction track.

Filtering both official training pools by the paper's exclusions yields the following inventory (these counts are not evidence of historical training):

| Regime | Eligible SFT rows | Eligible RL rows | Evaluation |
|---|---:|---:|---|
| In-Domain | 55,276 | 10,000 | Same full 30,000 test |
| Cross-Forgery: exclude implants | 33,887 | 6,987 | Same full 30,000 test |
| Cross-Model: exclude gemini, gpt, SD3.5-medium, SDXL-inpainting | 39,591 | 8,247 | Same full 30,000 test |

## Known launch blockers in the inspected upstream code

- [Detection evaluator](https://github.com/richardChenzhihui/ACL2026-MedForge/blob/f60505600d3de0959f4fcdc8bd1741754ed2c85f/evaluation/detection_eval/eval.py) calls `classify_image`, `extract_bbox`, `extract_model_name`, `compute_iou` and `visualize_result` without defining/importing them. A local synthetic 8×8 image check actually failed with `NameError: name 'classify_image' is not defined`. Restore intended implementations or document/test a minimal compatibility patch; do not silently substitute a different metric.
- [GSPO launcher](https://github.com/richardChenzhihui/ACL2026-MedForge/blob/f60505600d3de0959f4fcdc8bd1741754ed2c85f/medforge_reasoner/train_gspo.sh) supplies both `--beta 0.001` and `--beta 0`, and both `--vllm_mode colocate` and `--vllm_mode server`. Parser resolution and the intended published configuration need to be established. A chosen paper-faithful run should be labeled as such if the actual historical configuration is unknown.
- `setup_env.sh` installs an unpinned SWIFT checkout; requirements are unpinned. Do not assume the current dependency versions reproduce historical behavior.
- SFT uses `--split_dataset_ratio 0.05`, and RL uses `0.01`. Count the actual optimizer-training and validation records after these carve-outs. Record seeds and framework behavior.
- Training/inference scripts contain placeholder paths. Match the HF filenames explicitly; avoid inadvertently applying an adapter twice when choosing between merged weights and base-plus-adapter inference.
- The judge scripts expect `data/correct_samples.jsonl`; that file is absent from the inspected Git checkout. Exact Table 2 sample provenance remains unresolved. Judge code compares response/reference text; it is not itself a blinded clinician image review.

## Completion criteria

Report complete eligible training exposure; full-test prediction coverage; per-regime accuracy/F1 with reconciled aggregation; checkpoint/config/environment hashes; validation selection; and any paper/code/data discrepancies. Preserve raw predictions so results can be rescored. Report Table 2 evaluation independently.

Current status: official manifest inventory and a synthetic upstream-evaluator failure check completed. **No model inference, GPU training, raw-image verification or complete benchmark reproduction has been performed here.** The node operator still needs to run the protocol; no remote connection has been established.
