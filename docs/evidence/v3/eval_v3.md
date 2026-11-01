# eval_v3 / stats_v3: evaluation component for the v3 rebuild

Date: 2026-10-11. CPU only: no GPU was used. The GPU path was run end to end on CPU with a tiny random Qwen3-VL.
Every number below comes from a command run in this session. The commands are listed in section 6.

## 0. Bottom line

1. **`eval_v3.py`** is the GPU evaluator. It mirrors `circe/eval_medforge.py`: same loading, batched greedy decoding and flags. It supports `--prompt v3|legacy|legacy_v2|medforge`, where medforge defaults to max_new 2048. It saves every GT field per image and can resume and shard. It scores through stats_v3.
2. **`stats_v3.py`** is the CPU scorer. It reads new rows and old `eval_medforge` rows. It reports accuracy, binary macro-F1, balanced accuracy, per-class recall, organ and generator tables, mIoU / IoU@0.5 over fakes, twin consistency, per-field rationale accuracy, and parse and well-formed rates. CIs are group-bootstrap (1,000 resamples). Paired comparisons use exact McNemar plus a paired group-bootstrap CI of B − A. All of this is unit-tested on synthetic data with known answers (`--self_test` PASS).
3. **Old numbers reproduced exactly.** Under the legacy convention (unparsed → "real"), stats_v3 reproduces acc, macro-F1, mIoU, IoU@0.5, fake recall and parse rate of all 21 MedForge-90K `metrics_*.json` files in `v3_select/`. The largest absolute difference is 1.1e-16.
4. **Re-scored findings (old protocol):**
   - The interpolation, replay and GRPO stages have no measurable verdict effect. Every exact McNemar p is ≥ 0.118 within CF, CM and ID.
   - GRPO does raise mIoU: CF +2.65 pp [+1.94, +3.41], CM +1.88 [+1.23, +2.51], ID +1.95 [+1.40, +2.57].
   - Zero-shot on MF++ is an organ-constant predictor. Twin consistency is 3.2% for CF and 7.1% for ID over 253 pairs. Balanced accuracy is 48.5 and 53.6.
5. **Two issues for the lead (MF-Clean, not my files):**
   - (a) **Fundus fakes in the cores are missing their twin.** `twin` is null for 115/894 select fakes and 277/1748 test fakes, all fundus. Twin consistency on fundus is therefore computed over fewer pairs (section 4).
   - (b) **select_core is small:** 1,420 images, 526 groups. At about 90% accuracy, the group-bootstrap 95% CI half-width is ±1.5 to 2.1 pp on select_core and ±1.1 to 1.4 pp on test_core (simulation, section 4). A +0.5 pp gate cannot be resolved on select_core.
6. **No counterfactual results yet.** `v3_counterfactual/results/` was empty when I ran this. `cf_norm_cf_s1replay` was at 424/2994, 0.22 samples/s. `bash ~/CIRCE/v3/rescore_existing.sh` re-scores them once they exist, and adds the norm-vs-pasteback paired rows, paired by source path. That path was tested on simulated old-format files.

## 1. Files (all new, all under ~/CIRCE/v3/)

| file | role |
|---|---|
| `eval_v3.py` | GPU eval (CPU test modes: `--cpu` for a tiny model, `--dry_run oracle/const_real/const_fake/garbage`, `--self_test`) |
| `stats_v3.py` | CPU scorer, batch re-scorer, paired comparisons, `--self_test` |
| `legacy_meta_v3.py` | builds `cache/mf90k_legacy_meta.json` (group / organ / generator / source of every MF-90K image) so old files can be scored by group and twin |
| `rescore_existing.sh` | re-scores every existing old gen file plus the paired rows → `reports/rescore_existing.{md,json}` (and `rescore_counterfactual.*` once norm/pasteback gens exist) |
| `reports/rescore_existing.md/.json` | full re-scored output (the JSON has per-organ, per-generator and per-class tables with CIs for every file) |

Usage, for scheduler task lines:
```
# v3 checkpoint on MF-Clean (use the *_fields.jsonl copy: identical rows + GT evidence fields, verified equal)
cd ${MEDFORGE_WORKSPACE}/CIRCE/v3 && ${MEDFORGE_WORKSPACE}/.conda/envs/Qwen3.5/bin/python -u eval_v3.py \
  --base ${MEDFORGE_WORKSPACE}/models/Qwen3-VL-8B-Instruct --adapter <ckpt> --prompt v3 \
  --eval ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/mfclean/v3data/test_core_fields.jsonl \
  --save_gen <dir>/gen_<name>.jsonl --out <dir>/metrics_<name>.json
# old CIRCE checkpoints: --prompt legacy ; MedForge-Reasoner: --base ~/models_external/MedForge-Reasoner --prompt medforge
# compare two runs on the same images
~/.conda/envs/aiface/bin/python stats_v3.py --gen gen_A.jsonl --gen2 gen_B.jsonl --out_json AB.json --out_md AB.md
```

## 2. Metric conventions (also in the stats_v3.py README)

- **GT and classes.** The GT verdict is deepfake iff type ≠ real. Classes: real; implant (edit or inject); removal (remove).
- **Unparsed verdict.** It is a third output: wrong for every class and a positive for neither. `acc_legacy` and `f1_macro_legacy` use the old rule (unparsed → real) and exist only for comparability.
- **macro-F1** is the unweighted mean of F1(real) and F1(deepfake) on the binary verdict.
- **Balanced accuracy (binary)** = (recall_real + recall_deepfake) / 2. A constant predictor scores 0.5.
  - `bal_acc_typebal` = (recall_real + mean(recall_implant, recall_removal)) / 2. It is also 0.5 for any constant predictor.
- **Per-class recall** is the fraction of a class that gets the correct binary verdict.
- **mIoU and IoU@0.5** are computed over fakes with a GT box. A missing box scores 0, and IoU@0.5 means IoU ≥ 0.5.
  - Primary mIoU uses the parsed box whatever the verdict (old convention). `mIoU_gated` scores 0 unless the verdict is deepfake.
  - The old `eval_medforge` also averaged over *reals* that carry a box (MF++ real-tumour scans, 200 of its 600 boxed rows). That is the only reason the old MF++ metrics files differ in mIoU (0.0238 vs 0.0243 and 0.0920 vs 0.1198).
- **Twin consistency.** The pairs are (fake, its authentic-source real), counted only when both are in the file. A pair scores 1 iff the fake is called deepfake AND the twin is called real. The flagged and cleared rates on the pairs are reported too. The twin source:
  - v3 rows: the `twin` field.
  - old MF-90K rows: the source scan, or a verified duplicate of it, found through `legacy_meta_v3`.
  - old MF++ rows: kidney_real_tumor→kidney_removal, polyp_real→polyp_removal and polyp_real→colon_inject, matched by stem.
- **Rationale accuracy** is an exact match per field (type, location, size, intensity), over all images and over fakes.
  - The GT comes from `--fields`, or from the row's own `fields`. Without either, type, location and size are derived from the class and GT box (`prompts_v3.location_of/size_of`); intensity needs real GT.
  - A missing field counts as wrong.
  - Also reported: evidence-verdict consistency on v3 rows (type ≠ none or a box present ⇔ verdict deepfake).
- **CIs** are percentile bootstraps: 1,000 resamples of GROUPS with replacement, seed 2027. A twin pair belongs to its fake's group.
  - Group = forgery + source scan + patient + verified duplicate. The old-file groups equal the MF-Clean group ids on all 50,256 manifest rows (verified).
  - MF++ groups: KiTS case id for kidney, Kvasir stem for polyp/colon.
- **Paired comparison** runs on common images (`--pair_key key_orig` pairs counterfactual copies by source path).
  - Exact two-sided McNemar: p = min(1, 2·P[X ≤ min(b,c)]), X ~ Bin(b+c, ½).
  - The B − A CI uses the same resampled groups for both files.

## 3. Verification

| check | command (section 6) | result |
|---|---|---|
| stats_v3 unit tests | `stats_v3.py --self_test` | PASS. Perfect predictor: all metrics 1. **Constant "real": bal acc 0.5, twin 0, F1 0.25 (= n_r/(2n_r+n_f)).** **Constant "deepfake": bal acc 0.5, twin 0, F1 0.4.** Unparsed: acc 0, legacy acc = real share. Hand-computed confusion: acc 9/12, F1 (6/9+12/15)/2, mIoU (2·0.5+4/3)/8, IoU@0.5 2/8, twin 4/8. Twin pairs counted only if both images are present. Rationale fields (intensity 0.5 on fakes). The v3, legacy CIRCE and MedForge-Reasoner parsers. McNemar equals scipy.binomtest on 200 random (b,c), and (3,11) → 0.0574. **The bootstrap resamples groups:** 2 groups × 50 images (one all right, one all wrong) gives CI [0, 1]. With 2,000 singleton groups the CI width is 0.0345 vs analytic 0.0355. Paired diff (identical files → CI [0,0], b=c=0, p=1). The per-generator table scores twins that sit outside the generator subset. Legacy resolver (MF-90K and MF++). |
| eval_v3 unit tests | `eval_v3.py --self_test` (aiface and Qwen3.5 envs) | PASS. The prompt strings equal `eval_medforge` (legacy) and `eval_official_medforge_fair` (medforge). Dry-run oracle acc 1, twin 1. Constant predictors bal acc 0.5, twin 0. Garbage parse 0. Resume appends 7 → 18 rows. Shards 0/2 and 1/2 partition the rows. Every GT field is saved. |
| GPU code path on CPU | `eval_v3.py --cpu --base tiny_qwen3vl --adapter tiny_lora` (20M-param random Qwen3-VL + random LoRA, 4 MF-90K images re-encoded as single-channel L) for v3, legacy and medforge | Ran end to end with batch 2 and left padding. The PEFT adapter loads. The prompt is 1,301 tokens (image grid 64×64 → 1,024 image tokens). Random outputs → parse 0, `truncated`=True and n_tokens=12 recorded correctly. |
| MF-Clean structure | dry runs on `mfclean/test_core.jsonl` | oracle: every metric 1.0 (1,471 pairs, 1,008 groups). const_real: acc 0.3657, bal acc 0.5, F1 0.2678, twin 0. const_fake: acc 0.6343, bal acc 0.5, F1 0.3881, twin 0. With `test_core_fields.jsonl` the oracle gets type/location/size/intensity 1.0. A default-intensity oracle gets intensity 0.556 on fakes = (584+388)/1748, as expected from the GT counts. |
| old metrics reproduced | legacy convention vs `v3_select/metrics_*.json` | 21 MF-90K files: max abs diff 1.1e-16 over acc, F1, mIoU, IoU@0.5, fake recall and parse rate. The 2 MF++ files differ only in mIoU and IoU@0.5 (boxed reals, section 2). |
| vs audit `sel_stats.py` | McNemar | Identical for s1base (15/15), s2grpo (12/11) and α=0.5 (6/10). It differs by one discordant image for α=0.1/0.2/0.3 (ours b=4/6/7 vs audit 3/5/6). The cause is one degenerate output per file with no verdict sentence (ends in a broken `x1="204" y1="204 alt="deepfake region">`). The audit's loose parser ("fake" anywhere) counts it as deepfake; `eval_medforge` and stats_v3 count it as unparsed. **The α=0.1 p moves from 0.057 to 0.118.** |
| vs audit `twin_consistency.py` | MF++ zero-shot | CF s1replay: 8/253 pairs in both scripts. ID s1replay: ours 18/253 vs audit 20/253. The single unparsed image is `polyp_real/cju8cattbsivm0818p446wgel.png`, the twin of 2 fakes. The audit maps unparsed → real. |

## 4. Findings

### 4.1 Re-scored table: all existing old generation files (95% group-bootstrap CIs)

Files: `~/AAAI/MedForge++/v3_select/gen_*.jsonl` (23 files). `v3_counterfactual/results/` was empty. Old protocol: these select/test splits are NOT source-disjoint, so the CIs describe sampling noise and not leakage.

| file | n | groups | acc [95% CI] | bal acc [95% CI] | macro-F1 | rec real | rec implant | rec removal | mIoU [95% CI] | IoU@0.5 | twin (pairs) | parse | acc legacy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gen_sel_cf_a0.10 | 3000 | 2171 | 93.93 [92.8, 95.0] | 95.20 [94.3, 96.0] | 93.44 | 99.0 | 85.4 | 97.4 | 22.38 [21.0, 23.7] | 18.05 | 86.4 (103) | 100.0 | 93.93 |
| gen_sel_cf_a0.20 | 3000 | 2171 | 93.73 [92.6, 94.8] | 95.05 [94.2, 95.9] | 93.23 | 99.0 | 85.3 | 96.9 | 22.84 [21.5, 24.2] | 18.10 | 86.4 (103) | 100.0 | 93.73 |
| gen_sel_cf_a0.30 | 3000 | 2171 | 93.77 [92.6, 94.9] | 95.05 [94.2, 95.9] | 93.26 | 98.9 | 85.1 | 97.3 | 23.40 [22.0, 24.7] | 19.00 | 86.4 (103) | 100.0 | 93.77 |
| gen_sel_cf_a0.50 | 3000 | 2171 | 93.83 [92.8, 94.9] | 95.12 [94.3, 96.0] | 93.32 | 99.0 | 85.2 | 97.3 | 23.70 [22.4, 25.1] | 18.90 | 85.4 (103) | 100.0 | 93.83 |
| gen_sel_cf_s1base | 3000 | 2171 | 93.70 [92.6, 94.7] | 95.00 [94.2, 95.8] | 93.17 | 98.9 | 84.9 | 97.3 | 21.95 [20.7, 23.3] | 17.60 | 85.4 (103) | 100.0 | 93.70 |
| gen_sel_cf_s1replay | 3000 | 2171 | 93.70 [92.6, 94.8] | 95.03 [94.2, 95.9] | 93.18 | 99.0 | 85.1 | 97.0 | 21.89 [20.5, 23.2] | 17.95 | 87.4 (103) | 100.0 | 93.70 |
| gen_sel_cf_s2grpo | 3000 | 2171 | 93.67 [92.6, 94.8] | 94.95 [94.1, 95.8] | 93.14 | 98.8 | 84.7 | 97.5 | 24.54 [23.3, 25.9] | 19.60 | 86.4 (103) | 100.0 | 93.67 |
| gen_sel_cm_a0.10 | 3000 | 2171 | 94.07 [93.2, 94.9] | 95.38 [94.7, 96.0] | 93.59 | 99.3 | 92.8 | 90.1 | 28.64 [27.2, 30.1] | 24.90 | 95.1 (103) | 100.0 | 94.07 |
| gen_sel_cm_a0.15 | 3000 | 2171 | 94.13 [93.3, 95.0] | 95.43 [94.7, 96.1] | 93.66 | 99.3 | 92.9 | 90.2 | 28.50 [27.0, 30.0] | 24.90 | 95.1 (103) | 100.0 | 94.13 |
| gen_sel_cm_a0.20 | 3000 | 2171 | 94.10 [93.3, 95.0] | 95.40 [94.7, 96.0] | 93.62 | 99.3 | 92.8 | 90.2 | 28.85 [27.3, 30.3] | 25.10 | 95.1 (103) | 100.0 | 94.10 |
| gen_sel_cm_a0.30 | 3000 | 2171 | 94.07 [93.2, 94.9] | 95.38 [94.7, 96.0] | 93.59 | 99.3 | 92.6 | 90.3 | 28.85 [27.4, 30.3] | 24.90 | 95.1 (103) | 100.0 | 94.07 |
| gen_sel_cm_a0.50 | 3000 | 2171 | 94.20 [93.4, 95.0] | 95.50 [94.8, 96.2] | 93.73 | 99.4 | 92.8 | 90.4 | 29.23 [27.8, 30.7] | 25.35 | 95.1 (103) | 100.0 | 94.20 |
| gen_sel_cm_s1replay | 3000 | 2171 | 94.13 [93.3, 95.0] | 95.45 [94.8, 96.1] | 93.66 | 99.4 | 92.9 | 90.1 | 28.31 [26.9, 29.7] | 24.95 | 95.1 (103) | 100.0 | 94.13 |
| gen_sel_cm_s2grpo | 3000 | 2171 | 94.23 [93.4, 95.1] | 95.47 [94.8, 96.2] | 93.76 | 99.2 | 93.0 | 90.5 | 30.19 [28.8, 31.7] | 26.70 | 95.1 (103) | 100.0 | 94.23 |
| gen_sel_id_a0.10 | 3000 | 2171 | 99.10 [98.7, 99.4] | 99.08 [98.7, 99.5] | 98.99 | 99.0 | 99.5 | 98.8 | 31.50 [30.2, 32.9] | 27.65 | 99.0 (103) | 100.0 | 99.10 |
| gen_sel_id_a0.20 | 3000 | 2171 | 99.03 [98.7, 99.4] | 99.00 [98.6, 99.4] | 98.91 | 98.9 | 99.5 | 98.7 | 31.74 [30.4, 33.2] | 27.80 | 99.0 (103) | 100.0 | 99.03 |
| gen_sel_id_a0.30 | 3000 | 2171 | 99.07 [98.7, 99.4] | 99.05 [98.6, 99.4] | 98.95 | 99.0 | 99.4 | 98.8 | 32.03 [30.7, 33.4] | 28.15 | 99.0 (103) | 100.0 | 99.07 |
| gen_sel_id_a0.50 | 3000 | 2171 | 99.10 [98.7, 99.4] | 99.08 [98.7, 99.5] | 98.99 | 99.0 | 99.5 | 98.8 | 32.48 [31.1, 33.9] | 28.85 | 99.0 (103) | 100.0 | 99.10 |
| gen_sel_id_s1replay | 3000 | 2171 | 99.07 [98.7, 99.4] | 99.05 [98.6, 99.4] | 98.95 | 99.0 | 99.4 | 98.8 | 31.21 [29.9, 32.6] | 27.30 | 99.0 (103) | 100.0 | 99.07 |
| gen_sel_id_s2grpo | 3000 | 2171 | 99.00 [98.6, 99.4] | 99.00 [98.6, 99.4] | 98.88 | 99.0 | 99.5 | 98.5 | 33.16 [31.9, 34.6] | 29.55 | 99.0 (103) | 100.0 | 99.00 |
| gen_test_cf_s1replay | 2994 | 2166 | 94.05 [93.0, 95.1] | 95.30 [94.5, 96.1] | 93.54 | 99.0 | 85.7 | 97.5 | 21.73 [20.5, 23.0] | 17.55 | 80.9 (94) | 100.0 | 94.05 |
| gen_zs_pfp_cf_s1replay | 800 | 396 | 48.50 [45.6, 51.8] | 48.50 [45.6, 51.9] | 48.49 | 47.5 | 48.5 | 50.5 | 2.38 [1.3, 3.5] | 1.50 | 3.2 (253) | 100.0 | 48.50 |
| gen_zs_pfp_id_s1replay | 800 | 396 | 53.62 [50.7, 56.8] | 53.62 [51.2, 56.0] | 49.97 | 26.5 | 88.0 | 73.5 | 9.20 [7.0, 11.4] | 8.75 | 7.1 (253) | 99.9 | 53.75 |

Per-organ breakdown (from `rescore_existing.json`):

**gen_sel_cf_s1replay** (twin pairs: fake flagged / twin cleared = 88.35 / 99.03)

| organ | n | acc | bal acc | rec real | rec implant | rec removal | mIoU | IoU@0.5 | twin (pairs) |
|---|---|---|---|---|---|---|---|---|---|
| brain-mri | 1014 | 85.4 [82.8, 88.1] | 88.6 [86.6, 90.6] | 97.9 | 60.9 | 97.6 | 24.0 | 21.5 | 70.7 (41) |
| chest-xray | 993 | 97.5 [96.6, 98.4] | 97.9 [97.1, 98.7] | 99.1 | 96.7 | 96.7 | 17.6 | 13.3 | 100.0 (22) |
| fundus | 993 | 98.4 [97.6, 99.1] | 98.8 [98.2, 99.3] | 100.0 | 98.5 | 96.7 | 23.9 | 18.9 | 97.5 (40) |

**gen_test_cf_s1replay** (twin pairs: fake flagged / twin cleared = 80.85 / 100.00)

| organ | n | acc | bal acc | rec real | rec implant | rec removal | mIoU | IoU@0.5 | twin (pairs) |
|---|---|---|---|---|---|---|---|---|---|
| brain-mri | 1006 | 86.1 [83.4, 88.6] | 89.2 [87.2, 91.2] | 98.5 | 61.5 | 97.7 | 22.9 | 20.3 | 71.4 (63) |
| chest-xray | 1060 | 97.6 [96.5, 98.6] | 98.1 [97.2, 98.9] | 99.4 | 96.8 | 96.8 | 18.2 | 14.1 | 100.0 (10) |
| fundus | 928 | 98.6 [97.8, 99.4] | 98.7 [98.0, 99.4] | 99.1 | 98.6 | 98.1 | 24.7 | 18.6 | 100.0 (21) |

**gen_zs_pfp_cf_s1replay** (twin pairs: fake flagged / twin cleared = 41.50 / 56.92)

| organ | n | acc | bal acc | rec real | rec implant | rec removal | mIoU | IoU@0.5 | twin (pairs) |
|---|---|---|---|---|---|---|---|---|---|
| colonoscopy | 400 | 48.8 [44.9, 52.4] | 48.8 [46.3, 51.2] | 89.0 | 7.0 | 10.0 | 2.6 | 2.5 | 3.9 (153) |
| kidney-ct | 400 | 48.2 [43.8, 53.9] | 48.3 [45.8, 50.9] | 6.0 | 90.0 | 91.0 | 2.1 | 0.5 | 2.0 (100) |

**gen_zs_pfp_id_s1replay** (twin pairs: fake flagged / twin cleared = 75.89 / 20.95)

| organ | n | acc | bal acc | rec real | rec implant | rec removal | mIoU | IoU@0.5 | twin (pairs) |
|---|---|---|---|---|---|---|---|---|---|
| colonoscopy | 400 | 50.0 [46.1, 54.0] | 50.0 [46.2, 53.9] | 29.0 | 77.0 | 65.0 | 16.2 | 15.5 | 8.5 (153) |
| kidney-ct | 400 | 57.2 [52.8, 63.3] | 57.2 [54.6, 60.5] | 24.0 | 99.0 | 82.0 | 2.2 | 2.0 | 5.0 (100) |

### 4.2 Paired comparisons (exact McNemar + paired group-bootstrap CI of B − A)

| A (reference) | B | common | b (A only) | c (B only) | exact McNemar p | acc B-A [95% CI] (pp) | bal acc B-A [95% CI] (pp) | mIoU B-A [95% CI] (pp) |
|---|---|---|---|---|---|---|---|---|
| sel_cf_s1replay | sel_cf_a0.10 | 3000 | 4 | 11 | 0.118 | +0.23 [+0.00, +0.49] | +0.17 [-0.03, +0.40] | +0.49 [+0.10, +0.91] |
| sel_cf_s1replay | sel_cf_a0.20 | 3000 | 6 | 7 | 1 | +0.03 [-0.20, +0.27] | +0.02 [-0.18, +0.25] | +0.95 [+0.52, +1.41] |
| sel_cf_s1replay | sel_cf_a0.30 | 3000 | 7 | 9 | 0.804 | +0.07 [-0.19, +0.33] | +0.02 [-0.18, +0.23] | +1.51 [+1.00, +2.06] |
| sel_cf_s1replay | sel_cf_a0.50 | 3000 | 6 | 10 | 0.454 | +0.13 [-0.10, +0.40] | +0.10 [-0.12, +0.33] | +1.81 [+1.29, +2.39] |
| sel_cf_s1replay | sel_cf_s1base | 3000 | 15 | 15 | 1 | +0.00 [-0.34, +0.34] | -0.03 [-0.32, +0.27] | +0.06 [-0.47, +0.54] |
| sel_cf_s1replay | sel_cf_s2grpo | 3000 | 12 | 11 | 1 | -0.03 [-0.34, +0.27] | -0.08 [-0.34, +0.21] | +2.65 [+1.94, +3.41] |
| sel_cm_s1replay | sel_cm_a0.10 | 3000 | 2 | 0 | 0.5 | -0.07 [-0.17, +0.00] | -0.08 [-0.20, +0.00] | +0.33 [+0.05, +0.64] |
| sel_cm_s1replay | sel_cm_a0.15 | 3000 | 2 | 2 | 1 | +0.00 [-0.13, +0.13] | -0.02 [-0.16, +0.10] | +0.19 [-0.10, +0.51] |
| sel_cm_s1replay | sel_cm_a0.20 | 3000 | 3 | 2 | 1 | -0.03 [-0.20, +0.10] | -0.05 [-0.22, +0.08] | +0.54 [+0.20, +0.90] |
| sel_cm_s1replay | sel_cm_a0.30 | 3000 | 4 | 2 | 0.688 | -0.07 [-0.23, +0.10] | -0.08 [-0.23, +0.05] | +0.54 [+0.15, +0.94] |
| sel_cm_s1replay | sel_cm_a0.50 | 3000 | 3 | 5 | 0.727 | +0.07 [-0.10, +0.26] | +0.05 [-0.08, +0.20] | +0.92 [+0.43, +1.44] |
| sel_cm_s1replay | sel_cm_s2grpo | 3000 | 7 | 10 | 0.629 | +0.10 [-0.17, +0.40] | +0.02 [-0.23, +0.27] | +1.88 [+1.23, +2.51] |
| sel_id_s1replay | sel_id_a0.10 | 3000 | 0 | 1 | 1 | +0.03 [+0.00, +0.10] | +0.03 [+0.00, +0.08] | +0.30 [-0.01, +0.65] |
| sel_id_s1replay | sel_id_a0.20 | 3000 | 2 | 1 | 1 | -0.03 [-0.16, +0.07] | -0.05 [-0.18, +0.05] | +0.54 [+0.19, +0.91] |
| sel_id_s1replay | sel_id_a0.30 | 3000 | 1 | 1 | 1 | +0.00 [-0.10, +0.10] | +0.00 [-0.15, +0.15] | +0.82 [+0.45, +1.27] |
| sel_id_s1replay | sel_id_a0.50 | 3000 | 0 | 1 | 1 | +0.03 [+0.00, +0.10] | +0.03 [+0.00, +0.08] | +1.27 [+0.82, +1.77] |
| sel_id_s1replay | sel_id_s2grpo | 3000 | 5 | 3 | 0.727 | -0.07 [-0.27, +0.13] | -0.05 [-0.20, +0.10] | +1.95 [+1.40, +2.57] |
| zs_pfp_cf_s1replay | zs_pfp_id_s1replay | 800 | 146 | 187 | 0.0282 | +5.13 [+1.93, +8.49] | +5.13 [+1.71, +8.40] | +6.82 [+4.51, +9.31] |

What this shows:
- **Verdicts:** no checkpoint differs from its regime's s1replay SFT; every p ≥ 0.118.
- **Boxes:** GRPO and interpolation improve localization measurably (CF GRPO mIoU +2.65 pp, CI above 0).
- **Zero-shot MF++:** the ID model is significantly better than the CF model (p = 0.028). It is still near chance and organ-constant: kidney-ct real recall is 24% vs removal 82%, and on colonoscopy CF clears 89% of reals but flags only 7–10% of fakes.

### 4.3 MF-Clean cores: what eval will see (`build_mfclean.py` outputs, rc=0)

| core | images | real / implant / removal | groups | fakes with twin in file | fakes with `twin` = null |
|---|---|---|---|---|---|
| select_core | 1,420 | 526 / 460 / 434 | 526 | 779 / 894 | 115 (all fundus) |
| test_core | 2,756 | 1,008 / 924 / 824 | 1,008 | 1,471 / 1,748 | 277 (all fundus) |

- **Missing fundus twins.** The cause, read from `build_mfclean.py`: a core keeps only `real[:1]` per group. Fundus groups merge both eyes of a patient, so a fake whose source is the *other* eye loses its twin, because that real is never materialised.
  - Twin consistency stays well defined, since only pairs present are counted. But fundus twin numbers rest on fewer pairs.
  - If full twin coverage is wanted, the builder should keep every real that is some core fake's twin.
- **Resolution.** Simulation: each image right with p = 0.9 ("indep"), or half the images tied to a 10%-bad group ("group").
  - 95% CI half-width, acc / bal acc / twin: select_core ±1.53 / 1.51 / 3.20 pp (indep) and ±2.09 / 2.05 / 3.72 pp (group); test_core ±1.06 / 1.13 / 2.62 and ±1.43 / 1.38 / 2.94 pp.
  - Gate method decisions on paired McNemar over the same images, not on overlapping CIs, and consider pooling select_core over seeds.
- **Old vs new protocol.** The old selection split has only 103 twin pairs out of 2,000 fakes, and the old CF test split has 94 of 2,000. The MF-Clean cores are twin-structured: 779 and 1,471 pairs.

## 5. Known limitations

- eval_v3 was never run on a GPU or with real weights; that was not allowed here. The CPU tiny-model run exercises the same code path (processor, chat template, batched left-padded `generate`, PEFT, decoding, EOS and truncation bookkeeping), but it cannot show throughput or bf16/SDPA behaviour. The first real run should be checked against `eval_medforge.py` on a few hundred images: same checkpoint, `--prompt legacy`. The predictions should match exactly, since decoding is the same greedy path.
- Twin consistency for old files depends on `cache/mf90k_legacy_meta.json`. Rebuild it with `legacy_meta_v3.py` if `dataset_mapping.json` or the audit hash table changes. MF++ groups and twins come from filename rules that mirror the audit's `twin_consistency.py`.
- The MedForge-Reasoner box coordinate frame was not re-verified. Boxes are scored as parsed (`x1="…"` syntax). The old fair-eval mIoU (0.306) is in line with MedForge's published 31.55.
- The percentile bootstrap is coarse when discordant pairs are few (e.g. CF α=0.1: the CI lower bound is exactly 0.00 while McNemar p = 0.118). Use McNemar as the significance test and the CI as the effect size.
- Rationale accuracy is meaningful only for v3-format outputs. Legacy outputs have no evidence fields, so their field accuracy is reported but trivially low.

## 6. Commands run (all CPU, `nice`, aiface env unless noted; NUMPY_MADVISE_HUGEPAGE=0)

```
python legacy_meta_v3.py                                   # 95,276 MF90K items, 24,944 groups, 17 s
python stats_v3.py --self_test                              # PASS
python eval_v3.py --self_test                               # PASS (also with ~/.conda/envs/Qwen3.5/bin/python)
# tiny model (scratchpad): make_tiny.py (20.08M params), make_tiny_lora.py, make_tiny_eval.py, then for P in v3 legacy medforge:
CUDA_VISIBLE_DEVICES="" ~/.conda/envs/Qwen3.5/bin/python eval_v3.py --cpu --base tiny_qwen3vl --adapter tiny_lora \
   --eval tiny_eval.jsonl --prompt $P --batch 2 --max_new 12 --B 50 --save_gen tiny_gen_$P.jsonl --out tiny_metrics_$P.json
python eval_v3.py --eval ~/AAAI/MedForge++/mfclean/test_core.jsonl --dry_run {oracle,const_real,const_fake} ...
python eval_v3.py --eval ~/AAAI/MedForge++/mfclean/v3data/test_core_fields.jsonl --dry_run {oracle,const_real} ...
python stats_v3.py --gen dry_test_oracle.jsonl --fields ~/AAAI/MedForge++/mfclean/v3data/test_core_fields.jsonl ...
inline: legacy-convention reproduction of v3_select/metrics_*.json; legacy group == MF-Clean group on manifest.jsonl;
        missing-twin census of the cores; CI-resolution simulation on select_core/test_core;
        simulated old-format norm/pasteback gens -> stats_v3.py --rescore ... --pair_key key_orig
bash rescore_existing.sh                                    # 23 files scored, 18 paired rows, ~13 s
python ~/CIRCE/v3_audit/scripts/reviewer/twin_consistency.py v3_select/gen_zs_pfp_{cf,id}_s1replay.jsonl   # cross-check
```
