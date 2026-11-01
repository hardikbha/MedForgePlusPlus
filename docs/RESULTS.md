# Saved MF-Clean test results

Snapshot: 2026-11-01. Percentages; balanced accuracy = (real recall + fake recall)/2.
All rows use test_core: 2,756 images in 1,008 source groups. These are existing experiments, not new training runs.
Different regimes and training histories must be compared within matched conditions. See [the review](CVPR_REVIEW.md).

| Metric file | Accuracy | Balanced accuracy [95% group CI] | Macro-F1 | Removal recall | mIoU | IoU@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| [metrics_test_bridgeold_cf](../results/metrics/metrics_test_bridgeold_cf.json) | 50.33 | 47.07 [45.92, 48.31] | 47.08 | 76.33 | 22.88 | 23.57 |
| [metrics_test_contsft_cf](../results/metrics/metrics_test_contsft_cf.json) | 73.84 | 70.68 [69.15, 72.30] | 71.11 | 87.14 | 53.03 | 56.35 |
| [metrics_test_contsft_cf_bal](../results/metrics/metrics_test_contsft_cf_bal.json) | 77.36 | 79.93 [78.57, 81.30] | 77.03 | 71.97 | 49.95 | 53.95 |
| [metrics_test_ct_image_1697](../results/metrics/metrics_test_ct_image_1697.json) | 74.56 | 77.81 [76.45, 79.29] | 74.37 | 67.72 | 46.99 | 50.63 |
| [metrics_test_ct_image_s7](../results/metrics/metrics_test_ct_image_s7.json) | 74.56 | 77.70 [76.27, 79.08] | 74.35 | 68.45 | 46.87 | 50.80 |
| [metrics_test_ct_twin_1697](../results/metrics/metrics_test_ct_twin_1697.json) | 74.06 | 77.74 [76.37, 79.11] | 73.92 | 65.53 | 46.48 | 50.17 |
| [metrics_test_ct_twin_s7](../results/metrics/metrics_test_ct_twin_s7.json) | 74.09 | 77.43 [76.01, 78.87] | 73.91 | 67.35 | 46.55 | 50.51 |
| [metrics_test_ctgrpo_cf_image_lr5](../results/metrics/metrics_test_ctgrpo_cf_image_lr5.json) | 76.09 | 77.92 [76.52, 79.39] | 75.60 | 71.12 | 48.74 | 52.29 |
| [metrics_test_ctgrpo_cf_twin_lr5](../results/metrics/metrics_test_ctgrpo_cf_twin_lr5.json) | 74.82 | 77.23 [75.85, 78.61] | 74.47 | 70.15 | 47.81 | 50.92 |
| [metrics_test_ctgrpo_id_image](../results/metrics/metrics_test_ctgrpo_id_image.json) | 80.44 | 82.71 [81.47, 83.96] | 80.07 | 68.08 | 53.80 | 58.12 |
| [metrics_test_ctgrpo_id_twin](../results/metrics/metrics_test_ctgrpo_id_twin.json) | 80.55 | 82.99 [81.76, 84.20] | 80.21 | 67.48 | 53.26 | 57.38 |
| [metrics_test_medforge](../results/metrics/metrics_test_medforge.json) | 59.80 | 58.61 [57.17, 60.01] | 58.09 | 64.20 | 32.03 | 35.98 |
| [metrics_test_qwen_zs](../results/metrics/metrics_test_qwen_zs.json) | 61.21 | 53.19 [51.94, 54.39] | 51.82 | 83.01 | 34.63 | 38.44 |
| [metrics_test_sft_cf_s17](../results/metrics/metrics_test_sft_cf_s17.json) | 70.28 | 75.33 [73.93, 76.73] | 70.28 | 58.13 | 42.46 | 46.45 |
| [metrics_test_sft_cf_s7](../results/metrics/metrics_test_sft_cf_s7.json) | 69.09 | 75.02 [73.59, 76.26] | 69.07 | 54.25 | 41.07 | 45.25 |
| [metrics_test_sft_id_s17](../results/metrics/metrics_test_sft_id_s17.json) | 80.44 | 82.21 [80.96, 83.37] | 79.96 | 64.56 | 52.66 | 56.18 |
| [metrics_test_sft_id_s7](../results/metrics/metrics_test_sft_id_s7.json) | 79.72 | 81.95 [80.67, 83.18] | 79.33 | 62.74 | 52.04 | 55.84 |
| [metrics_test_v3_sft_cf](../results/metrics/metrics_test_v3_sft_cf.json) | 70.75 | 76.23 [74.94, 77.57] | 70.75 | 57.28 | 42.61 | 46.62 |
| [metrics_test_v3_sft_id](../results/metrics/metrics_test_v3_sft_id.json) | 79.46 | 81.82 [80.57, 83.04] | 79.10 | 61.77 | 51.82 | 55.89 |
