### gen_test_ct_image_1697.jsonl

n=2756 (real 1008, implant 924, removal 824), groups=1008, twin pairs=1471, formats={'v3': 2756}

| metric | value [95% group-bootstrap CI] |
|---|---|
| accuracy | 74.56 [72.9, 76.2] |
| balanced acc (binary) | 77.81 [76.5, 79.3] |
| balanced acc (type-balanced) | 77.86 [76.5, 79.3] |
| macro-F1 (binary) | 74.37 [72.7, 75.9] |
| recall real | 89.88 [88.0, 91.8] |
| recall implant | 63.96 [60.3, 67.4] |
| recall removal | 67.72 [64.6, 70.9] |
| mIoU (fakes) | 46.99 [44.8, 49.1] |
| mIoU gated by verdict | 46.99 [44.8, 49.1] |
| IoU@0.5 (fakes) | 50.63 [48.1, 53.4] |
| twin consistency | 55.81 [53.1, 58.7] |
| acc, legacy (unparsed->real) | 74.56 [72.9, 76.2] |
| twin: fake flagged / twin cleared | 64.38 / 90.48 |
| parse rate / well-formed / v3-format | 100.00 / 100.00 / 100.00 |
| box given when 'deepfake' / box given when 'real' | 100.00 / 0.00 |
| evidence-verdict consistency (v3 rows) | 100.00 |
| rationale type: all / fakes | 53.12 [50.5, 55.9] / 31.92 |
| rationale location: all / fakes | 65.49 [63.6, 67.3] / 51.43 |
| rationale size: all / fakes | 63.79 [61.9, 65.7] / 48.74 |
| rationale intensity: all / fakes | -- / -- |

field GT source: {'derived': 2756}

| organ | n | acc | bal acc | rec real | rec implant | rec removal | mIoU | IoU@0.5 | twin (pairs) |
|---|---|---|---|---|---|---|---|---|---|
| brain-mri | 1060 | 68.1 [65.5, 70.6] | 72.1 [69.8, 74.0] | 86.9 | 52.4 | 62.8 | 36.4 | 39.3 | 47.6 (672) |
| chest-xray | 808 | 73.6 [70.8, 76.4] | 75.6 [72.9, 78.1] | 86.4 | 64.4 | 64.9 | 41.6 | 43.5 | 53.4 (476) |
| fundus | 888 | 83.1 [80.7, 85.7] | 87.0 [85.2, 88.9] | 97.9 | 74.3 | 79.1 | 63.2 | 69.0 | 76.5 (323) |

| generator | n | acc | bal acc | rec real | rec implant | rec removal | mIoU | IoU@0.5 | twin (pairs) |
|---|---|---|---|---|---|---|---|---|---|
| flux.1-dev | 175 | 37.1 [30.3, 44.6] | -- | -- | 32.3 | 43.4 | 20.1 | 21.1 | 31.7 (145) |
| gemini | 175 | 51.4 [44.6, 58.9] | -- | -- | 51.6 | 51.2 | 31.7 | 33.7 | 48.6 (146) |
| gpt | 175 | 59.4 [51.4, 66.3] | -- | -- | 57.6 | 61.8 | 43.2 | 48.0 | 45.1 (144) |
| qwen | 174 | 83.3 [77.6, 88.5] | -- | -- | 81.4 | 85.2 | 61.7 | 66.7 | 70.2 (151) |
| real | 1008 | 89.9 [88.0, 91.8] | -- | 89.9 | -- | -- | -- | -- | -- (0) |
| seeddream | 175 | 89.7 [85.1, 93.7] | -- | -- | 93.9 | 84.4 | 75.0 | 81.1 | 80.3 (147) |
| stable-diffusion-3.5-large | 174 | 69.5 [62.6, 76.4] | -- | -- | 66.7 | 72.4 | 44.9 | 48.9 | 55.6 (144) |
| stable-diffusion-3.5-medium | 175 | 48.6 [41.1, 55.4] | -- | -- | 47.3 | 50.0 | 28.7 | 30.3 | 41.8 (146) |
| stable-diffusion-inpainting | 175 | 74.9 [68.6, 81.1] | -- | -- | 69.6 | 80.7 | 56.8 | 61.1 | 64.7 (153) |
| stable-diffusion-xl-1.0-inpainting-0.1 | 175 | 66.9 [60.0, 73.7] | -- | -- | 62.8 | 71.6 | 44.3 | 47.4 | 51.7 (143) |
| step1x-edit | 175 | 76.6 [70.3, 82.9] | -- | -- | 80.7 | 72.8 | 63.7 | 68.0 | 66.4 (152) |

### gen_test_contsft_cf_bal.jsonl

n=2756 (real 1008, implant 924, removal 824), groups=1008, twin pairs=1471, formats={'v3': 2756}

| metric | value [95% group-bootstrap CI] |
|---|---|
| accuracy | 77.36 [75.8, 78.9] |
| balanced acc (binary) | 79.93 [78.6, 81.3] |
| balanced acc (type-balanced) | 79.97 [78.6, 81.3] |
| macro-F1 (binary) | 77.03 [75.5, 78.6] |
| recall real | 89.48 [87.6, 91.4] |
| recall implant | 68.94 [65.4, 72.6] |
| recall removal | 71.97 [68.9, 75.0] |
| mIoU (fakes) | 49.95 [47.7, 52.1] |
| mIoU gated by verdict | 49.95 [47.7, 52.1] |
| IoU@0.5 (fakes) | 53.95 [51.3, 56.6] |
| twin consistency | 59.62 [56.8, 62.7] |
| acc, legacy (unparsed->real) | 77.36 [75.8, 78.9] |
| twin: fake flagged / twin cleared | 69.27 / 89.06 |
| parse rate / well-formed / v3-format | 100.00 / 100.00 / 100.00 |
| box given when 'deepfake' / box given when 'real' | 100.00 / 0.00 |
| evidence-verdict consistency (v3 rows) | 100.00 |
| rationale type: all / fakes | 54.25 [51.4, 57.1] / 33.92 |
| rationale location: all / fakes | 70.57 [68.8, 72.4] / 59.67 |
| rationale size: all / fakes | 66.51 [64.6, 68.3] / 53.26 |
| rationale intensity: all / fakes | -- / -- |

field GT source: {'derived': 2756}

| organ | n | acc | bal acc | rec real | rec implant | rec removal | mIoU | IoU@0.5 | twin (pairs) |
|---|---|---|---|---|---|---|---|---|---|
| brain-mri | 1060 | 71.3 [68.7, 73.8] | 73.7 [71.4, 76.0] | 82.7 | 60.6 | 69.4 | 39.7 | 43.3 | 50.9 (672) |
| chest-xray | 808 | 76.6 [73.8, 79.3] | 78.6 [75.9, 81.2] | 90.1 | 66.7 | 67.6 | 44.5 | 47.5 | 58.6 (476) |
| fundus | 888 | 85.2 [82.8, 87.6] | 88.5 [86.8, 90.3] | 97.9 | 77.6 | 82.0 | 65.7 | 71.0 | 79.3 (323) |

| generator | n | acc | bal acc | rec real | rec implant | rec removal | mIoU | IoU@0.5 | twin (pairs) |
|---|---|---|---|---|---|---|---|---|---|
| flux.1-dev | 175 | 50.3 [43.4, 57.7] | -- | -- | 49.5 | 51.3 | 25.2 | 24.0 | 42.8 (145) |
| gemini | 175 | 55.4 [48.0, 63.4] | -- | -- | 54.7 | 56.2 | 33.7 | 37.1 | 49.3 (146) |
| gpt | 175 | 61.1 [53.7, 68.6] | -- | -- | 59.6 | 63.2 | 46.0 | 50.3 | 47.2 (144) |
| qwen | 174 | 85.1 [79.9, 90.2] | -- | -- | 81.4 | 88.6 | 64.2 | 72.4 | 74.2 (151) |
| real | 1008 | 89.5 [87.6, 91.4] | -- | 89.5 | -- | -- | -- | -- | -- (0) |
| seeddream | 175 | 89.7 [85.1, 93.7] | -- | -- | 92.9 | 85.7 | 76.4 | 82.3 | 78.9 (147) |
| stable-diffusion-3.5-large | 174 | 74.1 [67.8, 80.5] | -- | -- | 72.4 | 75.9 | 48.6 | 53.4 | 60.4 (144) |
| stable-diffusion-3.5-medium | 175 | 58.3 [50.9, 65.7] | -- | -- | 56.0 | 60.7 | 34.3 | 36.0 | 46.6 (146) |
| stable-diffusion-inpainting | 175 | 78.9 [72.6, 84.6] | -- | -- | 73.9 | 84.3 | 58.6 | 65.1 | 68.6 (153) |
| stable-diffusion-xl-1.0-inpainting-0.1 | 175 | 72.6 [66.3, 78.9] | -- | -- | 68.1 | 77.8 | 47.1 | 50.3 | 60.1 (143) |
| step1x-edit | 175 | 78.3 [72.6, 84.6] | -- | -- | 84.3 | 72.8 | 65.4 | 68.6 | 66.4 (152) |

#### paired: A=gen_test_ct_image_1697.jsonl  vs  B=gen_test_contsft_cf_bal.jsonl

common images 2756 (only A 0, only B 0), groups 1008, GT mismatches 0. Exact McNemar: b(A only correct)=54, c(B only correct)=131, p=1.438e-08

| metric | A | B | B-A [95% paired group-bootstrap CI] |
|---|---|---|---|
| acc | 74.56 | 77.36 | +2.79 [+1.92, +3.64] |
| bal_acc | 77.81 | 79.93 | +2.12 [+1.24, +2.97] |
| f1_macro | 74.37 | 77.03 | +2.67 [+1.80, +3.51] |
| rec_real | 89.88 | 89.48 | -0.40 [-1.79, +0.99] |
| rec_fake | 65.73 | 70.37 | +4.63 [+3.37, +6.00] |
| mIoU | 46.99 | 49.95 | +2.95 [+2.27, +3.67] |
| IoU@0.5 | 50.63 | 53.95 | +3.32 [+2.15, +4.47] |
| twin | 55.81 | 59.62 | +3.81 [+2.10, +5.35] |
