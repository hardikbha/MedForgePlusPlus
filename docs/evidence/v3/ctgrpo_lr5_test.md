| file | n | groups | acc [95% CI] | bal acc [95% CI] | macro-F1 | rec real | rec implant | rec removal | mIoU [95% CI] | IoU@0.5 | twin (pairs) | parse | acc legacy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gen_test_ctgrpo_cf_image_lr5 | 2756 | 1008 | 76.09 [74.7, 77.7] | 77.92 [76.5, 79.4] | 75.60 | 84.7 | 71.1 | 71.1 | 48.74 [46.6, 50.9] | 52.29 | 56.6 (1471) | 100.00 | 76.09 |
| gen_test_ctgrpo_cf_twin_lr5 | 2756 | 1008 | 74.82 [73.3, 76.3] | 77.23 [75.8, 78.6] | 74.47 | 86.2 | 66.6 | 70.1 | 47.81 [45.7, 50.0] | 50.92 | 55.7 (1471) | 100.00 | 74.82 |

#### paired: A=gen_test_ctgrpo_cf_twin_lr5.jsonl  vs  B=gen_test_ctgrpo_cf_image_lr5.jsonl

common images 2756 (only A 0, only B 0), groups 1008, GT mismatches 0. Exact McNemar: b(A only correct)=82, c(B only correct)=117, p=0.01573

| metric | A | B | B-A [95% paired group-bootstrap CI] |
|---|---|---|---|
| acc | 74.82 | 76.09 | +1.27 [+0.32, +2.26] |
| bal_acc | 77.23 | 77.92 | +0.69 [-0.27, +1.71] |
| f1_macro | 74.47 | 75.60 | +1.14 [+0.17, +2.12] |
| rec_real | 86.21 | 84.72 | -1.49 [-3.27, +0.20] |
| rec_fake | 68.25 | 71.11 | +2.86 [+1.47, +4.18] |
| mIoU | 47.81 | 48.74 | +0.93 [+0.07, +1.70] |
| IoU@0.5 | 50.92 | 52.29 | +1.37 [+0.06, +2.57] |
| twin | 55.74 | 56.63 | +0.88 [-0.91, +2.63] |

#### paired: A=gen_test_v3_sft_cf.jsonl  vs  B=gen_test_ctgrpo_cf_twin_lr5.jsonl

common images 2756 (only A 0, only B 0), groups 1008, GT mismatches 0. Exact McNemar: b(A only correct)=121, c(B only correct)=233, p=2.71e-09

| metric | A | B | B-A [95% paired group-bootstrap CI] |
|---|---|---|---|
| acc | 70.75 | 74.82 | +4.06 [+2.81, +5.36] |
| bal_acc | 76.23 | 77.23 | +1.00 [-0.13, +2.17] |
| f1_macro | 70.75 | 74.47 | +3.71 [+2.48, +5.01] |
| rec_real | 96.63 | 86.21 | -10.42 [-12.30, -8.43] |
| rec_fake | 55.84 | 68.25 | +12.41 [+10.64, +14.39] |
| mIoU | 42.61 | 47.81 | +5.20 [+4.32, +6.18] |
| IoU@0.5 | 46.62 | 50.92 | +4.29 [+3.03, +5.70] |
| twin | 51.67 | 55.74 | +4.08 [+1.54, +6.70] |

#### paired: A=gen_test_v3_sft_cf.jsonl  vs  B=gen_test_ctgrpo_cf_image_lr5.jsonl

common images 2756 (only A 0, only B 0), groups 1008, GT mismatches 0. Exact McNemar: b(A only correct)=121, c(B only correct)=268, p=6.759e-14

| metric | A | B | B-A [95% paired group-bootstrap CI] |
|---|---|---|---|
| acc | 70.75 | 76.09 | +5.33 [+4.02, +6.63] |
| bal_acc | 76.23 | 77.92 | +1.68 [+0.49, +2.91] |
| f1_macro | 70.75 | 75.60 | +4.85 [+3.59, +6.11] |
| rec_real | 96.63 | 84.72 | -11.90 [-13.89, -9.92] |
| rec_fake | 55.84 | 71.11 | +15.27 [+13.42, +17.07] |
| mIoU | 42.61 | 48.74 | +6.13 [+5.22, +7.11] |
| IoU@0.5 | 46.62 | 52.29 | +5.66 [+4.35, +7.05] |
| twin | 51.67 | 56.63 | +4.96 [+2.19, +7.53] |
