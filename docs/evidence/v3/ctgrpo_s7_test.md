| file | n | groups | acc [95% CI] | bal acc [95% CI] | macro-F1 | rec real | rec implant | rec removal | mIoU [95% CI] | IoU@0.5 | twin (pairs) | parse | acc legacy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gen_test_ct_image_s7 | 2756 | 1008 | 74.56 [72.9, 76.1] | 77.70 [76.3, 79.1] | 74.35 | 89.4 | 63.9 | 68.4 | 46.87 [44.7, 49.0] | 50.80 | 56.1 (1471) | 100.00 | 74.56 |
| gen_test_ct_twin_s7 | 2756 | 1008 | 74.09 [72.4, 75.7] | 77.43 [76.0, 78.9] | 73.91 | 89.9 | 62.9 | 67.4 | 46.55 [44.4, 48.7] | 50.51 | 55.1 (1471) | 100.00 | 74.09 |

#### paired: A=gen_test_ct_twin_s7.jsonl  vs  B=gen_test_ct_image_s7.jsonl

common images 2756 (only A 0, only B 0), groups 1008, GT mismatches 0. Exact McNemar: b(A only correct)=8, c(B only correct)=21, p=0.02412

| metric | A | B | B-A [95% paired group-bootstrap CI] |
|---|---|---|---|
| acc | 74.09 | 74.56 | +0.47 [+0.10, +0.85] |
| bal_acc | 77.43 | 77.70 | +0.27 [-0.10, +0.61] |
| f1_macro | 73.91 | 74.35 | +0.44 [+0.07, +0.80] |
| rec_real | 89.88 | 89.38 | -0.50 [-0.99, +0.00] |
| rec_fake | 64.99 | 66.02 | +1.03 [+0.47, +1.56] |
| mIoU | 46.55 | 46.87 | +0.32 [-0.05, +0.69] |
| IoU@0.5 | 50.51 | 50.80 | +0.29 [-0.32, +0.93] |
| twin | 55.13 | 56.08 | +0.95 [+0.14, +1.88] |

#### paired: A=gen_test_ct_twin_s7.jsonl  vs  B=gen_test_v3_sft_cf.jsonl

common images 2756 (only A 0, only B 0), groups 1008, GT mismatches 0. Exact McNemar: b(A only correct)=161, c(B only correct)=69, p=1.193e-09

| metric | A | B | B-A [95% paired group-bootstrap CI] |
|---|---|---|---|
| acc | 74.09 | 70.75 | -3.34 [-4.36, -2.38] |
| bal_acc | 77.43 | 76.23 | -1.20 [-2.10, -0.26] |
| f1_macro | 73.91 | 70.75 | -3.16 [-4.18, -2.19] |
| rec_real | 89.88 | 96.63 | +6.75 [+5.26, +8.33] |
| rec_fake | 64.99 | 55.84 | -9.15 [-10.63, -7.76] |
| mIoU | 46.55 | 42.61 | -3.95 [-4.71, -3.25] |
| IoU@0.5 | 50.51 | 46.62 | -3.89 [-5.05, -2.82] |
| twin | 55.13 | 51.67 | -3.47 [-5.57, -1.43] |

#### paired: A=gen_test_ct_image_s7.jsonl  vs  B=gen_test_v3_sft_cf.jsonl

common images 2756 (only A 0, only B 0), groups 1008, GT mismatches 0. Exact McNemar: b(A only correct)=178, c(B only correct)=73, p=2.694e-11

| metric | A | B | B-A [95% paired group-bootstrap CI] |
|---|---|---|---|
| acc | 74.56 | 70.75 | -3.81 [-4.87, -2.79] |
| bal_acc | 77.70 | 76.23 | -1.47 [-2.35, -0.50] |
| f1_macro | 74.35 | 70.75 | -3.59 [-4.59, -2.57] |
| rec_real | 89.38 | 96.63 | +7.24 [+5.75, +8.83] |
| rec_fake | 66.02 | 55.84 | -10.18 [-11.65, -8.71] |
| mIoU | 46.87 | 42.61 | -4.27 [-5.03, -3.52] |
| IoU@0.5 | 50.80 | 46.62 | -4.18 [-5.25, -3.02] |
| twin | 56.08 | 51.67 | -4.42 [-6.40, -2.28] |
