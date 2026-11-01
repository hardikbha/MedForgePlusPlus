# Label / mask / edit-locality audit (MedForge-90K + MedForge++ addition)

Auditor role: labels, masks, edit locality. Date: 2026-10-11. CPU only (at most 16 processes, all under `nice`), no GPU, read-only on data, code and checkpoints.
Every number below comes from a script in `~/CIRCE/v3_audit/` that I ran. Where I did not measure something, I say "not measured".

Scripts and raw outputs (all in `~/CIRCE/v3_audit/`):

| Script | Output | What it does |
|---|---|---|
| `labels_pairdiff.py` | `labels_pairdiff.jsonl` (2,160 rows) | Compares forged and source images for a stratified MF90K sample (36 per generator x type x organ, 60 strata) |
| `labels_aggregate.py` | `labels_aggregate_pairdiff.md`, `labels_aggregate_boxes.md` | Builds the tables |
| `labels_boxstats.py` | `labels_boxstats.jsonl` (67,286 rows) | Statistics for every MF90K forgery mask and every MF++ kidney/polyp mask |
| `labels_jpegq.py` | `labels_jpegq.json` | Census of JPEG quantisation tables over all 95,276 MF90K images plus the MF++ files |
| `labels_kidney.py` | `labels_kidney.json`, `labels_kidney_pairs.json`, **`kidney_removal_montage.png`** | Kidney and polyp removal deep-dive |
| `labels_mfpp_implant.py` | `labels_mfpp_implant.json` | Locality of kidney and colon implants, plus a per-subset sharpness and format shortcut check |
| `labels_text_audit.py` | `labels_text_audit.json` | MF90K explanation texts and our SFT templates |
| `labels_esm.py` | `labels_esm_v2.jsonl`, `labels_esm_v2.log` | Proposed edit-support-mask target compared with the bbox (600 pairs) |

---

## TL;DR (ranked by how much each finding threatens the CVPR paper)

1. **None of the 10 MF90K generators edits locally. Every one re-encodes the whole image.**
   - Far from the box (more than 64 px away), the mean |forged - source| is 2.9 (flux-dev) to 14.0 (step1x remove) grey levels. The JPEG re-encode noise floor is 0.03 (Q95) or about 0.95 (Q75).
   - Only a median **33.3%** of the difference energy lies inside the annotated box. The box covers a median 4.75% of the frame.
   - **69.8%** of forgeries have less than half their change inside the box; **39.2%** have less than a quarter.
   - Gemini output is geometrically shifted by at least 4 px in 34% of pairs (max 67 px). The figures for gpt and step1x are 13% and 11%.
   - Consequences: the bbox is a noisy, partial localisation target, and a global generator fingerprint exists that a detector can use without looking at the lesion.
2. **JPEG-quality shortcut in MF90K.**
   - Five generators save at Q≈73-75: flux-dev, sd3.5-L, sd3.5-M, sdxl-inpaint, step1x. That is 31,728 forgeries, 50.1% of all fakes.
   - The other five save at Q95: gemini, gpt, qwen, seedream, sd-inpaint.
   - Real images are Q95, except 3,385 fundus reals at Q≈73. None of those 3,385 is ever used as a forgery source.
   - **In brain MRI and chest X-ray, every Q≈73 image is a forgery.**
   - Two of the four Cross-Model held-out generators (sd3.5-M, sdxl-inpaint) are Q≈73, like three of the six training generators. This shortcut therefore transfers across the Cross-Model split.
   - Also: all 2,975 brain "metastases" reals use a different chroma subsampling and are never a forgery source, so this is a source-dataset confound.
3. **Source-scan reuse and pairing leakage.**
   - The 63,286 forgeries are built from only **3,790 distinct real scans**: a mean of 16.7 forgeries per scan, a maximum of 110.
   - On the shared test set (2,000 forgeries):
     - In-Domain: 1,993 test forgeries share their source scan with a training forgery from another generator. For 920, the untouched source itself is in training labelled "real".
     - Cross-Forgery strict: 997 share a source and 918 have the source in training as "real".
   - So the test is not scan-disjoint from training.
4. **Our "expert-aligned" CoT is a constant per class.** Across all 52,766 In-Domain SFT records:
   - There are only **24 distinct assistant texts** after masking the box and description: 1 for implant, 1 for removal, and 22 for real, which differ only by the disease word.
   - The `<evidence cue="boundary_discontinuity">` tag is the same and **empty in 100%** of records, including removal and real records whose own Q1 says "No sharp discontinuity".
   - The description is one of 3 strings ("<organ> medical image").
   - In the Cross-Forgery strict GRPO outputs, **850/1,000 test implants get the removal explanation**: it claims over-smoothing or erasure for an added lesion.
   - The evidence reward is a constant +0.20 for any template output, so it has zero within-group variance in GRPO.
   - This directly confirms reviewers A5, A6 and C6.
5. **Kidney removal (3% accuracy) is close to unrecoverable from a single image, by construction.**
   - The LaMa edit is strictly local: 81% of pixels outside the box are bit-identical to the source, and for 80.3% of images the mean |d| outside is at most 1 grey level.
   - The lesion's contrast against its 8 px ring drops from a median |24.7| to |10.3| grey levels.
   - There is no over-smoothing signature. The in-box/ring high-frequency ratio is 1.06 in the removal versus 0.98 in the source, giving an AUC of 0.315 for "removal is smoother", the wrong direction.
   - `kidney_normal` comes from the **same KiTS patients** (226 of 320 tumour cases overlap). A good removal is therefore literally a normal slice of a patient who is in the real class.
   - Polyp removal, by contrast, is **globally altered in 95.9% of images** (mean |d| outside 5.4). That explains polyp 75% versus kidney 3%: the polyp signal is mostly a global resampling fingerprint, not forensic reasoning.
6. **Boxes: geometry is clean, the targets are not.**
   - All 63,286 MF90K masks exist, are 1024x1024, binary, single-component, solid rectangles, and none is empty.
   - The problem is target quality, not file integrity. Inside the bbox, only a median 57.7% of pixels show an above-noise change (z5 edit-support threshold). With MF90K's tight boxes, the edit spills past the box edge: mean |d| is 14.0 in the 0-32 px ring outside the box versus 6.4 more than 64 px away.
   - MF++ polyp boxes: 31.4-31.8% cover more than 25% of the frame, 8.4-9.1% more than 50%, maximum 99.8%.
   - MF++ kidney boxes: 33% cover less than 0.5% of the frame and 4.2-5.0% less than 0.1% (minimum 100 px, i.e. 10x10).
7. **Proposed localisation target (ESM-clip32).** The changed pixels touching the box, kept within 32 px of it, with the threshold set from the image's own far-field noise.
   - Compared with the bbox: pixel IoU 0.48, box IoU 0.67, ESM box 1.40x the bbox area, centre offset 11.5 px. 13.3% of samples have box IoU below 0.5 with the current GT. The empty rate is 3.2%, and those samples should be flagged.
   - So **even a perfect localiser of the true edit would score only about 0.67 median box IoU against the current bbox GT.** The current GT caps attainable mIoU and injects reward noise.
8. **MF++ implants are detectable without forensics.**
   - Kidney implants are whole-image FLUX outputs: no composite, a 512-px generation, a Kaggle source rather than KiTS, and JPEG rather than PNG. Mean |Laplacian| alone separates kidney_inject from kidney_normal with AUC 0.857.
   - Only **53 distinct kidney implant boxes** exist across 1,000 images, and the top 5 cover 60.6%.
   - Colon implants are composited, but still JPEG against PNG reals (Laplacian AUC 0.794 vs polyp_real).
   - This is reviewer A3/B4 ("generator shortcuts") made concrete.

---

## 1. MF90K forged-vs-source difference analysis (edit locality)

**Sample.** 2,160 forgeries, 36 per (generator x type x organ) stratum, with seed 20261011.
- Strata were filtered to pairs whose source file exists locally. In a separate random draw of 3,000 MF90K forgeries, 39 had a missing `real_image_path` and 256 a missing `explain_path`.
- All 2,160 forged, source and mask images are 1024x1024, so no resizing was needed.
- Diff = |gray(forged) - gray(source)|, in grey levels (0-255).
- "JPEG floor" means the source re-encoded with the forged image's own quantisation tables. That floor is what a perfect local paste-back saved the same way would leave outside the box.

### 1a. Label sanity of the source image

| Check | Result |
|---|---|
| `edit` sources in a `/normal/` folder | **100.0%** (1,080/1,080) |
| `remove` sources in a `/normal/` folder | **0.0%**, so removal sources are lesion images as intended |
| `remove` source folder equals the disease label | 91.5% |

All mismatches are fundus:
- 75 "Hypertension" removals use a source from the `Diabetes/` folder;
- 10 Glaucoma and 6 AMD removals also use `Diabetes/` sources;
- 1 Glaucoma removal uses an `AMD/` source.

That is about 25% of fundus removals (92 of 360 in the sample). ODIR is multi-label, so this is probably a multi-disease scan rather than a wrong pairing. But the disease named in the removal prompt and explanation is not the folder label of the source. This was not verified by eye.

The binary real/fake label and the edit/remove type are consistent with the source pairing.

### 1b. Per generator x type locality

Columns:
- **E_in**: median fraction of squared-difference energy inside the bbox.
- **out**: mean |d| outside the box.
- **floor**: JPEG re-encode floor.
- **>4**: fraction of outside pixels with |d| > 4.
- **shift**: fraction of pairs whose global registration shift is at least 1 px (phase correlation at 256 px, so a 4 px quantum).

| generator | type | box area % | E_in | mean \|d\| in box | out | floor | >4 | HF residual out | corr out | shift | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flux-dev | edit | 3.34 | 0.225 | 14.2 | 3.24 | 0.96 | 0.150 | 3.65 | 0.9949 | 0.01 | near-global (VAE round-trip, low amplitude) |
| flux-dev | remove | 3.70 | 0.231 | 13.9 | 3.22 | 0.98 | 0.154 | 3.82 | 0.9943 | 0.00 | near-global |
| gemini | edit | 4.96 | 0.187 | 34.6 | 10.95 | 0.03 | 0.419 | 4.99 | 0.9760 | 0.61 | GLOBAL regeneration and shift |
| gemini | remove | 3.45 | 0.194 | 31.7 | 9.03 | 0.03 | 0.367 | 4.89 | 0.9713 | 0.55 | GLOBAL regeneration and shift |
| gpt | edit | 4.36 | 0.322 | 37.6 | 8.56 | 0.03 | 0.433 | 3.60 | 0.9905 | 0.29 | GLOBAL |
| gpt | remove | 4.91 | 0.218 | 25.1 | 8.01 | 0.03 | 0.431 | 3.73 | 0.9884 | 0.30 | GLOBAL |
| qwen | edit | 5.50 | 0.620 | 48.8 | 5.50 | 0.03 | 0.264 | 4.79 | 0.9920 | 0.37 | GLOBAL (most concentrated) |
| qwen | remove | 4.53 | 0.446 | 33.8 | 5.39 | 0.03 | 0.259 | 4.24 | 0.9901 | 0.31 | GLOBAL |
| seedream | edit | 6.27 | 0.368 | 52.9 | 10.47 | 0.03 | 0.579 | 7.19 | 0.9847 | 0.72 | GLOBAL |
| seedream | remove | 5.26 | 0.295 | 43.1 | 6.93 | 0.03 | 0.474 | 7.14 | 0.9892 | 0.62 | GLOBAL |
| sd3.5-L | edit | 4.26 | 0.345 | 21.3 | 4.94 | 0.97 | 0.252 | 3.84 | 0.9914 | 0.00 | GLOBAL VAE |
| sd3.5-L | remove | 3.36 | 0.259 | 18.4 | 4.47 | 0.96 | 0.240 | 3.73 | 0.9909 | 0.00 | near-global |
| sd3.5-M | edit | 4.08 | 0.291 | 19.7 | 4.88 | 0.95 | 0.264 | 3.68 | 0.9934 | 0.00 | GLOBAL VAE |
| sd3.5-M | remove | 3.81 | 0.194 | 16.0 | 4.37 | 0.93 | 0.232 | 3.60 | 0.9917 | 0.01 | near-global |
| sd-inpaint | edit | 5.75 | 0.472 | 48.0 | 7.70 | 0.03 | 0.329 | 4.32 | 0.9806 | 0.00 | GLOBAL (no paste-back) |
| sd-inpaint | remove | 6.26 | 0.487 | 49.2 | 6.80 | 0.03 | 0.329 | 3.89 | 0.9795 | 0.05 | GLOBAL |
| sdxl-inpaint | edit | 5.08 | 0.394 | 34.9 | 6.88 | 0.94 | 0.385 | 3.97 | 0.9892 | 0.01 | GLOBAL (no paste-back) |
| sdxl-inpaint | remove | 4.79 | 0.415 | 30.5 | 5.64 | 0.97 | 0.312 | 4.02 | 0.9884 | 0.02 | GLOBAL |
| step1x | edit | 6.17 | 0.522 | 48.8 | 7.66 | 0.92 | 0.291 | 3.67 | 0.9816 | 0.17 | GLOBAL |
| step1x | remove | 5.96 | 0.356 | 54.6 | 14.78 | 0.99 | 0.391 | 4.15 | 0.9448 | 0.48 | GLOBAL (largest off-box change) |

(Thresholds behind the verdict column: "LOCAL" would need mean |d| outside below 2 and fewer than 10% of outside pixels above 4. **No cell qualifies.**)

Restricting to the 1,673 pairs with zero registration shift does not change the picture:
- median E_in 0.360;
- far-field (more than 64 px from the box) mean |d| per generator: flux-dev 2.9; sd3.5-L/M about 4.1-4.5; qwen 3.5-5.2; gpt 5.3-5.6; sdxl 5.1-6.2; sd-inpaint 5.4-6.8; gemini 7.1-9.4; seedream 5.9-8.4; step1x 5.3-14.0.

Geometric shift (phase correlation, in 1024-px units):

| generator | ≥2 px | ≥4 px | max |
|---|---|---|---|
| gemini | 44% | 34% | 67 px |
| gpt | 19% | 13% | 46 px |
| step1x | 18% | 11% | 42 px |
| seedream | 21% | 5% | 295 px |
| qwen | 16% | 7% | 14 px |
| flux and all SD models | about 0 | about 0 | 91 px (one sd-inpaint outlier) |

The mean global brightness offset outside the box is 2.5-7.4 grey levels for gemini, gpt, seedream and step1x.

**Interpretation**
- Every MF90K forgery carries a **whole-image generator fingerprint**: a VAE/decoder round trip plus, for 5 generators, Q≈75 re-compression, and for gemini, gpt, step1x and seedream, re-framing.
- This is why a detector can be right on "fake" without attending to the lesion. It also explains why localisation is hard to reward: the pixels that differ from the source are mostly *not* in the box.
- It also undermines MF90K's own claim of "pixel-level edit localization": the GT is a rectangle around the prompt target, not the changed region.
- The 0-32 px ring around the box has mean |d| 14.0, versus 6.4 far away. The edits spill past the box edge, so the boxes are slightly too tight around the true change. (The ESM-clip32 box is 1.40x the bbox area; section 5.)
- Edge cases:
  - 2.7% of forgeries change the box by less than 5 grey levels on average (nearly invisible edit; edit 2.4%, remove 3.1%).
  - 1.2% change the box *less* than the outside, i.e. the box misses the edit. These are mostly seedream/step1x removals and gemini edits.

### 1c. JPEG quantisation census (all 95,276 MF90K images; headers only)

| class | format / estimated quality / chroma subsampling | count |
|---|---|---|
| flux-dev, sd3.5-L, sd3.5-M, sdxl-inpaint, step1x | JPEG, Q≈73, 4:2:0 | 6,399 / 6,383 / 6,383 / 6,380 / 6,183 (31,728 total) |
| gemini, gpt, qwen, seedream, sd-inpaint | JPEG, Q95, 4:2:0 | 31,558 total |
| real | JPEG Q95 4:2:0 | 25,341 |
| real | JPEG Q≈73 4:2:0 | 3,385 (all fundus: normal 1,679, Diabetes 1,129, AMD 409, Glaucoma 168) |
| real | JPEG Q95, non-4:2:0 subsampling | 2,975 (**all brain metastases**) |
| real | unreadable or missing locally | 289 (243 chest Pneumonia, 38 fundus Myopia, 8 chest Lung_Opacity) |

- All 3,790 forgery source scans are Q95 4:2:0. The 3,385 Q≈73 reals and the 2,975 metastases reals are never used as sources.
- So in brain and chest, Q≈73 means fake with 100% precision, and the metastases look means real.
- MF++ addition:
  - every real, normal and removal image is PNG (5,800 files);
  - both implant sets are JPEG Q95 (kidney_inject 1,000, colon_inject 1,000).
  - So in MF++, implant means JPEG and everything else means PNG. A file-format shortcut for the implant class, previously noted in `DATASET_AUDIT.md`, is confirmed.

### 1d. Source reuse and pairing leakage (my scope: the pairing labels)

- There are 3,790 distinct `real_image_path` values for 63,286 forgeries: mean 16.7 uses, maximum 110.
- Distribution of uses: 2,686 sources are used 10x, 319 used 9x, 316 used 30x, 206 used 50x, 54 used 110x.

Shared test set `mfp_v2_test_shared.jsonl` (994 real, 2,000 fake) against each training file:

| train file | test fakes whose source scan is the source of a train forgery | test fakes whose source scan is a train *real* | test reals that are the source of a train forgery |
|---|---|---|---|
| mfp_v2_train_crossforgery_strict | 997 / 2000 | 918 / 2000 | 98 / 994 |
| mfp_v2_train_crossmodel | 1980 / 2000 | 921 / 2000 | 115 / 994 |
| mfp_v2_train_indomain | 1993 / 2000 | 920 / 2000 | 118 / 994 |
| mfp_train_crossforgery (old) | 998 / 2000 | 1037 / 2000 | 98 / 994 |

- No exact forged file overlaps.
- But the split is image-disjoint, not scan-disjoint. At test time the model has already seen the same anatomy, as an authentic image and/or as other generators' edits of it.
- This should be handed to the protocol auditor. A scan-disjoint re-split (group by `real_image_path`) is the clean fix.
- Its effect on accuracy is not measured.

---

## 2. Box statistics (all masks)

**MF90K (63,286 forgery masks)**
- 0 missing, 0 empty, 0 non-binary, 0 multi-component, 0 non-rectangular (fill of the bounding rectangle is 1.0 in every mask).
- All masks are 1024x1024.

**MF++ (4,000 masks)**: kidney_real_tumor, kidney_removal, polyp_real and polyp_removal.
- Same integrity results.
- The mapping-json `box` matches the mask bbox: median IoU 1.000, 0 cases below 0.5.

| set | organ | type | n | area % p10 | median | p90 | max | >50% frame | >25% | <0.5% | <0.1% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MF90K | brain-mri | edit | 10,267 | 1.70 | 3.89 | 10.80 | 57.7 | 0.02% | 0.02% | 0.06% | 0 |
| MF90K | brain-mri | remove | 9,821 | 1.57 | 3.16 | 10.17 | 22.7 | 0 | 0 | 0.22% | 0.01% |
| MF90K | chest-xray | edit | 11,162 | 1.86 | 5.27 | 12.32 | 28.4 | 0 | 0.02% | 0.08% | 0 |
| MF90K | chest-xray | remove | 11,170 | 1.86 | 5.56 | 12.97 | 44.0 | 0 | 0.04% | 0.09% | 0 |
| MF90K | fundus | edit | 10,965 | 1.55 | 5.64 | 13.16 | 47.2 | 0 | 0.07% | 0 | 0 |
| MF90K | fundus | remove | 9,901 | 1.65 | 5.87 | 13.39 | 58.9 | 0.02% | 0.04% | 0.03% | 0.01% |
| MF++ | kidney | real_tumor | 1,000 | 0.18 | 0.93 | 4.91 | 16.4 | 0 | 0 | **32.4%** | 4.2% |
| MF++ | kidney | removal | 1,000 | 0.17 | 0.94 | 4.91 | 39.4 | 0 | 0.2% | **33.2%** | 5.0% |
| MF++ | polyp | real | 1,000 | 4.32 | 15.86 | 48.01 | 99.8 | **9.1%** | **31.8%** | 0.1% | 0 |
| MF++ | polyp | removal | 1,000 | 4.28 | 15.64 | 47.23 | 99.3 | **8.4%** | **31.4%** | 0.1% | 0 |
| MF++ | kidney | inject (json box) | 1,000 | 1.08 | 1.32 | 1.32 | - | 0 | - | 0 | - |
| MF++ | colon | inject (json box) | 1,000 | 2.97 | 3.05 | 3.30 | - | 0 | - | 0 | - |

MF90K median box area per generator:

| generator | median box area | >25% of frame | <0.5% of frame |
|---|---|---|---|
| flux-dev | 3.70% | 0.00% | 0.00% |
| sd3.5-L | 4.03% | 0.00% | 0.00% |
| sd3.5-M | 4.00% | 0.00% | 0.00% |
| gemini | 4.16% | 0.05% | 0.03% |
| sdxl | 4.50% | 0.00% | 0.00% |
| gpt | 4.72% | 0.02% | 0.22% |
| qwen | 4.91% | 0.06% | 0.08% |
| sd-inpaint | 5.94% | 0.05% | 0.00% |
| seedream | 6.06% | 0.06% | 0.05% |
| step1x | 6.26% | 0.10% | 0.42% |

The per-generator box size differs by up to 1.7x. Box size is weakly generator-identifying, so not a meaningful shortcut.

MF++ kidney box absolute sizes:
- real_tumor: median 9,802 px, short side median 94 px; 4.6% have a short side under 32 px and 0.4% under 16 px; minimum 144 px.
- removal: median 9,830 px; 5.4% have a short side under 32 px and 0.5% under 16 px; minimum 100 px (10x10).

Polyp box absolute sizes: median about 165k px, short side median about 356 px.

**Localisation consequence.** In MF90K, boxes are moderate (median 3-6% of the frame) and well-formed, but they are not edit-tight (sections 1 and 5).

From the model's outputs on the shared test (CF strict GRPO, `gen_mfp_crossforgery_strict_grpo_8b.jsonl`):

| GT box area | mIoU |
|---|---|
| < 2% | 0.169 |
| 2-5% | 0.232 |
| 5-10% | 0.308 |
| 10-25% | 0.299 |

- The predicted box coordinates regress strongly to the dataset mean box. A linear fit of predicted against GT gives slope 0.30 / 0.21 / 0.33 / 0.38 for x1 / y1 / x2 / y2 in CF, and 0.51 / 0.41 / 0.53 / 0.50 in In-Domain replay GRPO.
- The ratio of mean predicted to mean GT coordinate is 0.94-1.05, so there is **no coordinate-frame (0-1000 vs 1024) bug**.
- Box placement is weakly image-conditioned. That matches a noisy GT plus a weak 0.25·IoU bonus next to a ±4 verdict reward.

---

## 3. Kidney removal deep-dive (paper Table 4: 3% accuracy)

**Pairing**
- `kidney_removal/images/<case>_slice_<n>.png` pairs 1:1 with `kidney_real_tumor/images/<same name>`: 1,000 of 1,000 names match, and the mapping jsons carry identical boxes.
- `polyp_removal` pairs with `polyp_real` by filename for 953 of 1,000. For 47, no same-name source exists in `polyp_real`.
- The generator is LaMa per `PROJECT_STATUS.md` (`model:"real-source"` in the mapping).
- All files are PNG at 1024x1024.

| metric (median unless noted) | kidney_removal (n=1000) | polyp_removal (n=953) |
|---|---|---|
| mask area (% of frame) | 0.94 (p10 0.17, p90 4.91) | 15.6 (p10 4.4, p90 47.9) |
| diff energy inside box | **0.914** | 0.660 |
| mean \|d\| inside box | 37.6 (p10 21.4) | 32.3 |
| max \|d\| inside box | 143 | 184 |
| mean \|d\| outside box | **0.23** | **5.45** |
| outside pixels bit-identical to source | **80.9%** | 28.5% |
| images with mean \|d\| outside > 1 (globally altered) | 19.7% (197 images, 102 cases) | **95.9%** |
| lesion \|contrast\| vs 8-px ring: source to forged | 24.7 to 10.3 | 24.0 to 9.9 |
| in-box/ring high-frequency ratio: forged vs source | 1.06 vs 0.98 | 0.87 vs 1.14 |
| AUC "removal is smoother than its own source" (oracle box) | **0.315** (inverted: no smoothing) | 0.787 |
| AUC in-box std, source vs removal | 0.369 | 0.692 |
| removals with mean \|d\| in box < 5 | 0.0% | 0.1% |

What these numbers mean:
- **The edit is real and strong.** The lesion is replaced: mean |d| in the box is 37.6 grey levels. So the label is not wrong in the sense that "nothing changed".
- **It leaves essentially no residue.**
  - The filled region has the same high-frequency texture as its surroundings; it is not over-smoothed.
  - The lesion's contrast collapses toward the parenchyma.
  - Outside the box the image is bit-identical in 81% of pixels.
  - So there is no global fingerprint either, except in the 19.7% globally-altered subset.
- **The real class contains the same thing.** `kidney_normal` comes from the same KiTS cohort: 326 normal cases versus 320 tumour cases, with 226 overlapping. A clean removal is therefore a plausible "tumour-free slice of a tumour patient", which is exactly what `kidney_normal` is.
- **The montage agrees** (`~/CIRCE/v3_audit/kidney_removal_montage.png`: 12 random kidney and 6 polyp triplets of forged-with-box, source crop, forged crop, mask crop and diff crop).
  - In the kidney rows the removed region looks like ordinary renal parenchyma or fat at crop scale. The diff maps follow the tumour outline: LaMa was driven by the KiTS segmentation, not the box.
  - The polyp rows show visible smudge or flat patches in several cases, and the diff maps show faint changes across the whole frame (global resampling).

**Verdict: for at least the roughly 80% of strictly-local kidney removals, the label is not recoverable from the single image.** It is a semantic forgery ("a tumour should be here"). The only evidence is the absence of a lesion that a normal kidney also lacks. The 3% accuracy is the expected behaviour of an honest detector, not a model bug.

Polyp removal's 75% is very likely the **global resampling fingerprint** (95.9% globally altered, mean |d| 5.45 outside the box) plus in-box smoothing (AUC 0.787), rather than lesion reasoning.

Options for the paper:
- (a) Report kidney removal as an *unrecoverable-by-design* control and exclude it from headline accuracy. Report removal AUC against same-source reals only.
- (b) Reframe it as a reference-based (paired) task.
- (c) Regenerate with an editor that leaves detectable traces. Not recommended: it creates the shortcut you would then be measuring.

Reviewer Q3 ("what accounts for 3%") can be answered with this table and the montage.

## 3b. MF++ implants: locality and format shortcut

Sources: kidney implants come from `~/AAAI/kidney/normal/Normal- (N).jpg`, a Kaggle CT-kidney set, **not KiTS**. Colon implants come from Kvasir-SEG originals (about 530x620 px).
I took 150 random pairs per set and resized each source to the 1024 canvas with INTER_AREA, exactly as the generator scripts do.

| metric (median) | kidney_inject (FLUX-Kontext, no composite) | colon_inject (FLUX-Kontext + feathered composite) |
|---|---|---|
| box area % | 1.32 | 3.04 |
| diff energy inside box | 0.527 | 0.707 |
| mean \|d\| inside / outside box | 29.3 / **2.19** | 18.4 / 1.40 |
| outside pixels with \|d\| > 4 | 17.8% | 4.7% |
| source resolution | 512x512 for 84 of 150; others about 650x800 | about 530x620 |

**Kidney implants are whole-image FLUX outputs.**
- `inject_flux_kidney.py` writes `res.resize((W,H))` without compositing back onto the original. Generation runs at 512 px, and most sources are 512 px themselves, so the 1024 canvas is upsampled and blurrier than KiTS.
- A single scalar (mean |Laplacian|) separates kidney_inject from kidney_normal with **AUC 0.857**, and from kidney_real_tumor with AUC 0.860. The spectral high-frequency fraction gives 0.733.
- Three shortcuts stack on the implant class: (i) a different source dataset (Kaggle versus KiTS), (ii) lower effective resolution, and (iii) JPEG versus PNG.
- Kidney implant boxes come from a fixed spot finder: **only 53 distinct boxes across 1,000 images**. The top box (264,476,388,588) is used 223 times, and the top 5 boxes cover 606/1,000. Sizes are 124x112 in 639 cases. Implant localisation is therefore close to a lookup of 5 positions.

**Colon implants are composited.**
- They are local (energy in box 0.707, mean |d| outside 1.40 after resize and Q95 JPEG).
- They are still separable from polyp_real by sharpness: Laplacian AUC 0.794, high-frequency fraction AUC 0.798, consistent with JPEG versus PNG.
- Boxes: 445 distinct out of 1,000, all about 175x182 px; the top box is used 31 times.
- 145 of 1,000 are flagged `faint` in the annotations.

**Removals carry no global sharpness signal.** Single-scalar AUC:

| comparison | AUC |
|---|---|
| kidney_removal vs kidney_real_tumor | 0.516 |
| kidney_removal vs kidney_normal | 0.535 |
| polyp_removal vs polyp_real | 0.594 |
| polyp_real vs polyp_normal (source confound from the July audit) | 0.974 (high-frequency fraction) |

So the two forgery types in MF++ are separable by trivially different routes. Implant can be detected without forensics (format, resolution, source dataset). Kidney removal cannot be detected at all. This asymmetry, not model capacity, is what Table 4 measures.


---

## 4. Explanation texts and our template annotations

### 4a. MF90K `explain_path` texts (1,500 sampled forgeries, 300 reals)

**Missing locally**
- Real explanations: 300 of 300 missing; `real/gt/` does not exist on disk.
- step1x-edit explanations: 163 of 163 missing.
- All other generators: 0 missing.
- Overall: 10.9% of fakes.

**Readable texts** (1,337)
- Mean length 205 words.
- 34.7% of sentences (after masking the box) appear verbatim in at least one other document.
- Three boilerplate sentences appear in 100% of them: the `<box>` line, "This image contains deepfake regions.", and "The anatomical errors present in this image are as follows:".
- Phrase document frequencies: "sticker" 62%, "lack" 81%, "texture" 62%, "smooth" 46%, "inpaint" 28%.

**Weakly type-specific**
- Removal words (remov/erase/absent/missing/obliterat/painted over) appear in 37.8% of *remove* explanations and in 27.2% of *edit* explanations.
- Insertion words (insert/implant/added/pasted/overlaid/superimpos) appear in 54.5% of *remove* and 58.4% of *edit* explanations.
- So the "expert" texts describe a generic "sticker-like, smooth, inconsistent noise" artifact regardless of forgery type.

**Coordinate frame**
- The `<box>` in the text is in the **0-1000 normalised frame**, not 1024 px. Mask/text ratio is about 1.024-1.027 on 8 spot-checked files, e.g. text 383,463,611,701 against mask 393,475,627,719.
- Against the mask bbox, IoU is mean 0.771 (n=400) and only 2.5% reach IoU 0.9 or more.
- Anyone mixing MedForge's text boxes with mask-derived boxes must rescale.

### 4b. Our SFT templates

| file | n | distinct assistant texts (box and description masked) | per label | evidence tag | description strings | repair turn |
|---|---|---|---|---|---|---|
| mfp_train_indomain.jsonl | 52,766 | **24** | edit 1, remove 1, real 22 | `cue="boundary_discontinuity"`, **empty body, 100%** | 3 | 2 distinct strings |
| mfp_train_crossforgery.jsonl | 34,839 | 23 | remove 1, real 22 | same, 100% empty | 3 | 2 |
| mfp_v2_train_crossforgery_strict.jsonl | 31,356 | 23 | remove 1, real 22 | same, 100% empty | 3 | 2 |
| mfp_v2_replay_crossforgery_strict.jsonl | 1,722 | 23 | remove 1, real 22 | same, 100% empty | 3 | 2 |

**The 22 real variants** differ only in the disease word of the conclusion, e.g. "Findings are consistent with a metastases region." So the real-class text even names the lesion category of an *authentic* image.

**The repair turn** always says "the initial box may be loose; the salient region is tightly bounded by the GT extent". It then repeats the *identical* GT box, so the self-correction target is a no-op.

**MF++ addition texts**
- colon_inject: 3 distinct assistant texts out of 1,000.
- kidney_inject: 3 distinct out of 190 in the first manifest shard.
- naitik subsets are LLM-written and more varied: kidney_normal 94 distinct out of 1,000; kidney_real_tumor 810; kidney_removal 860; polyp_normal 340 out of 800 (it also has the 29.5% byte-duplicates from the July audit); polyp_real 1,000; polyp_removal 998.

**Model behaviour inherits this.** First-answer template on the shared test:

| run | what it emitted |
|---|---|
| CF strict GRPO, 1,000 implants | removal template ("TOO smooth ... inpainting that erased a structure") 850 times; real template 150 times |
| ID replay GRPO, 1,000 removals | implant template ("an added focal structure") 199 times |

- The "reasoning" is a type classifier rendered as one of 3 fixed paragraphs. It contains no image-specific evidence.
- `circe_grpo_reward.py` pays `evidence_bonus = 0.20` when the cue is in `FORENSIC_CUES` or the text contains "boundary" or "texture". Every template satisfies this, so the term is constant and adds no GRPO advantage.
- **The claims "expert-aligned annotation" (main.tex §3.3) and "an evidence tag naming the dominant forensic cue" are not supported by the data.** No expert is involved, and the tag is a constant.

---

## 5. A better localisation target: Edit-Support Mask (ESM)

**Definition**, computed per forged/source pair (`labels_esm.py`):
1. Integer-pixel registration of forged to source (phase correlation; 20.2% of pairs needed a shift).
2. D = Gaussian_σ=2(|forged - source|).
3. Robust far-field noise: μ, σ = median and 1.4826·MAD of D over pixels more than 64 px from the bbox.
4. Threshold T = max(12, μ + 5σ). This is "z5"; the median T was 12.0. A "z3" variant uses max(8, μ + 3σ).
5. Connected components of D > T with at least 64 px that **touch the bbox dilated by 32 px**.
6. **ESM-clip32** = the step-5 components, clipped to the bbox dilated by 32 px. This is the proposed target. Its tight box is the box target.

Results on 600 pairs (10 per stratum, all 60 strata):

| target | empty | pixel IoU vs bbox | share of target outside bbox | bbox pixels covered | area / bbox | box IoU vs bbox | box IoU < 0.5 | box area / bbox | centre offset |
|---|---|---|---|---|---|---|---|---|---|
| ESM z5 (unclipped) | 3.2% | 0.315 | 37.2% | 57.7% | 0.90 (p10 0.15, p90 3.31) | 0.387 | 62.1% | 2.15 | 55.3 px |
| ESM z3 (unclipped) | 1.5% | 0.324 | 49.3% | 70.9% | 1.33 | 0.307 | 74.8% | 3.01 | 71.6 px |
| **ESM z5 clip32 (proposed)** | 3.2% | **0.482** | 17.3% | 57.7% | 0.71 (p10 0.14, p90 1.21) | **0.669** | **13.3%** | 1.40 | 11.5 px |
| ESM z3 clip32 | 1.5% | 0.550 | 21.6% | 70.9% | 0.89 | 0.650 | 12.5% | 1.47 | 8.1 px |

By type (z5 clip32):

| type | pixel IoU | box IoU |
|---|---|---|
| edit | 0.488 | 0.673 |
| remove | 0.464 | 0.663 |

By organ (z5 clip32):

| organ | pixel IoU | box IoU | bbox coverage |
|---|---|---|---|
| brain | 0.543 | 0.639 | 0.71 |
| chest | 0.480 | 0.676 | 0.57 |
| fundus | 0.329 | 0.687 | 0.33 |

In fundus, two-thirds of the box is unchanged tissue.

A fixed-threshold cross-check on the full 2,160 sample (diff > 16 after σ=2 smoothing, tight box within 32 px of the bbox) gives the same answer: box IoU vs bbox median 0.655, 13.9% below 0.5, area 1.44x.

**Why ESM is the better reward target**
- It marks the pixels that actually changed, rather than a rectangle drawn around the prompt.
- Its far-field-adaptive threshold discounts the global re-encode noise of section 1.
- Clipping to bbox+32 px keeps the annotator's intent and stops global VAE noise from leaking into the target.

**Recommended uses**
- (i) A pixel-F1 or soft-IoU reward against ESM-clip32, or a box-IoU reward against bbox(ESM-clip32).
- (ii) Flag the 3.2% of pairs with an empty ESM (invisible edits) and do not reward localisation on them. Consider dropping them from the forged class.
- (iii) For MF++ kidney removals, the LaMa diff gives the *exact tumour footprint*, because the edit is bit-exact outside. Use it directly.
- (iv) In the paper, report mIoU against both the bbox and ESM. Report the label ceiling too: **a perfect ESM localiser gets median IoU 0.67 against the published bbox.**

Not measured:
- training with ESM rewards (no GPU);
- ESM for the full 63k set. The script runs in about 1 s per pair; at 14 processes the full set is about 1.3 h of CPU.

---

## 6. Recommendations (ordered)

1. **Paper text**
   - Drop "expert-aligned" or substantiate it.
   - Describe the CoT honestly as a type-conditioned template.
   - Rebuild the evidence field from image-specific measurements, for example ESM area, location, contrast change and boundary statistics computed from the pair. Or remove the evidence reward.
2. **Localisation**
   - Replace bbox-IoU rewards with ESM-clip32 (pixel F1 or IoU against its tight box).
   - Report the 0.67 label ceiling alongside mIoU.
   - Raise the localisation weight relative to the ±4 verdict term, or the box will keep regressing to the mean (slope 0.21-0.38).
3. **Shortcut controls** (CVPR reviewers will ask)
   - Re-encode *every* image (real and fake) to a single JPEG quality and to PNG-to-JPEG parity for MF++. Then re-evaluate the Cross-Model and Cross-Forgery numbers. The Q≈73 versus Q95 split, the brain-metastases subsampling difference and MF++'s implant=JPEG / rest=PNG all need neutralising.
   - Report results on the scan-disjoint re-split.
   - Rebuild kidney implants:
     - composite the FLUX output back onto the original, as the colon script already does;
     - draw sources from KiTS, the same cohort as kidney_normal and kidney_real_tumor;
     - randomise placement over the actual kidney mask instead of about 5 fixed spots;
     - save all MF++ classes in one format.
4. **Kidney removal**
   - Present it as unrecoverable-by-design. Evidence: 81% bit-identical outside, no smoothing (AUC 0.315), same-cohort normals.
   - Exclude it from headline accuracy, or evaluate it as a paired (reference-based) task.
   - Explain polyp removal's 75% as a global-fingerprint effect (95.9% globally altered).
5. **Dataset hygiene**
   - 289 MF90K real files are unreadable or missing locally, and 39 of 3,000 sampled forgeries (1.3%) have a missing source file, and step1x explanations and all real explanations are absent.
   - The manifest `medforgepp_manifest.json` (103,067) describes the **old lung/breast "medical-defect" addition**: lung 5,091, breast 700, colon 2,000. It contains no kidney. It is not the kidney/colon 7,800 addition the paper describes as 103,076. Rebuild the manifest before release.
