# Shortcut audit: MedForge-90K and the MedForge++ addition

Auditor: SHORTCUT. Date: 2026-10-11. CPU only (14 or fewer worker processes, `nice 19`). Nothing under `~/AAAI` or `~/CIRCE/circe` was modified.
Scripts, raw features and CSVs are in `~/CIRCE/v3_audit/shortcut_work/` (see the last section). Every number below comes from those runs.

## 0. Verdict

**Both corpora can be solved without looking at the forgery.**

1. **MF90K has a one-line colour shortcut.** In the stratified sample, every real chest X-ray and 99% of real brain MRIs are exactly grayscale (R=G=B in every pixel). Over 99.7% of the fakes from every one of the 10 generators carry a faint colour cast.
   - Rule with no training, *"fake iff any pixel has R≠G or G≠B"*, on the shared test set (`mfp_v2_test_shared.jsonl`): **brain MRI 99.7% accuracy (n=1006), chest X-ray 100.0% (n=1060)**. That covers 69% of the test set.
2. **A 16-feature gradient-boosting model on global image statistics beats Bridge and MedForge on the official protocol.** It never localises anything. We trained it on 5,000 images from each regime's training file and tested on the shared test set (n=2,994):

   | Regime | Test acc | Macro-F1 |
   |---|---|---|
   | Cross-Forgery strict | **98.30** | 98.09 |
   | Cross-Model | **97.26** | 96.96 |
   | In-Domain | **98.30** | 98.08 |

   - For comparison: Bridge CF-strict 94.05 / 93.54; MedForge published CF 95.93 / 95.13 and CM 93.94 / 93.49.
   - In Cross-Forgery strict, the model **never sees an implant** and still scores **97.4%** on implants.
   - In Cross-Model, its AUC on the 4 held-out generators is **0.996**.
   - Adding a 16×16 thumbnail raises accuracy to 99.06–99.27%.
   - With the colour features removed it still reaches **93.59 / 93.02 on CF-strict**, about Bridge's number.
3. **Why the global statistics work: the generators re-render the whole frame.** Outside the GT box (with a 15 px halo excluded), fake and source differ by a median of **3.1–8.2 grey levels**, depending on the generator. A q75 JPEG re-encode of the source changes it by only 0.79–0.86. Between 7.6% and 25.5% of out-of-box pixels change by more than 8 levels. So Cross-Forgery and Cross-Model do not test what their names say: the same whole-image editor fingerprint (colour cast, smoothing, JPEG history) is present in both forgery types and in every generator.
4. **JPEG quality is a second, generator-tied cue.**
   - flux.1-dev, SD3.5-L, SD3.5-M, SDXL-inpainting and step1x-edit fakes are all JPEG **q75**.
   - gemini, gpt, qwen, seeddream and SD-inpainting fakes are **q95**.
   - Brain and chest reals are all q95.
   - Resolution is **not** a cue: all MF90K images are 1024×1024, and a width/height-only classifier gets AUC 0.47–0.49.
5. **In the MF++ addition, the implant class is trivially detectable from file format and dataset source.**
   - Every implant (colon and kidney) is a JPEG q95; every real and removal image is a PNG. File-format features alone give **AUC 1.000**.
   - The kidney implants come from a **different CT dataset**: `~/AAAI/kidney/normal/Normal- (N).jpg`, 64% at 512×512, JPEG q100. The kidney reals are KiTS slices.
   - The two *real* pools (KiTS normal vs the implant source pool) separate with **AUC 1.000 even from a 16×16 thumbnail**. The paper's "Implant 100%" on kidney is therefore not evidence of forensic skill.
   - polyp_real vs polyp_normal (both real) is re-confirmed at **AUC 1.000**. The separation survives every re-encoding recipe we tried.
6. **Removal behaves differently.**
   - Kidney removal is clean: out-of-box change is 0.19 grey levels, and there is no shortcut (metadata AUC 0.52, global statistics 0.70). It is genuinely almost undetectable, which is consistent with the 3% accuracy.
   - Polyp removal is **not** local: out-of-box change is 5.67 levels. A 16×16 thumbnail separates it from same-frame reals at **AUC 0.906**, because the polyp is either present or absent. That is a content cue, not forensic evidence.
7. **What fixes it.** Re-encoding alone (grayscale + resize + uniform JPEG or PNG) removes the file-level and colour cues. It does **not** remove the re-rendering fingerprint: global-statistic AUC stays 0.95–0.99.
   - **Pasting the generated region back into the real source** (only pixels inside the box come from the generator), then encoding real and fake identically, drops the pooled global-statistic AUC from **0.989 to 0.743** and the thumbnail AUC from 0.941 to 0.584.
   - That is the minimum fix before a CVPR resubmission can claim forensic detection on this benchmark.

AUC > 0.8 is treated as alarming. Almost every real-vs-fake comparison in MF90K is above 0.95 on non-forensic features.

---

## 1. What was measured

**Sample.** We sampled 46,358 images (`build_sample.py`, seed-fixed, deterministic; md5 of `sample.jsonl` = e8d80ff3...). 46,274 decoded; 84 MF90K reals are missing on disk.

- **MF90K, stratified:**
  - 200 per organ × forgery type × generator, giving 400 per organ × generator and 2,000 per organ × type.
  - Up to 2,400 reals per organ.
  - Plus up to 800 reals per organ that were used as the *source* of fakes, for a source-matched control.
- **Protocol files:**
  - The full shared test set, `mfp_v2_test_shared.jsonl` (2,994).
  - 5,000 random images from each of `mfp_train_crossforgery`, `mfp_v2_train_crossforgery_strict`, `mfp_train_crossmodel` and `mfp_train_indomain`. No path overlaps the test set (measured: overlap 0).
- **MF++ addition, all 7,800 images:**
  - The six `naitik_annotated/*` subsets (5,800).
  - `colon_injections/images` (1,000).
  - `~/AAAI/kidney/flux_injected/images` (1,000 kidney implants). These are the kidney implant images: `circe_sft_v3.jsonl` and `circe_eval_split_v2.jsonl` reference this folder.
- **Provenance references (not part of MF++):** Kvasir-SEG originals (1,000) and the kidney implant source pool `~/AAAI/kidney/normal` (1,000).

**Per-image features** (`extract.py`):

- **Metadata:** extension, PIL format and mode, width and height, file size, bytes per pixel, JPEG luma and chroma quality estimate, IJG-table residual (0 means a standard libjpeg table), number of quantisation tables, quantisation-table hash, chroma subsampling, EXIF, ICC, JFIF, Adobe and progressive flags, DPI, and the PIL info keys.
- **Global statistics at native resolution:**
  - `gbasic`: mean, std, log Laplacian variance, high-frequency FFT energy ratio (radius > 0.5 Nyquist), saturation.
  - `gext` adds: mid- and low-band FFT ratio, chroma = mean(|R−G|+|G−B|), median-filter noise residual, entropy, dark/bright/0/255 fractions, number of grey levels, and 8×8 JPEG blockiness.
- **`g512`:** the same statistics after resizing to 512×512.
- **`thumb`:** a 16×16 grayscale thumbnail, as a low-resolution appearance and source probe.

**Classifiers.**

- Logistic regression (imputer + standard scaler) and sklearn `HistGradientBoostingClassifier`.
- 5-fold `StratifiedGroupKFold`. Groups:
  - MF90K: the real source image, so a fake and its source are always in the same fold.
  - KiTS: case id.
  - Kvasir: frame stem.
- We report out-of-fold pooled ROC-AUC and the fold std.
- In the addition, exact byte duplicates (md5) are dropped before splitting.

**Note on what the VLM sees.** `~/CIRCE/circe/*.py` never sets `max_pixels` or `min_pixels` (grep: no hits). The Qwen3-VL preprocessor default is `longest_edge` 16,777,216 px, so 1024×1024 inputs reach the model at native resolution. JPEG 8×8 structure and per-pixel RGB differences are therefore present in its input. Whether Bridge actually uses them is **not measured** (that needs a GPU; see §8).

---

## 2. Metadata tables

Full tables: `shortcut_work/tables_meta.md` and `tab_*.csv`.

### 2.1 MF90K by organ × class (stratified sample)

| organ | class | n | ext | PIL mode | resolution | median size KB | JPEG q | subsampling | EXIF / ICC / progressive |
|---|---|---|---|---|---|---|---|---|---|
| brain-mri | real | 2400 | .jpg 70% / .jpeg 30% | RGB 70% / **L 30%** | 1024² 100% | 118.4 | 95 (100%) | 4:2:0 70% / 1-ch 30% | 0 / 0 / 0 |
| brain-mri | edit | 2000 | .jpeg 100% | RGB 100% | 1024² 100% | 114.6 | **75 50% / 95 50%** | 4:2:0 100% | 0 / 0 / 0 |
| brain-mri | remove | 2000 | .jpeg 100% | RGB 100% | 1024² 100% | 82.4 | 75 50% / 95 50% | 4:2:0 100% | 0 / 0 / 0 |
| chest-xray | real | 2351 | .jpeg 100% | RGB 100% | 1024² 100% | **217.7** | 95 (100%) | 4:2:0 | 0 / 0 / 0 |
| chest-xray | edit | 2000 | .jpeg | RGB | 1024² | 108.3 | 75 50% / 95 50% | 4:2:0 | 0 / 0 / 0 |
| chest-xray | remove | 2000 | .jpeg | RGB | 1024² | 111.3 | 75 50% / 95 50% | 4:2:0 | 0 / 0 / 0 |
| fundus | real | 2390 | .jpeg | RGB | 1024² | 144.3 | 95 67% / 75 33% | 4:2:0 | 0 / 0 / 0 |
| fundus | edit | 2000 | .jpeg | RGB | 1024² | 100.1 | 75 50% / 95 50% | 4:2:0 | 0 / 0 / 0 |
| fundus | remove | 2000 | .jpeg | RGB | 1024² | 90.0 | 75 50% / 95 50% | 4:2:0 | 0 / 0 / 0 |

All JPEGs use standard IJG tables (residual 0 in 100% of cases). No image has EXIF, ICC or progressive encoding. Every image is 1024×1024, so MF90K was already resolution-normalised.

### 2.2 MF90K fakes by generator (all organs, n=600 each)

| generator | JPEG q | median size KB | median chroma, brain fakes (edit / remove) | median chroma, chest fakes (edit / remove) |
|---|---|---|---|---|
| flux.1-dev | 75 | 70.2 | 0.029 / 0.048 | 0.012 / 0.013 |
| gemini | 95 | 188.1 | 0.086 / 0.054 | 0.073 / 0.070 |
| gpt | 95 | 158.2 | 0.108 / 0.056 | 0.056 / 0.055 |
| qwen | 95 | 178.7 | 0.298 / 0.130 | 0.271 / 0.283 |
| seeddream | 95 | 211.4 | 1.722 / 1.618 | 1.460 / 1.622 |
| stable-diffusion-3.5-large | 75 | 69.4 | 1.834 / 1.756 | 1.956 / 1.843 |
| stable-diffusion-3.5-medium | 75 | 66.4 | 2.245 / 2.276 | 2.219 / 2.302 |
| stable-diffusion-inpainting | 95 | 142.3 | 2.625 / 2.899 | 1.612 / 1.467 |
| stable-diffusion-xl-1.0-inpainting-0.1 | 75 | 69.2 | 1.477 / 1.170 | 1.419 / 1.512 |
| step1x-edit | 75 | 59.2 | 0.107 / 0.129 | 0.232 / 0.235 |
| *reals* | 95 | — | **0.000** | **0.000** |

Share of images with **any** pixel where R≠G or G≠B:

| group | share |
|---|---|
| Chest reals | 0.0000 (2,351 images) |
| Brain reals, YG metastases | 0.0000 (714 images) |
| Brain reals, Kaggle | 0.0142 (1,686 images) |
| Every generator × organ cell | 0.9975–1.0000 (400 images each) |

### 2.3 MF90K real sub-sources (never-edited populations)

Only **3,790** of the 31,990 reals were ever used as a fake source. Whole sub-populations have **zero** fake counterparts:

| sub-population | reals | used as a fake source | note |
|---|---|---|---|
| brain `metastases/YG_*` | 2,975 | 0 | grayscale `L` JPEG |
| fundus EyePACS | 2,538 | 0 | q75 |
| fundus "other" (`train-*`, `im*`, `*_G`, hash names) | 847 | 0 | q75 |
| fundus ODIR Cataract | 310 | 0 | |
| fundus ODIR Hypertension | 193 | 0 | |

- **213 of the 994 test reals** come from never-source sub-populations: YG 113, EyePACS 71, other fundus 29.
- **Source-matched control:** restricting reals to fake-source images does **not** reduce the shortcut. The global-statistic GB AUC is 1.000 (brain), 1.000 (chest) and 0.994 (fundus). So the shortcut is the generator fingerprint, not a sub-population prior.

### 2.4 MF++ addition and provenance references

| subset | n | format | mode | resolution | median KB | JPEG q | median chroma |
|---|---|---|---|---|---|---|---|
| kidney_normal | 1000 | PNG | RGB | 1024² | 782.3 | – | 0.000 |
| kidney_real_tumor | 1000 | PNG | RGB | 1024² | 792.7 | – | 0.000 |
| kidney_removal | 1000 | PNG | RGB | 1024² | 794.6 | – | 0.000 |
| **kidney_inject** | 1000 | **JPEG** | RGB | 1024² | **157.4** | **95** | **0.174** |
| polyp_normal | 800 | PNG | RGB | 1024² | 821.9 | – | 84.4 |
| polyp_real | 1000 | PNG | RGB | 1024² | 1008.1 | – | 81.4 |
| polyp_removal | 1000 | PNG | RGB | 1024² | 943.1 | – | 77.1 |
| **colon_inject** | 1000 | **JPEG** | RGB | 1024² | **169.1** | **95** | 80.4 |
| ref: Kvasir-SEG originals | 1000 | JPEG | RGB | 333 distinct (622×530 most common) | 33.8 | 75 (98%) | – |
| ref: kidney implant source pool | 1000 | JPEG | RGB | 512² 64%, 23 distinct | 106.3 | **100** | – |

**Provenance (measured).**

- **Kidney.**
  - kidney_removal has exactly the same 1,000 filenames as kidney_real_tumor.
  - Cases: normal 326, tumour 320, removal 320. Normal and tumour share 226 KiTS cases.
  - Median 2 slices per case, maximum 25.
  - All 1,000 kidney implants come from `~/AAAI/kidney/normal` (`Normal- (N).jpg`), not KiTS. None of them carries a `case_` name.
- **Colon.**
  - All 1,000 polyp_real frames and all 1,000 colon_inject frames are Kvasir-SEG stems.
  - polyp_real and polyp_removal share 953 stems.
  - polyp_real is Kvasir (q75, about 622×530) upsampled to a 1024 PNG. colon_inject is Kvasir → 1024 → FLUX → JPEG q95.

**Exact byte duplicates in our sample (md5).**

| subset | duplicate groups | rows |
|---|---|---|
| polyp_normal | 94 | 330 |
| MF90K ODIR reals | 105 | 214 |
| MF90K Kaggle brain reals | 88 | 201 |
| kidney implant source pool | 4 | 8 |

**Integrity.** 289 MF90K mapping entries point to missing files: 243 chest Pneumonia reals, 38 fundus Myopia, 8 chest Lung_Opacity.

---

## 3. MF90K: metadata-only and global-statistic classifiers

Out-of-fold AUC ± fold std, 5-fold group CV, within organ, real vs fake.

| organ (n / n_fake) | meta LR | meta GB | meta, no extension (GB) | size only | JPEG tables + size (GB) | gbasic (5 feats) LR | gbasic GB | gext LR | **gext GB** | g512 GB | thumb 16² LR | thumb GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| brain-mri (6400 / 4000) | 1.000±0.000 | 1.000 | 0.893±0.007 | 0.494 | 0.893 | 0.950 | 0.999 | 0.976 | **1.000±0.000** | 0.999 | 0.756 | 0.952 |
| chest-xray (6351 / 4000) | 0.833 | 0.834±0.010 | 0.834 | 0.480 | 0.834 | 0.951 | **1.000** | 0.996 | **1.000±0.000** | 1.000 | 0.674 | 0.984 |
| fundus (6390 / 4000) | 0.736 | 0.750±0.017 | 0.750 | 0.471 | 0.750 | 0.788 | 0.901 | 0.933 | **0.996±0.001** | 0.985 | 0.764 | 0.985 |
| all organs pooled (19141 / 12000) | – | 0.836 | – | – | – | – | – | – | **0.999** | 0.998 | – | 0.975 |
| brain, source-matched reals | – | 1.000 | – | 0.490 | – | – | – | – | 1.000 | 1.000 | – | 0.929 |
| chest, source-matched reals | – | 0.839 | – | 0.477 | – | – | – | – | 1.000 | 1.000 | – | 0.983 |
| fundus, source-matched reals | – | 0.830 | – | 0.478 | – | – | – | – | 0.994 | 0.976 | – | 0.981 |

**Single-feature AUCs** (`res_mf90k_singlefeat.csv`):

- **Brain:** saturation 0.996, chroma 0.995, `.jpg` extension 0.851, zero-pixel fraction 0.804, noise 0.754, JPEG q 0.75.
- **Chest:** saturation **1.000**, chroma **1.000**, blockiness 0.880, file size 0.834.
- **Fundus** (colour modality, so chroma is not a shortcut there): blockiness 0.78, saturation 0.707, file size 0.673.

### 3.1 Per generator

| | brain meta | brain gext | chest meta | chest gext | fundus meta | fundus gext | fundus g512 |
|---|---|---|---|---|---|---|---|
| flux.1-dev | 1.000 | 1.000 | 1.000 | 1.000 | 0.931 | 1.000 | 0.990 |
| gemini | 1.000 | 1.000 | 0.572 | 1.000 | 0.698 | 0.994 | 0.978 |
| gpt | 1.000 | 1.000 | 0.777 | 1.000 | 0.828 | 0.992 | 0.959 |
| qwen | 1.000 | 1.000 | 0.586 | 1.000 | 0.713 | 0.993 | 0.975 |
| seeddream | 1.000 | 1.000 | 0.635 | 1.000 | 0.897 | 1.000 | 1.000 |
| SD-3.5-large | 1.000 | 1.000 | 1.000 | 1.000 | 0.947 | 1.000 | 1.000 |
| SD-3.5-medium | 1.000 | 1.000 | 1.000 | 1.000 | 0.955 | 1.000 | 1.000 |
| SD-inpainting | 1.000 | 1.000 | 0.989 | 1.000 | 0.919 | 1.000 | 0.997 |
| SDXL-inpainting | 1.000 | 1.000 | 1.000 | 1.000 | 0.946 | 1.000 | 0.995 |
| step1x-edit | 1.000 | 1.000 | 1.000 | 1.000 | 0.959 | 1.000 | 0.989 |

These are dedicated real-vs-generator models (GB, 5-fold group CV, 2,000–2,400 reals vs 400 fakes). Width/height-only models score 0.44–0.49 in every cell.

- In chest, metadata AUC is 1.000 for every q75 generator. For the q95 generators it is 0.572 (gemini), 0.586 (qwen), 0.635 (seeddream), 0.777 (gpt) and 0.989 (SD-inpainting).
- Global statistics reach 0.99–1.00 for **every** generator.
- Slicing the out-of-fold scores of the pooled all-organ GB gives the same picture: per-generator gext AUC 0.998–1.000, metadata AUC 0.560 (seeddream) to 0.960 (SD3.5-M).

### 3.2 Per forgery type

| | meta | gext | g512 | thumb |
|---|---|---|---|---|
| brain: real vs edit | 1.000 | 1.000 | 0.999 | 0.966 |
| brain: real vs remove | 1.000 | 1.000 | 1.000 | 0.968 |
| chest: real vs edit | 0.833 | 1.000 | 1.000 | 0.980 |
| chest: real vs remove | 0.825 | 1.000 | 1.000 | 0.978 |
| fundus: real vs edit | 0.741 | 0.996 | 0.983 | 0.988 |
| fundus: real vs remove | 0.752 | 0.996 | 0.982 | 0.980 |

Implant and removal are equally easy. The cue does not depend on the forgery type.

---

## 4. The decisive test: shortcut models on the official protocol

We trained on 5,000 images from each regime's training file, used GB, and evaluated on the **full shared test set** (n=2,994: 994 real / 1,000 edit / 1,000 remove). Source: `res_protocol.csv`.

| regime (train file) | features | test acc | macro-F1 | AUC | acc real | acc edit (implant) | acc remove | AUC real vs CM-held-out generators |
|---|---|---|---|---|---|---|---|---|
| CF `mfp_train_crossforgery` | meta | 73.61 | 71.72 | 0.823 | 71.93 | 72.0 | 76.9 | 0.831 |
| CF | **gext (16)** | **98.26** | **98.06** | 0.999 | 98.79 | **97.2** | 98.8 | 0.999 |
| CF | thumb 16² | 86.84 | 85.83 | 0.950 | 90.54 | 77.1 | 92.9 | 0.937 |
| CF | gext+thumb | 99.06 | 98.95 | 0.999 | 99.40 | 98.1 | 99.7 | 0.999 |
| CF-strict `mfp_v2_train_crossforgery_strict` | **gext** | **98.30** | **98.09** | 0.998 | 98.79 | **97.4** | 98.7 | 0.999 |
| CF-strict | gext+thumb | 99.10 | 98.99 | 1.000 | 99.60 | 98.4 | 99.3 | 1.000 |
| CM `mfp_train_crossmodel` | **gext** | **97.26** | **96.96** | 0.998 | 99.09 | 96.2 | 96.5 | **0.996** |
| CM | gext+thumb | 97.96 | 97.73 | 0.997 | 99.50 | 96.3 | 98.1 | 0.993 |
| ID `mfp_train_indomain` | **gext** | **98.30** | **98.08** | 0.999 | 97.69 | 99.1 | 98.1 | 0.999 |
| ID | gext+thumb | 99.27 | 99.17 | 1.000 | 98.99 | 99.2 | 99.6 | 1.000 |

Reference points from the task brief (not re-measured here):

| model | ID | CF | CM |
|---|---|---|---|
| Bridge, leak-free CF-strict replay SFT | – | 94.05 / 93.54 | – |
| Bridge, clean selection split | 99.00–99.10 | 93.67–93.93 | 94.07–94.23 |
| MedForge, published | 99.23 / 98.98 | 95.93 / 95.13 | 93.94 / 93.49 |

### 4.1 Ablation: what carries the signal

Source: `res_protocol_ablate.csv`.

| train regime | feature set | test acc | macro-F1 | brain | chest | fundus |
|---|---|---|---|---|---|---|
| none: **rule "fake iff any R≠G or G≠B pixel"** | – | brain 99.7, chest 100.0 | – | 99.7 | 100.0 | n/a |
| CF-strict | gext (16) | 98.30 | 98.09 | 99.30 | 100.00 | 95.26 |
| CF-strict | gext without saturation/chroma | **93.59** | **93.02** | 89.56 | 96.89 | 94.18 |
| CF-strict | gext without colour and blockiness | 90.01 | 89.27 | 82.31 | 97.17 | 90.19 |
| CF-strict | only saturation + chroma | 86.81 | 85.75 | 99.20 | 99.62 | 58.73 |
| CF-strict | gbasic (5: mean, std, lapvar, fft_hf, saturation) | 93.55 | 92.82 | 98.21 | 99.91 | 81.25 |
| CM | gext without colour | 92.08 | 91.44 | 94.73 | 90.66 | 90.84 |
| ID | gext without colour | 96.66 | 96.22 | 97.42 | 97.64 | 94.72 |

**Reading.** Colour carries the gray modalities. Even without colour, smoothing, noise and FFT statistics from whole-frame re-rendering, plus JPEG-history blockiness, give 90–94%. That is close to what the 8B VLM reports on CF and CM. Cross-Forgery and Cross-Model therefore do **not** measure generalisation to an unseen forgery type or an unseen generator: the fingerprint is shared.

---

## 5. Is the forgery even local? (MF90K)

`locality.py`: 40 fakes per organ × generator × type (2,375 pairs; 25 skipped because the source real is missing). Each fake is compared with its real source inside the GT box and outside it, with a 15 px halo around the box excluded.

| generator | box fraction | MAD in box | **MAD outside box** | share outside > 8 levels | correlation outside | q75 re-encode MAD (reference) |
|---|---|---|---|---|---|---|
| flux.1-dev | 0.036 | 13.6 | 3.12 | 0.076 | 0.995 | 0.84 |
| gemini | 0.036 | 28.0 | **8.21** | 0.255 | 0.963 | 0.80 |
| gpt | 0.048 | 25.4 | 5.50 | 0.209 | 0.991 | 0.81 |
| qwen | 0.046 | 36.5 | 3.89 | 0.100 | 0.992 | 0.86 |
| seeddream | 0.055 | 36.5 | 6.52 | 0.202 | 0.989 | 0.84 |
| SD-3.5-large | 0.035 | 19.4 | 4.52 | 0.100 | 0.992 | 0.82 |
| SD-3.5-medium | 0.041 | 16.5 | 4.38 | 0.102 | 0.993 | 0.80 |
| SD-inpainting | 0.061 | 39.5 | 5.20 | 0.159 | 0.983 | 0.83 |
| SDXL-inpainting | 0.042 | 28.8 | 5.72 | 0.150 | 0.990 | 0.85 |
| step1x-edit | 0.053 | 38.7 | 5.93 | 0.157 | 0.974 | 0.79 |

All medians. Every generator, including the two "inpainting" pipelines, changes the whole frame (VAE round-trip or full re-generation). Pixels outside the box change 4–10× more than a q75 re-encode would change them. Any patch of a fake is therefore evidence, which is why global statistics and the VLM can both win without localising. It is also consistent with the low mIoU (24.7–28.8) in the reviews.

---

## 6. MedForge++ addition

Out-of-fold AUC, best of LR and GB per feature set, with exact duplicates removed. Source: `res_addition.csv`.

| comparison (pos / neg) | meta | meta, no extension | size | gbasic | gext | g512 | thumb 16² |
|---|---|---|---|---|---|---|---|
| **all 7,800: fake (4000) vs real (3564 after dedup)**, the paper's MF++ task | 0.766 | 0.766 | 0.487 | 0.826 | **0.893** | 0.877 | **0.895** |
| kidney: real (normal + tumour) vs removal | 0.503 | 0.503 | 0.482 | 0.548 | 0.705 | 0.713 | 0.573 |
| kidney: real_tumor vs removal (same slices) | 0.516 | 0.516 | 0.500 | 0.538 | 0.699 | 0.705 | 0.580 |
| kidney: normal vs removal | 0.515 | 0.515 | 0.460 | 0.550 | 0.713 | 0.711 | 0.601 |
| kidney: normal vs real_tumor (both real) | 0.524 | 0.524 | 0.460 | 0.525 | 0.530 | 0.547 | 0.563 |
| **kidney: real vs implant** | **1.000** | **1.000** | 0.486 | 1.000 | 1.000 | 1.000 | 1.000 |
| kidney: normal vs implant | 1.000 | 1.000 | 0.484 | 1.000 | 1.000 | 1.000 | 1.000 |
| kidney: implant vs its own source pool | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.974 |
| **kidney: KiTS normal vs implant source pool (both real)** | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.999 | **1.000** |
| **colon: polyp_real vs polyp_normal (both real)** | 0.820 | 0.820 | 0.499 | **1.000** | 1.000 | 1.000 | 1.000 |
| colon: real (real + normal) vs removal | 0.532 | 0.532 | 0.496 | 0.779 | 0.851 | 0.830 | **0.937** |
| colon: polyp_real vs removal (same frames) | 0.592 | 0.592 | 0.498 | 0.698 | 0.771 | 0.740 | **0.906** |
| colon: polyp_normal vs removal | 0.728 | 0.728 | 0.499 | 0.999 | 1.000 | 1.000 | 1.000 |
| **colon: real (real + normal) vs implant** | **1.000** | 1.000 | 0.500 | 0.976 | 0.999 | 0.984 | 0.999 |
| colon: polyp_real vs implant (same Kvasir frames) | 1.000 | 1.000 | 0.500 | 0.963 | 0.999 | 0.976 | 0.999 |
| colon: Kvasir original vs polyp_real (provenance) | 1.000 | 1.000 | 1.000 | 0.990 | 1.000 | 0.847 | 0.623 |

**Per-subset AUC slices of the all-7,800 GB model:**

| fake subset | meta | gext | thumb |
|---|---|---|---|
| kidney_inject | 1.000 | 1.000 | 1.000 |
| colon_inject | 1.000 | 0.997 | 0.994 |
| polyp_removal | 0.569 | 0.863 | 0.919 |
| kidney_removal | 0.497 | 0.712 | 0.666 |

**Top single features** (`res_addition_singlefeat.csv`):

- Implant vs real (both organs): `is_jpeg`, file size and bytes per pixel, each **AUC 1.000**.
- polyp_real vs polyp_normal: noise residual 0.981, high-frequency FFT 0.977, Laplacian variance 0.974.
- Kidney removal vs same-slice real: best single feature is std, at 0.529.

**Locality of the addition's forgeries** (`locality_add.py`, 150 pairs each, median MAD):

| pair | box fraction | MAD in box | MAD outside box | share outside > 8 levels |
|---|---|---|---|---|
| kidney_real_tumor vs kidney_removal | 0.005 | 39.8 | **0.19** | 0.000 |
| polyp_real vs polyp_removal | 0.141 | 31.3 | **5.67** | 0.133 |
| polyp_real vs colon_inject | 0.030 | 21.4 | 1.39 | 0.018 |

**Conclusions for the addition.**

- **Implant: shortcut (AUC 1.0).**
  - Format: JPEG vs PNG.
  - Colour: kidney implants have chroma 0.174; every KiTS image is exactly grayscale.
  - Source, for kidney: a different CT dataset, separable at AUC 1.0 from 16×16 thumbnails of *real* images alone.
  - The July finding that "Implant 100% is genuine forensic detection" rested on re-saving 20 reals as JPEG. That test checks neither the colour cue nor the source confound, so it does not support the claim.
- **polyp_normal: source confound (AUC 1.000).** Re-confirmed. It survives every re-encode in §7, so it cannot be fixed by preprocessing.
- **Kidney removal: no shortcut.** It is local and LaMa-clean (best AUC about 0.70). This is the only subset where a detector must find real evidence, and the evidence is nearly absent. That explains the 3% accuracy.
- **Polyp removal: content/semantic shortcut.** A 16×16 thumbnail gets 0.906 against same-frame reals, because "polyp present" means real in this data, plus out-of-box change from the inpainting. The colon task reduces to three rules:
  1. JPEG → fake.
  2. Kvasir style with no polyp → fake.
  3. Otherwise → real.

---

## 7. Does normalisation remove the shortcut?

`normalize_probe.py` re-encodes images **in memory**. Sample: 600 reals plus 60 per generator per MF90K organ, and 300 per addition subset; 5,988 images × 4 recipes. AUCs are from GB with 5-fold group CV (thumbnail with LR). Source: `res_norm.csv`.

Recipes:

- **R0:** as-is.
- **R1:** re-encode as JPEG q90.
- **R2:** force true grayscale for gray modalities, then JPEG q90 4:4:4.
- **R3:** R2 plus a non-integer resize to 720², which breaks the 8×8 grid.
- **R4:** R3 but saved as PNG.

| comparison | features | R0 | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|---|
| MF90K brain | file (size, q, mode) | 0.878 | 0.599 | 0.592 | 0.605 | 0.676 |
| MF90K brain | global (gext) | 1.000 | 1.000 | 0.994 | **0.965** | 0.963 |
| MF90K chest | file | 0.854 | 0.848 | 0.852 | 0.620 | 0.648 |
| MF90K chest | global | 1.000 | 1.000 | 0.998 | **0.990** | 0.992 |
| MF90K fundus | file | 0.745 | 0.725 | 0.709 | 0.623 | 0.651 |
| MF90K fundus | global | 0.986 | 0.979 | 0.979 | **0.954** | 0.962 |
| ADD kidney: real vs implant | file | 1.000 | 0.836 | 0.842 | 0.855 | 0.839 |
| ADD kidney: real vs implant | global | 1.000 | 1.000 | 1.000 | 0.997 | 1.000 |
| ADD kidney: real_tumor vs removal | global | 0.705 | 0.679 | 0.679 | 0.624 | 0.616 |
| ADD colon: polyp_real vs implant | file | 1.000 | 0.508 | 0.540 | 0.540 | 0.483 |
| ADD colon: polyp_real vs implant | global | 0.998 | 0.964 | 0.963 | 0.979 | 0.976 |
| ADD colon: polyp_real vs removal | thumb | 0.760 | 0.760 | 0.760 | 0.761 | 0.760 |
| ADD colon: polyp_real vs polyp_normal | global | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The thumbnail rows are unchanged by every recipe. Brain: 0.728 → 0.729; chest: 0.614 → 0.613; kidney implant: 1.000 → 0.997–1.000.

**Simple re-encoding is not enough.** It reduces file-level cues (to 0.60–0.68 for MF90K with R3/R4, and 0.48–0.54 for colon implants) and, with grayscale forcing, removes the colour cast. The re-rendering fingerprint (global AUC 0.95–0.99), the kidney source confound and the polyp_normal confound all survive.

**Paste-back works.** `paste_back.py` and `exp_pasteback.py` use 1,800 MF90K fakes (30 per organ × generator × type) and their 1,243 source reals. The generated pixels are kept only inside the GT box, the rest is taken from the real source, and real and fake are encoded identically (gray modalities as L, JPEG q90 4:4:4):

| organ | gext + file size: full-frame fake | **gext + file size: pasted-back fake** | thumb: full-frame | thumb: pasted-back |
|---|---|---|---|---|
| brain-mri | 0.989 | **0.813** | 0.892 | 0.609 |
| chest-xray | 0.998 | **0.750** | 0.958 | 0.601 |
| fundus | 0.979 | **0.642** | 0.962 | 0.558 |
| pooled | 0.989 | **0.743** | 0.941 | 0.584 |

The residual 0.64–0.81 comes from the edited 4.5% box itself (hard paste, lesion content). That is forensic evidence, which is what the benchmark should test. A feathered or Poisson blend at the box border would likely lower it further (**not measured**).

---

## 8. Conclusions and recommendations

### 8.1 Which results indicate a shortcut (AUC > 0.8)

| comparison | AUC | cue |
|---|---|---|
| MF90K, every organ, every generator, both forgery types | 0.99–1.00 (global), 0.75–1.00 (metadata) | colour cast, re-rendering, JPEG q |
| MF90K official test, shortcut GB trained on CF-strict, CM and ID | acc 97.3–98.3 (beats the VLM) | same cues |
| MF++ implants (kidney and colon) | 1.000 | format, colour (kidney), source (kidney) |
| polyp_real vs polyp_normal | 1.000 | source dataset |
| polyp removal vs real | 0.906–0.937 (thumbnail) | polyp presence, non-local inpainting |
| all-7,800 MF++ task | 0.893–0.895 | mixture of the above |

**Not a shortcut:**

| comparison | AUC |
|---|---|
| kidney removal vs real | ≤ 0.713 |
| kidney normal vs real_tumor | ≤ 0.563 |
| resolution, everywhere in MF++ and MF90K | 0.46–0.50 |

### 8.2 Data fixes, in priority order

1. **Paste-back compositing for every MF90K-style fake.** Copy only the generated region (the mask, dilated by a few px and feathered) into the original real; everything else stays byte-identical to the real. This is the only fix tested here that removes the whole-frame fingerprint: pooled AUC 0.989 → 0.743. It can be regenerated from existing files on CPU, because the source real and the box mask exist for every fake. 289 reals are missing on disk.
2. **One encoder for every image, real and fake.**
   - Decode, then convert gray modalities (MRI, X-ray, CT) to single-channel L. Equivalently, force R=G=B in the dataloader at train **and** test time.
   - Use the same size, and encode with one encoder at one quality (for example JPEG q90 4:4:4, or PNG).
   - This kills the one-line colour rule and weakens the q75/q95 cue. MF90K file-level AUC is 0.60–0.68 once a non-integer resize is included (R3/R4). Grayscale plus re-encode without the resize leaves chest at 0.852, because file size still tracks image noise.
3. **MF++ addition.**
   - Regenerate the kidney implants from KiTS slices (kidney_normal cases) instead of the external `Normal- (N).jpg` pool. As it stands, the kidney implant class is defined by dataset identity.
   - Save implants with the same encoder as reals.
   - Paste back the FLUX implants and the polyp LaMa removals (polyp removal changes out-of-box pixels: MAD 5.67).
   - Drop polyp_normal, or replace it with normals from the same Kvasir/HyperKvasir acquisition. It is a pure source confound, and it also has 94 duplicate groups.
4. **Split design.**
   - Cross-Forgery leaks through the generator fingerprint that both forgery types share. Report CF on **paste-back** data, and add a CF × CM cell (unseen type *and* unseen generator family).
   - Group every split by source image (MF90K real source, KiTS case, Kvasir stem).
   - Do not let the real class contain sub-populations with no fake counterparts (YG metastases, EyePACS, the other fundus sources, ODIR Cataract/Hypertension), or generate fakes from them.
   - Evaluate removal only against same-source reals.
5. **Put a shortcut baseline in the paper.**
   - Report the 16-feature GB row (98.3% on today's CF-strict test) and the one-line colour rule.
   - After the fixes, report them again to show they drop toward chance.
   - This directly answers reviewer A3/B4 ("editor-specific artifacts"). Today the data would *confirm* that concern.

### 8.3 GPU counterfactuals for Bridge (not measured here; cheap, 2,994 images each)

- **(a) Grayscale test set.** Force R=G=B on gray-modality test images, re-encode at uniform q, and re-evaluate. A large drop in fake recall means Bridge uses the colour cue.
- **(b) Colour-cast reals.** Add a ±1 tint to 1% of pixels in real images. If reals flip to fake, the shortcut is confirmed.
- **(c) Paste-back test set.** Evaluate on pasted-back fakes. Expect a drop if Bridge reads the whole-frame fingerprint.

---

## 9. Caveats

- Results come from the sample described in §1, not the full 95k. Stratification covers every organ × type × generator cell with 200 or more images.
- The shortcut models are trained on 5,000-image samples of each training file, not the full files. More data would only help them.
- The global statistics are computed on the decoded file. Qwen3-VL sees the same pixels at native 1024² (default `longest_edge`). Whether bf16 patch embedding preserves sub-1-level chroma differences is **not measured**. The SD-family casts (median 1.2–2.9 levels) are well above that resolution.
- The paste-back residual 0.743 uses a hard box paste. Blending was not tested.
- The JPEG quality estimate uses the sorted-table L1 fit to IJG tables. A residual of 0 for every MF90K image shows they are libjpeg/PIL-standard.

## 10. Reproducibility

All files are in `~/CIRCE/v3_audit/shortcut_work/`. Environment: `~/.conda/envs/aiface/bin/python` with `NUMPY_MADVISE_HUGEPAGE=0`; without it, numpy's hugepage madvise made every array operation 50–100× slower on this node.

| step | script | output |
|---|---|---|
| sample (46,358 images) | `build_sample.py` | `sample.jsonl` |
| per-image features | `extract.py` | `features.jsonl` |
| metadata tables | `tab_meta.py` | `tables_meta.md`, `tab_*.csv`, `tab_meta.log` |
| MF90K classifiers | `exp_mf90k.py` | `res_mf90k.csv`, `res_mf90k_slices.csv`, `res_mf90k_singlefeat.csv` |
| protocol transfer | `exp_protocol.py`, `exp_protocol_ablate.py` | `res_protocol.csv`, `res_protocol_ablate.csv` |
| addition | `exp_addition.py` | `res_addition.csv`, `res_addition_singlefeat.csv` |
| locality | `locality.py`, `locality_add.py` | `res_locality_*.csv`, `locality.json` |
| normalisation | `normalize_probe.py`, `exp_norm.py` | `features_norm.jsonl`, `res_norm.csv` |
| paste-back | `paste_back.py`, `exp_pasteback.py` | `features_pasteback.jsonl`, `res_pasteback.csv` |
| colour rule hit-rates | – | `rule_chroma.log` |
| addition provenance | – | `provenance.log` |
