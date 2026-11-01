# MedForge++ v3: data expansion plan (role: DATA EXPANSION planner)

Date: 2026-10-11. Author: expansion sub-agent. Read-only audit; no GPU used.
Scripts and logs are in `~/CIRCE/v3_audit/exp_*` and outputs in `~/CIRCE/v3_audit/exp_out/`.
Every number below comes from a command I ran (the log file is named next to it). Anything I did not
measure is marked **[estimate]** or **not measured**.

---

## 0. Short answers

| Question | Answer |
|---|---|
| How many samples do we have? | MF90K mapping 95,276 items, **289 of them missing on disk** (251 chest-xray real, 38 fundus real), so 94,987 usable. Kidney/colon addition: 7,800 files present (3,800 real / 2,000 implant / 2,000 removal). Paper total 103,076 is a file count, not a clean count. |
| Are they perfect? | **No.** The kidney implant class has the same source confound as polyp_normal. Implant-vs-real is separable at **AUC 1.000** with format-free features, because the implants were made on a different dataset (Kaggle CT-Kidney JPEGs, not KiTS). Other problems: polyp_normal is 564 unique images out of 800; 47 polyp "removals" were flagged `unresolved_` by the generator's own critic; all implants sit at **6 (kidney) / 8 (colon) fixed positions**; every colon implant is painted onto a frame that already has a real polyp; and colon implants are JPEG while all reals are PNG (AUC 1.000 with format features). About **5,808 of 7,800** addition images are usable as-is. |
| Can we increase the number of samples? | **Yes: about 16,800 new fakes for kidney and colon plus 10-15k same-source reals, in ~66 GPU-h (about 1 day on 3 GPUs). Optional lung-CT and source-disjoint MF90K edits add ~7k fakes for ~25 GPU-h more.** Every editor except FLUX-Kontext's full bf16 weights is already cached. Kvasir-SEG is fully used up: each of its 1,000 frames already appears as real, as removal and as implant background. KiTS23 is barely used: 420 of 489 cases contribute a median of **2 slices** each, out of 95,221 slices (13,059 normal-kidney, 11,164 tumor; 69 cases never touched). I reverse-engineered the exact KiTS preprocessing (window [-200, 300] HU, rot180+transpose, reversed slice index), so new slices will match v1 pixel for pixel. |
| What the reviewers asked for (data side) | A3/B4, "one editor per type": fixed with 4 new editors per forgery type plus 1 held-out editor per organ. A2/C1, "no zero-shot cross-organ": a lung-CT deepfake set with third-party editors (BTD: CT-GAN + SD) is **already on disk** (`~/medical defect`, 4,760 CT slices + 525 breast-MRI PNGs) and needs zero generation. C2, "CIs": an enlarged image-disjoint MF90K test of **11,916** images is **already built** (`exp_out/mfp_v3_test_large12k.jsonl`). Its 95% CI half-width at 94% accuracy is ±0.43 pp, versus ±0.85 pp now. |

---

## 1. What the MedForge++ addition was actually built from (measured)

| Subset (n) | Real source on disk | Editor / script | Evidence |
|---|---|---|---|
| kidney_normal (1,000), kidney_real_tumor (1,000) | **KiTS23** 2D slices (`case_XXXXX_slice_YYY.png`, 512x512 16-bit). 420 distinct cases (326 normal, 320 tumor, 226 shared); case ids 0-588 with **none in 300-399**, which is the KiTS23 id scheme | Naitik, `~/AAAI/naitik_data/file1.zip` | `ls`; PNG header |
| kidney_removal (1,000) | the same 1,000 KiTS tumor slices (identical file names) | LaMa via `~/AAAI/vcipher/Polyp_dataset/deepfake/removal` pipeline (simple-lama + YOLO critic) | set equality check |
| **kidney implant (1,000)** | **Kaggle "CT KIDNEY DATASET: Normal-Cyst-Tumor-Stone"** (`~/AAAI/kidney/normal/Normal- (N).jpg`, 785x634 JPEG) + 30 tumor exemplars (`Tumor- (N).jpg`). **Not KiTS** | `~/AAAI/kidney/inject_flux_kidney.py`: FLUX.1-Kontext GGUF-Q4 inpaint at <=512 px. The whole frame passes through the VAE (no composite) and is then stretched to 1024x1024 (aspect distorted) and saved as JPEG q95 | script read |
| polyp_real (1,000) | **Kvasir-SEG** (`cju*` ids). All 1,000 Kvasir-SEG frames | Naitik, `file2.zip` | 1000/1000 in `colon/kvasir_extracted` |
| polyp_removal (1,000) | the same Kvasir-SEG frames. 953 resolved + **47 `unresolved_`** (critic still detected a polyp after the max number of LaMa iterations) | LaMa + YOLO critic (vcipher) | filename diff |
| polyp_normal (800) | Kaggle **"WCE Curated Colon Disease Dataset"** normal class (`train_normal_ (N)`, 633x532). A **different source**. **564 unique** by md5 | Naitik | md5 count |
| colon implant (1,000) | the **same 1,000 Kvasir-SEG polyp frames** (1000/1000 overlap with polyp_real), so each implant image contains the real polyp plus the injected one | `~/AAAI/colon/inject_flux_batch.py`: FLUX-Kontext GGUF-Q4, feathered composite (background kept), JPEG q95 | stem overlap |
| (not in paper) BTD lung CT + breast MRI | `~/medical defect/` (7.0 GB). Kaggle "BTD - MRI and CT deepfake test sets" (Back-in-Time Diffusion, arXiv 2407.15169), license **AGPL-3.0** per its metadata. Series UIDs `1.3.6.1.4.1.14519.5.2.1.6279.6001.*` are LIDC-IDRI/LUNA16 | CT-GAN + fine-tuned SD (third-party) | `dataset-metadata.json`, CSVs |

BTD counts (from its CSVs): CT injection TM (real) 1,197 / FM-CTGAN 1,374 / FM-SD 1,374; CT removal TB (real) 382 / FB-CTGAN 382 / FB-SD 382; MRI 175 TM / 175 FM and 175 TB / 175 FB. Unique files on disk: 4,760 `.npy` CT slices (int16 HU, 512x512) and 525 MRI PNGs. The CSVs contain repeated rows.
**Bug in `medforgepp_manifest.json`:** the authentic BTD classes are labelled as forgeries: TM is labelled `inject` (lung 1,197 + breast 175) and TB is labelled `remove` (lung 382 + breast 175). Do not use that manifest for BTD labels.

---

## 2. Data-quality defects (all measured)

Source/format confound probe (`exp_confound.py`, log `exp_out_confound.log`). The features carry no forgery information: file format, 8x8 JPEG blockiness, global 16-bin gray histogram, Laplacian variance, saturation. Each pair gets a 5-fold logistic-regression AUC:

| Pair | All features | Histogram+sharpness only (no format) | Verdict |
|---|---|---|---|
| kidney implant vs kidney_normal (KiTS) | 1.000 | **1.000** | **source confound (new finding)** |
| kidney implant vs kidney_real_tumor | 1.000 | **1.000** | source confound |
| Kaggle CT-Kidney source (pre-injection) vs KiTS normal | 1.000 | 0.994 | confirms it is the source, not the edit |
| colon implant vs polyp_real (both Kvasir) | 1.000 | 0.674 | **file-format confound** (JPEG vs PNG); fixable |
| polyp_real vs polyp_normal | 0.992 | 0.987 | July audit's source confound reproduced |
| kidney removal vs kidney_real_tumor | 0.610 | 0.609 | clean (removal is semantic) |
| polyp removal vs polyp_real | 0.688 | 0.689 | clean |

**Clean count today.** About **5,808 of 7,800** addition images are usable: kidney normal/tumor/removal 3,000 (all KiTS), polyp_real 1,000, polyp_removal 953 (resolved only), and colon implants 855 ("clear" only; they still need format equalization). Drop all 800 polyp_normal and all 1,000 kidney implants. The 47 unresolved removals are labelled `remove` in both the mapping and `circe_sft_v3.jsonl`.

Other measured defects:
- **Fixed placement.** Kidney implants use **6** distinct box centres (32-px grid) across 1,000 images; colon implants use **8**. A localizer can memorize the templates. Measured in `exp_implant_quality.py`, log `exp_out_implant_quality.log`.
- **Faint lesions.** Kidney: 15.9% of implants have in-box mean|Δ| < 12 against the source. Colon: 16.9% on grayscale (the README reports 145/1,000 faint on its own metric). Outside-box mean|Δ| median is 1.96 (kidney, full VAE pass) and 1.41 (colon, composited).
- **Polyp boxes are huge.** polyp_real 31.8% >25% of the image, 9.1% >50%, 1.7% >90%. polyp_removal 31.4% / 8.4% / 1.6%. Kidney median box is 0.9% of the image.
- **Train/test overlap in the addition.** `circe_sft_v3.jsonl` (7,800 rows) contains **all 800/800** images of `circe_eval_split_v2.jsonl` verbatim. My regex matched 700; a substring check matched the 100 polyp_normal names, which contain spaces. Every eval item shares its KiTS case or Kvasir frame with training. This matches the note in `build_pfplus_failure_replay.py`. Log: `exp_out_mfpp_leak.log`.
- **MF90K source structure.** Fakes derive from only **3,790** source images, each edited 5-110 times; **28,200 of 31,990 reals have no derivative**. As a result:
  - **1,999/2,000** shared-test fakes have a same-source sibling in ID training (median 10 siblings), and 1,991/2,000 in CM training (median 6).
  - In CF, 1,037 test fakes have their own pre-edit source image in training labelled "real".
  - Effect measured in CF (existing gen files, `exp_out_sibling_effect.log`): on the shared test, implant accuracy is 83.5% (n=559) when the source was trained as real versus 88.4% (n=441) when it was not. On v3_select the same split gives 84.6% vs 85.7%.
  - For ID/CM the effect **cannot be measured**, because no sibling-free fakes exist. Only a source-disjoint test can settle it (§6).

---

## 3. Unused local raw sources

| Source | Local | Used | Unused |
|---|---|---|---|
| Kvasir-SEG (1,000 frames + masks) | `~/AAAI/colon/kvasir.zip`, `kvasir_extracted/` | 1,000 as real, 953+47 as removal, 1,000 as implant background | **0 frames** |
| KiTS23 (2D slices only, no volumes on disk) | `naitik_data/file1.zip` | 2,000 slices from 420 cases | volumes must be downloaded (§5). 95,221 slices; ~2.2-2.7k decorrelated tumor and ~2.3-2.6k normal-kidney slices; **69 cases never used** (§3a) |
| Kaggle CT-Kidney normal JPEGs | `~/AAAI/kidney/normal` (1,000) | 1,000 as implant source | do not reuse (confounded source) |
| WCE curated-colon normal | `naitik_data/file2.zip` (800, 564 unique) | all | do not reuse (confounded source) |
| BTD lung CT / breast MRI | `~/medical defect/` | **not used in the paper** | **all 4,760 CT + 525 MRI** |
| MF90K never-used | `medforge90k_full` | see §6 | **36,002** present on disk |
| Spine MRI DICOM | `~/medical_imaging` (17 GB) | unrelated project | not proposed |

No other KiTS, Kvasir, CVC, PolypGen, BraTS, LIDC, LiTS, ISIC, CheXpert, MIMIC or EyePACS copies exist under `~` (`find` to depth 6-7 for directory names and for `*.nii*/.mhd/.nrrd/.dcm`, excluding conda/caches).

### 3a. KiTS23 slice budget (from the 0.14 GB of segmentations, streamed into memory)
Source: `exp_kits_slices.py`, log `exp_out_kits_slices.log`. Labels are 1 kidney, 2 tumor, 3 cyst.
- **489 cases, 95,221 axial slices.** Slice thickness median 3.0 mm (range 0.5-5.0).
- **32,375** slices contain kidney.
- **11,164** slices have a tumor of at least 200 px.
- **13,059** "normal-kidney" slices: kidney at least 2,000 px and no tumor or cyst.
- Per case, the median is 14 tumor slices and 15 normal-kidney slices. 402 cases have 6 or more tumor slices; 399 have 6 or more normal slices.
- Adjacent slices are correlated, so the usable count is lower: about **2,245 tumor / 2,263 normal** slices at 10 mm or more apart, or **2,729 / 2,579** when capped at 6 per case.
- **69 cases were never used by v1** (675 tumor and 568 normal slices). Make them the patient-disjoint kidney **test** set.

The exact v1 preprocessing, recovered by `exp_kits_window_fit2.py` (log `exp_out_kits_window2.log`) on case_00061:
`PNG16 = clip((HU + 200) / 500, 0, 1) * 65535`, i.e. window **[-200, 300] HU**. The array transform is **rot180 + transpose**, and the slice index is **reversed** (`slice_022` is array index 6 of 29, i.e. n-1-k). Fit correlation is 1.0000. The 1024 export, recovered by `exp_kits_export_fit.py` (log `exp_out_kits_export.log`), is `uint8 = uint16 >> 8`, then a nearest-neighbour 2x upscale to 1024x1024, gray copied to 3 channels, saved as PNG; mean |error| against the v1 file is 0.0000. New KiTS slices extracted this way are pixel-identical in style to v1, so the new kidney implants need no source fix beyond "use KiTS".

---

## 4. Cached generative editors (no download needed)

All are complete (no `.incomplete` blobs). They live in `~/.cache/huggingface/hub` unless noted.

| Model | Size | Use | Pipeline (env with it) |
|---|---|---|---|
| FLUX.1-Kontext-dev **GGUF Q4_K_M** + base components | 6.9 GB GGUF (`~/AAAI/models/kontext_gguf`); full bf16 transformer **not** cached | current implant editor | `FluxKontextInpaintPipeline` (`~/AAAI/fluxenv`, diffusers 0.39.0.dev0) |
| FLUX.1-dev (bf16, full) | 54 GB | new implant + removal editor | `FluxInpaintPipeline` (fluxenv, gapl_env 0.38) |
| FLUX.1-schnell | 54 GB | optional fast variant | `FluxInpaintPipeline` |
| SD-1.5 inpainting (`booksforcharlie/stable-diffusion-inpainting`, 9-channel UNet, runwayml mirror) | 5.2 GB | new editor; same family as MF90K's `stable-diffusion-inpainting` | `StableDiffusionInpaintPipeline` |
| **Qwen-Image-Edit-2511** | 54 GB (5/5 transformer shards) | instruction editor (add/remove lesion) | `QwenImageEditPlusPipeline` (fluxenv, gapl_env; **not** in aiface 0.35) |
| SDXL base 1.0 (fp16) | 6.7 GB | held-out editor via `StableDiffusionXLInpaintPipeline` (masked img2img) | all envs |
| big-lama (`smartywu/big-lama/best.ckpt`, 410 MB) | 755 MB | removal | needs the advimman/lama code (not local). `simple_lama_inpainting` is installed in fluxenv but wants TorchScript `big-lama.pt` (**206 MB, not cached**: pre-fetch) |
| SAM 2.1 hiera-large | 1.7 GB | organ/mucosa masks for placement | - |
| Kandinsky 2.2 prior+decoder | 15 GB | text2img only (decoder in_channels=4); the inpaint decoder is **not** cached (15.6 GB repo, ungated) | skip |
| PixArt-Sigma, SD-1.4, SD-1.5, sd/sdxl-turbo, Kolors, Stable Cascade, IP-Adapter | - | no inpainting variant | skip |
| Not cached: FLUX.1-Fill-dev (gated, 58 GB), SDXL-inpainting-0.1 (ungated; MF90K CM held-out generator) | - | optional pre-fetch | - |

Measured throughput (July logs `~/AAAI/kidney/klog_*`, `~/AAAI/colon/plog_*`): FLUX-Kontext GGUF at <=512 px, 28-34 steps, takes **11.3-19.8 s per image per process**. Eval throughput: **1.6 s/image** (3,000 images in 78-82 min, `v3_select/log_sel_*`). Login-node download from Hugging Face: **17.8 MB/s** single stream (50 MB KiTS case piped to /dev/null).

---

## 5. Public datasets for expansion (verified on the web today)

| Dataset | What it adds | Size | License | Access |
|---|---|---|---|---|
| **KiTS23** (HF `neheller/KiTS-Challenge-Imaging`, ungated) | 489 cases with kidney/tumor/cyst masks | imaging **42.1 GB** + segs 0.14 GB (HF API) | **CC BY-NC-SA 4.0** (README: academic use needs no permission) | `kits23_download_data` or HF; about 40 min at the measured 17.8 MB/s |
| **HyperKvasir** (same Simula/Kvasir data family as Kvasir-SEG; its 1,000 'segmented images' are the Kvasir-SEG frames) | same-source **normal lower-GI** frames: normal-cecum **1,009**, bbps-2-3 **1,148**, retroflex-rectum 391 (official 2-fold split CSV); polyp 1,028 (it contains the Kvasir-SEG polyps) | labeled-images zip **3.9 GB** (full 58.6 GB) | **CC BY 4.0** | datasets.simula.no / OSF |
| **PolypGen** | 6 centres; 1,537 polyp frames + masks, 2,225 positive sequence frames, **4,275 negative frames** (from 4 of the centres) | ~2 GB | CC BY 4.0, educational/research use | **Synapse** (account + login required: risk) |
| CVC-ClinicDB | 612 frames (384x288) + masks, 29-31 sequences | small | research/education only, no commercial use | polyp.grand-challenge.org |
| LUNA16 / LIDC-IDRI | lung nodules (CT-GAN lineage) | LUNA16: 10 subsets of 6.0-7.3 GB (Zenodo), annotations.csv. LIDC-IDRI: 1,010 patients, 133 GB | LUNA16 CC BY 4.0; LIDC CC BY 3.0 + TCIA DUA | optional: BTD already covers lung |
| MSD (Task03 Liver = LiTS, Task06 Lung, Task10 Colon CT) | liver tumors etc. | Liver **28.9 GB**, Lung 9.2 GB, Colon 6.2 GB (S3 Content-Length) | CC BY-SA 4.0 | not recommended in 2 weeks |
| ISIC 2020 | dermoscopy | 32 GB, 33,126 images | CC BY-NC 4.0 | not recommended (new modality, out of scope for this window) |
| Kaggle CT-Kidney (Normal-Cyst-Tumor-Stone) | the source of the current kidney implants | 12,446 images | stated CC BY | **do not use for implants** (confound) |
| Kaggle BTD | lung CT + breast MRI deepfakes | already local | AGPL-3.0 | already on disk |

Release consequence: KiTS23 (NC-SA), Kvasir-SEG/CVC (research-only) and FLUX.1-dev (non-commercial) force a **CC BY-NC-SA** release of MedForge++ v3.

---

## 6. MF90K never-used pool and test enlargement (measured)

`exp_unused_protocol.py`, log `exp_out_protocol.log`. I took the union of all current protocol files (mfp_train_*, mfp_v2_*, all replay sets, raft, mix, shared test, v3_select, circe_*, CIRCE/v3) and found **36,291** MF90K images never referenced, of which **289 are missing on disk**, leaving **36,002 usable**. Usable composition: real 12,023 / edit 12,402 / remove 11,577; **9,585** are fakes from the CM held-out generators. This is in line with the 39,227 figure: 36,002 + the 3,000 now spent on v3_select is 39,002 (the 225 gap was not resolved).
- Caveat 1: all 36,002 also appear in the **old v1 pools** `mf90k_train.jsonl` / `mf90k_test_*.jsonl`. The current checkpoints' `adapter_config.json` give base = Qwen3-VL-8B-Instruct, and the launcher trains on `mfp_train_*`. I found no evidence the current lineage saw `mf90k_train.jsonl`, but **confirm this from the training logs** before reporting on the large test.
- Caveat 2: an image-disjoint test is **not source-disjoint**. A source-disjoint never-used pool contains **10,784 reals and 0 fakes**: fundus/normal 1,690, brain-mri/normal 614, chest-xray/normal 365, plus 8,115 diseased (`exp_out_disjoint.log`).

**Built (CPU, under v3_audit):** `exp_out/mfp_v3_test_large12k.jsonl`, **11,916 images**.
- Composition: 3,916 real, equal per organ (chest is short because of the missing files); 4,000 edit and 4,000 remove at **800 per generator**, so 3,200 are CM held-out fakes.
- Overlap is 0 with the shared test and 0 with v3_select. The format matches `mfp_v3_select_clean.jsonl`, with boxes from the bbox masks via the same function as `build_medforge_protocol.py`.

| n | 95% Wald half-width at p=0.94 |
|---|---|
| 2,994 (current) | ±0.85 pp |
| 6,000 | ±0.60 pp |
| 11,916 (built) | ±0.43 pp |
| 20,000 | ±0.33 pp |

Eval cost at the measured 1.6 s/img is **~5.3 GPU-h per checkpoint**.
Also written: `exp_out/mf90k_strict_real_pool.jsonl` (10,784 source-disjoint reals) as the base for a source-disjoint "MF90K-SD" test (plan item D2).

---

## 7. The two-week plan (3 x 80 GB GPUs, H100 per PROJECT_STATUS; ~1,008 GPU-h total)

Budget for data generation + QC + new-test evaluation is **about 190 GPU-h (~19%)**. That leaves ~800 GPU-h for training. Generation finishes by day 5.

### Design principle
One export pipeline for every class: decode, then 1024x1024 **PNG**, same resize kernel. Every fake is paired with a same-source real, and placement is inside an organ mask at random positions. Splits are group-disjoint (KiTS case, Kvasir/HyperKvasir pHash cluster, PolypGen centre). Each organ gets **4 training editors plus 1 held-out editor per forgery type**, which enables leave-one-editor-out (LOEO) evaluation and an editor x organ factorial.

### (a) Kidney CT and colonoscopy: editors, counts, GPU-hours

Throughput per process: Kontext is **measured** (15 s); all others are **[estimate]** for H100 bf16. SD-1.5-inpaint 512 px, 30 steps: ~2 s. SDXL inpaint 1024 px: ~5 s. FLUX.1-dev inpaint 1024 px, 28 steps: ~15 s. Qwen-Image-Edit-2511 1024 px, 40 steps with true CFG: ~40 s. LaMa ~0.2 s on GPU. Telea/Poisson run on CPU.

Per organ (kidney = KiTS slices; colon = HyperKvasir normal frames for implants, Kvasir-SEG/PolypGen polyp frames for removals). Targets are post-QC; generate 1.3x:

| Forgery | Editor | Role | Target | GPU-h (1.3x) |
|---|---|---|---|---|
| implant | FLUX-Kontext GGUF (existing editor; KiTS sources, mask-composited) | train | 800 | 4.3 (measured rate) |
| implant | FLUX.1-dev inpaint | train | 800 | 4.3 [estimate] |
| implant | SD-1.5 inpaint | train | 800 | 0.6 [estimate] |
| implant | Qwen-Image-Edit-2511 ("add a hypodense renal mass" / "add a sessile polyp"), composited by mask | train | 600 | 8.7 [estimate] |
| implant | Poisson splice of a real lesion patch (KiTS tumor / Kvasir polyp) via `cv2.seamlessClone`, a non-generative CT-GAN-style editor | train | 800 | CPU |
| implant | SDXL base inpaint | **held-out** | 400 | 0.7 [estimate] |
| removal | LaMa (existing editor) | train | 800 | <0.1 |
| removal | SD-1.5 inpaint ("normal renal parenchyma" / "normal colonic mucosa") | train | 800 | 0.6 |
| removal | FLUX.1-dev inpaint | train | 800 | 4.3 |
| removal | Qwen-Image-Edit-2511 ("remove the tumor/polyp") | train | 600 | 8.7 |
| removal | OpenCV Telea/NS inpaint | train | 800 | CPU |
| removal | SDXL base inpaint | **held-out** | 400 | 0.7 |
| **per organ** | | | **8,400 fakes** | **~33 GPU-h** |

Reals per organ:
- Kidney: about 2,500 normal-kidney + 2,700 tumor slices from KiTS23, at most 6 per case (§3a). Implants and removals reuse each source slice across editors (~1.6 edits per slice). This paired design lets editors be compared on identical anatomy, and case-level splits keep it leak-free.
- Colon: 1,000 Kvasir-SEG + up to 1,537 PolypGen polyp frames (+612 CVC-ClinicDB if research-only is acceptable). Normals: 2,157 HyperKvasir (cecum + bbps-2-3) + up to 4,275 PolypGen negatives.

**Kidney + colon total: ~16,800 new fakes + ~10-15k reals, about 66 GPU-h, roughly 1 day of wall time on 3 GPUs.** Qwen-Edit is the long pole (~35 GPU-h) and gets its own GPU.

### (b) New organ: lung CT, zero-shot cross-organ (recommended, nearly free)
- **B1 (0 GPU-h for data):** convert BTD to PNG on CPU (lung window; HU int16 npy to 1024 PNG). This gives an **external lung-CT test with third-party editors**: CT-GAN, the Mirsky lineage that reviewers know, plus fine-tuned SD.
  - Reals: TM 1,142 + TB 376 unique files. Fakes: FM-CTGAN/FM-SD 1,248 each, FB-CTGAN/FB-SD 376 each. Plus 525 breast-MRI PNGs.
  - Evaluate the MF90K-only CF/CM/ID checkpoints on it: about 2.1 GPU-h per checkpoint at 1.6 s/img. This is the clean answer to A2/C1: a single model trained only on MF90K, an unseen organ, unseen editors, no routing.
  - Note: TM and TB are **real**. The manifest mislabels them.
- B2 (optional, ~12 GPU-h [estimate]): apply 3 of our editors (SD-1.5, FLUX.1-dev, Qwen-Edit) to BTD real slices (implants on TB, removals on TM at the CSV x,y), 300 per editor per type. This completes an **editor x organ** grid (kidney, colon, lung) that separates editor fingerprints from organ shift.
- **B0 (0 GPU-h for data):** the MF90K-only CF/CM/ID checkpoints have never seen the kidney/colon addition. Paths from `naitik_annotated`, `colon_injections` and `kidney/flux_injected` appear only in `circe_*`, `pfplus_*`, `aggregate_test_medforgepp`, `mix8` and `leak_test` files, never in `mfp_*` files. So a zero-shot run on all 7,800 (better: on the rebuilt confound-free v3 sets) is a valid cross-organ test, at about 3.5 GPU-h per checkpoint. Someone is currently running zero-shot on the 800-image split (`v3_select/queue_zs.txt`).
- Liver (MSD Task03 = LiTS, 28.9 GB): **not recommended** in this window.

### (c) Fix the source confounds
1. **polyp_normal:** retire the WCE curated-colon normals (564 unique). Replace with HyperKvasir normal-cecum 1,009 + bbps-2-3 1,148 (same Simula/Kvasir data family as Kvasir-SEG) and, if Synapse access works, PolypGen negatives matched to the positive frames' centres.
2. **kidney implants:** retire the 1,000 Kaggle-CT-Kidney implants (keep them only as a documented "source-shortcut" stress test). Regenerate on KiTS normal-kidney slices with the recovered v1 window, placing lesions inside the KiTS kidney mask (label 1) with sizes drawn from KiTS tumor areas.
3. **colon implants:** place them on HyperKvasir/PolypGen **normal** frames, not on frames that already contain a polyp.
4. **Format:** one PNG export for everything. Then apply random JPEG(q 70-95) and resize augmentation identically to all classes at training time.
5. **Gate:** rerun `exp_confound.py` on the rebuilt sets. Accept only if every fake-vs-matched-real and real-vs-normal pair has format-free AUC ≤ 0.75 (today: 1.000 / 0.992).

### (d) MF90K test enlargement
- **D1 (done, CPU):** `mfp_v3_test_large12k.jsonl` (11,916). Report the MedForge-compatible shared test (n=2,994) for comparability and the 12k test with bootstrap CIs and paired McNemar tests for every claim. It also contains a 3,200-fake **CM held-out-generator-only** subset; the shared test has 832.
- **D2 (optional, ~13 GPU-h [estimate]):** build a **source-disjoint** MF90K test.
  - Edits: 2,669 never-used normal sources (fundus 1,690, brain 614, chest 365), with masks drawn from the MF90K bbox-mask bank of the same organ and edited by SD-1.5 + FLUX.1-dev, about 5,300 fakes.
  - Reals: the 10,784 strict reals.
  - Removal fakes are not possible without lesion localization on the 8,115 diseased reals, so D2 covers implant vs real only.
  - D2 decides whether the same-source siblings inflate ID/CM accuracy, which cannot be measured with the current data.

### Schedule
| Day | Where | Work |
|---|---|---|
| D0 (Oct 12) | login node mgmt01 (internet) | Pre-fetch: KiTS23 imaging 42.1 GB (~40 min measured rate), HyperKvasir labeled 3.9 GB, PolypGen ~2 GB (needs a Synapse login), `big-lama.pt` 206 MB, optional SDXL-inpaint-0.1. Smoke test of 10 images per editor on one GPU with `HF_HUB_OFFLINE=1`. |
| D1 | CPU (≤16 procs, nice) | KiTS slice extraction with the v1 transform (window, rot180+T, reversed index, >>8, nearest 2x). Split by case: the **69 never-used cases are the test set**; the other 420 go 90/10 train/val; md5+pHash dedup of HyperKvasir/PolypGen; BTD to PNG; real-vs-real confound gate. |
| D2-D3 | 3 GPUs | GPU-A: Qwen-Edit (both organs, ~35 h). GPU-B: FLUX.1-dev + Kontext (~26 h). GPU-C: SD-1.5, SDXL, LaMa, then lung B2 and MF90K D2 (~30 h). CPU in parallel: Poisson, Telea. |
| D4 | CPU + 1 GPU | QC gates (§8); regenerate rejects; human QC sample. |
| D5 | CPU | Freeze v3 data; protocol files (LOEO, cross-organ, cross-centre, group-disjoint); dataset card; then hand off to training. |
| D6-D14 | - | Training and evaluation (other roles). New-test eval is about 14 GPU-h per checkpoint (12k MF90K + 7,800 zero-shot MF++ + BTD + new kidney/colon test). |

---

## 8. Quality-control gates (automatic unless stated)

1. **Visibility:** implant in-box mean|Δ| vs source ≥ 12 (README threshold). Removal: no residual lesion. Colon uses the vcipher YOLO critic idea; kidney checks the tumor-HU fraction inside the original tumor mask.
2. **Locality:** outside-mask mean|Δ| ≈ JPEG/resize noise for masked editors (today's medians: 1.41 colon, 1.96 kidney). Record it for the global Qwen editor.
3. **Confound probe** (`exp_confound.py`): format-free AUC ≤ 0.75 for every fake vs matched real and every real vs normal.
4. **Placement diversity:** ≥ 200 distinct box centres per editor (today 6 / 8), all inside the organ mask.
5. **Duplicates:** md5 + pHash (today polyp_normal has 236 redundant copies).
6. **Group-disjoint splits:** KiTS case, Kvasir frame (its real/removal/implant triplets go to the same split), PolypGen centre.
7. **Labels:** drop or relabel the 47 `unresolved_` polyp removals. TM/TB in BTD are real.
8. **Human QC (also addresses A6):** 2 raters × 200 stratified images per organ, plausibility on a 1-5 scale plus "can you see the edit". Report Cohen's κ.

---

## 9. Risks

- **Compute nodes have no internet.** Everything must be pre-fetched on mgmt01 and run with `HF_HUB_OFFLINE=1`.
  - `simple_lama` downloads `big-lama.pt` on first use and will crash offline unless `LAMA_MODEL` points to a pre-fetched file.
  - Qwen-Image-Edit-2511 needs fluxenv or gapl_env (diffusers ≥ 0.36); aiface (0.35.1) lacks `QwenImageEditPlusPipeline`.
- **Disk quota.** `quota -s` reports **19.85 T used vs 800 G quota / 1 T limit** (flagged `*`) on /home. Writes work today, but enforcement would block the 42 GB KiTS download and the new images (~30k PNGs, roughly 30-60 GB **[estimate]**). Confirm with admins or write to a scratch filesystem.
- **PolypGen needs a Synapse account.** Fallback: HyperKvasir alone (2,157 normals) plus Kvasir-SEG/CVC for polyps.
- **Editor-fingerprint shortcut grows with editors.** Mitigate with leave-one-editor-out reporting and the held-out SDXL editor. Do not merge held-out editors into training.
- **Kidney removal stays semantically hard** (removal vs real AUC 0.61). More editors will not by themselves fix the 3% kidney-removal recall; diffusion editors may add detectable fingerprints that do not transfer to LaMa.
- **Licenses:** KiTS NC-SA, Kvasir-SEG/CVC research-only, FLUX.1-dev non-commercial, BTD AGPL-3.0. The release must be non-commercial.
- **Time:** about 4.5 weeks to the deadline. If generation slips past D5, cut Qwen-Edit to 300 per type first (it is the long pole) rather than cutting editors.
- **mf90k_train lineage** (§6 caveat 1) must be confirmed before reporting the 12k test.

---

## 10. Files produced (all under `~/CIRCE/v3_audit/`)
- Scripts: `exp_unused_scan.py`, `exp_unused_protocol.py`, `exp_source_disjoint.py`, `exp_test_siblings.py`, `exp_sibling_effect.py`, `exp_confound.py`, `exp_implant_quality.py`, `exp_mfpp_group_leak.py`, `exp_kits_slices.py`, `exp_kits_window_fit2.py`, `exp_kits_export_fit.py`, `exp_build_large_test.py`.
- Logs: `exp_out_*.log`.
- Data: `exp_out/mfp_v3_test_large12k.jsonl` (11,916), `exp_out/mf90k_strict_real_pool.jsonl` (10,784), `exp_out/mf90k_never_used_protocol.json`, `exp_out/mf90k_strict_pool.json`, `exp_out/kits23_slice_counts.json`.

**Not measured:**
- GPU throughput of every editor except FLUX-Kontext GGUF.
- LaMa speed.
- Bandwidth from Zenodo, Synapse and Simula.
- Exact on-disk size of the new images.
- Whether HyperKvasir normals pass the confound gate (needs the download).
- The verdicts in the July `leak_test` (outputs truncated before the verdict).
