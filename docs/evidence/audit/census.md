# MedForge++ / Bridge: data census and leakage audit

Auditor role: DATA CENSUS. Date: 2026-10-11. Everything here was read-only on data, code and checkpoints. No GPU was used. CPU jobs ran under `nice` with at most 16 processes.
Every number below comes from a script in `~/CIRCE/v3_audit/scripts/`. Its raw output is in `~/CIRCE/v3_audit/out/`. The image hash table (103,787 images: md5, 64-bit dHash, size, format, 32x32 thumbnail) is in `~/CIRCE/v3_audit/data/`.

---

## 0. Verdict (read this first)

1. **The image counts are about right. The independence of the images is not.** On disk there are **102,787 images**: 94,987 MF-90K plus 7,800 addition. The paper claims 103,076. Every protocol jsonl points at an existing file.
2. **The MF-90K protocol is split by image, not by source scan.** All 63,286 forgeries were made from only **3,790 authentic scans**. As a result:
   - **100%** of forged test images (1,999/2,000) have their source scan in In-Domain training, through the source real or a sibling forgery.
   - **99.9%** have the *same edit* (same scan, same type, same disease, made by another generator) in training.
   - The 3,000-image "clean" selection split has the same problem: 2,000/2,000. "Never used" does not mean independent.
   - In Cross-Model, **826/832** test forgeries from held-out generators have the identical edit from a *seen* generator in training.
3. **The MF++ 800-image eval split is inside its training set.** All 800 eval paths are in `circe_sft_v2.jsonl`, which trained the reported 82.25% model (log line: `[data] 7800 SFT records from .../circe_sft_v2.jsonl`). They are also all in `circe_sft_v3.jsonl`.
4. **The MF++ kidney-implant class has a dataset-origin confound.**
   - Kidney implants were made on a *different* CT dataset: Kaggle `Normal- (N).jpg`, 639/1000 of them 512 px upsampled. That dataset's originals are not in the benchmark. All kidney reals are KiTS PNG slices.
   - A linear model on 32x32 gray thumbnails separates kidney_inject from kidney_normal at **AUC 1.000**. It separates the *untouched* Kaggle normals from the KiTS normals at **AUC 1.000**. Kidney_inject against its own sources gives **AUC 0.530**.
   - Format is also a perfect cue: all 2,000 implants are JPEG and all 5,800 other addition images are PNG.
5. **The 64-bit dHash at Hamming ≤ 4 (the requested near-duplicate test) does not discriminate on this data.** It produces 377,173 "near-duplicate" chest real-real pairs, of which none passes a pixel check. I therefore report it together with a pixel-verified variant (thumbnail MAE ≤ 3; see Section 2).
6. **A leak-free grouped split is feasible.** Grouping by source scan, patient and duplicate cluster gives 24,718 groups, the largest with 225 images (Section 3.6).

---

## 1. How many samples do we have, exactly?

### 1.1 MedForge-90K on disk (`out/mf90k_census.txt`)

- The mapping lists 95,276 items. **94,987 exist on disk.** There are 0 files on disk that the mapping does not list.
- **289 files are missing, and all are reals:**
  - chest-xray: 251 (Pneumonia 243, Lung_Opacity 8)
  - fundus: 38 (Myopia)
- Real images on disk: **31,701**, against the 31,990 claimed. Implant 32,394 and removal 30,892 are complete.
- 631 forgeries have a source real that is missing on disk.
- The protocol builder silently dropped 165 SFT items with missing files (52,931 → 52,766).
- All 94,987 files are 1024x1024 JPEG (`out/near_dup.txt` A0).

Per organ, class and generator, all present on disk:

| organ / class | flux | gemini | gpt | qwen | seeddream | sd3.5-L | sd3.5-M | sd-inpaint | sdxl-inpaint | step1x | real | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| brain edit | 1027 | 1027 | 1027 | 1027 | 1027 | 1027 | 1027 | 1024 | 1027 | 1027 | | 10,267 |
| brain remove | 989 | 1004 | 998 | 1005 | 1005 | 973 | 973 | 961 | 973 | 940 | | 9,821 |
| brain real | | | | | | | | | | | 9,998 | 9,998 |
| chest edit | 1129 | 1129 | 1129 | 1124 | 1037 | 1129 | 1129 | 1098 | 1129 | 1129 | | 11,162 |
| chest remove | 1133 | 1133 | 1117 | 1133 | 1022 | 1133 | 1133 | 1100 | 1133 | 1133 | | 11,170 |
| chest real | | | | | | | | | | | 11,749 | 11,749 (12,000 listed) |
| fundus edit | 1117 | 1117 | 1067 | 1062 | 1067 | 1117 | 1117 | 1117 | 1117 | 1067 | | 10,965 |
| fundus remove | 1004 | 1003 | 1002 | 996 | 997 | 1004 | 1004 | 1003 | 1001 | 887 | | 9,901 |
| fundus real | | | | | | | | | | | 9,954 | 9,954 (9,992 listed) |

**Annotation assets:**
- 63,286 `gt/bbox_mask` PNGs, one per forgery. In a sample of 100, **100% are pure filled rectangles**, so these are box masks, not pixel masks (`out/mask_rectangularity.txt`).
- 57,103 `gt/triplet` PNGs: step1x-edit has none, and it also has no cot or explanation files.

**How many distinct scans:**
- **Duplicate reals.** There are 621 md5-identical groups of different real files (763 redundant files):
  - brain: 300 groups, Kaggle `Tr-*` vs `Te-*` duplicates
  - fundus: 321 groups, the same ODIR image filed in two disease folders (319 basenames appear in two folders)
- **After pixel verification:** 989 duplicate clusters cover 2,670 real files, with 1,681 redundant. That leaves **≈30,020 distinct real scans**.
- **Forgery sources:** the 63,286 forgeries come from **3,790 source scans** (brain 1,365, chest 1,238, fundus 1,187). Implants use 689 sources and removals 3,101. Each source has 4 to 110 forgeries; 2,686 sources have exactly 10.
- **28,200 reals are never manipulated.**

### 1.2 MedForge++ addition (`out/near_dup.txt` A0, `out/mfpp_leak.txt`, `out/mfpp_verified.txt`)

| subset | images | format | mask files | explanations | source / ID space | distinct after pixel dedup |
|---|---:|---|---:|---:|---|---:|
| kidney_normal | 1,000 | PNG 1024² | 0 | 1,000 | KiTS, 326 cases | 872 |
| kidney_real_tumor | 1,000 | PNG | 1,000 (rect) | 1,000 | KiTS, 320 cases | 791 |
| kidney_removal | 1,000 | PNG | 1,000 (rect) | 1,000 | **the same 1,000 slices** as real_tumor (LaMa) | 822 |
| polyp_normal | 800 | PNG | 0 | 800 | other colon set `train_normal_ (N)`; **564 unique md5, 236 byte-duplicates** | 494 |
| polyp_real | 1,000 | PNG | 1,000 (rect) | 1,000 | Kvasir, 1,000 frames | 997 |
| polyp_removal | 1,000 | PNG | 1,000 (rect) | 1,000 | **the same 1,000 Kvasir frames** (47 named `unresolved_*`) | 997 |
| colon_inject (`MedForge++/colon_injections/images`) | 1,000 | **JPEG** | **0** (boxes only) | — | **the same 1,000 Kvasir frames** as polyp_real; 145 rated "faint" | 997 |
| kidney_inject (**`~/AAAI/kidney/flux_injected/images`**, outside the MedForge++ root) | 1,000 | **JPEG** | **0** (boxes only) | — | **Kaggle CT `Normal- (N).jpg`**, not KiTS; originals in `~/AAAI/kidney/normal` (1,000 files, 996 unique md5, 499 distinct) | **541** |

- Totals: **7,800 files** (3,800 real / 2,000 implant / 2,000 removal). This matches the paper's file count.
- About **6,511** images are near-distinct.
- The authentic material underneath is small:
  - KiTS: 2,000 slices from **420 cases**
  - Kvasir: 1,000 frames, each used 3 times
  - colon normals: 564 unique
  - Kaggle kidney: 1,000 files, 499 distinct, and **their originals are not in the benchmark**
- There are 4,000 mask files and they are all rectangles. Implants have only box coordinates. The kidney manifests hold 1,006 records for 1,000 images (6 shard overlaps, with no box conflicts).

### 1.3 Reconciliation with the paper and the manifest (`out/manifest_audit.txt`)

| | paper | disk | note |
|---|---:|---:|---|
| MF-90K | 95,276 | **94,987** | 289 reals missing |
| addition | 7,800 | 7,800 | 2,000 implants have no masks; kidney implants live outside the data root |
| total | 103,076 | **102,787** | equals the number of unique image paths across all protocol files (102,787) |

About `medforgepp_manifest.json`:
- It has 103,067 records, 102,561 unique paths, **471 duplicated paths (506 extra records)**, and 353 duplicate groups whose fields conflict. In 175 of those, the same breast file is labelled both `inject` and `remove`.
- After deduplication: MF-90K 95,276 (paths under `~/AAAI/MedForge/data/MedForge-90K`), lung CT 4,760 (`.npy`, "medical-defect" CTGAN), breast MRI 525, colon 2,000 (Kvasir real plus FLUX).
- It contains **no kidney and none of `naitik_annotated`**. It describes a different dataset from the paper's MF++, and its total only happens to be close to 103,076.

**Paper text that the disk contradicts:**
- "Every forgery is paired with a pixel-level mask": the masks are rectangles, and the implants have none.
- "Implant forgeries start from an authentic normal image": colon implants start from polyp_real frames that already contain a polyp, and kidney implants come from an external dataset.
- "We adopt MedForge's 5:1:3 split": the split is a re-randomised image-level 5:1:3 split. Only 934 of the 2,994 test images are in MedForge's official test set, and our In-Domain training contains 16,680 images from MedForge's official test set (Section 3.5).

---

## 2. Exact and near-duplicates

### 2.1 Exact content (md5) (`out/near_dup.txt` A/B)

Within the data:
- 720 md5 groups with more than one file (1,004 redundant files): MF-90K real 621 groups / 763 redundant, polyp_normal 94 / 236, Kaggle kidney sources 4 / 4, MF-90K remove 1 / 1.
- **0** groups where identical bytes carry both a real and a fake label. **0** forgeries byte-identical to their source.

Training versus eval, identical bytes under a different path:
- shared test against every MF-90K training file: **31** (ID, CM and CF)
- v3_select: **27**
- circe_eval_split_v2: **70** against the PF+ replay files, which removed eval *paths* but kept byte-identical polyp_normal copies. All 800 eval paths are also path-identical to images in circe_sft_v2 and v3.
- Same-path overlap of the old `mf90k_train.jsonl` (80,225 rows): 2,543 of the shared-test images and 2,525 of the v3_select images.

### 2.2 64-bit dHash, Hamming ≤ 4 (as requested), and its calibration (`out/near_dup.txt` C/D, `out/near_dup_verify.log`)

Raw results as requested:

| eval | train file | eval images with a dHash≤4 neighbour in train |
|---|---|---|
| shared test | ID train | 2,362/2,994 (78.9%) |
| shared test | CF train | 2,218/2,994 (74.1%) |
| v3_select | ID train | 2,395/3,000 (79.8%) |
| circe_eval_split_v2 | circe_sft_v3 | 706/800 (88.2%) |

**This criterion is unreliable here.** Medical images share coarse gradient layouts:
- Real-real pairs within an organ at dHash ≤ 4: chest 377,173, fundus 422,103, brain 43,549.
- Their 32x32 thumbnail MAE has a median of 26.5.
- **Chest: 0 pairs with MAE < 4.**
- Kidney CT against colonoscopy or X-ray also produces dHash ≤ 4 pairs, with MAE ≥ 61.

So I added a verification step: a near-duplicate requires dHash ≤ 4 **and** thumbnail MAE ≤ 3. The threshold sits at the valley of the brain and fundus histograms, and 0% of the cross-modality false positives survive it.

Checking that the verified criterion catches real pairs (its recall):
- kidney_inject vs its Kaggle source: 100%
- colon_inject vs polyp_real: 99%
- kidney_removal vs tumor slice: 84%
- MF-90K forgery vs own source: 25.8% (generators repaint the whole image; out-of-mask difference is 3 to 13 gray levels, `out/edit_strength.txt`)

### 2.3 Verified near-duplicates, training file against eval (`out/near_dup_verified.csv`)

| eval vs train | verified near-dup | same source | **different source** (confirmed at 256 px) | opposite-label match |
|---|---:|---:|---:|---:|
| shared test vs ID | 765 (25.6%) | 688 | 140 (88; 20 byte-identical) | 349 |
| shared test vs CM | 691 (23.1%) | 614 | 138 | 345 |
| shared test vs CF | 502 (16.8%) | 417 | 132 (89; 22 byte-identical) | 322 |
| v3_select vs ID | 790 (26.3%) | 715 | 136 (84; 18 byte-identical) | 358 |
| v3_select vs CF | 510 (17.0%) | 426 | 125 | 320 |
| circe_eval_v2 vs circe_sft_v2/v3 | 800/800 identical paths, plus 523 more verified neighbours | 352 | 201 | 352 |
| circe_eval_v2 vs pfplus replays (eval paths removed) | 88 (11.0%) | 10 | 78 (mostly polyp_normal duplicates) | 10 |
| MF-90K evals vs MF++ training (and the reverse) | 0 | | | |

Where the confirmed different-source duplicates come from:
- Kaggle brain `Tr-*` vs `Te-*` copies of the same slice
- ODIR scans filed in two disease folders. For example, test `4077_left-Hypertension-remove` matches train `real/fundus/Hypertension/4077_left` while its own listed source is the Diabetes-folder copy. **Path-based source keys therefore *under*-count source leakage.**

---

## 3. Patient- and source-level leakage

### 3.1 The split was reproduced exactly (`out/source_leak.txt` §1–2)

- Rerunning `build_medforge_protocol.py`'s seed-42 shuffle on the local mapping (md5 eb4f6116…) reproduces `mfp_test_shared.jsonl` and `mfp_train_indomain.jsonl` exactly.
- The split is drawn **per image**.
- `mfp_v2_test_shared` equals `mfp_test_shared`.
- Each regime's v2 train ∪ v2 val equals its base file.
- **The v2 RL pool equals the v2 SFT train pool** in all three regimes, so there is no GSPO carve-out. The 10,586-item GSPO slice goes unused, except that 767 of its images ended up in v3_select.
- v3_select draws 767 images from that GSPO slice, 2,233 from the test pool and 0 from the SFT pool.

### 3.2 Source leakage per regime (`out/source_leak.csv`)

Definitions:
- **B** = forged eval image whose authentic source scan is itself a training real.
- **C** = forged eval image whose source scan appears in training through *any* image (the source real or a sibling forgery).
- **C′** = forged eval image whose *same edit* (same source, type and disease, but another generator) is in training.
- **E** = real eval image that is the source of at least one training forgery.

| eval | regime | B | C | C′ | E | implants C / C′ | removals C / C′ |
|---|---|---:|---:|---:|---:|---|---|
| shared test (2,000 forged / 994 real) | ID | 1,037 (51.8%) | **1,999 (100%)** | **1,998 (99.9%)** | 118 | 1000 / 1000 | 999 / 998 |
| | CM | 1,037 | 1,991 (99.6%) | 1,965 (98.2%) | 116 | 1000 / 979 | 991 / 986 |
| | CF | 1,037 | 1,558 (77.9%) | 998 (49.9%) | 98 | 559 / 0 | 999 / 998 |
| | CF + leaky implant replay | 1,037 | 1,959 (98.0%) | 1,445 (72.2%) | 115 | 960 / 447 | 999 / 998 |
| v3_select (2,000 / 1,000) | ID | 1,065 | **2,000 (100%)** | **2,000** | 126 | 1000 / 1000 | 1000 / 1000 |
| | CM | 1,065 | 1,993 | 1,970 | 126 | 1000 / 979 | 993 / 991 |
| | CF | 1,065 | 1,566 (78.3%) | 1,000 (50%) | 96 | 566 / 0 | 1000 / 1000 |

- **Cross-Model** (`§4`): 826/832 shared-test forgeries from the four held-out generators have the identical edit in CM training from a seen generator. For v3_select it is 796/800.
- **Cross-Forgery** (`§5`): the CF training set contains no implants. However, 559/1,000 test implants have their *untouched source scan* in CF training labelled real. The leaky replay adds the same edit for 447 implants and the same source, implanted with any disease, for 937.
- **Localisation is only mildly leaked** (`out/box_sibling.txt`). The best-matching sibling's training box has a median IoU of 0.37 with the test box (ID), and 32.4% of test forgeries have a sibling box at IoU ≥ 0.5. Each generator's box is different.

### 3.3 Patient level (`out/source_leak.txt` §6)

- **Fundus (ODIR / EyePACS):** the `<id>_left|right` naming identifies the patient, and 3,014 patients have both eyes in the dataset.
- **Brain (Yale metastases):** `YG_<pid>_<date>_<seq>` gives 591 patients, 2,418 exams and 2,975 images, up to 31 per patient.
- **Chest:** MIMIC study IDs only, so patient identity is **not measurable**. Kaggle brain `Tr/Te` files also carry no patient ID.

| eval | regime | rows with a patient key | patient has a *different* scan in train |
|---|---|---:|---|
| shared test | ID | 1,012 | **636 (62.8%)** (fundus 537, brain 99) |
| shared test | CF | 1,012 | 623 (61.6%) |
| v3_select | ID | 1,061 | 643 (60.6%) |

Yale same exam, different MR sequence, in CF training: 24/113 shared-test rows and 19/94 v3_select rows.

### 3.4 MF++ addition (`out/mfpp_leak.txt`, `out/mfpp_verified.txt`, `out/source_confound.txt`)

**Eval overlap with training files:**
- circe_eval_split_v2 shares **800/800 paths** with circe_sft_v2 and circe_sft_v3, and 600/800 with circe_sft_full.
- The dedicated MF++ SFT (82.25%, `ckpt circe_sft_v0.3_8b`) trained for 2 epochs on circe_sft_v2. **The number is training-set accuracy.**

**Even with eval paths removed (as the PF+ replay files do):**
- **42/100** polyp_real, polyp_removal and colon_inject eval images share their Kvasir frame with a training image.
- 70 eval images remain byte-identical to a training copy (polyp_normal).
- KiTS eval is almost case-disjoint: 1 to 2 shared cases per kidney subset. This is because the eval takes the last 100 cases in sorted order.

**Pairing structure inside the addition:**
- kidney_removal and kidney_real_tumor are the same 1,000 slices. kidney_normal shares 226 of its cases with them.
- polyp_real, polyp_removal and colon_inject are the same 1,000 frames.
- Within each subset there are many pixel-verified near-duplicate pairs, mostly adjacent slices of the same case: kidney_inject 634, Kaggle sources 710, kidney_real_tumor 280, kidney_normal 136.

**Confound test** (5-fold cross-validated logistic regression on 32x32 gray thumbnails, so neither file format nor high-frequency detail is available to it):

| pair | AUC |
|---|---:|
| polyp_real vs polyp_normal (positive control: two real sets from different sources) | 0.997 |
| **Kaggle kidney normals vs KiTS normals (both real)** | **1.000** |
| kidney_inject vs kidney_normal / vs kidney_real_tumor | **1.000 / 1.000** |
| kidney_inject vs its own Kaggle sources | 0.530 |
| polyp_removal vs polyp_normal | 0.998 |
| kidney_removal vs kidney_normal | 0.754 |

Pairs of the same images give cross-validated AUCs below 0.5 (0.25 to 0.28). That is an artefact of paired images being split across folds, and it means the classifier finds no signal.

Other properties:
- polyp_removal also changes the whole frame relative to its source: out-of-box |diff| median 5.45 gray levels, against 0.23 for kidney (`out/removal_diff.txt`).
- The 47 `unresolved_*` removals are genuinely edited: in-box |diff| median 25.96.

### 3.5 MedForge's official split, for context (`§7`)

- The official sft, grpo and test lists (55,276 / 10,000 / 30,000) share no paths.
- However, 19,990/20,000 official test forgeries have their source group in official training. **MedForge's published numbers carry the same source leakage.** That makes our protocol comparable to theirs, but it does not make either one sound for CVPR.
- Our test set overlaps theirs by only 934 images, so the head-to-head tables compare different test sets.

### 3.6 Is a leak-free split possible? (`out/group_components.txt`)

Joining images by forgery → source, patient key (both eyes; Yale patient), md5 duplicate and pixel-verified duplicate real gives:
- **24,718 groups** over 94,987 images
- median size 1, p90 11, **max 225**
- 3,400 groups contain forgeries, covering 68,305 images
- 21,318 groups are pure-real, covering 26,682 images

A grouped 5:1:3 split (by group, stratified on organ and type) is therefore straightforward.

---

## 4. Composition of every protocol file (`out/composition_table.md`, details in `out/composition.txt`)

Counts are rows by path-derived class. The organ columns count unique images.

| file | rows | unique | dup rows | real | implant | removal | chest | brain | fundus | kidney | colon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mfp_train_indomain | 52,766 | 52,766 | 0 | 17,604 | 17,927 | 17,235 | 18,930 | 16,740 | 17,096 | 0 | 0 |
| mfp_train_crossmodel | 38,534 | 38,534 | 0 | 17,604 | 10,643 | 10,287 | 13,832 | 12,273 | 12,429 | 0 | 0 |
| mfp_train_crossforgery | 34,839 | 34,839 | 0 | 17,604 | 0 | 17,235 | 12,786 | 11,039 | 11,014 | 0 | 0 |
| mfp_v2_train / rl _indomain | 47,490 | 47,490 | 0 | 15,845 | 16,157 | 15,488 | 17,059 | 15,039 | 15,392 | 0 | 0 |
| mfp_v2_train / rl _crossmodel | 34,681 | 34,681 | 0 | 15,814 | 9,612 | 9,255 | 12,428 | 11,034 | 11,219 | 0 | 0 |
| mfp_v2_train / rl _crossforgery_strict | 31,356 | 31,356 | 0 | 15,886 | 0 | 15,470 | 11,487 | 9,912 | 9,957 | 0 | 0 |
| mfp_v2_val_indomain (+_eval) | 5,276 | 5,276 | 0 | 1,759 | 1,770 | 1,747 | 1,871 | 1,701 | 1,704 | 0 | 0 |
| mfp_v2_val_crossmodel (+_eval) | 3,853 | 3,853 | 0 | 1,790 | 1,031 | 1,032 | 1,404 | 1,239 | 1,210 | 0 | 0 |
| mfp_v2_val_crossforgery_strict (+_eval) | 3,483 | 3,483 | 0 | 1,718 | 0 | 1,765 | 1,299 | 1,127 | 1,057 | 0 | 0 |
| mfp_replay_implant_focus_sft (LEAKY) | 6,400 | 4,000 | 2,400 | 800 | 4,678 | 922 | 1,673 | 1,341 | 986 | 0 | 0 |
| mfp_v2_replay_crossforgery_strict | 1,722 | 1,661 | 61 | 800 | 0 | 922 | 603 | 560 | 498 | 0 | 0 |
| mfp_crossmodel_replay | 77,068 | 38,534 | 38,534 | 35,208 | 21,286 | 20,574 | 13,832 | 12,273 | 12,429 | 0 | 0 |
| mfp_indomain_replay | 100,046 | 52,766 | 47,280 | 35,208 | 30,368 | 34,470 | 18,930 | 16,740 | 17,096 | 0 | 0 |
| raft_crossforgery / _cs | 1,490 / 1,322 | same | 0 | 746 / 696 | 0 | 744 / 626 | | | | | |
| raft_crossmodel / _cs | 997 / 1,354 | same | 0 | 464 / 638 | 272 / 369 | 261 / 347 | | | | | |
| mf90k_train (old, 8 gens) | 80,225 | 80,225 | 0 | 30,749 | 25,306 | 24,170 | 28,885 | 25,318 | 26,022 | 0 | 0 |
| circe_sft_v2 = circe_sft_v3 (images) | 7,800 | 7,800 | 0 | 3,800 | 2,000 | 2,000 | 0 | 0 | 0 | 4,000 | 3,800 |
| circe_sft_full | 5,800 | 5,800 | 0 | 3,800 | 0 | 2,000 | 0 | 0 | 0 | 3,000 | 2,800 |
| pfplus_replay_removal_focus_sft | 11,500 | 7,000 | 4,500 | 3,400 | 1,800 | 6,300 | 0 | 0 | 0 | 3,600 | 3,400 |
| pfplus_replay_v2_balanced_sft | 14,500 | 9,482 | 5,018 | 6,458 | 3,982 | 4,060 | 1,058 | 820 | 604 | 3,600 | 3,400 |
| **mfp_v2_test_shared** (= mfp_test_shared) | 2,994 | 2,994 | 0 | 994 | 1,000 | 1,000 | 1,060 | 1,006 | 928 | 0 | 0 |
| **v3_select/mfp_v3_select_clean** | 3,000 | 3,000 | 0 | 1,000 | 1,000 | 1,000 | 993 | 1,014 | 993 | 0 | 0 |
| **circe_eval_split_v2** | 800 | 800 | 0 | 400 | 200 | 200 | 0 | 0 | 0 | 400 | 400 |
| aggregate_test_medforgepp | 3,794 | 3,794 | 0 | 1,394 | 1,200 | 1,200 | 1,060 | 1,006 | 928 | 400 | 400 |

Generator coverage:
- The shared test has 179 to 219 images per generator. v3_select has exactly 200 per generator.
- The CM training files contain 6 generators (no gemini, gpt, sd3.5-M or sdxl-inpaint).
- CF training files contain 0 implants. The exception is the leaky replay, with 4,678 implant rows from 2,339 unique ID-train implants.

Label consistency: across all SFT files, the assistant's verdict disagrees with the path-derived label in 0 cases. The only exceptions are 2 `NONE` cases in RAFT files.

---

## 5. Other data and pipeline defects found while counting

1. **The "failure-targeted" replay for ID and CM is essentially uniform duplication.**
   - `mfp_crossmodel_replay` is every CM training row exactly twice.
   - `mfp_indomain_replay` is every row twice, except the 5,486 non-step1x fundus implants.
   - Cause: `build_regime_replay.image_type()` matches the substring `-edit/` in `step1x-edit/`. That puts every step1x image into a pseudo-stratum `('edit','step1x')`, and top_k=12 then covers almost all strata.
   - The CF strict replay also doubles its 61 step1x removals.
2. **The ID and CM validation eval files have `box: null` for every forged row** (3,517 and 2,063 rows). The paper's IoU-based five-way failure taxonomy therefore cannot have been computed for ID or CM.
3. **step1x-edit removals are labelled `edit`** in `mfp_v2_val_indomain_eval` (182 rows) and `mfp_v2_val_crossmodel_eval` (175 rows). In `mf90k_train_eval600`, all 192 removals are labelled `edit`.
4. **There is no separate RL pool.** `mfp_v2_rl_*` is byte-for-byte the same image set as `mfp_v2_train_*`.
5. **Format differences:**
   - MF-90K: all JPEG.
   - Addition: reals and removals are PNG (about 0.8 to 1.0 MB each); implants are JPEG (about 160 to 175 KB).
   - In MF-90K, real chest JPEGs have a median of 222 KB while forged chest JPEGs are about 100 to 109 KB. That is a compression-level difference a detector could exploit (`out/near_dup.txt` A0).
6. **Colon and polyp data quality:**
   - 145/1,000 colon injections are "faint" (Δ<12) and are all in training.
   - 47 polyp removals carry an unexplained `unresolved_` prefix.
7. **Unused images:**
   - 42,221 MF-90K images never appear in any `mfp_*` or `raft_*` training file.
   - Removing the shared test leaves 39,227, which matches the claim. Removing v3_select as well leaves 36,227.
   - If the old `mf90k_train.jsonl` also counts, only 6,558 are unused (`out/never_used.txt`).

---

## 6. Answers to the user's questions, from the data side

**How many samples?**
- 102,787 image files: MF-90K 94,987 (31,701 / 32,394 / 30,892) plus addition 7,800.
- About 30k distinct real MF-90K scans.
- All MF-90K forgeries come from 3,790 scans.
- The addition rests on 420 KiTS cases, 1,000 Kvasir frames, 564 unique colon normals and 499 distinct Kaggle kidney slices.

**Are they perfect?** No. The blocking issues are:
- the image-level split, which gives source leakage across 78 to 100% of forged test images
- the MF++ eval set sitting inside its training set
- the kidney-implant dataset-origin confound (AUC 1.0)
- the polyp_normal dataset-origin confound
- the box masks described as "pixel-level masks"

**Can we increase the number of samples?**
- Within MF-90K: 42,221 images are unused by the current pipeline, and the 10,586-image GSPO slice was never used for RL. But because each source scan appears in about 10 to 110 images, the honest unit of sample size is the **group**: 24,718 groups, 3,400 of them forgery-bearing.
- The cheapest real gain for MF++ is more *distinct sources*, not more edits per source:
  - more KiTS cases (420 used)
  - kidney implants made on KiTS normals
  - colon normals from the Kvasir source

## 7. Recommendations, in order

1. **Re-split MF-90K by group** (source scan + patient + verified duplicate; 24,718 groups). Re-run at least the CF-strict SFT and replay pipeline on it and report the gap. Keep the current split only as "MedForge-comparable".
2. **For the head-to-head with MedForge**, either evaluate on MedForge's official 30k test (and drop our 16,680 official-test images from training), or run MedForge-Reasoner on our grouped split. Stop saying "we adopt MedForge's 5:1:3 split".
3. **Rebuild the MF++ eval** so it is disjoint from training by KiTS case, by Kvasir frame stem, and by colon-normal md5 plus near-duplicate cluster. Retrain the dedicated SFT on the training part only, then re-report 82.25% and the aggregate 93.86%.
4. **Remove the kidney-implant confound.** Implant into KiTS kidney_normal slices, or add the Kaggle source normals as reals. Normalise all files to one format. Evaluate polyp removal only against Kvasir-source reals, never against polyp_normal.
5. **Fix the replay builder:** the step1x substring bug, add boxes to the ID and CM validation files, and do real failure targeting. **Give RL its own pool:** the unused GSPO slice, group-disjoint.
6. **Correct the paper and dataset card:**
   - total 102,787 on disk (or restore the 289 missing reals)
   - masks are rectangles; implants have boxes only
   - implant sources described accurately
   - regenerate or retire `medforgepp_manifest.json`

## 8. Reproduce

All scripts live in `~/CIRCE/v3_audit/scripts/`. Run them with `~/.conda/envs/aiface/bin/python`, in this order:

1. `census_hash.py` (16 processes, about 55 minutes on a shared node)
2. `protocol_parse.py`
3. `composition.py`
4. `mf90k_census.py`
5. `source_leak.py`
6. `mfpp_leak.py`
7. `near_dup.py`
8. `near_dup_verify.py`
9. `diffsrc_fullres.py`
10. `mfpp_verified.py`
11. `source_confound.py`
12. `group_components.py`
13. `box_sibling.py`
14. `edit_strength.py`
15. `removal_diff.py`
