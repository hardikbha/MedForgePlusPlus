# Dataset card: release snapshot

## Purpose and versions

MedForge++ studies detection and localization of inserted or removed lesions in medical images. The paper's extension adds kidney CT and colonoscopy to MedForge-90K's brain MRI, chest X-ray and fundus imaging. MF-Clean is a separate source-grouped, shortcut-controlled rebuild of the original organs. Early mixed lung/breast manifests are historical artifacts and must not be conflated with either dataset.

The paper's nominal corpus has 103,076 mapping entries/files; historical filesystem auditing found 102,787 existing image files (94,987 upstream plus 7,800 extension), with duplicates and missing upstream reals. See the [current filesystem audit](evidence/release_data_audit.json). These are image counts, not unique patients. Do not advertise the corpus as 103,076 independent scans.

## Extension composition

| Subset | Nominal files | Label | Source and concern |
|---|---:|---|---|
| Kidney normal | 1,000 | Real | KiTS slices; correlated patient/volume slices |
| Kidney real tumor | 1,000 | Real | KiTS slices |
| Kidney removal | 1,000 | Fake/removal | LaMa edits of the real-tumor slices |
| Kidney implantation | 1,000 | Fake/implant | FLUX edits of a different Kaggle kidney CT source pool |
| Polyp normal | 800 | Real | Different source from Kvasir; duplicates |
| Polyp real | 1,000 | Real | Kvasir-SEG frames |
| Polyp removal | 1,000 | Fake/removal | LaMa edits of those Kvasir frames |
| Colon implantation | 1,000 | Fake/implant | FLUX edits of those Kvasir frames |

Original implants are JPEG; the remaining extension files are PNG. Format and source origin confound label interpretation. The original 800-image evaluation overlaps training. **This version cannot establish clean generalization.**

## Annotations and schema

Evaluation rows typically contain `image`, `type`, `gen`, `box`; MF-Clean adds `organ`, `group`, `split`, `disease`, `twin`, and `orig`. `box` uses the 1024-pixel coordinate frame. SFT uses `messages` and `images`. Type aliases include `edit`/`inject` for implantation and `remove` for removal.

A rectangular mask file is bounding-box supervision, not lesion segmentation. Existing rationales may be templates or model-generated text. Neither constitutes independently validated clinician reasoning. The MF-Clean intensity target often measures change from an authentic source; wording that instead describes local surrounding contrast needs correction.

## Splits and evaluation

MF-Clean has train, selection, and test manifests with grouped source/duplicate relationships. The existing `train_all` has 46,080 rows, `select_core` 1,420, and `test_core` 2,756. The builder's actual hash thresholds are **55/15/30**; these are allocation thresholds, not exact final image proportions after filtering. Its introductory historical docstring says 60/10/30 and should not be treated as the implementation specification.

Known patient IDs are grouped where recoverable. Missing patient identity prevents a universal patient-disjoint claim. Missing/exact versus near-duplicate twins must be distinguished. Report balanced accuracy, macro-F1, organ/type/editor recalls, localization over all fakes, twin consistency, and grouped uncertainty. Report source-matched single-image detection and paired pre/post comparison as different tasks.

## Rights and release status

This repository publishes research code, author-provided paper drafts, aggregate metrics, and selected predictions. It does **not** distribute raw medical images, source volumes or pretrained model weights, and is not yet a complete dataset distribution.

Upstream MedForge-90K, KiTS, Kvasir, Kaggle-source data, editing models and pretrained detectors retain their own terms and attribution requirements. A per-source redistribution/license ledger, source approvals and de-identification documentation remain to be completed. No project-wide license has been selected; do not infer permission to redistribute all source data from the public GitHub repository.

The official CVPR template files are included for the historical manuscript and retain their original notices. A final software license should distinguish original code from third-party components. Clinical use has not been validated; the repository supports research review.
