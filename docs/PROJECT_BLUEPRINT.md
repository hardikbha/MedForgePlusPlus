# MedForge++ project blueprint and research memory

Last updated: **2026-11-03**. Historical snapshot reviewed: `1bcbc7393e8e0a3cbb803abf2747a1c1da0517d7`. This document is the recovery index for subsequent research conversations. Preserve evidence and uncertainty; never promote a plan, recorded historical claim or synthetic check into a completed experiment.

## Current state and immediate objective

**Latest priority correction (2026-11-03): run our Bridge method directly on the full official MedForge splits; defer MedForge baseline reproduction.** The user superseded the earlier baseline-first plan. Neither retraining MedForge-Reasoner nor evaluating its released checkpoint is a prerequisite. See [the Bridge full-protocol work order](ORIGINAL_PROTOCOL_REPRODUCTION.md). Keep the official test and regime exclusions, use our own training method, and label published baseline numbers as references rather than reproduced results. Kidney/colon augmentation follows this first comparison.

Official HF SFT/RL/test manifests have been downloaded and counted: 55,276 / 10,000 / 30,000 rows, with zero cross-file image-ID overlap. This supersedes the earlier assumption that obtaining official sample lists was still unresolved. Raw-image availability and source/patient independence remain unverified. The upstream evaluator failed a synthetic check because parsing helpers are missing; metric compatibility still needs resolution. Conflicting beta/vLLM flags in the upstream GSPO launcher are retained as historical findings but do not block our independent Bridge training pipeline. No training or model inference has started.

**GPU handoff:** the user requested a short, shareable Markdown prompt and received `MEDFORGE_BRIDGE_GPU_NODE_PROMPT.md` in the parent workspace. Its versioned copy is [GPU_NODE_PROMPT.md](GPU_NODE_PROMPT.md). It pins implementation commit `5baa6572f79ef9c272f17b7843de32314691601c`, requests ID first followed by independent CF/CM runs, and leaves Table 2 to a separate follow-up. An operator can run it directly on the data/GPU host. The user did not know the SSH host and chose to forward the prompt; no remote session or job was started by this assistant, and no operator run report/job ID has been received. A prepared prompt is not evidence of submission or execution.

For immediate recovery: read this section, the [full work order](ORIGINAL_PROTOCOL_REPRODUCTION.md), the [short prompt](GPU_NODE_PROMPT.md), and the [official inventory](evidence/official_manifest_inventory_20261103.json), then inspect the working tree and any newly supplied operator logs. Do not restart MedForge baseline reproduction, add kidney/colon to the first run, or substitute MF-Clean for the official 30,000-image evaluation unless the user changes direction.

The project addresses detecting and localizing lesion insertion/removal in medical images. The long-term goal is reliable, clinically meaningful evaluation across source populations, organs and editing models, supported by reproducible data lineage and uncertainty estimates.

The historical Bridge paper is **not submission-ready**. Its major problems are experimental validity, confounded extension data, unsupported cross-organ claims and insufficient evidence that RL/interpolation improves on appropriate controls. The best-supported direction is an **audit-led benchmark**, with Bridge and CT-GRPO treated as experimental baselines until replicated gains justify a method contribution.

The user requested: an independent technical review of the repository and PDFs; interpretation of the actual AAAI reviews; a basic experiment with results; a priority-ordered improvement plan; publishing that work to GitHub; and starting Change 1 (train/selection/test integrity). The latest request additionally requires comprehensive, evolving project memory. The requested actions and remaining work are preserved here.

Change 1 now has code checks, tests and a [protocol](SPLIT_PROTOCOL.md). **The private-data rebuild, final split freeze, retraining and new test inference are not complete.** Public predictions permit rescoring, not reconstruction of source provenance. Access to the private data host/manifests is the next execution dependency. The configured GitHub accounts lack upstream write permission; publication uses a fork/PR rather than claiming upstream main was changed.

Publication milestone: implementation commit `3e564fa` is pushed on [`hardiksharma6555/MedForgePlusPlus:fix/split-protocol`](https://github.com/hardiksharma6555/MedForgePlusPlus/tree/fix/split-protocol), with [draft upstream PR #1](https://github.com/hardikbha/MedForgePlusPlus/pull/1). It was not merged by this session. GitHub fork creation initially hit transport errors; retrying with HTTP/2 disabled succeeded. The local working repository is `MedForgePlusPlus/` beneath the user's `fairness_cvpr` workspace.

Maintenance incident during this memory update: a full disk prevented Git from creating its index lock. Approximately 242 MiB of review-created temporary PDF/renderings and downloaded manifest copies were removed; the repository's inventory, hashes, source revisions, reports and GPU prompt were retained. Original manifests can be downloaded again at the pinned HF revision; do not assume the large `/tmp` copies still exist. No user dataset or model checkpoint was removed.

## Milestones and superseded directions

| Date / revision | Work or discussion | Outcome / continuing relevance |
|---|---|---|
| 2026-11-01, review of `1bcbc739` | Independent repository/paper review and CPU verification | Archived metrics reproduced; serious protocol/data/claim limitations identified; no fresh GPU experiment |
| 2026-11-01 | User requested the most basic experiment | Rescored initial SFT, balanced continued SFT, image GRPO and constants; 5,000 group bootstraps. Results are preserved below. |
| 2026-11-01 | Actual AAAI correspondence supplied | Two human reviews distinguished from AI review; paraphrased priorities published; raw correspondence kept local |
| `3e564fa`, `fd89435` | Change 1, research blueprint and draft PR | Split gates, validation-only selector, resume-box repair and 15 passing tests; full private-data validation still pending |
| 2026-11-03 | User identified original MedForge Tables 1 and 2 as targets | Detection and judge-based explanation evaluations separated; no verified exact-protocol Bridge result found |
| `9411cbc` | User prioritized full original-data evaluation; baseline-first work order prepared | Official manifests actually downloaded and counted; upstream evaluator failure reproduced. Baseline-first execution was subsequently superseded. |
| `5baa657` | User explicitly deferred reproducing MedForge-Reasoner | Active plan became Bridge-only training/evaluation on official splits, with published baseline rows as references |
| 2026-11-03 | Short GPU-node prompt delivered | User will forward it to an operator. No returned run evidence yet. |

The existing readiness assessment and suggested audit-led paper direction are research judgments, not permission to override the user's latest execution priority. Keep negative findings visible while carrying out the full official-split Bridge experiment.

## Names and attribution

| Name | Meaning | Claim boundary |
|---|---|---|
| Original MedForge / MedForge-90K | Chen et al.'s prior benchmark and MedForge-Reasoner detector; [upstream paper](https://arxiv.org/abs/2603.18577) | Neither the original dataset nor detector is our invention |
| MedForge++ | Extension adding 7,800 kidney CT/colonoscopy files, annotation/construction work and broader experiments | Files are not independent scans or patients; extension validity needs repair |
| Bridge | Historical Qwen3-VL-8B pipeline: supervised training/replay, RL, cross-stage LoRA-factor interpolation | Claimed SOTA/generalization is superseded by leakage/control findings |
| CIRCE | Development codebase containing legacy construction, replay, reward, training and evaluation work | Do not describe every CIRCE experiment as a validated Bridge result |
| MF-Clean / v3 | Subsequent source-grouped, shortcut-controlled rebuild of the original brain MRI/chest X-ray/fundus organs | Does **not** repair kidney/colon; source-disjoint where identities are known does not imply universal patient independence |
| CT-GRPO | v3 experiments comparing image-normalized and twin-pooled advantage estimation | Twin pooling has no demonstrated robust advantage in current evidence |

Historical title: *MedForge++: Bridging Reinforcement Learning for Generalizable Medical Forgery Detection Across Organ Domains*. The PDFs in [paper/legacy_cvpr_draft](../paper/legacy_cvpr_draft/) are historical drafts with superseded claims. The repository's [CVPR_REVIEW](CVPR_REVIEW.md) is an assessment to verify, not a substitute for source evidence.

## Architecture and evolution

Original intended pipeline: acquire authentic source images → edit lesions using FLUX insertion / externally supplied LaMa removal → construct boxes and structured descriptions → train Qwen3-VL with LoRA SFT and failure replay → RL with verdict/localization/evidence/format rewards → interpolate SFT/RL adapters → evaluate classification/localization, including organ routing.

Implemented areas:

| Area | Code / evidence | What exists |
|---|---|---|
| Extension construction | [construction](../construction/), especially [kidney insertion](../construction/inject_flux_kidney.py), [colon insertion](../construction/inject_flux_colon.py), [annotations](../construction/annotate_kidney_colon.py) | Insertion scripts, rectangular labels and template/LLM annotation pipeline; removal-generation provenance incomplete |
| Legacy methods | [circe](../circe/), [patched RL objective](../circe/vlm_grpo_patch.py), [replay builder](../circe/build_regime_replay.py), [adapter interpolation](../circe/interpolate_adapters.py) | SFT, replay, RL, factor interpolation, evaluation scripts and archived results |
| Audits | [v3_audit](../v3_audit/), [audit reports](evidence/audit/), [release audit](evidence/release_data_audit.json) | Census, hashes/duplicates, family leakage, format/source probes, mask/annotation review |
| MF-Clean | [builder](../v3/build_mfclean.py), [SFT/RL preparation](../v3/build_sft_v3.py) | Grouping, paste-back compositing, shared encoder, train/select/test and SFT/RL pools |
| v3 models/evaluation | [trainer](../v3/train_ctgrpo.py), [evaluator](../v3/eval_v3.py), [scorer](../v3/stats_v3.py) | Image/twin RL, structured prompts, group bootstrap, per-type/organ scores and saved predictions |
| Reproducibility | [reproduction guide](REPRODUCIBILITY.md), [validation](VALIDATION.md), [result hashes](../results/SOURCE_SNAPSHOT.json) | Portable path mapping, CPU scoring, environment inventory and selected predictions; full training/data release pending |

The audit-driven redesign groups sources and known patient/duplicate identities before splitting, pastes edited regions back onto authentic backgrounds, normalizes encoding, and compares RL with continued supervised training. It changes the evaluation population; legacy headline accuracy and MF-Clean balanced accuracy must not be compared as if measured on the same test.

## Dataset findings and limitations

Unless explicitly marked independently rerun below, raw-data figures are **recorded audit evidence**, not a new local filesystem census.

| Finding | Evidence / interpretation |
|---|---|
| Mapping 95,276; existing upstream files 94,987; 289 missing, all real | [Release audit](evidence/release_data_audit.json), [census](evidence/audit/census.md). Adding 7,800 yields 102,787 existing files, not 103,076 independent scans. |
| 63,286 upstream forgeries derive from 3,790 source scans | Many generated files share a source; image-level splitting/uncertainty is inadequate. |
| Extension: 3,800 real, 2,000 insertion, 2,000 removal | Kidney has 1,000 normal + 1,000 real tumor + 1,000 removal from KiTS (420 cases), while 1,000 implants use a different Kaggle CT-kidney source. Colon normals use WCE; real/removal/insertion use Kvasir. |
| 800 colon normals have only 564 byte-unique files | 236 duplicate copies; count files, unique images, groups and patients separately. |
| All 2,000 implants JPEG; other 5,800 extension files PNG | Label-correlated encoding. Source cohorts also differ by class; authentic-versus-authentic kidney source probe reportedly AUC 1.0. Canonical encoding alone cannot remove source confounding. |
| 800/800 extension evaluation paths occur in both 7,800-row SFT sets | Historical 82.25% is training-exposed, not held-out generalization. |
| Legacy source-family overlap | 1,999/2,000 ID fake test examples have a source family in training; 826/832 held-out-generator examples share the underlying edit with a trained generator. |
| Legacy CF targeted replay | 4,678 rows, 2,339 unique implants; violates held-out forgery type. Distinguish forbidden-type exposure from exact test-image overlap. |
| One editor per extension manipulation type | FLUX versus LaMa makes editor/type inseparable; leave-editor-out generalization missing. |
| Localization/clinical labels | Rectangle masks and saved boxes; no complete expert study evidence. Kidney insertion uses limited centers/boxes and a soft-tissue heuristic rather than organ segmentation. Colon insertion avoids an existing polyp in already diseased frames. |
| Clinical success unknown | Pixel changes do not prove successful lesion insertion/removal; 47 unresolved removal cases noted in the audit. Expert review, source rights and release governance remain open. |

Detailed sources: [expansion](evidence/audit/expansion.md), [labels](evidence/audit/labels.md), [shortcuts](evidence/audit/shortcut.md), [dataset card](DATASET_CARD.md).

The original MedForge is **classification + localization + explanation** data, not classification-only. Our extension also provides verdict/type labels, boxes and generated explanations; localization itself is therefore not a new task contribution. Upstream masks/boxes are source-difference-derived, while many extension mask exports are filled rectangles. Neither establishes clinically precise segmentation. In the extension, authentic tumor/polyp examples may carry real-lesion boxes: these must be distinguished from forgery-region boxes. Extension file totals are kidney 4,000 (2,000 real/1,000 implant/1,000 removal) and colon 3,800 (1,800/1,000/1,000).

### Official released manifests: verified 2026-11-03

Dataset revision: `7bbf54f870584b37d6594593da76b2fc78fe4c5b`. These are separate from MF-Clean and our old locally randomized subset. [Inventory script](../scripts/audit_official_manifests.py) and [aggregate report with SHA-256 hashes](evidence/official_manifest_inventory_20261103.json) preserve the check.

| Released pool | Total records | Real | Implant | Removal |
|---|---:|---:|---:|---:|
| SFT | 55,276 | 16,990 | 21,389 | 16,897 |
| RL | 10,000 | 5,000 | 3,013 | 1,987 |
| Test | 30,000 | 10,000 | 10,952 | 9,048 |

Each pool has unique release-relative image IDs; all pairwise ID intersections are zero. Types were derived from image paths. This does not verify raw-file existence, pixel duplicates, source-family overlap, patient independence or whether these manifests produced the paper's scores. The release differs from the prose's nominal 50K/10K/30K and equal class proportions; do not silently resample it to match those statements.

Before a documented training-side validation carve-out, eligible SFT/RL counts are ID **55,276/10,000**, CF **33,887/6,987**, CM **39,591/8,247**. CF excludes implants; CM excludes `gemini`, `gpt`, `stable-diffusion-3.5-medium`, and `stable-diffusion-xl-1.0-inpainting-0.1` from every training stage. All regimes use the unchanged full 30,000-image test. These are computed eligibility inventories, not completed training runs.

MF-Clean recorded protocol:

- Train: **46,080 rows / 11,283 groups**; selection: **1,420 / 526**; test: **2,756 / 1,008**. Test consists of 1,008 real, 924 implants and 824 removals.
- Group hash thresholds are **55/15/30**, not the old docstring's 60/10/30. Union-find combines source/forgery relations, available patient IDs, exact real duplicates and near duplicates verified by dHash ≤4 plus 32-pixel thumbnail MAE ≤3.
- Inside train, `SHA1("sftrl-v1:" + group) % 6 == 0` goes to RL; other groups go to SFT. Recorded CF SFT has 15,592 records (10,580 real/5,012 removal); ID has 20,518 (additional 4,926 implants). Recorded CF RL has 3,017 pairs plus 754 singleton reals; 299 distinct pair reals are reused approximately ten times.
- Core construction keeps only one real per group, excludes real-only groups, and sets a forgery sampling budget from the first fake's type (`5` for edit, otherwise `1`). This is a curated, order-sensitive population. Do not silently change it while retaining archived result labels.
- Only **1,471/1,748** test fake twins are present; 277 fundus twins are missing. Some twins are near-duplicate representatives rather than the exact source (recorded 1,360 nonexact across splits, including 44 test brain examples). Report coverage and exact-source semantics.
- Compositing/encoding reduces shortcuts but does not prove their elimination. Recorded generator-extraction shortcut BA remains approximately 55.43% CF / 63.53% CM. Boundary, sham, outside-mask and alternative-compositing tests remain needed.
- Many patient IDs are unavailable. Heuristic mask agreement is neither clinical gold truth nor a hard upper bound on localization. Weak probe failure is not proof of no shortcut.

## Method implementation versus paper

The legacy patched loss in [vlm_grpo_patch.py](../circe/vlm_grpo_patch.py), `compute_loss`, uses `-exp(lp-lp.detach()) * advantage`, averaged over masked tokens. It has no PPO clipping or KL penalty, and no per-sequence length normalization. Legacy beta settings therefore do not establish an implemented KL ablation. The newer trainer's explicit beta-zero guard is more faithful to the code.

[build_regime_replay.py](../circe/build_regime_replay.py) ranks failures using verdict/type/organ behavior; this is not the manuscript's five-way replay taxonomy. [interpolate_adapters.py](../circe/interpolate_adapters.py) blends LoRA factors: the bilinear expansion follows algebraically, but does not establish that cross terms cause useful complementarity. Compare against effective-weight merging and matched continued training before making that claim.

Evidence/format rewards often have little within-group variation; formatting is not clinical reasoning. Gated bonuses are not universally mathematical no-ops, however. v3 intensity labels derive from a source delta while prompts describe local contrast; selected released prediction files lack intensity ground truth. Do not infer intensity accuracy from format compliance.

Legacy SFT dry-run assumptions (five messages versus v3's three), defaults (4B/rank16 versus recorded 8B/rank32), seeding and balancing recipe need reconciliation for fresh training. [build_rl_as_sft.py](../v3/build_rl_as_sft.py) deduplicates continuation images but does not fully specify the balanced-control training recipe. Existing weights/results alone do not establish matched compute or data exposure.

## Results and what was actually verified

### Original-paper targets and current comparison status

The [original MedForge v2 paper](https://arxiv.org/html/2603.18577v2) was read in HTML and its rendered PDF tables were inspected. Its Table 1 is detection; Table 2 is explanation judging. The following Bridge entries are historical manuscript claims, **not verified results from the full official test**:

| Regime | Published MedForge accuracy / F1 | Historical Bridge accuracy / F1 |
|---|---:|---:|
| In-Domain | 99.23 / 98.98 | 99.03 / 98.87 |
| Cross-Forgery | 95.93 / 95.13 | 96.96 / 96.18 |
| Cross-Model | 93.94 / 93.49 | 94.66 / 94.62 |

The historical Bridge rows use 2,994 examples and have the selection/training/numerical problems documented below; the Cross-Model displayed class values imply approximately 94.36 accuracy / 94.34 average F1. The original Table 1 comparison remains pending. Adding kidney/colon has not produced a verified augmentation gain on this exact test.

For Table 2, MedForge with GSPO reports LC/VH/MP/average **71.1/67.4/83.2/73.9** under Gemini 3 Pro and **93.3/79.9/97.5/90.2** under Qwen3-VL-Plus. VH is a faithfulness score: higher is better. The highest average anywhere in that table is **76.2** under Gemini and **90.2** under Qwen. Evaluation uses 100 forgery examples correctly detected by all compared models. No matching Bridge judge artifacts were found. Parsing compliance, localization and structured-field accuracy cannot fill these columns.

The exact 100-example sample and judge-version/configuration provenance remain unresolved. The released judge code compares generated and reference text, not images directly. A new sample can support a separately identified evaluation, not an undisclosed replacement for the original table.

### Existing MF-Clean results

The following are **rescored saved predictions**, not new training or model inference. All values below are percentages on the same MF-Clean CF test.

| Model | Accuracy | Balanced accuracy | Real recall | Implant recall | Removal recall | mIoU | Twin success |
|---|---:|---:|---:|---:|---:|---:|---:|
| Initial SFT | 70.75 | 76.23 | 96.63 | 54.55 | 57.28 | 42.61 | 51.67 |
| Balanced continued SFT | 77.36 | **79.93** | 89.48 | 68.94 | 71.97 | **49.95** | **59.62** |
| Image GRPO | 74.56 | 77.81 | 89.88 | 63.96 | 67.72 | 46.99 | 55.81 |
| Twin GRPO | 74.06 | 77.74 | — | — | — | 46.48 | 54.79 |
| Always real | 36.57 | 50.00 | 100 | 0 | 0 | 0 | 0 |
| Always fake | 63.43 | 50.00 | 0 | 100 | 100 | 0 | 0 |

The basic experiment requested by the user compared initial SFT, balanced continued SFT, image GRPO and constants on saved predictions. It checked exact image/GT metadata agreement, no duplicate identifiers, 2,756 images / 1,008 groups, valid present twins and an independent verdict tally. Balanced SFT correctly classifies 902/1,008 reals and 1,230/1,748 fakes; image GRPO 906/1,008 and 1,149/1,748.

Paired **source-group bootstrap**, 5,000 replicates, seed 2027, fixed checkpoints:

| Comparison | BA difference, percentage points | 95% interval |
|---|---:|---:|
| Balanced continued SFT − initial SFT | +3.6940 | [+2.5972, +4.8111] |
| Image GRPO − initial SFT | +1.5755 | [+0.6181, +2.5364] |
| Balanced continued SFT − image GRPO | +2.1185 | [+1.2376, +2.9943] |

Source: [machine-readable basic experiment](../results/baseline_rescore_20261101/results.json), [CSV](../results/baseline_rescore_20261101/metrics_percent.csv), [reusable script](../scripts/rescore_baseline.py). Balanced SFT improves fake detection at a cost in real recall; it is not uniformly better on every endpoint. Its 71.97% removal recall is **not kidney removal performance**.

The earlier 1,000-bootstrap paired review found CF twin-minus-image BA −0.0645 pp (CI −0.536 to +0.436), mIoU −0.5167 (−0.925 to −0.166), and twin success −1.0197 (−1.792 to −0.203). ID twin-minus-image BA +0.2748 (−0.223 to +0.730), mIoU −0.5375 (−0.967 to −0.141), and twin success +0.3399 (−0.543 to +1.153). Evidence: [independent review artifacts](evidence/independent_review_20261101/).

Other recorded results: ID SFT/image/twin BA approximately 81.82/82.71/82.99; other CF RL seed or learning-rate trials approximately 77.23–77.92 BA; frozen MedForge-Reasoner BA 58.61 and frozen Qwen approximately 53.19; historical Bridge evaluated on MF-Clean approximately 47.07. Historical zero-shot extension results around 48.50/53.62 BA are weak and measured on a confounded extension. See [result inventory](RESULTS.md) for exact run names/populations. Unbalanced continued SFT has high removal recall (~87.14%) but low real recall (~58.93%), illustrating why balanced accuracy and per-class results matter.

Interpretation: RL beats initial SFT on the available CF outputs, but **does not beat balanced continued SFT**. Twin pooling and interpolation have no established robust advantage under a corrected, matched protocol. Bootstrap CIs condition on fixed models and declared groups; they do not quantify training-seed variation or prove unknown patient independence. Image-wise McNemar is not the primary inference for correlated source families.

Independent verification on 2026-11-01 at the historical snapshot:

- All 127 Python files parsed; scorer and evaluator synthetic CPU self-tests passed (private legacy-metadata resolver branch skipped).
- Eight prediction files rescored; six primary point metrics matched archived values with maximum difference zero. All 49 result-release hashes matched.
- A synthetic cached-build case reproduced a fake box changing from `[320,320,624,624]` to null. Downstream `recover_box` already mitigates this; the observation does **not** establish that saved runs used corrupt labels. The current patch fixes resume and adds a regression test with a differently sized mask.
- No GPU training, real-model inference, raw-data census, full rebuild, clinical study or processor-dependent CT-GRPO test suite was run in this independent review. Earlier repository validation reports those environment-specific checks separately; do not conflate them with this session's work.

## Manuscript/review conclusions

The actual review file describes two human reviews plus a separate AI review and a Phase-1 AAAI rejection. Human concerns prioritize test selection, new-organ adaptation mislabeled generalization, editor diversity, kidney removal, localization and expert/reasoning evidence. The second review cites historical SFT 95.76 → RL 95.26 → interpolation 96.96 accuracy, questioning independent RL value. Review sentiment cannot validate the protocol; one review's favorable description is contradicted by the data audits.

Specific historical numerical/terminology issues:

- Main Cross-Model recalls 99.40 (n=994), 91.90 (n=1,000), 91.80 (n=1,000) imply weighted accuracy **94.3566**, not 94.66. The three displayed F1 values 92.12/95.48/95.43 average **94.3433**, not 94.62; clarify intended weighting/population from predictions.
- Supplement S11's 83.80% real “agreement” is both-correct; adding 5.63% both-wrong yields 89.43% binary agreement.
- Legacy `per_class_breakdown.py` aggregates F1 across different class subsets; it is not standard binary or three-class macro-F1. Correction from the 2026-11-03 upstream v2 check: Appendix D.3 explicitly defines Real versus all fakes, Implant versus Real, and Remove versus Real. These match the subgroup choices in our code; its comment claiming the upstream per-class methodology is unpublished is outdated. Exact table aggregation and prediction-level reproduction still require verification. Different F1 definitions must not be substituted for one another.
- “Full corpus” reporting sometimes means a 3,794-example test aggregate. Some advertised 50/10/30 counts total 90K, not 95,276. “Accuracy” for a single ground-truth class is recall. Define “upper bound” and all routing populations explicitly.
- Updated text repairs some KL-sign/replay-order/label wording and acknowledges SFT/RL pool reuse, but does not repair the experimental foundation. Main Tables 2–7 and supplementary S1–S11 require review against current code, populations and predictions.
- Novelty comparisons suggested by the AI review are research leads, not verified literature conclusions. Read primary sources before claiming prior work does or does not subsume the method.

The [priority action plan](REVIEW_ACTION_PLAN.md) maps concerns to corrections, experiments and completion evidence. Rewrite the paper from verified results after the blocking experiments; do not merely polish the historical headline narrative.

## Decisions and rationale

| Decision | Rationale / boundary |
|---|---|
| Keep original MedForge, extension, Bridge and MF-Clean distinct | Avoid attribution errors and claiming clean original-organ results fix extension failures |
| Use balanced accuracy plus class recalls and localization | Class imbalance can inflate plain accuracy and conceal removal collapse |
| Retain balanced continued SFT as the essential method control | Current evidence favors that control; compute/exposure matching and replication remain needed after the user's first full Bridge run |
| Run Bridge directly on official splits; defer original-baseline reproduction | Latest user decision; use published MedForge rows as references and disclose unresolved comparability limits |
| Resample declared source groups | Generated siblings are correlated; record limitations where patients are unknown |
| Preserve archived predictions and historical core sampling | A silent dataset change would invalidate result provenance; new sampling/clinical repairs need a new version |
| Fail data preparation on overlap or incomplete materialization | A partial/contaminated manifest should not become a valid training input |
| Keep missing patient IDs/twins visible | Filling or inferring them without evidence creates false independence/pairing claims |
| Never call CPU rescoring a new trained experiment | Local raw data, checkpoint weights and GPU access are absent |
| Publish paraphrased action plan, not review correspondence | The user authorized sharing the work; confidential source text stays local |

## Active implementation and validation

Change 1 adds [split_protocol.py](../v3/split_protocol.py), [select_checkpoint.py](../v3/select_checkpoint.py), builder integration, annotation-resume correction and [regression tests](../tests/test_split_protocol.py). It checks declared identity consistency, records provenance where known, freezes exact manifest bytes, and rejects test predictions during selection. It does not automatically enforce every legacy training entry point or retrospectively cleanse exposed results. Follow the complete [split protocol](SPLIT_PROTOCOL.md) on the data host.

At the Change 1 milestone, the new regression suite passed **15 tests** on CPU; scorer/evaluator self-tests passed, all **131 Python files then present** parsed, and new documentation links and diff checks passed. The later official-manifest inventory script was syntax-checked and run against the complete downloaded JSONs, with aggregate consistency checks passing. This is code/manifest evidence, not the private-data completion gate. Check results are recorded in [VALIDATION](VALIDATION.md) and the [official inventory](evidence/official_manifest_inventory_20261103.json). Source snapshot hashes describe the original release; new files are later work and should not be falsely included in that historical snapshot's provenance.

Inspected upstream code revision: `f60505600d3de0959f4fcdc8bd1741754ed2c85f`. Its detection evaluator lacks `classify_image`, `extract_bbox`, `extract_model_name`, `compute_iou` and `visualize_result`; a synthetic 8×8 image invocation actually raised `NameError` at `classify_image`. Repair/test compatible evaluation before the full run. Upstream GSPO duplicate beta/vLLM flags and unpinned SWIFT dependencies are deferred reproduction findings, not requirements to use that trainer for Bridge. Bridge must still record an unambiguous objective, environment and validation policy.

## Roadmap, hypotheses and open questions

2026-11-03 user steering: prioritize a direct comparison with **Tables 1 and 2 of the original MedForge paper**, before evaluating the benefit of the added kidney/colon images. Read the [upstream v2 paper](https://arxiv.org/html/2603.18577v2), including the rendered PDF tables and Appendix D.3. Table 1 covers detection; Table 2 covers judge-scored explanation quality. Our historical Table-1-style claims use a locally constructed 2,994-image test subset, not a verified reproduction of the paper's stated 30,000-image test. Training-pool reuse, forbidden CF replay and test-based model selection also block exact-protocol claims. No matching Gemini/Qwen Table-2 judge results were found in the available repository. MF-Clean's 2,756-image evaluation is a separate benchmark. No new model inference or training was run for this check.

The official detection split identities are now obtained. Next steps are verifying their raw images, enforcing regime-specific training exclusions, reconciling the scorer and recording checkpoint provenance. Explanation evaluation additionally requires the original judge rubric, reference annotations, sample identities and judge configurations. An added-data experiment should be labeled separately and retain the original test when measuring augmentation benefit. The private-node audit remains unreceived as of this update; no remote execution has been started from this workspace.

### Handoff acceptance and next conversation

When the operator responds, obtain the code/dataset revisions, resolved data paths, scheduler job IDs/state, exact commands/configs, validation manifest hashes, checkpoint locations, full-test prediction counts and report paths. Verify train/validation/test membership and CF/CM exclusions across replay and RL. Check that test labels were not supplied in prompts, malformed outputs were not silently dropped, and checkpoint/alpha selection preceded test inspection. Compare actual results only after confirming coverage and metric definitions. Missing raw images are a blocker to claiming the full official test, not permission to silently reduce it.

If no operator response has arrived, report the handoff as prepared and results as pending. Do not invent progress, infer a running job from the prompt, or repeatedly demand SSH information the user already said they do not know. The user chose an operator handoff; direct execution can resume if connection details are supplied later.

| Order | Work / hypothesis | Evidence to obtain |
|---|---|---|
| Now | Train Bridge directly on the official eligible SFT/RL pools and evaluate the full official test; defer MedForge baseline runs | Verified image coverage, pinned Bridge environment, validated scorer, validation-selected ID/CF/CM checkpoints and complete 30,000-image evaluations |
| Parallel audit | Can source/patient/duplicate grouping eliminate measured leakage? Keep any repaired benchmark separate from original-protocol reproduction. | Full-data lineage table, unknown-ID coverage, duplicate decisions, new versioned split lock and successful pool checks |
| After first full Bridge run | Does balanced continued SFT explain the gain attributed to RL? | Matched three-or-more-seed experiment; identical base/continuation eligibility; configs, steps, image exposure, FLOPs/time and grouped CIs |
| Next | Do paste-back edges or nonclinical traces drive MF-Clean performance? | Boundary-only/outside-mask/sham controls, alternate compositing/encoding, unseen editors and source-matched authentic controls |
| Major | Can the kidney/colon extension become a defensible benchmark? | Same-source real/edit/remove construction, source/video/case disjointness, diverse editors, deduplication and blinded clinical QC |
| Major | Does a single fixed model transfer to genuinely unseen organs? | No-target-tuning kidney/colon results, per-type/source/organ metrics and external confirmatory set |
| Conditional | Does twin pooling help localization or robustness at matched budget? | Replicated paired comparisons; ablate sampling/reward normalization separately from extra data |
| Conditional | Does factor interpolation offer benefit beyond effective-weight merging? | Validation-selected alphas, matched endpoints, strong merge/control baselines and test-once evaluation |
| Parallel | Are edits/labels clinically valid and legally releasable? | Expert rubric/readers/agreement/adjudication; acquisition/license inventory and release decision |
| Later | What paper story survives the evidence? | Audit-led benchmark unless method/control results justify stronger claims; regenerate tables and rewrite title/abstract |

Open questions include missing patient/video identifiers; unreleased removal-generation settings; exact balanced-SFT recipe; whether existing clinical labels distinguish target lesion from edit rectangle; how to handle near-duplicate substitute twins; how much the current core sampling changes deployment prevalence; how to reserve a genuinely confirmatory evaluation after extensive test exposure; and whether the strongest publishable contribution is the audit itself, the rebuilt benchmark, or a reproducible method gain. No answers should be invented to make the project look complete.

## Maintenance and recovery rules

Start future work here, then read the linked evidence and `git status`. Update this document at each meaningful milestone with date, code commit, dataset/protocol version, exact command/config, inputs/checkpoint hashes, metrics and uncertainty, interpretation, and remaining dependencies. Append failures and superseded decisions with reasons; do not delete inconvenient negative results.

Treat machine-readable artifacts and source code as evidence. This narrative is the project index, not authority to overwrite contradictory evidence. If a claim conflicts with an artifact, record the discrepancy and resolve it. Separate **proposed**, **implemented**, **synthetically tested**, **run on private data**, and **clinically validated** states. Record any future publication URL/PR and closure of the private-data gates explicitly. Never infer new experiments from a plausible narrative or from the passage of time.
