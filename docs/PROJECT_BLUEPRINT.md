# MedForge++ project blueprint and research memory

Last updated: **2026-11-01**. Historical snapshot reviewed: `1bcbc7393e8e0a3cbb803abf2747a1c1da0517d7`. This document is the recovery index for subsequent research conversations. Preserve evidence and uncertainty; never promote a plan, recorded historical claim or synthetic check into a completed experiment.

## Current state and immediate objective

The project addresses detecting and localizing lesion insertion/removal in medical images. The long-term goal is reliable, clinically meaningful evaluation across source populations, organs and editing models, supported by reproducible data lineage and uncertainty estimates.

The historical Bridge paper is **not submission-ready**. Its major problems are experimental validity, confounded extension data, unsupported cross-organ claims and insufficient evidence that RL/interpolation improves on appropriate controls. The best-supported direction is an **audit-led benchmark**, with Bridge and CT-GRPO treated as experimental baselines until replicated gains justify a method contribution.

The user requested: an independent technical review of the repository and PDFs; interpretation of the actual AAAI reviews; a basic experiment with results; a priority-ordered improvement plan; publishing that work to GitHub; and starting Change 1 (train/selection/test integrity). The latest request additionally requires comprehensive, evolving project memory. The requested actions and remaining work are preserved here.

Change 1 has now been executed on the private data host: raw identity reconstruction, versioned split rebuilds, regenerated pools and split/pool freeze/verification are complete for available identities. **The overall issue remains partially resolved:** unknown patient/video identities, historical test exposure, prospective retraining and confirmatory evaluation remain open. The execution below supersedes earlier statements that access to private data was pending. Historical publication-account limitations describe the earlier session only.

Publication milestone: implementation commit `3e564fa` is pushed on [`hardiksharma6555/MedForgePlusPlus:fix/split-protocol`](https://github.com/hardiksharma6555/MedForgePlusPlus/tree/fix/split-protocol), with [draft upstream PR #1](https://github.com/hardikbha/MedForgePlusPlus/pull/1). It was not merged by this session. GitHub fork creation initially hit transport errors; retrying with HTTP/2 disabled succeeded. The local working repository is `MedForgePlusPlus/` beneath the user's `fairness_cvpr` workspace.

## Change 1 private-data execution — 2026-11-01

**Status: partially resolved.** The private-data split/pool repair was executed from requested commit `fd894354bebc50595a6d7206058eea84a61997d0` in an isolated checkout. See [SPLIT_AUDIT_REPORT.md](../SPLIT_AUDIT_REPORT.md) and [aggregate evidence](evidence/change1_20261101/). No model training, alpha tuning, checkpoint selection or new model-performance evaluation occurred.

- Rehashed and decoded **154,043** raw/source/historical-materialized image files. Existing raw census checksums matched for all 102,787 previously censused images. **3,870** near-duplicate links passed the full-pixel rule; **two** reconstructed components crossed historical MF-Clean partitions despite a declared-metadata PASS.
- Preserved all **50,256** historical MF-Clean rows, merged whole connected identity components, added **286** exact-source real images, and moved **four** historical rows between partitions. New counts: **46,155 train / 1,482 validation / 2,905 test**. Every fake has an exact mapping-source twin; no arbitrary test-row deletion or row exclusion was used.
- Kidney/colon were rebuilt separately: all **7,800** original extension rows plus **1,000** authentic kidney insertion sources. New counts: **5,206 train / 1,110 validation / 2,484 test**. KiTS IDs are case/volume fields; patient/study/video fields remain unknown. MF-Clean still covers only the original organs.
- All **800/800** historical extension evaluation images remain recorded as training-exposed. Regrouping and new file paths do not make an untouched test set. Both new protocols explicitly label evaluation as exploratory/historically exposed.
- The initial kidney-source lookup missed the generator's documented space-to-underscore conversion. It was corrected; the 1,000 authentic sources were actually rehashed/decoded and linked. Prior attempts are preserved. The source-complete extension pool build then failed on missing surrounding rings; final extension **v2r3** clips 15 declared grid-boundary boxes while retaining original annotations and all rows, and uses source delta or explicit unknown evidence when no ring exists.
- **21** final regression tests passed. Scorer/evaluator CPU self-tests passed; the absent private legacy-resolver cache branch was explicitly skipped. Historical `v3data_b` and both regenerated actual-output pools passed their self-tests. New CF/CM exclusions, SFT/RL separation and validation/test exposure checks passed across the regenerated inventories.
- **88** historical/new candidate stage files were audited. Historical consumer evidence confirms the old CF replay file has **4,678 implant references / 2,339 unique implants**. A file inventory is not by itself proof that every historical checkpoint consumed that file; the confirmed replay log is distinguished from other candidate manifests.
- Split locks were frozen/verified after provenance and pool gates, and a SHA-256 pool lock was saved. An independent post-freeze verification re-read every final image and checked hashes, decoded content, raw identity components and exact twins. Complete identifiers, raw images, row transitions, logs and full manifests remain private; aggregate reports and code are shareable.

Private execution paths:

- mfclean: `/home/sachin.chaudhary/medforge_change1_20261101T111042Z/data/mfclean_change1_v2r2_20261101`
- extension: `/home/sachin.chaudhary/medforge_change1_20261101T111042Z/data/extension_split_v2r3_20261101`
- run: `/home/sachin.chaudhary/medforge_change1_20261101T111042Z/run`
- checkout: `/home/sachin.chaudhary/medforge_change1_20261101T111042Z/checkout`

Next runnable baseline wrapper (prepared, **not executed**): `/home/sachin.chaudhary/medforge_change1_20261101T111042Z/run/next_baseline.sh`, inside an authorized GPU allocation. It verifies the split and pool locks and initializes a fresh CF SFT baseline from the pretrained base model. Do not reuse historical adapters as clean evidence. The complete balanced-continuation recipe now distinguishes unique-image controls from pair-exposure controls; no method gain is established by producing those files.

Remaining blockers: unavailable original-organ patient/study IDs and extension patient/video linkage; an independent confirmatory population after historical inspection; extension source/format/editor confounds and clinical QC; prospective base-initialized training and validation-only selection. Source/duplicate separation is repaired for reconstructed identities, not universally certified patient independence.

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
- Legacy `per_class_breakdown.py` aggregates F1 across incompatible class subsets; it is not standard binary or three-class macro-F1. Some table differences may reflect definitions rather than simple arithmetic.
- “Full corpus” reporting sometimes means a 3,794-example test aggregate. Some advertised 50/10/30 counts total 90K, not 95,276. “Accuracy” for a single ground-truth class is recall. Define “upper bound” and all routing populations explicitly.
- Updated text repairs some KL-sign/replay-order/label wording and acknowledges SFT/RL pool reuse, but does not repair the experimental foundation. Main Tables 2–7 and supplementary S1–S11 require review against current code, populations and predictions.
- Novelty comparisons suggested by the AI review are research leads, not verified literature conclusions. Read primary sources before claiming prior work does or does not subsume the method.

The [priority action plan](REVIEW_ACTION_PLAN.md) maps concerns to corrections, experiments and completion evidence. Rewrite the paper from verified results after the blocking experiments; do not merely polish the historical headline narrative.

## Decisions and rationale

| Decision | Rationale / boundary |
|---|---|
| Keep original MedForge, extension, Bridge and MF-Clean distinct | Avoid attribution errors and claiming clean original-organ results fix extension failures |
| Use balanced accuracy plus class recalls and localization | Class imbalance can inflate plain accuracy and conceal removal collapse |
| Compare against balanced continued SFT before adding RL complexity | Current evidence favors that control; compute/exposure matching and replication still needed |
| Resample declared source groups | Generated siblings are correlated; record limitations where patients are unknown |
| Preserve archived predictions and historical core sampling | A silent dataset change would invalidate result provenance; new sampling/clinical repairs need a new version |
| Fail data preparation on overlap or incomplete materialization | A partial/contaminated manifest should not become a valid training input |
| Keep missing patient IDs/twins visible | Filling or inferring them without evidence creates false independence/pairing claims |
| Never call CPU rescoring a new trained experiment | Local raw data, checkpoint weights and GPU access are absent |
| Publish paraphrased action plan, not review correspondence | The user authorized sharing the work; confidential source text stays local |

## Active implementation and validation

Change 1 adds [split_protocol.py](../v3/split_protocol.py), [select_checkpoint.py](../v3/select_checkpoint.py), builder integration, annotation-resume correction and [regression tests](../tests/test_split_protocol.py). It checks declared identity consistency, records provenance where known, freezes exact manifest bytes, and rejects test predictions during selection. It does not automatically enforce every legacy training entry point or retrospectively cleanse exposed results. Follow the complete [split protocol](SPLIT_PROTOCOL.md) on the data host.

The new regression suite passed **15 tests** on CPU; scorer/evaluator self-tests passed, all **131** current Python files parsed, and new documentation links and diff checks passed. This is code behavior evidence, not the private-data completion gate. Current check results are appended to [VALIDATION](VALIDATION.md). Source snapshot hashes describe the original release; these new files are later work and should not be falsely included in that historical snapshot's provenance.

## Roadmap, hypotheses and open questions

| Order | Work / hypothesis | Evidence to obtain |
|---|---|---|
| Now | Review published draft PR #1; obtain data-host access | Reviewed code/PR, private-data location and runnable environment |
| Next | Can source/patient/duplicate grouping eliminate measured leakage? | Full-data lineage table, unknown-ID coverage, duplicate decisions, new versioned split lock and successful pool checks |
| Next | Does balanced continued SFT explain the gain attributed to RL? | Matched three-or-more-seed experiment; identical base/continuation eligibility; configs, steps, image exposure, FLOPs/time and grouped CIs |
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
