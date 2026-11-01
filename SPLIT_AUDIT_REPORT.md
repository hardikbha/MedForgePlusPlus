# SPLIT_AUDIT_REPORT — Change 1, private-data execution

**Status: partially resolved.** Rebuilt prospective splits/pools are separated for the reconstructed identities. Unknown patient/study/video identities and historical test inspection remain unresolved. No training, checkpoint selection, alpha tuning or new model-performance evaluation was run.

## Code, resources and checks

Started from the requested fork branch `fix/split-protocol` at `fd894354bebc50595a6d7206058eea84a61997d0`, in an isolated checkout. CPU work ran inside the existing PBS workq allocation `43937.mgmt01` on `gpu01` through its append-only task scheduler, with CUDA disabled, one BLAS thread and four worker processes. No running user job was stopped. Two initial worker setup attempts failed before tests because the compute node lacked Git; corrected wrappers read the already-verified checkout HEAD.

Base checks: 15 regression tests passed; the final suite passed 21 tests. Scorer/evaluator self-tests passed; the scorer explicitly skipped the absent private legacy-resolver cache. The actual historical `v3data_b` self-tests passed. Both final actual-output pool self-tests passed. Command statuses and private logs retain earlier failures and interrupted attempts.

## Original mapping and raw evidence

The audit rehashed and decoded **154,043** actual files, including raw source/forgery files and historical MF-Clean materializations. Unreadable inputs: **0**. The mapping contains 95,276 records; missing mapping files by type: `{'real': 289}`; repeated mapping paths: 0.

Verified near-duplicate links: **3870**; rejected candidates: **34721**. Candidate rule and full-pixel acceptance thresholds are in raw_identity_audit.json. These are conservative split-safety links, not invented patient identities. Existing declared MF-Clean partitions contain **2** cross-partition reconstructed components. The metadata-only PASS did not establish this raw-file property. The complete raw census was checkpointed; one own audit subprocess was intentionally stopped to resume the same verification rule with vectorized candidate pruning. The first source-incomplete orchestration attempt was stopped before freeze. These are recorded interrupted attempts, not passed data releases.

Known filename-derived original-organ patient identities are retained where available. KiTS case keys are recorded as volume/case IDs, not asserted patient IDs. Study IDs, Kvasir/WCE patient/video linkage, and Kaggle kidney patient IDs are unavailable. Zero overlap of declared/reconstructed identities cannot certify those missing identities.

Unavailable authentic-source files/links by corpus: `{'original': 631}`. The original mapping can identify an unavailable source, but that does not provide pixels for construction. Historical twins by exact mapping path versus identical decoded aliases versus different decoded sources: `{'exact_mapping_source_path': 33605, 'different_path_identical_decoded_source': 687, 'different_path_different_decoded_source': 1360, 'missing_or_dangling': 392}`. This distinguishes a substituted filename from an actual pixel mismatch.

## Rebuild and population changes

| Scope | Preserved historical rows | Added exact-source real images | Quarantined rows | Historical rows moved to a different split |
|---|---:|---:|---:|---:|
| mfclean | 50256 | 286 | 0 | 4 |
| extension | 7800 | 1000 | 0 | 0 |


| Rebuilt scope | Train | Validation | Test |
|---|---:|---:|---:|
| MF-Clean v2r2 | 46,155 | 1,482 | 2,905 |
| Kidney/colon split-only v2r3 | 5,206 | 1,110 | 2,484 |

Every historical MF-Clean example is preserved. Connected source, known patient/case, exact-content, decoded-content and verified-near-duplicate components are merged; conservative historical group unions are retained. Newly rendered identical outputs are also joined before assigning partitions. The canonical existing group and its deterministic hash assignment are retained where possible; actual component merges can move all linked examples together. No accuracy result informed grouping or exclusions.

Missing and substituted authentic twins are replaced by materialized **exact mapping sources**. The final attempt reuses earlier own-run original-organ renderings only after matching input/source/mask hashes, unchanged renderer code, and output-byte hashes; historical datasets are not modified. Added source images and any reassignment change the evaluation population, so this is protocol **change1-v2r2-20261101**. Every new row is explicitly labelled historically exposed/exploratory. A new directory or split assignment does not create an untouched test set.

Kidney/colon were rebuilt separately as a split-only version. Their historical 800-image evaluation remains training-exposed; the exact overlap counts are in raw_identity_audit.json. Unresolved sources or missing localization provenance are quarantined with private row-level reasons. All 1,000 kidney injection sources were linked using the generator’s documented space-to-underscore filename conversion and independently rehashed/decoded. The initial incomplete lookup and intermediate build were preserved and not frozen. Format/source/editor confounds and clinical label validity are not repaired by this task. MF-Clean still covers the original organs and does not substitute for extension validation.

The source-complete extension r2 pool build failed because full-frame lesion boxes left no surrounding ring for the intensity calculation. That failed attempt is retained. Final extension version `change1-extension-v2r3-20261101` clips **15** declared 1024-grid boxes to inclusive [0,1023], preserves their original values, and retains every image. If a ring is unavailable, the evidence label uses measured source intensity change; zero/unavailable change is explicitly unknown (`none`). This is target construction, not clinical annotation validation. Exact source-copy checks passed 300/300, replacing an inapplicable JPEG-reconstruction test.

The complete before/after split × organ × type × editor counts are supplied as CSV and JSON. Identity coverage is supplied by split and organ. Patient identifiers, raw image paths, component membership and quarantine details remain in the private versioned directories.

## Training stages, locks and historical exposure

Audited 88 candidate historical/new training-stage files. Historical CF/CM exclusion violations occur in one file inventory. Consumer-provenance checks confirm **4,678 implant references / 2,339 unique implants** in the old CF replay file; this is distinguished from unconfirmed candidate-file exposure. Full stage inventories distinguish reference counts, unique images and source-component exposure.

New SFT/RL pools, unique continuation controls, pair-exposure continuation controls and SFT-pool replay eligibility inventories were generated. The latter are eligibility lists, not model-mined failures. The label-construction threshold was fixed at the recorded 1.0 value; the explicit override no longer computes an implant-based tuning curve. SFT versus RL groups are disjoint under the reconstructed grouping. CF excludes implants and CM excludes the declared held-out editors in every new applicable inventory. Each continuation control intentionally uses eligible RL identities and represents an alternative experiment, not another sequential SFT-pool stage.

Manifest freeze and immediate verification ran only after successful raw provenance, pool self-tests and new-stage exposure checks. Both scope-specific split locks and a SHA-256 pool lock are in the private run directory; aggregate split locks are copied here. Image and input hashes are retained in private manifests/census records. No old checkpoint is certified for the new protocol; original training/replay exposure and prior test inspection cannot be undone.

An additional independent post-freeze check re-read every final image, verified SHA-256 and decoded-pixel identities, reconstructed raw identity-component membership and source hashes, and required exact-source twin links in the same partition/group. Both scopes passed with zero cross-partition or cross-group decoded-content/raw-identity components, zero changed image hashes, and zero missing/substituted twins. This does not fill unknown patient, study or video fields. Its aggregate evidence is in materialized_provenance.json.

## Artifacts and next step

Private artifact locations for this execution:

- Isolated code: `/home/sachin.chaudhary/medforge_change1_20261101T111042Z/checkout`
- MF-Clean v2: `/home/sachin.chaudhary/medforge_change1_20261101T111042Z/data/mfclean_change1_v2r2_20261101`
- Kidney/colon split-only v2: `/home/sachin.chaudhary/medforge_change1_20261101T111042Z/data/extension_split_v2r3_20261101`
- Logs, complete identities, duplicate decisions and locks: `/home/sachin.chaudhary/medforge_change1_20261101T111042Z/run`

Publishable aggregate evidence is alongside this report. Full command stdout/stderr, source membership, exact paths and image hashes remain private. `code.diff` records the changes against the requested input commit. Historical manifest bytes were checked against the initial audit; checkpoints and historical image data were not written.

Next: train a new original-organ CF baseline from the base pretrained model, using only the newly frozen SFT-CF pool, inside an allocated GPU task. The saved `next_baseline.sh` verifies the split and pool locks first; it was **not executed**. Checkpoint/config candidates and a validation-only selection policy must be fixed before comparing candidates. Evaluation on these already-inspected test identities remains exploratory. A confirmatory claim requires independent identities and, for patient/video independence, the missing source linkage information.
