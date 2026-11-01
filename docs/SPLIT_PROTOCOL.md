# Change 1: split integrity and validation-only selection

Status (2026-11-01): the private raw-file audit, new versioned split builds, regenerated pools and split/pool locks have been executed. See [the execution report](../SPLIT_AUDIT_REPORT.md). Overall **partially resolved**: unknown patient/study/video identities, historical exposure and fresh training/confirmatory evaluation remain open. Kidney/colon now have a separate split-only version; source/editor/format and clinical validity remain unrepaired.

## Fixed protocol

1. Form connected components before splitting: original scan, every derived forgery, known patient/study/volume, exact duplicates and verified near duplicates. Use globally namespaced IDs; e.g. different datasets' patient `123` must not be conflated. Kidney CT requires case/volume grouping; video-derived colonoscopy requires patient/video grouping where available. Missing identifiers are an explicit limitation, not evidence of independence.
2. Assign each component to exactly one partition. The existing MF-Clean hash is `SHA1("mfclean-v1:" + group) % 100`: train below 55, selection 55–69, test 70–99. These are group hash thresholds, not guaranteed image percentages. The base implementation preserved that assignment and historical core sampling. The private-data repair merges independently verified connected components, retains historical examples, and adds exact-source twins; all population/group changes are versioned. Neither action manufactures an untouched test set.
3. Inside train, use the existing `SHA1("sftrl-v1:" + group) % 6 == 0` RL-pool rule, otherwise SFT. Apply CF (no implants) and CM (held-out editors) exclusions to every training, replay and continued-SFT stage. Balanced continued SFT uses the same eligible continuation pool as RL, with its sampling recipe and exposure recorded.
4. Choose checkpoints, replay settings, alpha, reward weights and all other hyperparameters using selection data only. Pre-register candidate IDs, budgets, configs, metric and tie-break. Primary selection metric here is binary balanced accuracy, with candidate ID as the deterministic exact-tie breaker. Localization and per-type recalls remain reported secondary outcomes.
5. Freeze the chosen configuration and checkpoint hashes before final test inference. Use one predeclared policy per regime; do not pick a different stage using each test table's maximum. After test inspection, subsequent development needs a separately identified confirmatory evaluation. Existing public test results are historical/exploratory, not newly untouched evidence.

## Implemented checks

[split_protocol.py](../v3/split_protocol.py) rejects duplicate image identifiers (including lexical path aliases), missing groups, incorrect split declarations, empty partitions, unknown types, shared group/source/patient/duplicate/content identities across groups or splits, and invalid declared twin links. It reports optional-identity coverage and missing twins. It does not infer identity from raw images, resolve filesystem aliases, or certify that supplied IDs are truthful or complete.

[build_mfclean.py](../v3/build_mfclean.py) now retains source IDs, known filename-derived patient IDs, duplicate-cluster IDs and input content hashes; validates partitions before publishing manifests; and refuses to publish partially materialized splits. Cached-image resume now reconstructs fake boxes instead of returning null annotations. Reuse an output directory only with unchanged source data, masks and rendering settings; use a new versioned directory after any such change.

[build_sft_v3.py](../v3/build_sft_v3.py) checks the base manifests before pixel processing and before indexing rows by image. `--selftest_only` now fails with a nonzero exit code when a check fails. The normal build already failed on its final self-tests; that behavior is retained. Existing checks cover SFT/RL group separation, regime exclusions and target schema. A failed build may leave diagnostic/output files; only a successful run is usable.

The generic audit can inspect old manifests, but missing optional identity columns are reported as missing coverage. A PASS means that the supplied metadata satisfies the checks. It is not a universal patient-disjointness claim. A missing twin remains reported rather than silently changing the historical test population.

## Run on the data host

After construction and provenance review, with the actual private manifests:

```bash
python v3/split_protocol.py --mfclean "$MEDFORGE_DATA_ROOT/mfclean"
python v3/split_protocol.py --mfclean "$MEDFORGE_DATA_ROOT/mfclean" \
  --freeze /path/to/versioned-protocol/split-lock.json
python v3/split_protocol.py --mfclean "$MEDFORGE_DATA_ROOT/mfclean" \
  --verify-lock /path/to/versioned-protocol/split-lock.json
python v3/build_sft_v3.py --mfclean "$MEDFORGE_DATA_ROOT/mfclean" --selftest_only
```

The lock contains SHA-256 hashes of the exact three manifest files, counts and audit scope. It refuses overwrite and detects even whitespace changes. Verify it before and after each training/evaluation stage. It does not freeze image bytes or prevent a researcher from deliberately rewriting files; preserve the lock, image hashes, candidate plan and run logs in version control or immutable storage.

Prepare a candidate JSON **before the sweep** (paths relative to this file):

```json
[
  {"id": "sft-step100", "predictions": "select/sft-step100.jsonl", "artifact": "weights/step100.safetensors", "config": "configs/step100.json"},
  {"id": "sft-step200", "predictions": "select/sft-step200.jsonl", "artifact": "weights/step200.safetensors", "config": "configs/step200.json"}
]
```

Run selection after producing those selection predictions:

```bash
python v3/select_checkpoint.py --mfclean "$MEDFORGE_DATA_ROOT/mfclean" \
  --lock /path/to/versioned-protocol/split-lock.json \
  --candidates /path/to/versioned-protocol/candidates.json \
  --out /path/to/versioned-protocol/selection-decision.json
```

The selector requires exact selection-set coverage and matching manifest metadata, re-parses raw outputs using the released scorer, treats unparsed outputs as wrong, rejects test rows and records artifact/config/prediction hashes. The decision file cannot be overwritten. For sharded checkpoints supply a checksum inventory as `artifact` and independently verify every shard before use. This tool cannot prove that training respected exclusions, that predictions came from the named checkpoint, that the candidate plan was registered in advance, or that someone has never viewed test results. Those require run provenance and review.

## Completion evidence still required

| Gate | Required artifact | Current status |
|---|---|---|
| Code behavior | Adversarial synthetic tests for leakage, selection and resume | Implemented; see [validation](VALIDATION.md) |
| Identity reconstruction | Mapping with exact source, patient/study where known, source hashes and duplicate decisions | Completed for available identities; unknown fields remain explicit |
| Actual disjointness | Full-data audit, zero cross-split components, documented unknown-ID coverage | Passed for reconstructed identities; independent materialized-pixel check passed |
| Locked experiment | Versioned split lock, candidate plan, run config, environment and immutable checkpoint hashes | Data/pool locks frozen and verified; prospective candidate/checkpoint plan remains future work |
| Regime validity | SFT/replay/RL/continued-SFT inventories, zero forbidden types/editors and zero held-out identities | New pool inventories passed; historical exposure retained |
| Confirmatory results | Retrained baselines, validation-selected checkpoints, fixed test predictions and grouped CIs | No new training/inference performed |

Do not label Change 1 fully resolved until the data and experiment gates are complete. Run synthetic checks with `python -m unittest discover -s tests -v`.
