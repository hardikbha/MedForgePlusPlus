# Reproduction and validation

## Scope

This is a curated snapshot of the original MedForge++ and CIRCE working trees, including local uncommitted changes. [Source hashes](SOURCE_SNAPSHOT.json) record the inputs. Absolute executable paths were adapted through [release_paths.py](../release_paths.py); source comments/documentation may retain historical layout descriptions. [Result provenance](../results/SOURCE_SNAPSHOT.json) records the saved-run inputs. Workstation prefixes in published prediction files are relative identifiers; scoring uses those identifiers and does not open images.

GPU training, editing-model generation, and the complete historical audit were not rerun for this public release. Selected existing predictions permit CPU rescoring without access to medical images. See [validation results](VALIDATION.md) for checks actually performed.

## CPU scoring

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python v3/stats_v3.py --self_test
python v3/eval_v3.py --self_test
python scripts/summarize_results.py
python v3/stats_v3.py \
  --gen results/predictions/gen_test_ct_image_1697.jsonl \
  --gen2 results/predictions/gen_test_contsft_cf_bal.jsonl \
  --B 1000 --out_json /tmp/medforge_comparison.json \
  --out_md /tmp/medforge_comparison.md
```

`--B` controls bootstrap replicates. The paired output reports **B minus A**, so the command above reports balanced continued SFT minus GRPO. Historical McNemar output is included by the scorer; source-group bootstrap intervals are the appropriate primary inference for correlated image families. A passed synthetic self-test establishes code behavior, not dataset quality or model validity.

## Data and construction

Export variables rather than copying workstation paths:

```bash
export MEDFORGE_DATA_ROOT=/absolute/path/to/MedForge++
export MEDFORGE_MODEL_ROOT=/absolute/path/to/models
export MEDFORGE_WORKSPACE=/absolute/path/to/source-workspace
```

The expected data root holds `medforge90k_full/dataset_mapping.json`, upstream `real/` and `deepfake/` trees, the original extension's `naitik_annotated/` and `colon_injections/`, and `mfclean/` output manifests. Additional source folders live under `$MEDFORGE_WORKSPACE/AAAI/`, including `colon`, `kidney`, `naitik_data` and editing models. The six annotation prompts are in `prompts/`.

To rerun the release's read-only checks against the original files:

```bash
python scripts/audit_local_data.py \
  --data-root "$MEDFORGE_DATA_ROOT" \
  --kidney-root "$MEDFORGE_WORKSPACE/AAAI/kidney" \
  --out /tmp/medforge_data_audit.json
```

MF-Clean construction requires the original upstream images/mapping **and** the hash/thumbnail artifacts produced by `v3_audit/scripts/census_hash.py` under `v3_audit/data/`. The census expects the extension folders too. It creates `image_hashes.tsv` and `thumbs32.npy`; use the same corpus enumeration for both. Then `v3/build_mfclean.py` builds the clean image and split manifests; `v3/build_sft_v3.py` constructs the SFT/RL pools. Study their `--help` options and reports before rebuilding; construction writes outputs and can be expensive.

Follow the [split protocol](SPLIT_PROTOCOL.md) to audit/freeze those manifests and select checkpoints using validation only. The builders now reject declared identity overlaps, and `--selftest_only` propagates failure. New synthetic regression tests run with `python -m unittest discover -s tests -v`. These safeguards have not been run against the private full dataset in this independent follow-up.

The early `construction/legacy/build_medforgepp.py` describes a different mixed corpus and must not be used to recreate the paper's kidney/colon release. Original removal images were ingested as external artifacts; their complete generation implementation and version record are still missing from this snapshot.

## Training environment

`requirements-training.txt` records versions present in the research environment, including TRL 0.16.1 and a custom VLM patch. This is an observed configuration, not a validated fresh-install lock file. Editing uses additional packages such as Diffusers, OpenCV and GGUF plus separately obtained model weights; no editing-environment lock is supplied yet.

`circe/train_sft.py` is the SFT entry point. `v3/train_ctgrpo.py` provides the ordinary image and twin-pooling variants. `v3/eval_v3.py` runs inference; `v3/stats_v3.py` scores saved outputs. All model/data/output arguments should point to your own resources. Do not launch heavy work implicitly from the CPU verification commands.

The implemented patched RL objective uses a token-level mean with no KL term and no clipping. The current trainer documents this and rejects incompatible beta settings. The legacy manuscript's KL/clipping equation does not describe that implementation. CT-GRPO is retained as experimental code; the current CF results do not establish an advantage over its matched image baseline.

## Paper

`paper/legacy_cvpr_draft/` contains the original and lightly updated CVPR-format drafts. Both retain superseded claims. To rebuild the corresponding sources, compile main first, then the supplement:

```bash
cd paper/legacy_cvpr_draft
latexmk -pdf main.tex
latexmk -pdf supplementary.tex
```

The existing PDFs are supplied for historical supervisor review. A complete scientific rewrite and current CVPR template/policy check are still required before submission. No author identities, citation metadata, conference acceptance or arXiv identifier have been invented for this release.

## Private Change 1 execution

See [the execution report](../SPLIT_AUDIT_REPORT.md) for actual paths, package versions, command outcomes and frozen artifacts. The raw-source checkpoint can be recreated with `scripts/private_split_audit.py --census-only`, then completed with `scripts/resume_identity_audit.py`. Both require the documented private input roots and a fresh output directory. `scripts/rebuild_split_protocol.py` forms versioned connected-component splits and exact-source twins; `scripts/prepare_continuation_pools.py` exports alternative unique/pair-exposure continuation controls and replay eligibility. `scripts/audit_training_exposure.py` checks pools against actual reconstructed identities and distinguishes a confirmed historical CF consumer from filename-only regime inference. `scripts/verify_rebuilt_artifacts.py` independently checks actual final image hashes/pixels, raw provenance and twins. Keep all row-level outputs private.

The split-only extension keeps original encoding and authentic source copies, so its replica check compares exact source bytes, not MF-Clean JPEG reconstruction. Full-frame boxes with unavailable contrast rings use source delta or explicit unknown intensity; clipping decisions preserve original box annotations in the private v2r3 manifests. These rules do not establish clinical validity. A prepared baseline wrapper is recorded privately and was not executed.
