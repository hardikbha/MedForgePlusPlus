# Release validation — 2026-11-01

The following checks were run on the public snapshot before publication:

| Check | Outcome |
|---|---|
| Python source syntax | All 127 Python files parse successfully |
| `v3/stats_v3.py --self_test` | Passed confusion/IoU/twin checks, parsing, constant/oracle cases, group bootstrap and paired comparison checks; legacy metadata resolver branch skipped because its private cache is absent |
| `v3/eval_v3.py --self_test` | Passed prompt parity, oracle/constant/garbage dry runs, saved-row bookkeeping, resume and sharding |
| `v3/test_ctgrpo.py` | All 19 tests passed in the recorded training environment, with local Qwen3-VL processor files and network access disabled for Hugging Face |
| Published prediction rescoring | Accuracy, balanced accuracy, macro-F1, mIoU, IoU@0.5 and twin consistency match saved originals to 1e-12 for both compared models |
| Fresh local-data audit | Confirmed file counts, format signatures, byte duplicates, original evaluation/training overlap, and disjoint declared MF-Clean groups |
| New documentation links | All relative file links checked |
| Publication contents | Common credential patterns absent; no model-weight artifacts or raw medical-image dataset trees included |

CPU scorer/evaluation self-tests ran in the existing `aiface` environment. The trainer/core tests ran in the existing `Qwen3.5` environment with `MEDFORGE_MODEL_ROOT` set to the local model directory. An initial test attempt without that variable failed at the processor-dependent prompt-parity case; setting the documented local processor path allowed the complete suite to pass. The published training requirements record observed versions rather than a newly installed environment.

The [reproduced paired report](../results/reproduced_balanced_sft_vs_grpo.md) uses 1,000 source-group bootstrap replicates. Balanced continued SFT minus image-normalized GRPO is **+2.12 percentage points [95% CI +1.24, +2.97]**. This recomputes existing saved predictions; it is not a new model training result. Intensity-field ground truth is not included with those selected prediction files, so the rescore leaves intensity accuracy unavailable; the six primary reproduced metrics do not require it.

Not validated here: a clean-environment installation, full GPU training, editing-model generation, a full raw-data reconstruction, clinical correctness, patient independence where IDs are unavailable, licensing clearance, or rebuilding the preserved historical manuscript PDFs. Those remain explicit limits of the snapshot.

## Independent follow-up and Change 1 — 2026-11-01

The earlier section records original-release validation in its historical environments. The independent follow-up used macOS arm64, Python 3.14.7, NumPy 2.5.4 and Pillow 12.3.0. A fresh CPU virtual environment was installed from `requirements-cpu.txt`; this is not a GPU training-environment reproduction.

| Check actually run in the follow-up | Outcome |
|---|---|
| `python -m unittest discover -s tests -v` | 15 tests passed: duplicate paths/aliases, shared declared identities, wrong split, invalid/missing twins, manifest freeze/tampering, failed-selftest exit, pre-processing overlap rejection, validation-only checkpoint choice and cached localization-box preservation |
| `python v3/stats_v3.py --self_test` | Passed; private legacy resolver branch skipped |
| `python v3/eval_v3.py --self_test` | Passed synthetic oracle/constants/garbage, prompt checks, resume and sharding |
| Independent historical-snapshot review | 127 Python AST parses; eight prediction files' primary metrics matched archived values exactly; 49 result hashes matched. [Recorded verification](evidence/independent_review_20261101/verification.json) |
| Basic baseline comparison | Existing predictions only, 5,000 paired group-bootstrap replicates, seed 2027; [full results](../results/baseline_rescore_20261101/results.json) and [CSV](../results/baseline_rescore_20261101/metrics_percent.csv) |

The earlier synthetic resume failure in `verification.json` is historical evidence of the bug fixed in this change. Its downstream recovery path means the failure does not establish corrupt archived training labels. The regression test now verifies identical first-run/resumed boxes, including mask scaling.

No fresh training, real-model inference, private-data audit/rebuild, clinical validation or processor-dependent trainer test suite was performed in this follow-up. The real split lock and checkpoint selection decision cannot be produced from the public prediction files alone. See [completion gates](SPLIT_PROTOCOL.md#completion-evidence-still-required).
