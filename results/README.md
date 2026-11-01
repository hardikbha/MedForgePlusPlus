# Existing experimental artifacts

`metrics/` contains saved selection/test metric files from the MF-Clean experiments. `predictions/` contains eight selected saved test-generation files so the main comparisons can be recomputed without downloading medical images or model weights. All test files have 2,756 rows from 1,008 source groups; selection metrics are a separate protocol.

Workstation path prefixes were replaced consistently with relative `dataset/` identifiers. These identifiers preserve joins and twin references but are not download URLs. Original source hashes are in [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json).

[reproduced_balanced_sft_vs_grpo.md](reproduced_balanced_sft_vs_grpo.md) was newly generated with the released scorer and selected predictions. It compares **A = image-normalized GRPO** with **B = balanced continued SFT**; differences are **B minus A**. Use the paired source-group intervals for inference, rather than the historical image-level McNemar statistic.

Selected predictions do not include every ground-truth evidence-field file; intensity-field results in the archived aggregate JSONs cannot be fully reproduced from these files alone. The published primary detection/localization/twin results reproduce exactly for the comparison described in [validation](../docs/VALIDATION.md). No training was rerun during this release.
