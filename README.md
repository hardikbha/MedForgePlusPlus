# MedForge++

Research code, paper drafts, dataset construction tools, and an independent readiness review for **MedForge++**, an extension of medical image forgery benchmarks with kidney CT and colonoscopy data.

**Status: research snapshot, not submission-ready.** The original paper's headline results are affected by training/evaluation overlap and shortcut cues. The newer **MF-Clean** rebuild addresses several of these problems and provides a stronger basis for further work. Read the [CVPR review and prioritized improvement plan](docs/CVPR_REVIEW.md) before interpreting the paper.

## Start here

- [Review: what is complete, what is pending, and how to improve it](docs/CVPR_REVIEW.md)
- [Current results generated from saved metric files](docs/RESULTS.md)
- [Dataset card and release limitations](docs/DATASET_CARD.md)
- [Reproduction instructions](docs/REPRODUCIBILITY.md)
- [Original CVPR-format paper PDF](paper/legacy_cvpr_draft/main_updated.pdf) and [supplement PDF](paper/legacy_cvpr_draft/supplementary.pdf) — historical drafts with superseded claims
- [Fresh filesystem/manifest audit](docs/evidence/release_data_audit.json), [earlier audit evidence](docs/evidence/audit/census.md), and [MF-Clean data checks](docs/evidence/v3/data_v3b.md)

## What is included

| Directory | Contents |
|---|---|
| `construction/` | FLUX kidney/colon implantation and annotation scripts; early manifest builders in `construction/legacy/` |
| `circe/` | Original SFT, replay, RL, interpolation, and evaluation code used during development |
| `v3/` | MF-Clean builder, structured prompts, SFT/RL data construction, CT-GRPO experiments, and grouped scoring |
| `v3_audit/` | Census, leakage, duplicate, annotation, and shortcut analysis code |
| `results/` | Existing aggregate metrics and selected saved predictions, with workstation path prefixes replaced |
| `paper/legacy_cvpr_draft/` | CVPR-format LaTeX sources and existing compiled drafts |
| `docs/` | Review, dataset card, reproduction notes, provenance, and historical technical evidence |

This snapshot includes previously uncommitted local work. It uses a fresh Git history and does not publish the private CIRCE repository's history. Raw medical images, source volumes, model weights, credentials, job logs, and confidential reviewer correspondence are not included. A complete image-dataset release remains pending.

## Quick verification

Python 3.10+; scoring requires no GPU, images, or model downloads:

```bash
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

The last command compares existing GRPO and class-balanced continued-SFT runs. Balanced accuracy is approximately **77.81% versus 79.93%**, respectively. These values do not support a GRPO advantage over that control. They use the **MF-Clean** test protocol and cannot be compared directly with the older paper's accuracy numbers.

In the recorded training environment, run `python v3/test_ctgrpo.py` for the reward/sampler tests. Its final prompt-parity check also needs local Qwen3-VL processor files under `$MEDFORGE_MODEL_ROOT/Qwen3-VL-8B-Instruct`. See [reproduction notes](docs/REPRODUCIBILITY.md) for versions and data paths.

## Relationship to prior work

The original [MedForge paper](https://arxiv.org/abs/2603.18577) and MedForge-90K were created by Chen et al. MedForge++ builds on that work; the upstream dataset and detector are not original contributions of this repository. Our snapshot includes the extension work, Bridge/CIRCE development, and subsequent MF-Clean audit/rebuild. Preserve attribution and follow each source dataset and model's terms.

No project-wide software or dataset license has yet been chosen by the authors. Public availability alone does not grant an open-source license; see [rights and release status](docs/DATASET_CARD.md#rights-and-release-status).
