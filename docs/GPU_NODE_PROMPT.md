# GPU-node task: Bridge on the full MedForge protocol

Run **our Bridge method**, without retraining MedForge-Reasoner or adding kidney/colon images yet. Execute the work; do not stop at a plan.

## Setup

- Use an isolated checkout of `https://github.com/hardiksharma6555/MedForgePlusPlus`, branch `fix/split-protocol`, commit `5baa6572f79ef9c272f17b7843de32314691601c`.
- Read `docs/ORIGINAL_PROTOCOL_REPRODUCTION.md`, `docs/PROJECT_BLUEPRINT.md` and applicable `AGENTS.md` instructions.
- Locate the dataset, Qwen3-VL-8B base weights and compatible training environment. Follow the node's scheduler/GPU allocation rules. Preserve existing files and use a new run directory.

## Execute

1. Use official dataset `RichardChenZH/MedForge-90K`, revision `7bbf54f870584b37d6594593da76b2fc78fe4c5b`: **55,276 SFT**, **10,000 RL**, **30,000 test** records. Verify manifest hashes and every referenced image. Do not silently skip missing files or recreate the split.
2. Freeze training-side validation, seed, budgets and checkpoint/alpha selection rules. Keep SFT/RL pools separate. Repair and test evaluator compatibility, parsing and metric definitions; reference answers must never enter inference prompts.
3. Run Bridge's **SFT → eligible training-only replay → RL → validation-selected interpolation/stage**. Train ID first, then independent CF and CM models:
   - **ID:** all eligible training types/editors.
   - **CF:** exclude implants from every training/replay/RL stage.
   - **CM:** exclude `gemini`, `gpt`, `stable-diffusion-3.5-medium` and `stable-diffusion-xl-1.0-inpainting-0.1` from every training stage.
4. Use complete eligible pools, except documented validation carve-outs. Run a small validation smoke test before full jobs. Evaluate each selected model on **all 30,000 official test images**. Never tune on test results.

## Deliver

Save configs, environment, hashes, job IDs/logs, checkpoints, per-image predictions and Table-1-style accuracy/F1, per-type recalls and localization metrics. Label MedForge's published scores as reference results. Table 2 judge evaluation is a separate follow-up.

Update the project blueprint and return a concise status report with actual results and artifact paths. If blocked, identify the exact missing resource or failure; never claim an unexecuted run succeeded. Keep medical images, patient identifiers and credentials private.
