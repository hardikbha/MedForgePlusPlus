# CT-GRPO trainer review (train_ctgrpo.py / ctgrpo_core.py), 2026-10-21

Reviewer pass on `~/CIRCE/v3/{train_ctgrpo.py, ctgrpo_core.py, test_ctgrpo.py, dryrun_ctgrpo.py}` before the 8B GPU
run. Everything ran on CPU (`CUDA_VISIBLE_DEVICES=""`, nice, at most 16 threads in total). Env: `~/.conda/envs/Qwen3.5`
(transformers 5.8.1, trl 0.16.1, peft 0.19.1, torch 2.5.1). `prompts_v3.py` and `~/CIRCE/circe/vlm_grpo_patch.py` were
not modified.

## Verdict

**Ready to launch once the SFT adapter exists.** Do a 5-step GPU smoke run first (command in section 6) to measure memory
and step time, and to see the new step-1 log-prob self-check pass on GPU. One real correctness bug was found and fixed:
EOS handling. The other changes are guards and robustness. The core maths (rewards, the 4 advantage modes, sampler
layout, image threading, the loss) matched independent references.

## 1. Bugs found and fixed

| # | severity | finding | fix |
|---|---|---|---|
| 1 | **real bug** (audit B6, never fixed) | Rollouts stopped only on `<|im_end|>` 151645. trl 0.16.1 builds `GenerationConfig(eos_token_id=processing_class.eos_token_id)`, and transformers 5.8.1 does not merge `model.generation_config`'s `[151645, 151643]` into it, so `_model_shims` had no effect. The patch builds the loss mask from 151645 only. If the policy emitted `<|endoftext|>` 151643, generation kept going and the mask covered the endoftext token and every token sampled after it, all trained with that rollout's advantage. Reproduced on the tiny model (test D, "record of the bug"). | `CTGRPOTrainer.__init__` sets `self.generation_config.eos_token_id = [151645, 151643]` (pad 151643). `_generate_and_score_completions` rebuilds `completion_mask` = up to and including the first of either id (`completion_mask_multi_eos`) and corrects trl's `completion_length` metric. New diag fields: `n_stop_on_endoftext`, `n_mask_rebuilt`, `pad_tokens_in_mask` (pads inside the mask other than a terminating 151643; must be 0). |
| 2 | guard (silent fallback) | `--adv_mode cogroup_bonus` with the default `--pair_bonus 0` is exactly `image`. `--adv_mode gdpo --scale_std 0` is exactly `image` (the existing test even asserts this). twin_pooled / cogroup_bonus on data without pairs is exactly `image`. Each of these would have run standard GRPO under a CT label. | `parse_args` and `CTGRPOTrainer.__init__` refuse all three. |
| 3 | robustness | `TwinPairSampler` had no `set_epoch`; epoch order depended on how many iterators had been created. | `set_epoch(e)` added (transformers -> accelerate `DataLoaderShard.set_epoch` -> sampler). Verified through the real Trainer dataloader over 2 epochs (test A). |
| 4 | test-harness drift | `dryrun_ctgrpo.py` built its own hand-copied `GRPOConfig`, so it never exercised `make_trainer`, the config the GPU run uses. | `make_trainer(..., trainer_cls=)` added. The dry run now goes through `make_trainer` (only `save_strategy="no"` is overridden). |
| 5 | missing option | Sampler support for capping pairs per real twin existed but the CLI did not expose it (rl_cf reuses one real in up to 20 pairs; 263 reals are used 10x). | `--max_pairs_per_real` (default 0 = off, so behaviour is unchanged). `--save_total_limit` also added (default 0 = keep all). |
| 6 | safety net (new) | On CPU, batched vs single-row log-probs disagree by up to 0.9 nats when torch uses >1 thread (section 3). This can't be verified on GPU from here. | `--logp_selfcheck 1` (default): at step 1, 4 rows (shortest and longest completions, distinct images) are run batched vs each alone, unpadded, with their own pixels, under no_grad. It prints the result, logs `logp_selfcheck_maxdiff` in diag, and on GPU aborts if the diff is > `--logp_selfcheck_abort` (0.5). Cost: one no-grad forward of 4 rows plus 4 single rows, once. |
| 7 | doc | The usage line pointed at a non-existent `mfclean/rl_CF.jsonl`. The target-length comment said 73-83 tokens; measured on rl_cf + sft_cf it is 56 (real) and 76-79 (fake) incl. `<|im_end|>`. | Fixed. |

Not a bug, but a decision for the lead. `--adv_mode twin_pooled` defaults to `--pool_channels verdict`: the verdict
reward is centred over the 2G rollouts of the pair, the localisation reward is centred per image, and the result is
divided by one std over the pair. The literal spec ("normalise jointly over the 2G rollouts") is `--pool_channels all`.
I recommend keeping `verdict`. Under `all`, a fully solved pair still has an offset: a correct real scores 1.5 and a
correct fake scores 1 + 0.5·IoU < 1.5. That offset pushes the correct fake rollouts down and the reals up, a spurious
"say real" signal (shown by hand in `test_review_core` and `test_ctgrpo.adv_twin_pooled_solved_pair_offset`). The
launch command passes `--pool_channels verdict` explicitly so the paper can state it. `all` remains available as an
ablation.

Untouched, correct as written (verified): reward rules; the image / twin_pooled / cogroup_bonus / gdpo formulas; the
layout check from pair_id/role columns; the per-step diagnostics; loss micro-batching (gradient equals one-shot, max
diff 8e-9); `beta` refused unless 0; the honest objective banner.

## 2. Independent tests written (`~/CIRCE/v3/review_trainer/`)

`test_review_core.py` (pure, about 40 s). **21/21 pass.**
* Reward: 16 hand cases covering IoU 1, 1/3 and 0, wrong verdicts, the real-with-box and deepfake-without-box
  contradictions, unparseable inputs (empty, bad verdict word, box without verdict), case-insensitive verdicts, and the
  field bonus on correct fakes only.
* Advantages: G=4 pair + singleton fixture, hand-computed for every mode (image centred/scaled, twin_pooled verdict/all,
  cogroup_bonus λ=0.3, gdpo). Also my own spec-level reference (plain Python + `statistics.stdev`, written
  independently of the engineer's `ref_adv`) on 200 random layouts x 4 modes x scale on/off: max diff 2.3e-7.
* No silent fallback: on a twin-contrast pair, image gives 0 and twin_pooled is non-zero. The CLI refuses the configs
  in bug #2.
* Sampler: full epochs for (pairs, singletons, K) in {(37,9,2), (5,0,2), (1,3,2), (11,1,3), (0,7,2), (13,13,4),
  (101,25,2)} over 4 epochs. Checked: every image is a run of G identical indices; each fake is immediately followed by
  its own real in the same step; no image repeats within an epoch; singletons come in coupled pairs; used slots =
  floor(n/K)·K; epochs are reshuffled. Also seed and `set_epoch` replay, and the `max_pairs_per_real` cap.
* EOS mask unit test. Real data: rl_cf loads under `--regime CF` (3017 pairs + 754 singleton reals -> **3394 slots ->
  1697 steps/epoch at K=2**); every pair is (fake, its real); a pyarrow round trip keeps `gt_box`; the longest GT target
  is 79 tokens, so `max_completion 160` is 2x; token ids are 151645 = `<|im_end|>` and 151643 = `<|endoftext|>` = pad.

`test_review_model.py` (tiny random Qwen3-VL + LoRA, real trl + patch + CTGRPOTrainer via `make_trainer`). **17/17 pass.**
* A: the real Trainer/accelerate dataloader over 2 epochs on odd sizes (7 pairs + 5 singletons, K=2) gives batches
  identical to `epoch_slots(e)`.
* B: the pixel slice that `generate()` receives for each row is exactly that row's own image (2K=6 images, 2 sizes),
  and the `<|image_pad|>` count matches its grid. A policy that answers from the *pixels* gets per-row rewards equal to
  that row's GT. The advantages that reach `compute_loss` are the CT ones: in image mode all 0; in twin_pooled,
  contrast pairs give fake < 0 and real > 0.
* C: the loss forward with left-padded rows: batched per-token logps equal the unpadded single row with its own pixels
  (9.5e-7, 1 thread). Swapping in the twin's pixels changes a row's logps by 0.12, so the image really is used.
* D: real tiny-model `generate` with forced tokens. It stops on 151643 and on 151645; the mask covers up to and
  including the first EOS; no pad tokens are inside the mask. The pre-fix config is also shown running past 151643.
* E: 2 steps x 3 runs. `checkpoint-1` and `checkpoint-2` contain adapter, optimizer, scheduler, rng and trainer_state.
  The final adapter equals checkpoint-2, which equals the trained LoRA. With the same `--seed`, step order, rollout RNG
  draws and final LoRA are bit-identical (1 thread); a different seed gives different results.
* F: the real `build_model()` path on CPU (saved tiny base + a saved non-identity "SFT" adapter). Adapter weights load
  exactly, only LoRA is trainable, and one CT-GRPO step updates the loaded adapter.

Existing suites after the fixes: `test_ctgrpo.py` **19/19 pass** (16 before + 3 new); `dryrun_ctgrpo.py` **ALL CHECKS
PASSED** in all 5 mode configurations (image, twin_pooled, twin_pooled/all, cogroup_bonus, gdpo). In every mode: 28/28
LoRA tensors updated, advantages at `compute_loss` == reference (<= 9.2e-8), identical raw rewards across modes, every
mode differs from image, micro-batch gradient == one-shot. It passed before the fixes too. The CLI `--dry_run` on the
real rl_cf (3394 slots -> 1697 steps; GT reward 1.5, garbage -1.5) and the CLI `--tiny` end-to-end via `main()` for image
and twin_pooled (2 steps, checkpoints + final adapter written, self-check 0.0) both exit 0. Logs: `review_trainer/logs/*_final.log`.

## 3. Findings that are not code bugs but matter

* **CPU attention is thread-dependent in this env.** Inside Qwen3-VL on CPU with >1 intra-op thread, any masked
  (padded) sequence gives logits that differ from the unmasked computation by 0.3-0.9 nats, even with identical
  position ids. I confirmed the 4D bool mask is correctly causal. With 1 thread the difference is 8e-7. I could not
  reproduce it with standalone `F.scaled_dot_product_attention`. Consequences: (a) CPU dry runs are only numerically
  meaningful single-threaded, so the review tests use 1 thread where it matters; (b) multi-thread CPU training is not
  bit-reproducible (LoRA differs by up to 1e-2 after Adam even though rollouts, rewards, advantages and losses are
  identical). GPU kernels are different code, but I couldn't check them, hence the step-1 self-check (fix #6). Real
  rl_cf images are all 1024x1024, so prompts are never left-padded, but completions (56 vs about 78 tokens) are always
  right-padded.
* **Exact resume is impossible in this env.** `trainer.train(resume_from_checkpoint=...)` fails: transformers 5.8.1
  refuses `torch.load` for optimizer.pt / scheduler.pt / rng_state.pth with torch < 2.6. I did not add a `--resume`
  flag. If a run dies, the checkpoint adapters are still fine for evaluation or a warm restart (`--sft_adapter
  <out>/checkpoint-N`, which resets Adam/LR state and replays the sampler from the start).
* **Data protocol checked:** rl_cf is disjoint from sft_cf, test_core and select_core by image (0 overlap) and by group
  (0/983 RL groups in SFT/test/select). All 4070 images have split=train and are 1024x1024.
* LoRA dropout 0.05 from the SFT adapter is active in the loss forward (train mode) but not during generation (eval).
  This is standard trl/peft behaviour and the same as the older runs, so it is noted only.
* top_k=50 / top_p 0.95 / T 0.7 are trl defaults, not the model defaults (audit B8). Log-probs are untruncated
  softmax(logits/T).
* rl_cf real-image reuse: 299 distinct reals in 3017 pairs, so each pair-real is trained about 10x per epoch (x8
  rollouts). `--max_pairs_per_real` exists if this turns out to matter. It is off in the recommended runs, to keep
  "1 pass over all pairs".

## 4. Step count

1 epoch = 3017 pairs + floor(754/2) = 377 singleton slots = 3394 two-image slots. At K=2 that is **1697 optimizer
steps**, which covers every pair exactly once (6788 images, 54,304 rollouts). The linear LR decay runs over 1697
steps. Use the same `--seed 42` for both modes so both see the identical pair order (paired comparison).

## 5. Memory and time estimate (8B, LoRA r32, K=2, G=8 -> 32 rollouts/step, prompt 1360 incl. 1024 image tokens, completion about 80, max 160)

Analytic estimate. The smoke run records `cuda_max_mem_gb` per step in the diag jsonl.

| component | GB |
|---|---|
| base weights bf16 (8.77B params incl. vision) | 17.5 |
| LoRA 87.3M fp32 params + grads + Adam m,v | 1.4 |
| **generation phase**: KV cache 147 KiB/token x 32 x (1360+160) | 7.2 (worst) |
| prefill transients (LM MLP on 43.5k tokens about 3.2; vision 32x4096 patches about 1.1; deepstack 0.8) | about 4.5 |
| **generation peak** | **about 30-31** |
| **loss phase, one forward of 32 seqs** (grad-ckpt): stored layer inputs 36 x 32 x about 1450 x 4096 x 2 B | 13.7 |
| one-layer recompute + backward (MLP 4 x N x 12288 x 2 B, attn, grads) | about 8-9 |
| completion logits 32 x 161 x 151936 bf16, plus /T copy, log-softmax and grad | about 3-6 |
| embeds + deepstack features | about 1.2 |
| **loss-phase peak, `--loss_microbatch 0`** | **about 45-50** (allocator reserve about 55) |
| **loss-phase peak, `--loss_microbatch 16`** (2 chunks) | about 33 -> step peak **about 33-36** |
| loss-phase peak, `--loss_microbatch 8` (1 image group per chunk) | about 26 -> step peak about 31 (generation-bound) |

Micro-batching changes nothing numerically (dry run: gradient diff 8e-9 vs one-shot) and costs little throughput. On
the shared gpu01 I recommend `--loss_microbatch 16` with scheduler `min_free` 50000 MiB. On an idle 80 GB GPU,
`--loss_microbatch 0` (about 55 GB) is also fine.

Time (estimate from the audit's measured 12-14 s/step for 8 rollouts x 245 tokens and about 0.5-0.95 s per
1500-token fwd+bwd): generation about 5-8 s (about 80 decode steps at batch 32, plus prefill) and loss about 16-30 s,
so **about 25-35 s/step and about 12-17 h per 1697-step run** (two runs about 24-34 GPU-h). Confirm with the smoke run
(`gen_score_s`, `step_s` in diag).

## 6. Launch commands

Precondition: `${MEDFORGE_WORKSPACE}/AAAI/MedForge++/ckpt/v3_sft_cf/adapter_model.safetensors` exists (SFT `sft_cf`
started 18:39 on gpu1, 1949 steps at about 7.6 s/it, ETA about 22:45). One GPU per run (the scheduler sets
`CUDA_VISIBLE_DEVICES`; `device_map="auto"` would shard over several visible GPUs).

Smoke (5 steps, about 3 min; check: the log-prob self-check line, `cuda_max_mem_gb`, `step_s`, `parse_rate`,
`frac_no_eos`, `pad_tokens_in_mask` == 0):
```
cd ${MEDFORGE_WORKSPACE}/CIRCE/v3 && ${MEDFORGE_WORKSPACE}/.conda/envs/Qwen3.5/bin/python -u train_ctgrpo.py --sft_adapter ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/ckpt/v3_sft_cf --data ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/mfclean/v3data_b/rl_cf.jsonl --regime CF --adv_mode twin_pooled --pool_channels verdict --pairs_per_step 2 --num_gen 8 --max_steps 5 --save_steps 1000 --loss_microbatch 16 --seed 42 --out ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/ckpt/v3_ctgrpo_cf_smoke
```

Control (standard per-image GRPO):
```
cd ${MEDFORGE_WORKSPACE}/CIRCE/v3 && ${MEDFORGE_WORKSPACE}/.conda/envs/Qwen3.5/bin/python -u train_ctgrpo.py --sft_adapter ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/ckpt/v3_sft_cf --data ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/mfclean/v3data_b/rl_cf.jsonl --regime CF --adv_mode image --pairs_per_step 2 --num_gen 8 --max_steps 1697 --lr 1e-6 --temperature 0.7 --top_p 0.95 --top_k 50 --max_completion 160 --scale_std 1 --field_bonus 0 --loss_microbatch 16 --save_steps 200 --seed 42 --out ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/ckpt/v3_ctgrpo_cf_image
```

Ours (twin-pooled):
```
cd ${MEDFORGE_WORKSPACE}/CIRCE/v3 && ${MEDFORGE_WORKSPACE}/.conda/envs/Qwen3.5/bin/python -u train_ctgrpo.py --sft_adapter ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/ckpt/v3_sft_cf --data ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/mfclean/v3data_b/rl_cf.jsonl --regime CF --adv_mode twin_pooled --pool_channels verdict --pairs_per_step 2 --num_gen 8 --max_steps 1697 --lr 1e-6 --temperature 0.7 --top_p 0.95 --top_k 50 --max_completion 160 --scale_std 1 --field_bonus 0 --loss_microbatch 16 --save_steps 200 --seed 42 --out ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/ckpt/v3_ctgrpo_cf_twin
```

Scheduler lines (`~/CIRCE/v3/sched/tasks2.txt` format, waiting on SFT):
```
ctgrpo_cf_image|50000|after:sft_cf;cd ${MEDFORGE_WORKSPACE}/CIRCE/v3 && ${MEDFORGE_WORKSPACE}/.conda/envs/Qwen3.5/bin/python -u train_ctgrpo.py --sft_adapter ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/ckpt/v3_sft_cf --data ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/mfclean/v3data_b/rl_cf.jsonl --regime CF --adv_mode image --pairs_per_step 2 --num_gen 8 --max_steps 1697 --lr 1e-6 --temperature 0.7 --top_p 0.95 --top_k 50 --max_completion 160 --scale_std 1 --field_bonus 0 --loss_microbatch 16 --save_steps 200 --seed 42 --out ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/ckpt/v3_ctgrpo_cf_image
ctgrpo_cf_twin|50000|after:sft_cf;cd ${MEDFORGE_WORKSPACE}/CIRCE/v3 && ${MEDFORGE_WORKSPACE}/.conda/envs/Qwen3.5/bin/python -u train_ctgrpo.py --sft_adapter ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/ckpt/v3_sft_cf --data ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/mfclean/v3data_b/rl_cf.jsonl --regime CF --adv_mode twin_pooled --pool_channels verdict --pairs_per_step 2 --num_gen 8 --max_steps 1697 --lr 1e-6 --temperature 0.7 --top_p 0.95 --top_k 50 --max_completion 160 --scale_std 1 --field_bonus 0 --loss_microbatch 16 --save_steps 200 --seed 42 --out ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/ckpt/v3_ctgrpo_cf_twin
```
Outputs: adapter in `--out` (plus `checkpoint-{200,...,1600}`, about 1 GB each), per-step diagnostics in
`<out>_diag.jsonl`, args in `<out>/ctgrpo_args.json`.

## 7. Remaining risks

1. GPU numerics of padded batches can't be verified from here. The step-1 self-check covers it (aborts at > 0.5
   nats). bf16 batch-vs-single noise of about 1e-2 to 5e-2 is normal.
2. No exact resume (env limitation above). A crash at hour 10 means a warm restart from the last adapter, or a rerun.
3. Memory and time are estimates; run the smoke first.
4. If the SFT policy is near-deterministic on rl_cf, most groups will be zero-std in image mode. That is the motivating
   case for twin_pooled, but if both twins are always solved, both modes do little. Watch `frac_zero_std_img/twin`,
   `frac_twin_contrast_only` and `acc_fake/acc_real` in the first 50 steps.
5. Real-image reuse (about 10x per epoch) could overfit the real class; `--max_pairs_per_real` is available.

## 8. Files changed / added

* `train_ctgrpo.py`: EOS fix, mask rebuild + diag fields, silent-fallback guards (CLI and trainer), `--max_pairs_per_real`,
  `--save_total_limit`, `--logp_selfcheck[_abort]` + `_ct_logp_selfcheck`, `make_trainer(trainer_cls=, unit_real=)`,
  doc fixes, resume note.
* `ctgrpo_core.py`: `TwinPairSampler.set_epoch`.
* `test_ctgrpo.py`: +3 tests (set_epoch replay, EOS mask, CLI fallback guards) -> 19 tests.
* `dryrun_ctgrpo.py`: builds through `make_trainer`.
* New: `review_trainer/test_review_core.py`, `review_trainer/test_review_model.py`, `review_trainer/scratch/diag_C*.py`
  (the CPU-attention investigation), logs in `review_trainer/logs/`, pre-review copies `review_trainer/*.orig` (core,
  tests, dryrun; the original `train_ctgrpo.py` is described in the table above).
