# GRPO internals audit + v3 implementation spec (Bridge / MedForge++)

Role: GRPO-internals engineer. Everything below is read-only analysis of existing code, logs, checkpoints and
saved generations, plus new scratch code under `~/CIRCE/v3_audit/code_scripts/`. No GPU was used, no existing
file was modified. Every number was produced by a command run in this session; anything not measured is marked
**not measured** or **estimate**. Scripts that produced each number are listed in section 7.

---

## 0. TL;DR

1. **The RL stage hardly trains the verdict at all, and the logs show why.** With `reward_classification_dominant`
   + trl's default `scale_rewards=True`, the strict Cross-Forgery GRPO run (400 steps, 1 image/step) had
   **45% zero-variance steps** (logged `grad_norm == 0`), only **9-10 of 400 steps (2.3-2.5%)** containing any
   verdict disagreement, and **~52%** of steps whose only variance came from the 0-0.25 IoU bonus. That bonus is then
   std-normalised up to unit advantage, the same size as a verdict flip. Across 9 classification-dominant runs, verdict-split steps are 0.3-9.3% of steps.
   Ablating the localisation bonus (`grpo_ablate_loc`) makes **92.5%** of logged steps zero-gradient.
2. **Outcome check:** on the clean selection split, SFT to GRPO changed **23/3000 verdicts in CF (11 fixed, 12 broken)**,
   17 in CM and 8 in ID. It changed the **box on 1806/2000 boxed fakes** (mean dIoU +0.027). GRPO here is a box optimiser.
3. **The RL pool is the SFT training pool.** `mfp_v2_rl_crossforgery_strict.jsonl` is byte-identical (md5
   `19b3d432…`) to `mfp_v2_train_crossforgery_strict.jsonl`. The CM and ID RL pools are set-identical to their SFT
   train sets (`audit_protocol.py:215`: *"no separate GSPO carve-out exists yet; reuse train pool"*). All 400 images the
   strict GRPO actually consumed were SFT-trained images, which is why rollout accuracy is about 98%. The paper says
   "50K SFT / 10K alignment / 30K held out". **That is not what the code did. This blocks honest reporting.**
4. **KL was never applied.** `vlm_grpo_patch.compute_loss` has no KL term and forces `ref_per_token_logps=None`.
   Runs saved with `beta=0.001` (`grpo_v2_classdom_beta001_8b`) and `beta=0.03` (`klonly`, `mixdomain_beta`) trained
   with no KL; 0 `kl` log entries. The paper reports `beta ∈ {0, 1e-3, 3e-2}` and a "beta=0.001" table row, and its
   Eq. has clipping and 1/|y_i| normalisation that the code does not implement. Even if KL were switched on, trl's PEFT
   reference is `disable_adapter()`, i.e. the **base** model, not the SFT policy.
5. **Two of the four reward terms are constants.** On 3000 outputs per checkpoint, the evidence bonus fires on **100%**
   (every completion contains the template `<evidence cue="boundary_discontinuity">`), the format bonus fires on
   **100%** (2999/3000 for CM), and the contradiction penalty fires on **0%**. So the paper's statement that removing
   R_evid hurts accuracy most is attributing seed noise to a no-op term.
6. **The "reasoning" is a 2-4-way template.** After masking box digits, organ word and disease word, every checkpoint
   produces **2-4 distinct completions over 3000 images**. The verdict is determined 100% by the first two tokens of the
   `A1:` answer (token positions 39-40). The box is 12 digit tokens. So each ~212-token completion has 2 decision
   tokens + 12 box tokens; the rest is deterministic. This matters for (d), (f) and for an order-of-magnitude rollout speed-up.
7. **Measured cost.** GRPO takes **12.2-13.8 s/step** uncontended (8 rollouts of a 1267-token prompt + about 245
   completion tokens), roughly **261-295 images/hour**. SFT runs at **1.52-1.97 samples/s**. Memory was **not measured**
   (nothing on disk); an analytic estimate puts the current run at about 25-30 GB of 80 GB.
8. **Multi-prompt steps already work with the existing patch.** A CPU dry run with a tiny random Qwen3-VL + LoRA +
   `vlm_grpo_patch` + trl 0.16.1, using 2 prompts x 4 rollouts per step with different image sizes, trained and updated
   28/28 LoRA tensors. Batched vs unbatched log-probs under left padding differ by at most **9.5e-7**.
9. **Data constraints for (c)/(e).** Counterfactual pairs are plentiful: 15,470/15,470 RL fakes have a source path,
   15,149 of those exist on disk, 3,096 distinct sources. But the split is image-level, not source-level:
   **511 RL fakes are forgeries of shared-test real images**, and 1,037/2,000 test fakes have their source in CF training.
   The raw difference map |forged − source| is **not** a usable standalone target (median IoU 0.09-0.11 vs the GT
   box). Restricted to the dilated GT box, it gives a tighter change region in 91-94% of fakes.

---

## 1. How the current GRPO works (code walk)

Files: `~/CIRCE/circe/train_grpo.py`, `vlm_grpo_patch.py`, `circe_grpo_reward.py`, `circe_reward.py`,
trl `~/.conda/envs/Qwen3.5/lib/python3.10/site-packages/trl/trainer/grpo_trainer.py` (0.16.1), transformers 5.8.1.

### 1.1 Dataset rows (`train_grpo._rows_from_file`, l.38-54)
* One row per SFT record: `prompt` = [system(SYS), user(image placeholder + USER1)], `image` = path,
  `gt_verdict` = `"deepfake" if "deepfake" in <all assistant text>`, `gt_box` = first `x1="…" y1=… x2=… y2=…`
  in the assistant text (pixel coordinates; see 2.6).
* Verified on the strict CF pool: 31,356 rows (15,886 real / 15,470 remove), label vs path mismatches = 0,
  boxes on 15,470/15,470 fakes, 0 boxes on reals. `--limit 6400` = `rows[:6400]`.

### 1.2 Group / batch formation (trl `_get_train_sampler`, l.514-553; `RepeatRandomSampler`, l.75-160)
* `per_device_train_batch_size = num_generations = 8`, `gradient_accumulation_steps = 1`, one GPU, so
  `effective_batch_size // G = 8 // 8 = 1` distinct prompt per step. The sampler yields that index 8 times
  consecutively (`mini_repeat_count=G`).
* `torch.randperm(6400, generator(seed=42))` reproduces the consumed order (assuming no sampler wrapping by
  accelerate). The strict run used **400 of 31,356 rows (1.28%)**: 197 remove + 203 real, organs 155 CXR / 134 fundus /
  111 MRI, and 0 implants (Cross-Forgery by design). 400/400 of those images are in the base CF SFT data.
* `_prepare_inputs` (l.643-655): since `num_iterations == 1`, it calls `_generate_and_score_completions` every step.
  No reuse and no buffering in effect.

### 1.3 How the VLM patch threads images (`vlm_grpo_patch.py`)
* `_generate_and_score_completions` (patched, l.148-260) replaces trl's text-only version:
  1. `apply_chat_template(_clean_msgs(prompt), add_generation_prompt=True)` per row. `_clean_msgs` keeps a bare
     `{"type":"image"}` placeholder.
  2. Each row's `image` path is opened again (8 times for the same image) and passed as nested `[[img], …]` to the
     processor. The outputs are `input_ids` (1267 prompt tokens, 1024 of them `<|image_pad|>`), `pixel_values`
     (8x4096 patches x 1536), `image_grid_thw` (8,3) and `mm_token_type_ids`. Measured: the loss forward shapes in the
     strict log are `input_ids=(8,1512) pixel_values=(32768,1536) image_grid_thw=(8,3)`.
  3. `unwrapped_model.generate(prompt_ids, attention_mask, generation_config=self.generation_config, **vis)`.
  4. The EOS mask uses `processing_class.eos_token_id` (151645 only).
  5. Rewards are computed (step 1.4), then advantages, metrics, and `out.update(vis)` so that
     `pixel_values / image_grid_thw / mm_token_type_ids` reach the loss.
* `compute_loss` (patched, l.263-301): it zero-extends `mm_token_type_ids` over the completion, calls
  `_get_per_token_logps(model, ids, mask, logits_to_keep, **vis)` (logits/temperature, `selective_log_softmax`), then
  `per_token_loss = -exp(lp - lp.detach()) * A`, and `loss = Σ(per_token_loss·mask) / Σmask`. That is token-level
  normalisation over the whole batch. There is **no clipping, no KL, and no clip_ratio/kl metrics**.
* `apply_vlm_grpo_patch` also overrides `G.split_tensor_dict` / `G.shuffle_tensor_dict`. Those functions **do not
  exist in trl 0.16.1**, so both overrides are inert (dead code today; reusable for micro-batching, see 4b).

### 1.4 Where rewards become advantages (patched copy of trl l.816-839)
```
rewards_per_func[:, i] = reward_func(prompts, completions, gt_verdict=…, gt_box=…, image=…)   # (B, n_funcs)
rewards  = (rewards_per_func * reward_weights).nansum(1)                                         # (B,)
mean_g   = rewards.view(-1, G).mean(1).repeat_interleave(G)
std_g    = rewards.view(-1, G).std(1).repeat_interleave(G)       # unbiased (n-1)
A        = rewards - mean_g
A        = A / (std_g + 1e-4)          # scale_rewards defaults to True in GRPOConfig 0.16.1 and is never overridden
```
`reward_classification_dominant`: correct = +4 + 0.25·IoU (fakes only) + 0.2 evidence + 0.2 format; wrong = −4;
contradiction −1. **On real outputs the evidence/format/contradiction parts are constants** (section 3, B3), so in
practice reward = ±4 (+0.4 if correct) + 0.25·IoU.
Std of a verdict-split group of 8 with +4.4/−4 rewards is 2.97 (7:1) to 4.49 (4:4). An all-correct real group has
std **exactly 0**. An all-correct fake group has std of order 0.01-0.05 (IoU spread), which `/(std+1e-4)` turns into
unit-scale advantages.

### 1.5 What is logged (every `logging_steps=5`, averaged over the window by `GRPOTrainer.log`)
`loss, grad_norm` (from the **last** step of the window), `learning_rate, num_tokens, completion_length` (appended
twice per step, same value), `rewards/<fn>, reward, reward_std` (mean of per-group std), `epoch`, and
`train_runtime` at the end. Also one-off prints of the first 4 completions and the loss tensor shapes.
**Not logged:** per-group statistics, verdict accuracy of rollouts, IoU, kl, clip_ratio, zero-std fraction.
`log_completions` was off in the strict PBS job.

### 1.6 Generation config actually used
trl builds `GenerationConfig(max_new_tokens=640, do_sample=True, temperature=0.7, top_p=0.95, top_k=50,
eos_token_id=processing_class.eos_token_id=151645, pad_token_id=151643)`. In transformers 5.8.1,
`_prepare_generation_config` fills only `None` fields from `model.generation_config`. So `gc.eos_token_id = [151645,
151643]` in `train_grpo.py` l.188-191 has **no effect** on GRPO rollouts; the line that matters is
`proc.eos_token_id = 151645`. The model's own defaults (top_k 20, top_p 0.8) are overridden. Eval is greedy
(`do_sample=False`, `max_new=448`).

---

## 2. Measurements from existing logs and artifacts

### 2.1 GRPO wall-clock (8B, LoRA r32, 1 prompt/step)
| run | G | steps | mean s/step | 1st/2nd half | train_runtime |
|---|---|---|---|---|---|
| grpo_v2_classdom_beta0_8b | 8 | 400 | 13.79 | 14.04 / 13.54 | 5517 s |
| grpo_v2_classdom_beta001_8b | 8 | 400 | 13.76 | 14.07 / 13.46 | 5507 s |
| grpo_v2_classdom_beta0_extended_8b | 8 | 400 | 13.03 | – | 5216 s |
| grpo_v2_classdom_lowlr_8b | 8 | 600 | 12.98 | – | 7793 s |
| grpo_ablate_{evid,fmt,loc} | 8 | 400 | 12.56 / 12.30 / 13.13 | – | 5024 / 4923 / 5257 s |
| grpo_indomain_beta0_8b / _replay_8b | 8 | 400 | 12.88 / 12.18 | – | 5154 / 4875 s |
| grpo_crossmodel_beta0_8b_v2 / _replay_8b | 8 | 400 | 18.27 / 17.34 | – | 7311 / 6937 s |
| **strict CF (pbs strict_grpo)** | 8 | 400 | **23.74** | **14.5 / 33.0** (contention) | 9502 s |
| guarded / klonly / mix runs | 4 | 200 | 10.19-11.06 | – | 2040-2214 s |

* Tokens per step: 12,022 (classdom_beta0: 4.809e6/400) and 11,797 (strict). Completion length window means are
  170-246 tokens, window max 246.1. **No run with the current SFT format came near `max_completion`** (640/2048).
* Uncontended throughput: **261-295 images/hour** (one image per step).
* Rough split, an inference rather than a measurement: greedy eval generation of 8 images at batch 8 runs at
  **0.58-0.83 samples/s** (6 eval logs), i.e. 9.6-13.8 s per 8 generations. SFT forward+backward costs about
  0.51-0.66 s per ~1500-token sequence (2.2). So in a 12-14 s step, generation is most of the cost and the backward on
  8 sequences is about 4-5 s. A per-phase timer is part of spec (a).
* G=4 to G=8 adds only about 2-3 s/step (10.2-11.1 vs 12.2-13.8; different runs, indicative only), so extra rollouts
  are cheap to generate. The backward pass scales linearly.

### 2.2 SFT throughput (bs 1 x grad-accum 8, gradient checkpointing, LoRA r32)
| log | records | optimizer steps | wall | samples/s |
|---|---|---|---|---|
| train_mfp_crossforgery | 34,839 | 4,355 | 5:58:00 | 1.62 |
| train_mfp_indomain | 52,766 | 6,596 | 8:58:23 | 1.63 |
| train_mfp_crossmodel (shared gpu02 with indomain) | 38,534 | 4,817 | 11:41:21 | 0.92 |
| train_full90k_8b | 80,225 | – | 40,830 s | 1.965 |
| train_crossmodel_replay_sft | 77,068 | 9,634 | 49,660 s | 1.552 |
| train_indomain_replay_sft | 100,046 | 12,506 | 65,620 s | 1.525 |
| mfp_crossforgery_implant_replay_sft_8b | 4,000 used | 500 | 2,195 s | 1.822 |
| strict_replay_sft (pbs) | 4,000 used | 500 | 3,294 s | 1.21 |

One SFT record = 1,503 tokens (1,267 prompt incl. 1,024 image tokens), 199 supervised tokens covering both
assistant turns. Measured with the real collator on a strict-pool record; not truncated at 4096.

### 2.3 Memory
**Not measured**: no `nvidia-smi`, `max_memory_allocated` or OOM line exists in any GRPO/SFT log. Analytic
estimate for the current config:
base weights 17.53 GB (safetensors `total_size` 17,534,247,392 B) + LoRA 87.3 M fp32 params (0.35 GB) + grads
(0.35 GB) + Adam (0.70 GB; `optimizer.pt` is 698,777,210 B) + generation KV
(2·36 layers·8 kv-heads·128·2 B = **144 KiB/token**, x 1,513 tokens x 8 = 1.8 GB) + training activations with
gradient checkpointing (36·4096·2 B·1,513 = 446 MB/seq, about 3.6 GB for 8) + completion logits (247 x 151,936 x bf16,
plus copies, about 1.2 GB) comes to **about 25-30 GB**. The H100 is mostly idle.

### 2.4 Group statistics reconstructed from the 5-step logging windows
Method (`group_bounds.py`): a verdict-split group has std in [2.97, 4.49]; a non-split group has std ≤ ~0.5.
Bounds on split steps come from 5·reward_std per window. The zero-std fraction is estimated from logged
`grad_norm == 0`, which is exact for the window's last step, so it is an unbiased 1-in-5 sample.

| run | verdict-split steps | zero-std steps (est.) | loc/other-only steps (est.) | implied rollout acc |
|---|---|---|---|---|
| strict CF | 9-10 / 400 (2.5%) | 45% | 52.5% | 0.98 |
| classdom_beta0 | 24-27 / 400 (6.8%) | 8.7% | 84.5% | 0.97 |
| classdom_beta001 | 24-25 (6.2%) | 11.3% | 82.5% | 0.96 |
| classdom_beta0_extended | 19-20 (5.0%) | 12.5% | 82.5% | 0.98 |
| classdom_lowlr (600) | 54-56 (9.3%) | 10.8% | 79.8% | 0.97 |
| crossmodel_beta0_v2 | 10 (2.5%) | 52.5% | 45% | 0.99 |
| crossmodel_replay | **1 (0.3%)** | 50% | 49.7% | 1.00 |
| indomain_beta0 / replay | 17 (4.3%) / 5 (1.3%) | 43.8% / 38.8% | 52% / 60% | 0.97 / 1.00 |
| **ablate_loc** (no IoU bonus) | 20-22 (5.5%) | **92.5%** | 2% | 0.97 |

Strict run reward_std: mean 0.108, median 0.0148; 88.7% of windows < 0.05. classdom_beta0 mean 0.271 (this is the
"0.27" quoted in the task). A 7:1 split has std 2.97, not 2.6.
Grad norm in loc-only windows has median 0.19-0.29, against 1.14 in ablate_loc windows that only contain verdict
splits. So each loc-only step pushes about 1/4-1/5 as hard as a verdict step, and there are 10-30x more of them.
Proxy check on real text, using 7 CF checkpoints as pseudo-rollouts on the same 3000 images
(`test_grpo_v3_reference.py`): zero-std 49.1%, verdict-split 1.7%, loc-only 49.2%; **3.2% of |advantage| mass comes
from the verdict term and 96.6% from the IoU bonus.**

### 2.5 What GRPO changed (clean selection split, n=3000 per regime)
| regime | verdict flips SFT to GRPO | fixed / broken | boxed fakes with a changed box | mean dIoU |
|---|---|---|---|---|
| CF | 23 | 11 / 12 | 1806 / 2000 | +0.0265 |
| CM | 17 | 10 / 7 | 1789 / 2000 | +0.0188 |
| ID | 8 | 3 / 5 | 1931 / 2000 | +0.0195 |

### 2.6 Output structure (saved greedy generations; `gen_structure.py`, `branch_tokens.py`)
* Distinct completions after masking box digits, organ word and disease word: strict CF GRPO test **3**, CF s1replay test
  **2**, CF sel SFT/GRPO **3/3**, CM GRPO **4**, ID GRPO **3**.
* Verdict vs first `A1:` answer: real-branch to real 1149/1149, remove-branch to deepfake 1845/1845 (strict CF test).
  CM has an implant branch: implant 565 deepfake, remove 1278 deepfake (one unparsed), real 1156 real.
  **The verdict is a deterministic function of the A1 branch.**
* Token positions (Qwen3 tokenizer): A1 branch tokens at index 39-40 (`" No"," abnormal"` / `" No"," sharp"` /
  `" Yes"," —"`). The box is **12 value-digit tokens** (digits are tokenised one per token; the `1`/`2` of `x1`/`y2` are
  separate tokens). The verdict line is 7 tokens. Totals: 168 (real), 223 (implant) and 244 (remove) tokens.
* All 2994/3000 outputs end with `<|im_end|>`. Median 245 tokens, max 246. No truncation.
* The CF model never emits the implant branch: test implants are routed to the remove branch (850/1000) or to real (150).
* GT boxes use **pixel coordinates** of the 1024x1024 images: IoU 0.990 vs the mask bbox in pixels, 0.798 vs the 0-1000
  normalised bbox, n=136. All 236 sampled fakes are 1024x1024.

---

## 3. Bugs and train/eval mismatches (ranked)

| # | severity | finding | evidence | fix |
|---|---|---|---|---|
| B1 | blocking | **RL pool = SFT train pool.** No held-out alignment split. | md5 `19b3d432a4c357995e77634a12374bc4` is shared by `mfp_v2_rl_crossforgery_strict.jsonl` and `mfp_v2_train_crossforgery_strict.jsonl`. CM (34,681) and ID (47,490) RL pools are set-identical to their train manifests. `audit_protocol.py:215` comment. 400/400 consumed images were SFT-trained. Paper: "50K SFT / 10K alignment / 30K held out". | Build the RL pool from never-used images (39,227 exist per the project notes; must exclude test, selection and val), or correct the paper. Re-run RL on held-out images. |
| B2 | blocking (paper) | **KL never applied**; PEFT reference would be the base model, not SFT. | Patched `compute_loss` has no KL term and `ref_per_token_logps=None`. `training_args.bin`: beta=0.001 (classdom_beta001), beta=0.03 (klonly, mixdomain_beta). `'kl'` occurs 0 times in those logs. Paper Eq. (clip, 1/|y_i|, −beta·KL) and "beta ∈ {0,1e-3,3e-2}". | Either report beta=0 everywhere and fix the Eq. (token-level mean, no clip because ratio ≡ 1), or implement KL against a frozen copy of the **SFT** adapter (spec 4.7). Reviewer M1 (KL sign) gets a correct answer either way. |
| B3 | major | **Evidence and format bonuses are constants; contradiction penalty never fires.** | `bonus_constancy.py`: grounded=True 2994/2994, 3000/3000, 3000/3000, 3000/3000. Format True 100% (one exception in 3000). Contradiction 0/3000 everywhere. Root cause: the SFT template always writes `<evidence cue="boundary_discontinuity">` (in `FORENSIC_CUES`) and "consistent". | Drop the per-term ablation claims for R_evid/R_fmt (they measure seed noise), or redefine evidence as a verifiable signal (e.g. the diff-map term in 4e). |
| B4 | major | **`scale_rewards=True` inflates IoU noise to unit advantage**; the verdict signal is a few percent of the update. | Section 2.4 (52-85% loc-only steps; ablate_loc gives 92.5% zero-grad) and 2.5 (23 verdict flips vs 1806 changed boxes). | Dr.GRPO (`scale_rewards=False`) or decoupled advantages (4d). |
| B5 | major | **1 prompt per step** and only 400 steps: 400 images (1.3% of the pool), 2.5% of them verdict-informative. | `training_args`: per_device 8 = G 8, GA 1; sampler reproduction. | Spec 4b and 4f. |
| B6 | minor | EOS: rollouts stop only at `<|im_end|>`; the `<|endoftext|>` EOS in `model.generation_config` is ignored. If 151643 were emitted, the completion mask would extend over pad tokens (pad = 151643). | trl builds GenerationConfig with eos=processing_class.eos_token_id; transformers 5.8.1 `_prepare_generation_config` fills only None fields. All logged completions `eos=True`, so not triggered. | Pass `eos_token_id=[151645,151643]` into trl's generation config and build the EOS mask from the same list. |
| B7 | minor | Silent-assumption guards missing: `num_iterations>1` would silently reuse ratio=1 (old logps never computed); `max_prompt_length` other than None would left-truncate image tokens (trl default is 512 < 1267). | Patch l.193-195; `train_grpo` passes None. | Add asserts in the patch. |
| B8 | minor | Rollout sampling (T 0.7, top_p 0.95, **top_k 50**) differs from the model defaults (top_k 20, top_p 0.8) and from greedy eval. | trl GRPOConfig defaults, `generation_config.json`. | Document it. Consider T=1.0 on a hard pool to increase verdict exploration. |
| B9 | minor | Unparsed verdict: eval maps None to "real", GRPO gives −4. | `eval_medforge.py` l.161. Parse rate 1.0 (one exception in 3000). | Harmless now; log it. |
| B10 | info | Prompt parity verified, **no mismatch**: GRPO == eval == SFT prefix, 1,267 identical token ids. No completion truncation (max 246 < 448 < 640). | `prompt_parity.py`, `gen_structure.py`. | none |
| B11 | info | Efficiency: the same image is decoded and processed 8 times per step, and the vision tower runs on 8 copies. Inert split/shuffle overrides. | Patch l.154, l.308-311. | Dedupe per group (small gain). |
| B12 | protocol | Image-level, not source-level split: 1,037/2,000 test fakes have their authentic source in CF training; 4,490 training fakes are siblings (same source) of test fakes; 511 RL fakes are forgeries of test reals (503 for the selection split). | `pair_leakage.py`. | Report to the data audit. Pairs and diff maps (4c/4e) must exclude any source in test/select/val. Consider a source-disjoint split. |

---

## 4. Implementation spec

All items go into **one subclass** `GRPOTrainerV3(GRPOTrainer)` in a new file
(`circe/grpo_v3_trainer.py`), plus a replacement patch module. `vlm_grpo_patch.py` stays untouched for
reproducibility. The pure pieces are already written and unit-tested in
`~/CIRCE/v3_audit/code_scripts/grpo_v3_reference.py` (section 5).

Methods to override (trl 0.16.1 / transformers 5.8.1):

| method | why |
|---|---|
| `GRPOTrainer._get_train_sampler(self, dataset=None)` | paired sampler (c) and hard-pool sampler (f); keep the dataset=None shim |
| `GRPOTrainer._prepare_inputs(self, inputs)` | dynamic-sampling refill loop (f) |
| `GRPOTrainer._generate_and_score_completions(self, inputs)` | split into `_generate()` and `_score()`; diagnostics (a), pair advantages (c), token masks and decoupled advantages (d), diff reward (e) |
| `GRPOTrainer.compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None)` | per-token advantage tensor, global token normaliser, optional KL, extra metrics |
| `Trainer.training_step(self, model, inputs, num_items_in_batch=None)` | generate once for k·G rollouts, then micro-batched backward (b) |
| reuse `vlm_grpo_patch._split_tensor_dict_vlm` | chunking that understands `pixel_values` via `image_grid_thw` |
| new `_concat_tensor_dict_vlm` | refill batches (f): left-pad prompts, right-pad completions, concatenate `pixel_values` / `image_grid_thw` in order |

New `GRPOConfig`-side flags (plain attributes on the trainer): `v3_prompts_per_step`, `v3_train_microbatch`,
`v3_adv_mode ∈ {trl, drgrpo, decoupled}`, `v3_pair_mode ∈ {off, independent, coupled, joint}`, `v3_lam_box`,
`v3_dapo ∈ {off, any_variance, verdict_split}`, `v3_oversample`, `v3_max_refill_rounds`, `v3_diag_jsonl`,
`v3_ref_adapter`, `v3_beta`.

### 4a. Per-group diagnostics
**Purpose:** turn the "advantage collapse" hypothesis into per-step facts, and make it a figure for the paper.
**Where:** at the end of `_score()`, right after `rewards` (patch l.223) is computed. This is the point where the
reward per function, the group view and the advantages all exist.
```python
comps = [classdom_components(t, gv, gb, parse_output, iou, _cue_of, FORENSIC_CUES)      # reference impl.
         for t, gv, gb in zip(completions_text, gt_verdict, gt_box)]                       # sum == reward (tested)
per_group, summ, A_check = group_diagnostics(comps, G, scale_rewards=self.args.scale_rewards)
assert torch.allclose(A_check.float().view(-1), advantages_full.cpu(), atol=1e-4)          # only when adv_mode=='trl'
for k, v in summ.items(): self._metrics[mode][f"diag/{k}"].append(v)
self._metrics[mode]["diag/rollout_acc"].append(mean(c["correct"] for c in comps))
self._metrics[mode]["diag/rollout_iou_fakes"].append(mean(c["iou"] for c, gb in zip(comps, gt_box) if gb))
self._metrics[mode]["time/generate_s"].append(t_gen); self._metrics[mode]["time/score_s"].append(t_score)
if self.v3_diag_jsonl and self.accelerator.is_main_process:     # exact per-group record (~1 KB/group)
    for g, pg in enumerate(per_group):
        sl = slice(g*G, (g+1)*G)
        write_jsonl(dict(step=self.state.global_step, image=inputs[g*G]["image"], gt=gt_verdict[g*G],
                         rewards=rewards[sl].tolist(), correct=[c["correct"] for c in comps[sl]],
                         iou=[c["iou"] for c in comps[sl]], adv=advantages_full[sl].tolist(),
                         ntok=completion_mask[sl].sum(1).tolist(), **pg))
```
Logged keys: `diag/frac_zero_std, frac_verdict_split, frac_all_correct, frac_all_wrong, frac_groups_loc_only,
adv_share_{verdict,loc,const,contra}, rollout_acc, rollout_iou_fakes, time/generate_s, time/score_s,
time/backward_s` (the last one is timed in `training_step`). Memory and time: negligible (CPU regex on 8-64
strings; about 1 KB per group of JSONL). Also log `torch.cuda.max_memory_allocated()` each logging step, which
fills the "memory: not measured" gap.
**Risks:** the decomposition is exact only for `reward_classification_dominant`. Other reward sets need their own
component split. `grad_norm` remains last-step-only, so use the JSONL for exact counts.
**CPU test (done):** `test_grpo_v3_reference.py [a]`. The decomposition equals `reward_classification_dominant` on
3000 real completions (0 mismatches), the advantage equals the trl formula, and the proxy groups give the numbers in
2.4. Next step: call it inside `cpu_dryrun_tiny_qwen3vl.py` and check that the `diag/*` keys appear in the log dicts.

### 4b. Multi-prompt steps (k prompts x G rollouts per optimizer step)
**Level 1 (config only, verified on CPU):** `per_device_train_batch_size = k·G`, `GA = 1`. The current patch handles
it: the CPU dry run with 2 prompts x 4 rollouts and two image sizes trained 2 steps, updated 28/28 LoRA tensors,
and showed batched vs single log-prob difference ≤ 9.5e-7 under left padding (prompt lengths 307 vs 323). The
limit is memory: the loss forward runs on all k·G sequences at once, so **k ≤ 4 at G=8** (estimate: 32 seqs means
about 14 GB activations + 5 GB logits + 7 GB generation KV on top of about 19 GB fixed).

**Level 2 (decoupled generation batch and training micro-batch):** override `training_step`:
```python
def training_step(self, model, inputs, num_items_in_batch=None):
    model.train()
    t0 = time.time(); batch = self._prepare_inputs(inputs)   # ONE generate() for k*G rollouts (+ (f) refills)
    keep = batch.pop("_keep_groups", None)                   # from (f); drop zero-adv groups BEFORE backward
    if keep is not None: batch = select_groups_vlm(batch, keep, self.num_generations)
    tot_tok = batch["completion_mask"].sum().clamp(min=1)
    n_chunks = max(1, batch["completion_ids"].size(0) // self.v3_train_microbatch)   # chunk at group boundaries
    loss_sum = 0.0
    for ch in _split_tensor_dict_vlm(batch, n_chunks):
        with self.compute_loss_context_manager():
            l = self.compute_loss(model, ch, token_normaliser=tot_tok)   # Σ(loss·mask)/tot_tok -> exact global mean
        self.accelerator.backward(l); loss_sum += l.detach()
    self._metrics["train"]["time/backward_s"].append(time.time() - t0)
    return loss_sum
```
(`GA=1`, so `Trainer` does not rescale the loss. `compute_loss` takes an optional `token_normaliser`. The
`_split_tensor_dict_vlm` already in the patch slices `pixel_values` by `image_grid_thw` patch counts.)
**Level 3 (the big wall-clock lever, "decision-point rollouts"):** the policy is deterministic except at the A1
branch (2 tokens) and the box digits (12 tokens) (section 2.6). A custom `_generate()` can:
(i) prefill prompt + fixed prefix up to `A1:`; (ii) sample 2 tokens; (iii) append the branch's fixed span in one
cached forward; (iv) sample the box digits under a digit-only logits mask; (v) append the verdict line +
`<|im_end|>`. That is about 14-16 decode steps instead of about 210-245, so generation should get **5-10x cheaper
(estimate)**. Forced tokens get `completion_mask = 0` in the loss (they were not sampled from the policy). Log the
minimum policy probability of forced tokens and fall back to free sampling if it drops below 0.9.

**Throughput and memory model.** Measured inputs: 12.2-13.8 s/step today; 0.51-0.66 s fwd+bwd per sequence (from
SFT); KV cache 144 KiB/token. Everything else in this table is an **estimate** and must be confirmed by a 20-step GPU
smoke run with the (a) timers.

| config | gen rollouts/step | train seqs/step | est. s/step | images/hour | x now | est. peak GB |
|---|---|---|---|---|---|---|
| now (k=1, G=8) | 8 | 8 | 12.2-13.8 (meas.) | 261-295 | 1 | ~25-30 |
| L1: k=4, G=8 | 32 | 32 | 28-35 | 410-510 | ~1.6 | ~45-50 |
| L2: k=8, G=8, micro-batch 8 | 64 | 64 | 45-55 | 520-640 | ~2 | ~35 (KV 14 GB) |
| L2 + (f) drop zero-std groups (45% in strict run) | 64 | ~35 | 30-38 | 760-960 | ~3 | ~35 |
| L3 decision-point gen + L2 + (f), k=16 | 128 | ~40-60 | 25-40 | 1,400-2,300 | ~5-8 | ~50 (KV 28 GB) |
| + teacher-forced screening of the pool (4f) | prefill only | – | ~0.1-0.2 s/image scored | 18k-36k scored/h | screening ≥ 50x | ~25 |

"10-50x more images in the same wall-clock" is reachable only for images **scored** (screening) or with L3. Images
**trained** on rise about 2-8x. More importantly, the fraction of trained groups that carry a verdict signal can rise
from 2.5% to most of the batch. That is the quantity that matters.
**Risks:** fewer optimizer steps per image (keep total images fixed when comparing; retune LR to about 2e-6 if k=8);
class balance per step (draw k/2 reals + k/2 fakes); different image sizes for MF++ kidney/colon (parity verified on
CPU); HF generate memory at 128 rollouts (generate in chunks of 64); L3 couples the trainer to the output template,
so the mask builder must support both the legacy and the v2 format.
**CPU test:** `cpu_dryrun_tiny_qwen3vl.py` (done for L1). For L2: same harness with
`per_device=16, v3_train_microbatch=4`; assert the summed gradient equals the unchunked gradient (compare LoRA grads
after one backward, atol 1e-6 in fp32). For L3: with a tiny model and a toy template, assert that forced and free
sampling give identical token sequences when the template tokens are given probability 1 (inject a logits bias).

### 4c. Counterfactual-paired groups (forged image + its authentic source in the same step)
**Data (measured):** the strict CF RL pool has 15,470 fakes, all with `real_image_path`; 15,149 sources exist on disk;
3,096 distinct sources (mean 5.0, max 13 fakes per source); 1,460 sources are also RL-pool reals. **Exclusions are
mandatory:** 511 RL fakes have their source in `mfp_v2_test_shared.jsonl` and 503 in the clean selection split. Also
cap at 1-2 pairs per source per epoch.
**Dataset rows:** add `pair_id, role ∈ {forged, source}, src_image, change_map_path`. The source row is a normal real
row (gt_verdict=real, gt_box=None). Both images are 1024x1024, so prompts have identical length and need no padding.
**Sampler** (override `_get_train_sampler`): `PairedRepeatSampler(pairs, G, pairs_per_step=m)` yields
`[f]*G + [s]*G` for each pair, contiguously, with `per_device_train_batch_size = 2·G·m`. trl's group contiguity is
preserved, so `rewards.view(-1, G)` still means one group per image.
**Advantage modes** (in `_score()`, replacing patch l.224-230):
```python
if pair_mode == "independent": A = groupnorm(rewards, G)                       # control
elif pair_mode == "coupled":   A = pair_coupled_advantages(rewards, G)          # baseline/scale over 2G (reference impl.)
elif pair_mode == "joint":     # expected joint reward: forged rollout i vs ALL G source rollouts, and vice versa
    r_f, r_s = pair_joint_rewards(correct_f, correct_s, loc_f, G)               # (reference impl.)
    A = groupnorm(interleave(r_f, r_s), G)
```
Why it helps (tested on synthetic data): a pair whose forged group is all correct while its source group is all
wrong has **zero** advantage under independent groups, but non-zero under coupled normalisation (|A| sum = 7.48 in the
unit test). The "always say deepfake" bias then gets a gradient. The joint reward pays a forged rollout only as much as
the source is also classified correctly, i.e. it rewards *discrimination* rather than a prior. Inference stays
single-image, so there is no train/test mismatch.
**Memory/time:** one pair is equivalent to k=2 (16 sequences at G=8). Estimate: about 17-20 s/step at L1, about 25 GB
peak. Use L2 for m ≥ 2.
**Risks:** (1) leakage via sources (above); (2) a source shared by up to 13 fakes means repeated authentic images,
so overfitting to those reals (cap per epoch); (3) coupled normalisation mixes two images' difficulties (Dr.GRPO-style
"no std" is safer: try `scale=False`); (4) 321 fakes with missing sources fall back to unpaired; (5) the CF regime
has no implants in RL, so pairing can only teach implant transfer indirectly. That is a hypothesis to test on the CF
implant column, not a claim.
**CPU test:** done for the math (`[c]`). Sampler test: iterate 100 batches and assert each 2G block is
(forged x G, its source x G), no pair is split across steps, and no source ∈ test ∪ select ∪ val. Then a tiny-model
dry run with `pair_mode=coupled`.

### 4d. Decoupled advantages for verdict tokens vs box tokens
**Why (measured):** each completion has 2 decision tokens + 12 box digits; the IoU bonus drives 52-85% of steps; and
a single scalar advantage per sequence ties "good box" to "correct verdict".
**Rewards:** `r_v = ±4` (or ±1); `r_b = IoU(pred, GT)` (optionally the diff-mass reward from 4e);
`valid = correct verdict AND GT box`.
**Advantages** (reference `decoupled_token_advantages`): `A_v = r_v − mean_G(r_v)` (Dr.GRPO, no std);
`A_b = (IoU − mean_valid)/max(std_valid, 0.05)` over valid rollouts only, 0 otherwise.
Per token: `A_t = A_v·m_v + λ_b·A_b·m_b·valid + A_v·m_b·(1−valid) + A_v·m_rest`.
A box emitted with a wrong verdict therefore gets the negative verdict advantage, and a wrong verdict never earns
box credit. This keeps the classification gating the reviewers liked.
**Token masks:** in `_score()`, decode each completion token (`tok.decode([id])`) and locate the spans with
`token_masks_legacy` (A1 branch = first two words after `A1:`; verdict line; the four coordinate **value** digits).
Return `adv_tok (B,T)` in the output dict. In `compute_loss`, replace `advantages.unsqueeze(1)` with
`inputs["adv_tok"]`.
**Memory/time:** one extra (B,T) float tensor; CPU masking about 1 ms per completion. Negligible.
**Risks:** the token masks depend on the template. If masks are empty, fall back to the sequence-level `A_v`. Count
fallbacks in `diag/mask_fallback`. After a re-SFT on the v2 format, `<box>x y x y</box>` is already supported. Changing
the effective loss weight between verdict and box changes the LR sensitivity; sweep λ_b ∈ {0.25, 1}.
**CPU test (done):** on 3000 real completions, masks are disjoint 3000/3000 and the verdict mask is non-empty
3000/3000. The box mask equals the number of coordinate digits on 1843/1843 boxed outputs, and is empty on 1157/1157
unboxed ones (mean 9 verdict-mask tokens, 12 box tokens, 211.9 tokens/completion). λ_b=0 reduces exactly to Dr.GRPO
on non-box tokens, and wrong-verdict rollouts get no box credit.

### 4e. Difference-map localisation reward (training only)
**Measured feasibility** (236 stratified MF90K fakes, 12 per generator x type; `diffmap_feasibility.py`,
`diffmap_inbox.py`):
* A global thresholded |forged − source| is unusable as a target. Median IoU of the largest-diff-blob box vs the GT
  box is 0.109 (edit) / 0.086 (remove); only 11.7% / 14.7% exceed 0.5. Just 34% / 32% of strongly changed pixels
  (>25 grey levels) lie inside the GT box, and the mean absolute diff outside the box is about 5.3 grey levels
  (generators re-render the whole image; step1x-edit changes 21.6% of pixels by >25).
* Restricted to the GT box dilated by 25%, with the global median offset removed and a threshold at the 99th
  percentile of the outside-box diff, a change region exists in 94% / 91% of fakes. Its box covers a median 40% / 31%
  of the GT box area (IoU 0.37 / 0.31 with GT). The changed fraction of the GT box has median 0.20 / 0.22. Agreement
  is generator-dependent: median IoU 0.70 (qwen, seeddream), 0.54-0.56 (gemini, SDXL-inpaint), 0.14-0.19 (SD3.5-L/M,
  flux).
* Precompute cost: 2.19 s per pair single-process, measured on a login node at load ~95/64. That is about 35 min for
  15k pairs at 16 processes. 8/58 maps came out empty (falls back to IoU).

**Reward (reference `change_map`, `diffmass_reward`):** precompute `C` = a normalised change mass inside
dilate(GT box), stored as a float16 128x128 map or an integral image (about 0.5 GB for 15k fakes). Then
`r_loc' = w·IoU(pred, GT) + (1−w)·[mass_C(pred) − κ·max(0, area(pred)/area(GT) − 1)]`, gated on a correct verdict
and used as `r_b` in 4d. With w=0.5 the GT box remains optimal (synthetic test: GT box 1.00 > tight-edit box 0.73 >
whole image −2.63). So the reward cannot push the model below the GT box that evaluation uses. A "tight" variant adds
a precision term but would be expected to **lower** reported mIoU against GT boxes, since diff boxes cover only
31-40% of GT boxes.
**Risks:** generator misalignment (gate: use C only if the outside-box p99 < 40); sources in test/select (exclude;
same 511/503 fakes); the GT boxes are loose, so whether the diff region is "more correct" cannot be validated without
pixel masks (not available); an extra training-only target is a methods contribution only if it moves the CF implant
or MF++ columns.
**CPU test:** synthetic pasted square (done: 0.965 of the mass recovered). Real-data smoke on 20 pairs; assert that
maps are None-or-normalised and that rewards lie in [−κ·k, 1].

### 4f. DAPO-style dynamic sampling
**Online filter** (override `_prepare_inputs`, train mode):
```python
def _prepare_inputs(self, inputs):
    if self.control.should_evaluate: return self._generate_and_score_completions(inputs)
    kept, rounds = [], 0
    batch = self._generate_and_score_completions(inputs)                  # k*G rollouts
    while True:
        keep = dapo_keep_mask(batch["_rewards"], G, correct=batch["_correct"], mode=self.v3_dapo)
        kept.append(select_groups_vlm(batch, keep, G))
        if n_groups(kept) >= self.v3_min_groups or rounds >= self.v3_max_refill_rounds: break
        batch = self._generate_and_score_completions(next(self._hard_iter))   # refill from a hard pool
        rounds += 1
    out = _concat_tensor_dict_vlm(kept)[: self.v3_min_groups * G]        # left-pad prompts, right-pad completions
    self._metrics["train"]["dapo/kept_frac"].append(...); self._metrics["train"]["dapo/rounds"].append(rounds)
    return out
```
(Use `mode="verdict_split"` to train only on verdict-informative groups for the verdict term. Keep box-informative
groups for the 4d box term. If nothing survives after the maximum number of rounds, skip the optimizer step by
returning a zero-loss batch, and count it.)
**Offline pre-screening (cheap and high-value, enabled by section 2.6):** the completion prefix up to `A1:` is
deterministic (39 tokens). One teacher-forced forward per image gives p(" Yes"/" No") and p(" sharp"/" abnormal"),
i.e. the policy's exact verdict-branch probability, **without generation**. Score the never-used pool (not the SFT
pool, B1) and sample RL prompts with p_correct ∈ [0.02, 0.98], stratified by class and generator. Estimated cost: one
prefill of about 1,306 tokens, roughly 0.1-0.2 s/image on an H100, so about 1-2 GPU-hours per 30k images (estimate).
Re-screen every about 200 steps, because the pool drifts with the policy.
**Memory/time:** generation scales with the oversampling factor ρ (KV 144 KiB/token per rollout); backward runs only
on kept groups. Strict run: 45% of steps were zero-std and would have been skipped, but at 1 group/step there was
nothing to refill, which is why (f) needs (b).
**Risks:** focusing on label-noisy images (cap reuse per image; inspect the top-hard list; MF++ polyp/kidney have
known label/source issues); class imbalance among kept groups (balance quotas); conditioning on informative groups
changes the effective step size (zero-std groups contribute exactly zero gradient anyway, so the estimator direction
is unchanged); `epsilon_high` (clip-higher) is a no-op at num_iterations=1.
**CPU test:** `[f]` (done for the mask). Tiny-model harness with a reward that is constant for even-indexed prompts:
assert that only odd prompts survive, the refill loop terminates, the concatenated `pixel_values` row count equals
Σ`image_grid_thw.prod(1)`, and the loss matches an unfiltered run restricted to the same groups.

### 4.7 KL against the SFT policy (only if beta > 0 is kept in the paper)
Load the SFT adapter twice: `PeftModel.from_pretrained(base, sft, adapter_name="default", is_trainable=True)` and
`model.load_adapter(sft, adapter_name="ref", is_trainable=False)`. In `_score()`, under `no_grad`: `set_adapter("ref")`,
compute logps, then `set_adapter("default")`. Assert that the "ref" params have `requires_grad=False` and are absent
from `optimizer.param_groups` after **every** switch (peft's `set_adapter` toggles requires_grad). In `compute_loss`,
add `beta·(exp(ref−lp) − (ref−lp) − 1)` with the correct sign (reviewer M1). Cost: +0.35 GB; one extra no-grad forward
over k·G sequences, about +10-20% step time (estimate).
CPU test on the tiny model: at step 0, ref logps == policy logps (atol 1e-6); after 2 steps "ref" weights are
bit-identical; KL > 0.

---

## 5. What was built and verified on CPU (all under `~/CIRCE/v3_audit/code_scripts/`)
* `cpu_dryrun_tiny_qwen3vl.py`: tiny random Qwen3-VL (19.7 M params, 2+2 layers) + LoRA r4 + the **existing**
  `vlm_grpo_patch` + trl 0.16.1 GRPOTrainer on CPU. Two multi-prompt steps (2 prompts x 4 rollouts, mixed image
  sizes); reward called on 8 completions per step; 28/28 LoRA tensors updated; batched vs single log-probs within
  9.5e-7. This is the harness for every spec item's dry run.
* `grpo_v3_reference.py` + `test_grpo_v3_reference.py`: pure implementations of (a), (c), (d), (e) and (f);
  **all tests pass**.
* Measurement scripts: `parse_grpo_logs.py` (timing and windows), `group_bounds.py` (split/zero-std bounds),
  `rl_pool_stats.py` + `used_idx_strict_cf.txt` (consumed prompts), `prompt_parity.py`, `gen_structure.py`,
  `branch_tokens.py`, `bonus_constancy.py`, `pair_leakage.py`, `diffmap_feasibility.py`, `diffmap_inbox.py`
  (outputs in `out/`).

## 6. Recommendations (priority order, for the CVPR rebuild)
1. **Fix the protocol before any new RL run:** RL pool = never-used images (not the SFT pool), excluding test,
   selection, val and any image whose source is in those sets. Correct the paper's split description.
2. **Fix the paper/code mismatch on the GRPO objective:** state beta=0 (or implement 4.7), drop clipping and 1/|y_i|
   from the Eq. (or implement them), and remove the R_evid/R_fmt ablation claims.
3. Implement **(a)** first (half a day of work) and log memory and phase timers. Run a 20-step GPU smoke test to
   replace the estimates in 4b.
4. Then **(d)+(f)+(b-L2)** together: Dr.GRPO verdict advantage on decision tokens, a separate box advantage, k=8
   prompts/step, and a pre-screened hard pool. This combination directly targets reviewer B1 ("RL adds no gain"),
   with the diagnostics as evidence.
5. **(c) coupled pairs** and **(e) diff-mass** as the candidate novel components, ablated on the CF implant column and
   MF++. Check their novelty against 2025-26 literature (not done here).
6. Consider **decision-point rollouts (b-L3)**. They are the cheapest large speed-up, but they make explicit that the
   CoT is a template (reviewers A5/C6). Either own that framing (structured decision + localisation) or train
   non-template evidence text.

## 7. Reproduction
```
cd ~/CIRCE/v3_audit/code_scripts
nice ~/.conda/envs/aiface/bin/python parse_grpo_logs.py ~/AAAI/MedForge++/grpo_*.log ~/CIRCE/pbs_jobs/logs/strict_grpo.{err,out}
nice ~/.conda/envs/aiface/bin/python group_bounds.py ~/CIRCE/pbs_jobs/logs/strict_grpo.out ~/AAAI/MedForge++/grpo_v2_classdom_*.log ...
nice ~/.conda/envs/aiface/bin/python rl_pool_stats.py used_idx_strict_cf.txt
CUDA_VISIBLE_DEVICES= nice ~/.conda/envs/Qwen3.5/bin/python prompt_parity.py
nice ~/.conda/envs/aiface/bin/python gen_structure.py ~/AAAI/MedForge++/v3_select/gen_sel_cf_s2grpo.jsonl
nice ~/.conda/envs/aiface/bin/python bonus_constancy.py <gen files>
nice ~/.conda/envs/aiface/bin/python pair_leakage.py
nice -n 15 ~/.conda/envs/aiface/bin/python diffmap_feasibility.py 12 && nice -n 15 ~/.conda/envs/aiface/bin/python diffmap_inbox.py
nice ~/.conda/envs/Qwen3.5/bin/python cpu_dryrun_tiny_qwen3vl.py        # CPU only
nice ~/.conda/envs/aiface/bin/python test_grpo_v3_reference.py
md5sum ~/AAAI/MedForge++/mfp_v2_rl_crossforgery_strict.jsonl ~/AAAI/MedForge++/mfp_v2_train_crossforgery_strict.jsonl
```
