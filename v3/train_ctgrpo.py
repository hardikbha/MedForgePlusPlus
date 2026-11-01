"""
train_ctgrpo.py -- Counterfactual-Twin GRPO (CT-GRPO) for Qwen3-VL-8B + LoRA on MF-Clean (v3).

README
------
Built on trl 0.16.1 GRPOTrainer + the EXISTING ~/CIRCE/circe/vlm_grpo_patch.py (imported and applied unchanged; it
threads pixel_values/image_grid_thw through generation and the loss forward). Pure pieces live in ctgrpo_core.py.

What this file adds on top of the patch (class CTGRPOTrainer):
  * _get_train_sampler  -> ctgrpo_core.TwinPairSampler: one optimizer step = K twin pairs (--pairs_per_step) = 2K
                           images x G rollouts (--num_gen); pairs (not images) are shuffled with --seed; each image is
                           repeated G times consecutively; the fake's G rows are directly followed by its real twin's
                           G rows; singleton reals are coupled two per slot (each is its own group).
  * _generate_and_score_completions -> calls the patched method (generation + reward), then REPLACES
                           out["advantages"] with ctgrpo_core.compute_advantages(--adv_mode), using per-row group ids
                           built from the dataset columns pair_id/role (layout is checked, not assumed), and writes one
                           diagnostics record per optimizer step (--diag_jsonl, default <out>_diag.jsonl).
  * training_step       -> optional --loss_microbatch: generation once for all 2K*G rollouts, loss forward/backward
                           in row chunks, each chunk loss re-weighted by its token share, so the summed gradient equals
                           the one-shot gradient (same global token mean; checked in dryrun_ctgrpo.py).

The objective that is ACTUALLY optimised (patched compute_loss; nothing else is claimed):
    L = - (1/N) * sum_i sum_t m_it * exp(logp_it - stopgrad(logp_it)) * A_i ,   N = sum_it m_it over the whole step
  - ratio exp(logp - stopgrad(logp)) == 1 (num_iterations = 1): gradient = -(1/N) sum_it m_it A_i grad logp_it.
  - NO clipping (trl's epsilon is never used by the patched loss), NO KL term (beta must be 0; the patch forces
    ref_per_token_logps=None), NO per-sequence 1/|y_i| normalisation: it is a token-level mean over all completion
    tokens of the step, so a longer completion carries proportionally more weight (v3 targets: real 56 tokens, fake
    76-79 tokens incl. <|im_end|>, measured on rl_cf.jsonl with the Qwen3-VL tokenizer).
  - log-probs use logits / temperature (same temperature as sampling).

Reward (ctgrpo_core.score_completion): verdict +1 / -1 / -1.5 (unparseable); if the verdict is correct: fake
+0.5*IoU, real without box +0.5, real with a box -0.5, deepfake without a box -0.5; optional --field_bonus w per
correct evidence field on correct fakes. No constant terms.

Advantage modes (--adv_mode): image (standard GRPO baseline) | twin_pooled (ours) | cogroup_bonus (ConsistRoll-style,
--pair_bonus) | gdpo (decoupled per-channel normalisation). Exact formulas: ctgrpo_core.compute_advantages docstring.

Usage (GPU node, launched by the lead's scheduler; cwd anywhere):
  ~/.conda/envs/Qwen3.5/bin/python -u ~/CIRCE/v3/train_ctgrpo.py --sft_adapter <sft_v3_ckpt> \
      --data ~/AAAI/MedForge++/mfclean/v3data_b/rl_cf.jsonl --regime CF --adv_mode twin_pooled \
      --out ~/AAAI/MedForge++/ckpt/v3_ctgrpo_cf_twin
  (exact recommended commands: ~/CIRCE/v3/reports/trainer_review.md)
CPU checks (no GPU): --dry_run (data + reward wiring, no model); --tiny (random tiny Qwen3-VL end-to-end on CPU).
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import argparse
import inspect
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CIRCE = _release_path('CIRCE/circe')
sys.path.insert(0, HERE)
sys.path.insert(0, CIRCE)

import torch  # noqa: E402

import ctgrpo_core as core  # noqa: E402

DEFAULT_BASE = _release_path('models/Qwen3-VL-8B-Instruct')
IM_END, PAD = 151645, 151643
EOS_IDS = (IM_END, PAD)          # <|im_end|>, <|endoftext|> (= pad); generation stops on either


def completion_mask_multi_eos(completion_ids, eos_ids=EOS_IDS):
    """1 for tokens up to and including the first EOS (any id in eos_ids), 0 after; all 1 if no EOS (truncated)."""
    is_eos = torch.zeros_like(completion_ids, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= completion_ids == e
    T = completion_ids.size(1)
    first = torch.where(is_eos.any(1), is_eos.int().argmax(1), torch.full_like(is_eos[:, 0], T, dtype=torch.long))
    return (torch.arange(T, device=completion_ids.device).unsqueeze(0) <= first.unsqueeze(1)).int()


# measured (reports/trainer_review.md): longest target in rl_cf / sft_cf = 79 tokens incl. <|im_end|> -> 2x margin
DEFAULT_MAX_COMPLETION = 160


def _patch_trl():
    """Apply the existing VLM patch (unchanged) + the transformers-5.8 sampler-signature shim of train_grpo.py."""
    from trl import GRPOTrainer
    import vlm_grpo_patch
    if GRPOTrainer._generate_and_score_completions is not vlm_grpo_patch._generate_and_score_completions:
        vlm_grpo_patch.apply_vlm_grpo_patch()
    assert GRPOTrainer.compute_loss is vlm_grpo_patch.compute_loss
    return GRPOTrainer, vlm_grpo_patch


GRPOTrainer, vlm_grpo_patch = _patch_trl()


class CTGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, ct_units=None, ct_reward=None, ct_adv_mode="twin_pooled", ct_pairs_per_step=2,
                 ct_scale_std=True, ct_std_floor=0.0, ct_pair_bonus=0.0, ct_pool_channels="verdict",
                 ct_diag_path=None, ct_loss_microbatch=0, ct_strict_layout=True, ct_unit_real=None,
                 ct_max_pairs_per_real=0, ct_logp_selfcheck=1, ct_logp_selfcheck_abort=0.5, **kw):
        if ct_adv_mode not in core.ADV_MODES:
            raise ValueError(f"adv_mode {ct_adv_mode!r} not in {core.ADV_MODES}")
        # [review fix] no mode may silently reduce to standard per-image GRPO
        if ct_adv_mode == "cogroup_bonus" and not float(ct_pair_bonus) > 0:
            raise ValueError("--adv_mode cogroup_bonus with --pair_bonus <= 0 is exactly --adv_mode image")
        if ct_adv_mode == "gdpo" and not ct_scale_std:
            raise ValueError("--adv_mode gdpo with --scale_std 0 is exactly --adv_mode image (Dr.GRPO centring)")
        if ct_adv_mode in ("twin_pooled", "cogroup_bonus") and not any(len(u) == 2 for u in (ct_units or [])):
            raise ValueError(f"--adv_mode {ct_adv_mode} needs twin pairs, but the data has none (it would reduce to "
                             "--adv_mode image)")
        self._ct_unit_real, self._ct_cap = ct_unit_real, int(ct_max_pairs_per_real or 0)
        self._ct_units, self._ct_reward = ct_units, ct_reward
        self._ct_mode, self._ct_K = ct_adv_mode, int(ct_pairs_per_step)
        self._ct_scale, self._ct_floor, self._ct_bonus = bool(ct_scale_std), float(ct_std_floor), float(ct_pair_bonus)
        self._ct_pool, self._ct_diag_path = ct_pool_channels, ct_diag_path
        self._ct_mb, self._ct_strict = int(ct_loss_microbatch), ct_strict_layout
        self._ct_pending, self._ct_last_adv = None, None
        self._ct_selfcheck, self._ct_selfcheck_abort = int(ct_logp_selfcheck), float(ct_logp_selfcheck_abort)
        super().__init__(*args, **kw)
        a = self.args
        G = self.num_generations
        if a.beta != 0.0:
            raise ValueError("beta != 0 requested, but the vlm_grpo_patch loss has NO KL term (ref logps forced to "
                             "None). Refusing to run a config that would silently ignore beta.")
        if self.num_iterations != 1 or a.gradient_accumulation_steps != 1:
            raise ValueError("CT-GRPO assumes num_iterations == 1 and gradient_accumulation_steps == 1 "
                             "(the patch never computes old log-probs; one dataloader batch = one optimizer step)")
        if self.accelerator.num_processes != 1:
            raise ValueError("single-process only: twin groups are not gathered across processes")
        if self.max_prompt_length is not None:
            raise ValueError("max_prompt_length must be None (left-truncation would cut image tokens)")
        if a.per_device_train_batch_size != 2 * self._ct_K * G:
            raise ValueError(f"per_device_train_batch_size must be 2*K*G = {2 * self._ct_K * G}")
        if self._ct_mb and (a.per_device_train_batch_size % self._ct_mb):
            raise ValueError("--loss_microbatch must divide 2*K*G")
        if self._ct_mode != "cogroup_bonus" and self._ct_bonus:
            raise ValueError("--pair_bonus is only used by --adv_mode cogroup_bonus")
        # [review fix, audit B6] trl 0.16.1 builds GenerationConfig(eos_token_id=processing_class.eos_token_id) =
        # 151645 only, and transformers 5.8.1 does not merge model.generation_config's EOS list into it. Stop on BOTH
        # <|im_end|> (151645) and <|endoftext|> (151643); the completion mask is rebuilt from the same list below.
        if getattr(self, "generation_config", None) is not None:
            self.generation_config.eos_token_id = list(EOS_IDS)
            self.generation_config.pad_token_id = PAD

    # ---------------------------------------------------------------------------------------------------- sampler
    def _get_train_sampler(self, dataset=None):
        return core.TwinPairSampler(self._ct_units, self.num_generations, self._ct_K, seed=self.args.seed,
                                    unit_real=self._ct_unit_real, max_pairs_per_real=self._ct_cap)

    # ------------------------------------------------------------------------------------------ rewards -> adv
    def _generate_and_score_completions(self, inputs):
        t0 = time.time()
        out = super()._generate_and_score_completions(inputs)      # patched: generate + reward (+ trl advantages)
        t_gen = time.time() - t0
        rec = self._ct_reward.pop()
        comps = rec["comps"]
        pair_ids, roles = [x["pair_id"] for x in inputs], [x["role"] for x in inputs]
        if rec["pair_id"] != pair_ids or rec["role"] != roles:
            raise RuntimeError("reward call rows do not match the batch rows")
        G = self.num_generations
        lay = core.layout_group_ids(pair_ids, roles, G, strict=self._ct_strict)
        rv = torch.tensor([c["r_verdict"] for c in comps], dtype=torch.float32)
        rl = torch.tensor([c["r_loc"] for c in comps], dtype=torch.float32)
        correct = torch.tensor([c["correct"] for c in comps])
        adv = core.compute_advantages(rv, rl, lay["img_gid"], lay["pair_gid"], self._ct_mode, correct=correct,
                                      partner=lay["partner"], scale_std=self._ct_scale, std_floor=self._ct_floor,
                                      pair_bonus=self._ct_bonus, pool_channels=self._ct_pool)
        out["advantages"] = adv["adv"].to(out["advantages"].device, out["advantages"].dtype)
        self._ct_last_adv = adv
        # [review fix, audit B6] the patch masks on 151645 only; a rollout that stopped on 151643 would get a mask
        # over all its trailing pad tokens. Rebuild: mask = tokens up to and including the FIRST of {151645, 151643}.
        cm_patch = out["completion_mask"]
        cm = completion_mask_multi_eos(out["completion_ids"])
        n_mask_changed = int((cm != cm_patch.to(cm.dtype)).any(1).sum())
        out["completion_mask"] = cm.to(cm_patch.dtype)
        comp_len = cm.sum(1).tolist()
        if self._metrics["train"].get("completion_length"):
            self._metrics["train"]["completion_length"][-1] = float(sum(comp_len) / max(1, len(comp_len)))
        is_eos = torch.zeros_like(out["completion_ids"], dtype=torch.bool)
        for e in EOS_IDS:
            is_eos |= out["completion_ids"] == e
        has_eos = is_eos.any(1).tolist()
        stop_on_endoftext = int(((out["completion_ids"] == PAD) & cm.bool()).any(1).sum())
        # pad tokens inside the mask other than a terminating 151643 must be 0 (would mean a broken mask)
        pad_in_mask = int(((out["completion_ids"] == PAD) & cm.bool()).sum()) - stop_on_endoftext
        d = core.step_diagnostics(comps, lay, adv, [x["gt_verdict"] for x in inputs], [x["ftype"] for x in inputs],
                                  comp_len, has_eos, G)
        d.update({"mode": self._ct_mode, "gen_score_s": round(t_gen, 3), "pad_tokens_in_mask": pad_in_mask,
                  "n_stop_on_endoftext": stop_on_endoftext, "n_mask_rebuilt": n_mask_changed,
                  "layout_errors": lay["errors"], "rows": [f"{p}|{r}" for p, r in zip(pair_ids, roles)][::G],
                  "images": [x["image"] for x in inputs][::G],
                  # per-row arrays (2K*G values each) so every step's advantages can be recomputed offline
                  "row_r_verdict": rv.tolist(), "row_r_loc": rl.tolist(), "row_correct": correct.tolist(),
                  "row_parsed": [c["parsed"] for c in comps], "row_iou": [c["iou"] for c in comps],
                  "row_adv": [round(x, 6) for x in adv["adv"].tolist()], "row_len": comp_len})
        if self._ct_selfcheck:
            self._ct_selfcheck = 0                       # once, on the first optimizer step
            d["logp_selfcheck_maxdiff"] = self._ct_logp_selfcheck(out)
        self._ct_pending = d
        m = self._metrics["train"]
        for k in ("frac_zero_std_img", "frac_zero_std_twin", "frac_verdict_split_img", "frac_verdict_split_twin",
                  "frac_twin_contrast_only", "adv_share_verdict", "acc_fake", "acc_real", "parse_rate",
                  "mean_iou_correct_fakes", "frac_no_eos"):
            if d.get(k) is not None:
                m[f"ct/{k}"].append(d[k])
        return out

    # ------------------------------------------------------------------------------------ log-prob self-check
    def _ct_forward_logps(self, model, b):
        """per-token log-probs exactly as the patched compute_loss builds its forward (same vis / mm extension)."""
        ids = torch.cat([b["prompt_ids"], b["completion_ids"]], 1)
        am = torch.cat([b["prompt_mask"], b["completion_mask"]], 1)
        vis = {k: b[k] for k in ("pixel_values", "image_grid_thw") if k in b}
        if b.get("mm_token_type_ids") is not None:
            mm = b["mm_token_type_ids"]
            vis["mm_token_type_ids"] = torch.cat([mm, mm.new_zeros(b["completion_ids"].shape)], 1)
        return self._get_per_token_logps(model, ids, am, b["completion_ids"].size(1), **vis)

    @torch.no_grad()
    def _ct_logp_selfcheck(self, out):
        """[review addition] Step-1 check that the BATCHED loss forward (left-padded prompts, right-padded completions,
        2K images' pixel_values concatenated) gives the same per-token log-probs as each row ALONE, unpadded, with its
        own pixels. On CPU (torch 2.5.1, >1 thread) masked attention inside Qwen3-VL was found to be thread-dependent
        (reports/trainer_review.md), so this is the cheap on-GPU confirmation. Uses 4 rows (shortest/longest
        completions, distinct images). Returns max |diff| over completion tokens; aborts above the threshold on GPU."""
        cm, pm = out["completion_mask"], out["prompt_mask"]
        lens = cm.sum(1)
        G = self.num_generations
        cand = [int(lens.argmin()), int(lens.argmax())] + [g * G for g in range(cm.size(0) // G)]
        rows = []
        for r in cand:
            if r not in rows and all(r // G != x // G for x in rows):
                rows.append(r)
            if len(rows) == 4:
                break
        counts = out["image_grid_thw"].prod(1).tolist()
        starts = [0]
        for c in counts:
            starts.append(starts[-1] + c)

        def sub(rs, unpad=False):
            if unpad:
                (r,) = rs
                keep, L = pm[r].bool(), int(cm[r].sum())
                b = {"prompt_ids": out["prompt_ids"][r][keep][None], "prompt_mask": pm[r][keep][None],
                     "completion_ids": out["completion_ids"][r, :L][None], "completion_mask": cm[r, :L][None]}
                if out.get("mm_token_type_ids") is not None:
                    b["mm_token_type_ids"] = out["mm_token_type_ids"][r][keep][None]
            else:
                b = {k: out[k][rs] for k in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask")}
                if out.get("mm_token_type_ids") is not None:
                    b["mm_token_type_ids"] = out["mm_token_type_ids"][rs]
            b["pixel_values"] = torch.cat([out["pixel_values"][starts[r]:starts[r + 1]] for r in rs])
            b["image_grid_thw"] = out["image_grid_thw"][rs]
            return b
        model = self.model
        was_training = model.training
        model.eval()
        try:
            batched = self._ct_forward_logps(model, sub(rows)).float()
            md = 0.0
            for j, r in enumerate(rows):
                L = int(cm[r].sum())
                single = self._ct_forward_logps(model, sub([r], unpad=True)).float()[0]
                md = max(md, float((batched[j, :L] - single[:L]).abs().max()))
        finally:
            if was_training:
                model.train()
        on_gpu = out["completion_ids"].device.type == "cuda"
        print(f"[ctgrpo] log-prob self-check (rows {rows}, completion lens {[int(lens[r]) for r in rows]}): batched vs "
              f"unpadded single-row max |dlogp| = {md:.3e} on {out['completion_ids'].device.type}", flush=True)
        if on_gpu and self._ct_selfcheck_abort > 0 and md > self._ct_selfcheck_abort:
            raise RuntimeError(f"log-prob self-check failed: batched vs single-row |dlogp| {md:.3f} > "
                               f"{self._ct_selfcheck_abort} (padding/mask/pixel threading is broken on this setup)")
        return md

    # --------------------------------------------------------------------------------------------- train step
    def training_step(self, model, inputs, num_items_in_batch=None):
        t0 = time.time()
        if not self._ct_mb:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
            batch = self._prepare_inputs(inputs)
            B = batch["completion_ids"].size(0)
            tot = batch["completion_mask"].sum().clamp(min=1).float()
            loss = torch.zeros((), device=batch["completion_ids"].device)
            for ch in vlm_grpo_patch._split_tensor_dict_vlm(batch, B // self._ct_mb):
                with self.compute_loss_context_manager():
                    lc = self.compute_loss(model, ch) * (ch["completion_mask"].sum().float() / tot)
                self.accelerator.backward(lc)
                loss = loss + lc.detach()
            del batch
        d, self._ct_pending = self._ct_pending, None
        if d is not None and self._ct_diag_path and self.accelerator.is_main_process:
            d["step"] = int(self.state.global_step) + 1
            d["step_s"] = round(time.time() - t0, 3)
            d["loss"] = float(loss.detach().float().item())
            d["lr"] = float(self._get_learning_rate()) if self.lr_scheduler is not None else None
            if torch.cuda.is_available():
                d["cuda_max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            with open(self._ct_diag_path, "a") as fh:
                fh.write(json.dumps(d) + "\n")
            print(f"[ctgrpo] step {d['step']} mode={d['mode']} acc fake/real={d['acc_fake']}/{d['acc_real']} "
                  f"zero-std img/twin={d['frac_zero_std_img']}/{d['frac_zero_std_twin']} "
                  f"split img/twin={d['frac_verdict_split_img']}/{d['frac_verdict_split_twin']} "
                  f"adv_share_verdict={d['adv_share_verdict']:.3f} len={d['comp_len_mean']:.1f} "
                  f"gen={d['gen_score_s']}s step={d['step_s']}s", flush=True)
        return loss


# ------------------------------------------------------------------------------------------------- model utils
def load_processor(base):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id", "pad_token", "eos_token"):
        if getattr(proc, attr, None) is None:
            try:
                setattr(proc, attr, getattr(proc.tokenizer, attr))
            except Exception:
                pass
    proc.eos_token_id = IM_END       # trl builds GenerationConfig(eos_token_id=proc.eos_token_id); patch masks on it
    return proc


def _model_shims(model):
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    if not hasattr(model, "add_model_tags"):
        model.add_model_tags = lambda *a, **k: None
    model.generation_config.eos_token_id = [IM_END, PAD]
    model.generation_config.pad_token_id = PAD
    return model


def build_tiny_model(base, lora_r=4, seed=0):
    """Random-weight tiny Qwen3-VL (same recipe as ~/CIRCE/v3_audit/code_scripts/cpu_dryrun_tiny_qwen3vl.py) + LoRA."""
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    cfg = Qwen3VLConfig.from_pretrained(base)
    tc, vc = cfg.text_config, cfg.vision_config
    tc.hidden_size, tc.num_hidden_layers, tc.num_attention_heads, tc.num_key_value_heads = 64, 2, 4, 2
    tc.head_dim, tc.intermediate_size = 16, 128
    tc.rope_parameters["mrope_section"] = [4, 2, 2]
    vc.depth, vc.hidden_size, vc.intermediate_size, vc.num_heads, vc.out_hidden_size = 2, 32, 64, 2, 64
    vc.deepstack_visual_indexes = [0, 1]
    model = _model_shims(Qwen3VLForConditionalGeneration(cfg).float())
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    return model


def build_model(args):
    from transformers import AutoModelForImageTextToText
    from peft import PeftModel, LoraConfig, get_peft_model
    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, device_map="auto",
                                                        trust_remote_code=True)
    model = _model_shims(model)
    if args.sft_adapter:
        model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    else:
        print("[ctgrpo] WARNING: no --sft_adapter, starting from a FRESH LoRA on the base model")
        from train_sft import select_lora_targets
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                                 target_modules=select_lora_targets(model), task_type="CAUSAL_LM"))
    return model


def make_trainer(model, proc, rows, units, args, use_cpu=False, extra_cfg=None, unit_real=None, trainer_cls=None):
    from datasets import Dataset
    from trl import GRPOConfig
    G, K = args.num_gen, args.pairs_per_step
    cfg_kw = dict(output_dir=args.out, learning_rate=args.lr, num_generations=G,
                  per_device_train_batch_size=2 * K * G, gradient_accumulation_steps=1,
                  num_train_epochs=args.epochs, max_steps=args.max_steps, max_prompt_length=None,
                  max_completion_length=args.max_completion, bf16=not use_cpu, logging_steps=args.logging_steps,
                  save_steps=args.save_steps, save_total_limit=args.save_total_limit or None, beta=0.0, report_to=[], remove_unused_columns=False,
                  gradient_checkpointing=args.grad_ckpt, temperature=args.temperature, top_p=args.top_p,
                  top_k=args.top_k, seed=args.seed, log_completions=False, dataloader_num_workers=0,
                  use_cpu=use_cpu, num_iterations=1)
    if extra_cfg:
        cfg_kw.update(extra_cfg)
    cfg = GRPOConfig(**cfg_kw)
    if args.grad_ckpt and getattr(model, "enable_input_require_grads", None):
        model.enable_input_require_grads()
    reward = core.CTReward(field_bonus=args.field_bonus)
    tr = (trainer_cls or CTGRPOTrainer)(model=model, reward_funcs=[reward], args=cfg, train_dataset=Dataset.from_list(rows),
                       processing_class=proc, ct_units=units, ct_reward=reward, ct_adv_mode=args.adv_mode,
                       ct_pairs_per_step=K, ct_scale_std=bool(args.scale_std), ct_std_floor=args.std_floor,
                       ct_pair_bonus=args.pair_bonus, ct_pool_channels=args.pool_channels,
                       ct_diag_path=args.diag_jsonl, ct_loss_microbatch=args.loss_microbatch,
                       ct_unit_real=unit_real, ct_max_pairs_per_real=args.max_pairs_per_real,
                       ct_logp_selfcheck=args.logp_selfcheck, ct_logp_selfcheck_abort=args.logp_selfcheck_abort)
    return tr


def honesty_banner(args):
    print("[ctgrpo] OBJECTIVE: L = -(1/N) sum_it m_it * exp(logp - sg(logp)) * A_i, N = completion tokens of the step "
          "(token-level mean; ratio == 1).")
    print("[ctgrpo] NO KL is applied (beta=0; the patched loss has no KL term). NO clipping (num_iterations=1, trl "
          "epsilon unused). NO 1/|y_i| per-sequence normalisation.")
    print(f"[ctgrpo] adv_mode={args.adv_mode} scale_std={args.scale_std} std_floor={args.std_floor} "
          f"pool_channels={args.pool_channels} pair_bonus={args.pair_bonus} field_bonus={args.field_bonus} "
          f"K={args.pairs_per_step} G={args.num_gen} -> {2 * args.pairs_per_step} images, "
          f"{2 * args.pairs_per_step * args.num_gen} rollouts per optimizer step")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sft_adapter", default=None)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--data", required=True, help="rl_<regime>.jsonl (twin pairs + singletons)")
    ap.add_argument("--regime", choices=list(core.REGIMES), default=None,
                    help="if given, assert the data obeys the regime's exclusions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--diag_jsonl", default=None, help="default: <out>_diag.jsonl (next to the output dir)")
    ap.add_argument("--adv_mode", choices=core.ADV_MODES, default="twin_pooled")
    ap.add_argument("--pool_channels", choices=("verdict", "all"), default="verdict")
    ap.add_argument("--pair_bonus", type=float, default=0.0, help="lambda for cogroup_bonus")
    ap.add_argument("--field_bonus", type=float, default=0.0)
    ap.add_argument("--scale_std", type=int, default=1, help="1: divide by group std (GRPO); 0: Dr.GRPO centring")
    ap.add_argument("--std_floor", type=float, default=0.0, help="denominator = max(std, floor) + 1e-4")
    ap.add_argument("--pairs_per_step", type=int, default=2)
    ap.add_argument("--num_gen", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--save_total_limit", type=int, default=0, help="0 = keep every checkpoint")
    ap.add_argument("--logging_steps", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50, help="trl 0.16.1 default (model default is 20; audit B8)")
    ap.add_argument("--max_completion", type=int, default=DEFAULT_MAX_COMPLETION)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beta", type=float, default=0.0, help="must be 0: the patched loss has no KL term")
    ap.add_argument("--lora_r", type=int, default=32, help="only used without --sft_adapter")
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--loss_microbatch", type=int, default=0, help="rows per loss chunk (0 = one forward)")
    ap.add_argument("--logp_selfcheck", type=int, default=1,
                    help="step 1: batched vs single-row log-prob consistency check (4 rows, no grad)")
    ap.add_argument("--logp_selfcheck_abort", type=float, default=0.5,
                    help="abort (GPU only) if the self-check max |dlogp| exceeds this; 0 = never abort")
    ap.add_argument("--no_singletons", action="store_true")
    ap.add_argument("--max_pairs_per_real", type=int, default=0,
                    help="per epoch keep at most c pairs per real twin image (0 = all; rl_cf reuses a real <= 20x)")
    ap.add_argument("--limit_units", type=int, default=0)
    ap.add_argument("--allow_nontrain", action="store_true", help="DANGER: allow rows whose split != train")
    ap.add_argument("--dry_run", action="store_true", help="data + reward wiring only, no model")
    ap.add_argument("--tiny", action="store_true", help="CPU: random tiny Qwen3-VL end-to-end (no real weights)")
    args = ap.parse_args(argv)
    if args.beta != 0.0:
        ap.error("--beta must be 0: vlm_grpo_patch.compute_loss has no KL term (and trl's PEFT reference would be "
                 "the base model, not the SFT policy)")
    if args.pair_bonus and args.adv_mode != "cogroup_bonus":
        ap.error("--pair_bonus is only used by --adv_mode cogroup_bonus")
    if args.adv_mode == "cogroup_bonus" and not args.pair_bonus > 0:
        ap.error("--adv_mode cogroup_bonus needs --pair_bonus > 0 (otherwise it is exactly --adv_mode image)")
    if args.adv_mode == "gdpo" and not args.scale_std:
        ap.error("--adv_mode gdpo needs --scale_std 1 (otherwise it is exactly --adv_mode image)")
    if args.pool_channels != "verdict" and args.adv_mode != "twin_pooled":
        ap.error("--pool_channels is only used by --adv_mode twin_pooled")
    if args.diag_jsonl is None:
        args.diag_jsonl = args.out.rstrip("/") + "_diag.jsonl"
    return args


def main(argv=None):
    args = parse_args(argv)
    honesty_banner(args)
    rows, units, st = core.load_rl_rows(args.data, regime=args.regime, allow_nontrain=args.allow_nontrain,
                                        use_singletons=not args.no_singletons, limit_units=args.limit_units)
    print(f"[ctgrpo] data {args.data}: {json.dumps(st)}")
    samp = core.TwinPairSampler(units, args.num_gen, args.pairs_per_step, seed=args.seed, unit_real=st["unit_real"],
                                max_pairs_per_real=args.max_pairs_per_real)
    print(f"[ctgrpo] {samp.n_slots} two-image slots -> {samp.steps_per_epoch} optimizer steps/epoch")
    if args.dry_run:
        from prompts_v3 import format_target
        r0 = rows[units[0][0]]
        t = format_target(r0["gt_verdict"], r0["gt_box"], r0["gt_type"], r0["gt_intensity"] or "none")
        rw = core.CTReward(args.field_bonus)
        kw = {k: [r0[k]] for k in ("gt_verdict", "gt_box", "gt_type", "gt_location", "gt_size", "gt_intensity",
                                    "pair_id", "role")}
        print("[dry_run] GT-target reward:", rw(completions=[[{"role": "assistant", "content": t}]], **kw),
              "| garbage reward:", rw(completions=["blah"], **kw))
        it = iter(samp)
        first = [next(it) for _ in range(2 * args.pairs_per_step * args.num_gen)]
        print("[dry_run] first step rows:", [f"{rows[i]['pair_id']}|{rows[i]['role']}" for i in first[::args.num_gen]])
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.diag_jsonl)), exist_ok=True)
    if os.path.exists(args.diag_jsonl):
        os.rename(args.diag_jsonl, args.diag_jsonl + f".bak{int(time.time())}")
    proc = load_processor(args.base)
    if args.tiny:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        model = build_tiny_model(args.base)
    else:
        model = build_model(args)
    tr = make_trainer(model, proc, rows, units, args, use_cpu=args.tiny, unit_real=st["unit_real"])
    os.makedirs(args.out, exist_ok=True)
    json.dump({**vars(args), "data_stats": st}, open(os.path.join(args.out, "ctgrpo_args.json"), "w"), indent=1)
    # NOTE (review): exact resume (trainer.train(resume_from_checkpoint=...)) is NOT possible in this env:
    # transformers 5.8.1 refuses torch.load (optimizer.pt / scheduler.pt / rng_state.pth) with torch 2.5.1.
    # Checkpoint adapters are still usable for evaluation / a warm restart (--sft_adapter <ckpt>).
    tr.train()
    tr.save_model(args.out)
    print(f"[done] CT-GRPO ({args.adv_mode}) -> {args.out} ; diagnostics -> {args.diag_jsonl}")


if __name__ == "__main__":
    main()
