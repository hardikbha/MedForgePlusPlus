"""
S1 GRPO for CIRCE — continue from the SFT LoRA adapter, optimize the VERIFIABLE
reward stack (R_verdict + R_loc) with trl GRPOTrainer.

This is the run that targets BEATING MEDFORGE: the V1 config uses CIRCE's r_loc
(gated-IoU + DIoU + area-penalty); the V0 baseline swaps in reward_coverage_baseline
to reproduce MedForge's coverage reward for a controlled comparison.

Env: Qwen3.5 (transformers 5.8.1, trl 0.16.1, peft 0.19.1).
Run (GPU node):
  python train_grpo.py --sft_adapter <ckpt> --data circe_sft_full.jsonl \
     --reward v1 --out ckpt/circe_grpo_v1

NOTE: not yet GPU-validated (needs a sanity-passing SFT checkpoint + trl VLM-GRPO check).
`--dry_run` validates dataset build + reward-func wiring on CPU.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, argparse, sys, re
sys.path.insert(0, os.path.dirname(__file__))
from circe_grpo_reward import (reward_verdict, reward_localization, reward_coverage_baseline,
                               reward_cue_consistency, reward_calibration, reward_forgery_grounded,
                               reward_balanced_verdict_guard, reward_format,
                               reward_evidence_consistency, reward_classification_dominant)

DEEPFAKE_TYPES = {"remove", "inject", "deepfake", "removal", "injection", "edit"}
# single source of truth: GRPO rollout prompt MUST match the SFT training prompt (train/inference match)
from build_sft_data import SYS, USER1 as _U
USER1 = _U.replace("<image>", "").strip()
CONF_SUFFIX = " Finally, on a new line output your calibrated confidence as 'Confidence: 0.xx'."


def _c(text=None, image=None):
    # Keep actual image payloads in the row-level `image` column. Native TRL VLM
    # support fills empty image placeholders from that column; embedding the
    # path here creates duplicate/mismatched placeholders.
    return {"type": "image"} if image else {"type": "text", "text": text}


def _rows_from_file(sft_path, user_text):
    rows = []
    for l in open(sft_path):
        r = json.loads(l)
        img = r["images"][0]
        asst = " ".join(m["content"] for m in r["messages"] if m["role"] == "assistant")
        gt_v = "deepfake" if "deepfake" in asst else "real"
        bm = re.search(r'x1="(\d+)" y1="(\d+)" x2="(\d+)" y2="(\d+)"', asst)
        gt_box = [int(x) for x in bm.groups()] if bm else None
        rows.append({
            "prompt": [{"role": "system", "content": [_c(text=SYS)]},
                       {"role": "user", "content": [_c(image=img), _c(text=user_text)]}],
            "image": img,
            "gt_verdict": gt_v,
            "gt_box": gt_box,
        })
    return rows


def build_prompts(sft_path, limit=0, conf=False, data2=None, mix_ratio=0.0, seed=42):
    """GRPO dataset: prompt (chat) + gt columns the reward funcs read (gt_verdict, gt_box).

    If data2 is given, interleave a mix_ratio fraction of rows from data2 (a second domain,
    e.g. PF+ kidney/polyp) into the primary set, so RL reward pressure can't just optimize
    one distribution and drift off the other -> portability across domains, not just the
    benchmark the reward was tuned on.
    """
    import random
    user_text = USER1 + (CONF_SUFFIX if conf else "")
    rows = _rows_from_file(sft_path, user_text)
    if limit:
        rows = rows[:limit]
    if data2 and mix_ratio > 0:
        rows2 = _rows_from_file(data2, user_text)
        rng = random.Random(seed)
        rng.shuffle(rows2)
        n2 = max(1, int(len(rows) * mix_ratio / (1 - mix_ratio)))
        rows = rows + rows2[:n2]
        rng.shuffle(rows)
        print(f"[grpo] domain mix: +{min(n2, len(rows2))} rows from {data2} "
              f"(target mix_ratio={mix_ratio})")
    return rows


REWARD_SETS = {
    "v0": ([reward_verdict, reward_coverage_baseline], [1.0, 1.0]),   # MedForge coverage baseline
    "v1": ([reward_verdict, reward_localization], [1.0, 1.0]),        # CIRCE r_loc (beats coverage)
    "v2": ([reward_verdict, reward_localization, reward_cue_consistency], [1.0, 1.0, 0.5]),  # + cue-invariance
    "calib": ([reward_calibration], [1.0]),  # hack-proof Stage-2: verdict-correct - Brier(confidence)
    # GSPO analogue: forgery-grounded reward + format-bootstrap (rewards concluding -> variance while the
    # policy learns to terminate). reward_evidence_consistency adds the grounded-cue transfer signal.
    "forgery": ([reward_forgery_grounded, reward_localization, reward_format, reward_evidence_consistency],
                [1.0, 1.0, 0.5, 0.5]),
    "forgery_guarded": ([reward_balanced_verdict_guard, reward_localization,
                         reward_format, reward_evidence_consistency],
                        [1.2, 0.8, 0.3, 0.4]),
    # CIRCE-v2 Phase 10: single reward fn, internally gated so localization/evidence/format
    # bonuses are ONLY paid on a correct verdict -- cannot compensate for a wrong one (unlike
    # every set above, which sums independent terms). See reward_classification_dominant docstring.
    "circe_v2_classification_dominant": ([reward_classification_dominant], [1.0]),
    # Ablation 4: localization-only reward, no verdict term at all -- isolates whether RL
    # improves localization even with zero classification signal in the reward.
    "ablation_localization_only": ([reward_localization], [1.0]),
    # Ablation 4: explanation/evidence-only reward, no verdict or box term -- isolates whether
    # RL improves reasoning/format quality with zero classification or localization signal.
    "ablation_explanation_only": ([reward_evidence_consistency, reward_format], [1.0, 1.0]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_adapter", required=False)
    ap.add_argument("--base", default=_release_path('models/Qwen3-VL-4B-Instruct'))
    ap.add_argument("--data", default=_release_path('AAAI/MedForge++/circe_sft_full.jsonl'))
    ap.add_argument("--data2", default=None, help="second-domain jsonl to mix in (portability)")
    ap.add_argument("--mix_ratio", type=float, default=0.0, help="fraction of batch drawn from --data2")
    ap.add_argument("--beta", type=float, default=0.0, help="KL penalty to reference (SFT) policy")
    ap.add_argument("--reward", choices=list(REWARD_SETS), default="v1")
    ap.add_argument("--out", default=_release_path('AAAI/MedForge++/ckpt/circe_grpo_v1'))
    ap.add_argument("--num_gen", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_completion", type=int, default=640)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--log_completions", action="store_true")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--save_steps", type=int, default=300)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    rows = build_prompts(args.data, args.limit, conf=(args.reward == "calib"),
                         data2=args.data2, mix_ratio=args.mix_ratio)
    funcs, weights = REWARD_SETS[args.reward]
    print(f"[grpo] {len(rows)} prompts | reward={args.reward} funcs={[f.__name__ for f in funcs]} weights={weights}")

    if args.dry_run:
        # verify reward funcs run on a fake completion batch with the gt columns
        demo = ['<think>...</think>\nThis image is a deepfake. '
                '<|box_start|>x1="10" y1="10" x2="30" y2="30"<|box_end|>']
        for f in funcs:
            out = f(completions=demo, gt_verdict=["deepfake"], gt_box=[[10, 10, 30, 30]])
            assert isinstance(out, list) and isinstance(out[0], float), f"{f.__name__} bad output"
            print(f"  {f.__name__}({demo[0][:30]}...) -> {out}")
        assert rows and "prompt" in rows[0] and rows[0]["prompt"][1]["content"][0]["type"] == "image"
        print("[dry_run] dataset + reward wiring OK. Skipping model load.")
        return

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel, LoraConfig
    import trl
    from packaging.version import Version
    from trl import GRPOTrainer, GRPOConfig
    if Version(trl.__version__) < Version("0.20.0"):
        from vlm_grpo_patch import apply_vlm_grpo_patch
        apply_vlm_grpo_patch()   # thread images into VLM-GRPO generation + loss
    from datasets import Dataset
    import inspect
    # shim: transformers-5.8 Trainer calls _get_train_sampler(dataset); trl-0.16 override takes only self
    if "dataset" not in inspect.signature(GRPOTrainer._get_train_sampler).parameters:
        _orig_sampler = GRPOTrainer._get_train_sampler
        GRPOTrainer._get_train_sampler = lambda self, dataset=None: _orig_sampler(self)
    if hasattr(GRPOTrainer, "_get_eval_sampler") and\
       "dataset" not in inspect.signature(GRPOTrainer._get_eval_sampler).parameters:
        _orig_esampler = GRPOTrainer._get_eval_sampler
        GRPOTrainer._get_eval_sampler = lambda self, eval_dataset=None: _orig_esampler(self, eval_dataset)

    proc = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"   # generation needs left-pad
    # shim: trl 0.16 reads these off the processor; transformers-5.8 keeps them on .tokenizer
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id", "pad_token", "eos_token"):
        if not hasattr(proc, attr) or getattr(proc, attr, None) is None:
            try:
                setattr(proc, attr, getattr(proc.tokenizer, attr))
            except Exception:
                pass
    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16,
                                                        device_map="auto", trust_remote_code=True)
    # shim: trl 0.16 expects PreTrainedModel.warnings_issued (dropped in transformers 5.8)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    if not hasattr(model, "add_model_tags"):
        model.add_model_tags = lambda *a, **k: None
    if args.sft_adapter:
        model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)

    # FIX non-termination: force generation to STOP at <|im_end|> (151645) / <|endoftext|> (151643).
    # Without this, trl's sampled rollouts run to max_completion without ever emitting the verdict+EOS,
    # so reward_verdict is always 0 and reward_std=0 (no gradient). Greedy eval stopped fine; sampling did not.
    gc = model.generation_config
    gc.eos_token_id = [151645, 151643]
    gc.pad_token_id = 151643
    proc.eos_token_id = 151645

    cfg = GRPOConfig(output_dir=args.out, learning_rate=args.lr, num_generations=args.num_gen,
                     per_device_train_batch_size=args.num_gen, gradient_accumulation_steps=1,
                     num_train_epochs=args.epochs, max_steps=args.max_steps,
                     max_prompt_length=None, max_completion_length=args.max_completion,
                     bf16=True, logging_steps=5, save_steps=args.save_steps,
                     beta=args.beta, report_to=[], remove_unused_columns=False, reward_weights=weights,
                     gradient_checkpointing=True, temperature=args.temperature, top_p=0.95,
                     log_completions=args.log_completions)
    if getattr(model, "enable_input_require_grads", None):
        model.enable_input_require_grads()   # needed for grad-checkpointing through a frozen base + LoRA
    trainer = GRPOTrainer(model=model, reward_funcs=funcs, args=cfg,
                          train_dataset=Dataset.from_list(rows),
                          processing_class=proc)
    trainer.train()
    trainer.save_model(args.out)
    print(f"[done] GRPO ({args.reward}) -> {args.out}")


if __name__ == "__main__":
    main()
