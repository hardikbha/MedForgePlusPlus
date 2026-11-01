"""
S0 SFT cold-start for CIRCE — Qwen3-VL-4B-Thinking, LoRA on LANGUAGE layers only
(vision tower + merger FROZEN; our verified mode-collapse fix).

Env: Qwen3.5 (transformers 5.8.1, trl 0.16.1, peft 0.19.1).
Run (on a GPU node):
  python train_sft.py --data circe_sft_full.jsonl \
    --model ${MEDFORGE_WORKSPACE}/models/Qwen3-VL-4B-Thinking \
    --out ${MEDFORGE_WORKSPACE}/AAAI/MedForge++/ckpt/circe_sft_v0.2 --epochs 1

NOTE: not yet GPU-validated — this is the v0.2 training scaffold. `--dry_run` checks data +
LoRA target selection on CPU without loading weights.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, argparse


def load_dataset(path):
    rows = [json.loads(l) for l in open(path)]
    print(f"[data] {len(rows)} SFT records from {path}")
    return rows


def select_lora_targets(model):
    """LoRA on language-model linear layers only; skip vision tower + merger."""
    import torch.nn as nn
    targets, frozen = set(), 0
    for name, mod in model.named_modules():
        low = name.lower()
        is_vision = ("visual" in low or "vision" in low or "merger" in low or "patch_embed" in low)
        if isinstance(mod, nn.Linear) and not is_vision:
            targets.add(name.split(".")[-1])
        if is_vision:
            frozen += 1
    # keep the usual attention/mlp projections
    keep = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = sorted(t for t in targets if t in keep)
    print(f"[lora] targets={targets}  (vision/merger modules seen: {frozen})")
    return targets


def _structure_messages(msgs, img_paths):
    """Qwen3-VL needs the image as a structured content entry (not a literal '<image>'
    string) so apply_chat_template inserts the real <|vision_start|>..<|image_pad|> tokens.
    Inject the image into the FIRST user turn; strip any literal '<image>' text."""
    out, injected = [], False
    for m in msgs:
        content = m["content"]
        if m["role"] == "user" and not injected and img_paths:
            text = content.replace("<image>", "").strip() if isinstance(content, str) else content
            out.append({"role": "user",
                        "content": [{"type": "image", "image": img_paths[0]},
                                    {"type": "text", "text": text}]})
            injected = True
        else:
            if isinstance(content, str):
                content = content.replace("<image>", "").strip()
            out.append({"role": m["role"], "content": content})
    return out


def _subseq_positions(row, sub):
    """indices where list `sub` starts inside 1-D tensor `row`."""
    out, n = [], len(sub)
    r = row.tolist()
    for i in range(len(r) - n + 1):
        if r[i:i + n] == sub:
            out.append(i)
    return out


def build_collator(processor):
    from PIL import Image
    tok = processor.tokenizer
    # assistant-turn header "<|im_start|>assistant\n" and terminator "<|im_end|>"
    asst_header = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

    def collate(batch):
        texts, images = [], []
        for ex in batch:
            paths = ex.get("images", [])
            msgs = _structure_messages(ex["messages"], paths)
            texts.append(processor.apply_chat_template(msgs, tokenize=False,
                                                       add_generation_prompt=False))
            images.append([Image.open(p).convert("RGB") for p in paths] if paths else None)
        model_inputs = processor(text=texts, images=images, return_tensors="pt",
                                 padding=True, truncation=True, max_length=4096)
        ids = model_inputs["input_ids"]
        labels = ids.clone().fill_(-100)   # mask everything, then UNMASK assistant spans only
        for b in range(ids.size(0)):
            row = ids[b]
            for start in _subseq_positions(row, asst_header):
                s = start + len(asst_header)
                e = s
                while e < row.size(0) and row[e].item() != im_end_id:
                    e += 1
                e = min(e + 1, row.size(0))  # include <|im_end|> so the model learns to stop
                labels[b, s:e] = row[s:e]
        model_inputs["labels"] = labels
        return model_inputs
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=_release_path('models/Qwen3-VL-4B-Thinking'))
    ap.add_argument("--out", default=_release_path('AAAI/MedForge++/ckpt/circe_sft_v0.2'))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=-1, help="cap steps (smoke test)")
    ap.add_argument("--limit", type=int, default=0, help="cap dataset size (smoke test)")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--resume_adapter", default=None,
                    help="continue-train an existing LoRA adapter (Stage-2 RAFT on top of Stage-1 SFT)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed LoRA init + data order (default: unseeded LoRA init, Trainer seed 42 = old behaviour)")
    args = ap.parse_args()

    rows = load_dataset(args.data)
    if args.limit:
        rows = rows[:args.limit]
        print(f"[data] capped to {len(rows)} records (smoke)")
    if args.dry_run:
        assert all(len(r["messages"]) == 5 for r in rows[:50]), "expected 5-msg two-turn records"
        assert all(os.path.exists(r["images"][0]) for r in rows[:20]), "image paths must exist"
        print("[dry_run] data OK (5-msg two-turn, images exist). Skipping model load.")
        return

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    if args.seed is not None:
        from transformers import set_seed
        set_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    # freeze vision tower + merger explicitly
    for name, p in model.named_parameters():
        low = name.lower()
        if "visual" in low or "vision" in low or "merger" in low:
            p.requires_grad = False

    if args.resume_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
        print(f"[stage2] resumed LoRA adapter from {args.resume_adapter}")
    else:
        lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                          target_modules=select_lora_targets(model), task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    cfg = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs,
                    max_steps=args.max_steps,
                    per_device_train_batch_size=args.bs,
                    gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
                    bf16=True, logging_steps=1, save_steps=500, save_total_limit=2,
                    gradient_checkpointing=True, report_to=[], remove_unused_columns=False,
                    dataset_kwargs={"skip_prepare_dataset": True},
                    **({"seed": args.seed} if args.seed is not None else {}))
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=rows,
                         data_collator=build_collator(processor))
    trainer.train()
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    print(f"[done] SFT checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
