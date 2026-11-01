"""CPU dry-run harness: tiny random Qwen3-VL + LoRA + the EXISTING vlm_grpo_patch + trl 0.16.1 GRPOTrainer.
Purpose: (1) prove multi-prompt steps (k prompts x G rollouts) run through the current patch,
(2) check batched-vs-unbatched per-token log-prob parity with LEFT padding and DIFFERENT image sizes,
(3) give the template every spec item's unit test plugs into. No GPU, no real weights, no writes outside v3_audit."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import os, sys, json, random, inspect
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, _release_path('CIRCE/circe'))
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
torch.manual_seed(0); random.seed(0)
OUT = _release_path('CIRCE/v3_audit/code_scripts/out/dry')
BASE = _release_path('models/Qwen3-VL-8B-Instruct')
proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
proc.tokenizer.padding_side = "left"
for attr in ("pad_token_id", "eos_token_id", "bos_token_id", "pad_token", "eos_token"):
    if getattr(proc, attr, None) is None:
        try: setattr(proc, attr, getattr(proc.tokenizer, attr))
        except Exception: pass
proc.eos_token_id = 151645
cfg = Qwen3VLConfig.from_pretrained(BASE)
tc, vc = cfg.text_config, cfg.vision_config
tc.hidden_size, tc.num_hidden_layers, tc.num_attention_heads, tc.num_key_value_heads = 64, 2, 4, 2
tc.head_dim, tc.intermediate_size = 16, 128
tc.rope_parameters["mrope_section"] = [4, 2, 2]   # must sum to head_dim/2
vc.depth, vc.hidden_size, vc.intermediate_size, vc.num_heads, vc.out_hidden_size = 2, 32, 64, 2, 64
vc.deepstack_visual_indexes = [0, 1]
model = Qwen3VLForConditionalGeneration(cfg).float()
model.generation_config.eos_token_id = [151645, 151643]; model.generation_config.pad_token_id = 151643
if not hasattr(model, "warnings_issued"): model.warnings_issued = {}
if not hasattr(model, "add_model_tags"): model.add_model_tags = lambda *a, **k: None
model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
print("[dry] tiny model params:", sum(p.numel() for p in model.parameters()), "trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# tiny dataset in the exact SFT/RL jsonl schema; two image sizes -> different prompt lengths -> left padding
recs = []
for i, (w, h) in enumerate([(256, 256), (256, 320), (320, 256), (256, 256)]):
    p = f"{OUT}/img{i}.png"; Image.new("RGB", (w, h), (i * 40, 80, 120)).save(p)
    fake = i % 2 == 0
    a1 = ('<think>x</think>\n\nThis image is a deepfake. <|box_start|>x1="10" y1="10" x2="60" y2="60"<|box_end|>' if fake
          else '<think>x</think>\n\nThis image is a real image.')
    recs.append({"images": [p], "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "<image>q"},
                                             {"role": "assistant", "content": a1}]})
jp = f"{OUT}/tiny.jsonl"; open(jp, "w").write("\n".join(json.dumps(r) for r in recs) + "\n")
import train_grpo
rows = train_grpo.build_prompts(jp)
from vlm_grpo_patch import apply_vlm_grpo_patch
apply_vlm_grpo_patch()
from trl import GRPOTrainer, GRPOConfig
if "dataset" not in inspect.signature(GRPOTrainer._get_train_sampler).parameters:
    _o = GRPOTrainer._get_train_sampler; GRPOTrainer._get_train_sampler = lambda self, dataset=None: _o(self)
from datasets import Dataset

calls = []
def reward_random(prompts=None, completions=None, gt_verdict=None, **kw):
    calls.append(len(completions)); return [random.random() for _ in completions]
K, G = 2, 4   # 2 prompts x 4 rollouts per optimizer step
args = GRPOConfig(output_dir=f"{OUT}/run", per_device_train_batch_size=K * G, num_generations=G, max_steps=2,
                  max_prompt_length=None, max_completion_length=6, learning_rate=1e-4, logging_steps=1, report_to=[],
                  remove_unused_columns=False, use_cpu=True, bf16=False, save_strategy="no", beta=0.0, temperature=0.7, top_p=0.95)
tr = GRPOTrainer(model=model, reward_funcs=[reward_random], args=args, train_dataset=Dataset.from_list(rows), processing_class=proc)
before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
tr.train()
changed = sum(int(not torch.equal(before[n], p.detach())) for n, p in model.named_parameters() if p.requires_grad)
print(f"[dry] multi-prompt GRPO ran: reward calls={calls} (each = K*G={K*G} completions); LoRA tensors changed={changed}/{len(before)}")

# batched-vs-unbatched log-prob parity with left padding + different image sizes
model.eval()
from vlm_grpo_patch import _clean_msgs
texts = [proc.apply_chat_template(_clean_msgs(r["prompt"]), tokenize=False, add_generation_prompt=True) for r in rows[:2]]
ims = [[Image.open(r["image"]).convert("RGB")] for r in rows[:2]]
comp = torch.tensor([[1, 2, 3, 4, 5]] * 2)
def logps(text_list, im_list, comp_ids):
    pi = proc(text=text_list, images=im_list, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    ids = torch.cat([pi["input_ids"], comp_ids], 1); am = torch.cat([pi["attention_mask"], torch.ones_like(comp_ids)], 1)
    vis = {k: pi[k] for k in ("pixel_values", "image_grid_thw") if k in pi}
    if "mm_token_type_ids" in pi:
        vis["mm_token_type_ids"] = torch.cat([pi["mm_token_type_ids"], torch.zeros_like(comp_ids)], 1)
    with torch.no_grad():
        return tr._get_per_token_logps(model, ids, am, comp_ids.size(1), **vis), pi["input_ids"].shape
lb, shp = logps(texts, ims, comp)
l0, s0 = logps(texts[:1], ims[:1], comp[:1]); l1, s1 = logps(texts[1:], ims[1:], comp[1:])
print(f"[dry] prompt lens: batch={tuple(shp)} single0={tuple(s0)} single1={tuple(s1)}")
print(f"[dry] max |logp batched - single| row0={float((lb[0]-l0[0]).abs().max()):.2e} row1={float((lb[1]-l1[0]).abs().max()):.2e}")
