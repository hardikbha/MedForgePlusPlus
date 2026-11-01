"""Train/eval prompt parity: render the GRPO rollout prompt, the eval prompt and the SFT prompt prefix with the
real Qwen3-VL processor (CPU only, no weights) and compare token ids. READ-ONLY."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import sys, json
sys.path.insert(0, _release_path('CIRCE/circe'))
from transformers import AutoProcessor
from PIL import Image
import train_grpo, vlm_grpo_patch, train_sft, eval_medforge
from build_sft_data import SYS
BASE = _release_path('models/Qwen3-VL-8B-Instruct')
proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
RL = _release_path('AAAI/MedForge++/mfp_v2_rl_crossforgery_strict.jsonl')
rec = json.loads(open(RL).readline())
img = rec["images"][0]
# GRPO
row = train_grpo._rows_from_file.__wrapped__(RL, train_grpo.USER1) if hasattr(train_grpo._rows_from_file, "__wrapped__") else None
rows = []
for i, l in enumerate(open(RL)):
    if i >= 1: break
    tmp = _release_path('CIRCE/v3_audit/code_scripts/out/_one.jsonl')
    open(tmp, "w").write(l)
    rows = train_grpo.build_prompts(tmp)
g_text = proc.apply_chat_template(vlm_grpo_patch._clean_msgs(rows[0]["prompt"]), tokenize=False, add_generation_prompt=True)
# eval
e_msgs = [{"role": "system", "content": SYS},
          {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": eval_medforge.USER1}]}]
e_text = proc.apply_chat_template(e_msgs, tokenize=False, add_generation_prompt=True)
# SFT (full conversation) -> take prefix up to first assistant header
s_msgs = train_sft._structure_messages(rec["messages"], rec["images"])
s_full = proc.apply_chat_template(s_msgs, tokenize=False, add_generation_prompt=False)
hdr = "<|im_start|>assistant\n"
s_prefix = s_full[: s_full.index(hdr) + len(hdr)]
print("GRPO==EVAL text:", g_text == e_text)
print("GRPO==SFT-prefix text:", g_text == s_prefix)
if g_text != e_text or g_text != s_prefix:
    import difflib
    for a, b, n in ((g_text, e_text, "eval"), (g_text, s_prefix, "sft")):
        if a != b:
            sm = difflib.SequenceMatcher(None, a, b)
            for op in sm.get_opcodes():
                if op[0] != "equal":
                    print(n, op[0], repr(a[op[1]:op[2]][:200]), "->", repr(b[op[3]:op[4]][:200]))
im = Image.open(img).convert("RGB")
ids_g = proc(text=[g_text], images=[[im]], return_tensors="pt")["input_ids"][0]
ids_e = proc(text=[e_text], images=[[im]], return_tensors="pt")["input_ids"][0]
print("prompt tokens GRPO/EVAL:", len(ids_g), len(ids_e), "identical ids:", bool((len(ids_g) == len(ids_e)) and (ids_g == ids_e).all()))
print("image size:", im.size, "image_pad tokens:", int((ids_g == 151655).sum()))
# SFT training sequence + label coverage with the real collator
coll = train_sft.build_collator(proc)
batch = coll([rec])
lab = batch["labels"][0]; ids = batch["input_ids"][0]
print("SFT seq len:", len(ids), "supervised tokens:", int((lab != -100).sum()), "truncated at 4096:", len(ids) >= 4096)
# assistant turn-1 length in tokens (what GRPO must reproduce)
a1 = rec["messages"][2]["content"]
print("turn-1 assistant tokens:", len(proc.tokenizer(a1)["input_ids"]) + 1, "(+<|im_end|>)")
print("GRPO user text == eval user text:", train_grpo.USER1 == eval_medforge.USER1)
print(repr(g_text[-420:]))
