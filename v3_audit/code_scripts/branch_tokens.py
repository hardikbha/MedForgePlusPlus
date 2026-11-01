"""Locate the decision tokens inside the templated completion (verdict branch at A1, box digits, final verdict line)."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, sys, re
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(_release_path('models/Qwen3-VL-8B-Instruct'))
rows = [json.loads(l) for l in open(_release_path('AAAI/MedForge++/v3_select/gen_sel_cm_s2grpo.jsonl'))]
seen = {}
for r in rows:
    o = r["output"].split("<|im_end|>")[0]
    m = re.search(r"A1: (\S+ \S+)", o)
    key = m.group(1) if m else None
    if key in seen: continue
    ids = tok(o)["input_ids"]
    toks = [tok.decode([i]) for i in ids]
    # position of first token after "A1:"
    acc = ""; pos_a1 = None; box_pos = []; vline = None
    for j, t in enumerate(toks):
        acc += t
        if pos_a1 is None and acc.endswith("A1:"): pos_a1 = j + 1
    txt = ""; spans = []
    for j, t in enumerate(toks):
        spans.append((len(txt), len(txt) + len(t))); txt += t
    bm = re.search(r'x1="\d+" y1="\d+" x2="\d+" y2="\d+"', txt)
    if bm:
        box_pos = [j for j, (a, b) in enumerate(spans) if a >= bm.start() and b <= bm.end() and toks[j].strip().isdigit()]
    vm = re.search(r"This image is a (deepfake|real image)\.", txt)
    if vm:
        vline = [j for j, (a, b) in enumerate(spans) if a >= vm.start() and b <= vm.end()]
    seen[key] = dict(n_tokens=len(ids), a1_pos=pos_a1, a1_tokens=toks[pos_a1:pos_a1 + 3] if pos_a1 else None,
                     n_box_digit_tokens=len(box_pos), box_digit_positions=(box_pos[0], box_pos[-1]) if box_pos else None,
                     verdict_line_positions=(vline[0], vline[-1]) if vline else None,
                     verdict_line_tokens=[toks[j] for j in vline] if vline else None)
for k, v in seen.items():
    print(repr(k), json.dumps(v, ensure_ascii=False))
