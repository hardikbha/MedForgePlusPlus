"""Structure of saved eval generations: where is the verdict decided, truncation/EOS, token lengths. READ-ONLY."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, sys, re, collections
sys.path.insert(0, _release_path('CIRCE/circe'))
from circe_reward import parse_output
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(_release_path('models/Qwen3-VL-8B-Instruct'))
A1 = {"Yes — an added": "implant", "No sharp discontinuity": "remove", "No abnormal boundary": "real"}
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    c = collections.Counter(); lens = []; a1_vs_verdict = collections.Counter(); a1_vs_gt = collections.Counter()
    distinct_think = collections.Counter()
    for r in rows:
        o = r["output"]
        body = o.split("<|im_end|>")[0]
        n = len(tok(body)["input_ids"]) + (1 if "<|im_end|>" in o else 0)
        lens.append(n)
        c["has_im_end"] += "<|im_end|>" in o
        p = parse_output(o)
        m = re.search(r"A1: (.{0,22})", o)
        a1 = next((v for k, v in A1.items() if m and m.group(1).startswith(k[:len(m.group(1))][:15])), "other") if m else "none"
        a1_type = a1
        v = p["verdict"]
        a1_vs_verdict[(a1_type, v)] += 1
        a1_vs_gt[(a1_type, r["type"])] += 1
        # canonical think without box coords / description
        t = re.sub(r'x1="\d+" y1="\d+" x2="\d+" y2="\d+"', "BOX", body)
        t = re.sub(r"<description>.*?</description>", "<description/>", t, flags=re.S)
        t = re.sub(r"<conclusion>.*?</conclusion>", "<conclusion/>", t, flags=re.S)
        distinct_think[t] += 1
    lens.sort()
    print(f"== {path.split('/')[-1]} n={len(rows)} has_<|im_end|>={c['has_im_end']} tokens: median={lens[len(lens)//2]} p99={lens[int(.99*len(lens))]} max={lens[-1]}")
    print("  (A1-implied type, parsed verdict):", dict(a1_vs_verdict))
    print("  (A1-implied type, GT type):", dict(a1_vs_gt))
    print(f"  distinct completions after masking box coords/description/conclusion: {len(distinct_think)}; top-3 cover "
          f"{sum(v for _, v in distinct_think.most_common(3))}/{len(rows)}")
