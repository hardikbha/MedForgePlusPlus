"""
check_sft_collator_v3.py -- CPU-only check that v3 SFT records go through train_sft.py's collator correctly.

README
------
train_sft.py's `--dry_run` asserts 5-message (two-turn) records, so it rejects the 3-message v3 records by design;
its real training path has no such assertion. This script exercises the real path instead, WITHOUT model weights
and WITHOUT a GPU (CUDA_VISIBLE_DEVICES is cleared before torch is imported):
  1. loads only the Qwen3-VL processor (tokenizer + image processor) from --model;
  2. imports train_sft.build_collator (read-only import of ~/CIRCE/circe/train_sft.py) and collates --n records;
  3. asserts, per record, that the label-unmasked tokens decode to exactly  assistant_target + "<|im_end|>"
     (i.e. only the v3 target is supervised) and that no truncation happened;
  4. prints prompt / image / target token counts.
Run: nice -n 19 ~/.conda/envs/Qwen3.5/bin/python check_sft_collator_v3.py --data <v3data>/sft_id.jsonl --n 16
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import argparse, json, os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, _release_path('CIRCE/circe'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=_release_path('models/Qwen3-VL-8B-Instruct'))
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    assert not torch.cuda.is_available(), "this check must not see a GPU"
    from transformers import AutoProcessor
    from train_sft import build_collator
    rows = [json.loads(l) for l, _ in zip(open(args.data), range(10 ** 9))]
    real = [r for r in rows if "<verdict>real</verdict>" in r["messages"][2]["content"]][: args.n // 2]
    fake = [r for r in rows if "<verdict>deepfake</verdict>" in r["messages"][2]["content"]][: args.n - len(real)]
    batch = real + fake
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = proc.tokenizer
    coll = build_collator(proc)
    pad_id = tok.pad_token_id
    img_id = tok.convert_tokens_to_ids("<|image_pad|>")
    n_ok = 0
    for i in range(0, len(batch), 4):       # small batches -> exercises padding too
        sub = batch[i:i + 4]
        mi = coll(sub)
        for b, ex in enumerate(sub):
            ids, lab = mi["input_ids"][b], mi["labels"][b]
            sup = tok.decode(ids[lab != -100].tolist(), skip_special_tokens=False)
            want = ex["messages"][2]["content"] + "<|im_end|>"
            n_tok = int((ids != pad_id).sum()) if pad_id is not None else ids.numel()
            n_img = int((ids == img_id).sum())
            ok = sup == want and n_tok < 4096
            n_ok += ok
            print(f"[{i + b:02d}] {'OK ' if ok else 'BAD'} tokens={n_tok} image_tokens={n_img} "
                  f"supervised={int((lab != -100).sum())} verdict={'real' if 'verdict>real' in want else 'deepfake'}")
            if not ok:
                print("   supervised:", repr(sup[:300]))
                print("   expected  :", repr(want[:300]))
    print(f"[collator] {n_ok}/{len(batch)} records supervise exactly the v3 target (+<|im_end|>)")
    sys.exit(0 if n_ok == len(batch) else 1)


if __name__ == "__main__":
    main()
