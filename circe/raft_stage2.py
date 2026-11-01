"""
Stage-2 RAFT (Reward-rAnked Fine-Tuning) — the reliable alternative to GRPO whose training-mode
generation is broken on this stack. Uses the PROVEN eval-generation path (model.eval, cache on) to
sample K reasonings per training image, scores each with the forgery-grounded reward, and keeps the
best CORRECT+GROUNDED reasoning per image -> a self-distilled, reward-selected SFT set.

Fine-tuning on this set = "reward-ranked grounded-reasoning optimization" (our Stage-2 novelty),
which teaches the model from its own best grounded reasoning and transfers to held-out manipulations.

Usage:
  python raft_stage2.py --base <8B> --adapter <sft_ckpt> --data <train.jsonl> \
      --out <raft.jsonl> --k 6 --limit 1500 --temp 0.8
"""
import json, os, argparse, sys
sys.path.insert(0, os.path.dirname(__file__))
from circe_grpo_reward import reward_forgery_grounded
from circe_reward import parse_output
from build_sft_data import SYS, USER1 as _U
USER1 = _U.replace("<image>", "").strip()
DEEPFAKE_TYPES = {"remove", "inject", "deepfake", "removal", "injection", "edit"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)          # SFT-format jsonl (has messages + images + type via gt)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--start", type=int, default=0, help="slice offset (for multi-GPU sharding)")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--max_new", type=int, default=512)
    args = ap.parse_args()

    import torch, time
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    from PIL import Image

    rows = [json.loads(l) for l in open(args.data)]
    rows = rows[args.start:args.start + args.limit] if args.limit else rows[args.start:]
    # ground-truth verdict from the SFT assistant text
    for r in rows:
        asst = " ".join(m["content"] for m in r["messages"] if m["role"] == "assistant")
        r["_gt"] = "deepfake" if "deepfake" in asst else "real"

    proc = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()   # <-- the proven clean-generation state (NOT GRPO's broken train-mode gen)

    kept, seen_ok, t0 = [], 0, time.time()
    B = args.batch
    for bi in range(0, len(rows), B):
        batch = rows[bi:bi + B]
        texts, imgs, metas = [], [], []
        for m in batch:
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": [{"type": "image", "image": m["images"][0]},
                                                 {"type": "text", "text": USER1}]}]
            texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            imgs.append([Image.open(m["images"][0]).convert("RGB")])
            metas.append(m)
        inp = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(model.device)
        best = {i: None for i in range(len(batch))}   # idx -> (reward, completion)
        for _ in range(args.k):
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=args.max_new, do_sample=True,
                                     temperature=args.temp, top_p=0.95)
            outs = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)
            gts = [metas[i]["_gt"] for i in range(len(batch))]
            rs = reward_forgery_grounded(completions=outs, gt_verdict=gts)
            for i, (o, r, g) in enumerate(zip(outs, rs, gts)):
                v = parse_output(o)["verdict"]
                if v == g and (best[i] is None or r > best[i][0]):   # keep best CORRECT rollout
                    best[i] = (r, o)
        for i, m in enumerate(metas):
            if best[i] is not None:
                seen_ok += 1
                out = best[i][1]
                # strip trailing pad/eos noise; keep up to the verdict sentence
                cut = out.find("<|im_end|>")
                clean = out[:cut] if cut > 0 else out
                kept.append({"messages": [{"role": "system", "content": SYS},
                                          {"role": "user", "content": USER1},
                                          {"role": "assistant", "content": clean.strip()}],
                             "images": m["images"]})
        done = min(bi + B, len(rows))
        print(f"  [{done}/{len(rows)}] kept={len(kept)} ok_rate={seen_ok/done:.2f} "
              f"{done/(time.time()-t0):.2f} img/s", flush=True)

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"[raft] wrote {len(kept)} reward-selected examples -> {args.out} "
          f"(from {len(rows)} imgs, keep-rate {len(kept)/len(rows):.2f})")


if __name__ == "__main__":
    main()
