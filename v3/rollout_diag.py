"""
v3 diagnostic: how much learning signal does GRPO actually get?

Samples G rollouts per image with the SAME sampling settings as train_grpo.py (T=0.7, top_p=0.95,
max_new=640), scores each rollout with reward_classification_dominant, and reports per-group
statistics: zero-variance groups, groups with any verdict disagreement, pass@G, mean reward std,
split by class / organ / generator / subset. No training; read-only on checkpoints.

Usage (GPU node):
  python rollout_diag.py --adapter <ckpt> --eval <jsonl {image,type,box,...}> --n 480 --G 8 \
      --out <json> --save <jsonl>
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import argparse, json, os, random, sys, time, collections, statistics
sys.path.insert(0, _release_path('CIRCE/circe'))
from circe_reward import parse_output
from circe_grpo_reward import reward_classification_dominant
from build_sft_data import SYS, USER1 as _U

USER1 = _U.replace("<image>", "").strip()
DEEPFAKE_TYPES = {"remove", "inject", "deepfake", "removal", "injection", "edit"}


def organ_of(p):
    for k in ("chest", "brain", "fundus", "kidney", "polyp", "colon"):
        if k in p:
            return k
    return "?"


def stratified(rows, n, seed=0):
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["type"], organ_of(r["image"]), r.get("subset") or "")].append(r)
    for v in by.values():
        rng.shuffle(v)
    keys = sorted(by)
    out, i = [], 0
    while len(out) < min(n, len(rows)):
        k = keys[i % len(keys)]
        if by[k]:
            out.append(by[k].pop())
        i += 1
    return out


def summarize(groups):
    def agg(gs):
        if not gs:
            return {}
        stds = [g["std"] for g in gs]
        return {"n": len(gs),
                "frac_zero_std": round(sum(s < 1e-6 for s in stds) / len(gs), 4),
                "frac_verdict_disagree": round(sum(g["n_correct"] not in (0, g["G"]) for g in gs) / len(gs), 4),
                "frac_all_correct": round(sum(g["n_correct"] == g["G"] for g in gs) / len(gs), 4),
                "frac_all_wrong": round(sum(g["n_correct"] == 0 for g in gs) / len(gs), 4),
                "mean_std": round(statistics.mean(stds), 4),
                "mean_acc_rollout": round(statistics.mean(g["n_correct"] / g["G"] for g in gs), 4)}
    out = {"all": agg(groups)}
    for key in ("type", "organ", "gen", "subset"):
        d = collections.defaultdict(list)
        for g in groups:
            d[g.get(key) or "-"].append(g)
        out[f"by_{key}"] = {k: agg(v) for k, v in sorted(d.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=_release_path('models/Qwen3-VL-8B-Instruct'))
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--imgs_per_batch", type=int, default=2)
    ap.add_argument("--max_new", type=int, default=640)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval) if l.strip()]
    rows = stratified(rows, args.n, args.seed)
    print(f"[diag] {len(rows)} images x G={args.G}", flush=True)

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    from PIL import Image
    torch.manual_seed(args.seed)
    proc = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, device_map="auto",
                                                        trust_remote_code=True, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    gc = model.generation_config
    gc.eos_token_id = [151645, 151643]
    gc.pad_token_id = 151643

    groups, t0 = [], time.time()
    fsave = open(args.save, "w") if args.save else None
    B = args.imgs_per_batch
    for bi in range(0, len(rows), B):
        batch = rows[bi:bi + B]
        texts, imgs = [], []
        for m in batch:
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": [{"type": "image", "image": m["image"]},
                                                 {"type": "text", "text": USER1}]}]
            texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            imgs.append([Image.open(m["image"]).convert("RGB")])
        inputs = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=True,
                                 temperature=args.temperature, top_p=args.top_p,
                                 num_return_sequences=args.G)
        outs = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, m in enumerate(batch):
            o = outs[j * args.G:(j + 1) * args.G]
            gt_v = "deepfake" if m["type"] in DEEPFAKE_TYPES else "real"
            rew = reward_classification_dominant(completions=o, gt_verdict=[gt_v] * len(o),
                                                 gt_box=[m.get("box")] * len(o))
            verdicts = [parse_output(t)["verdict"] for t in o]
            n_correct = sum(v == gt_v for v in verdicts)
            g = {"image": m["image"], "type": m["type"], "organ": organ_of(m["image"]),
                 "gen": m.get("gen", ""), "subset": m.get("subset", ""), "G": len(o),
                 "n_correct": n_correct, "rewards": rew, "std": statistics.pstdev(rew),
                 "verdicts": verdicts}
            groups.append(g)
            if fsave:
                fsave.write(json.dumps({**g, "outputs": o}) + "\n")
        done = min(bi + B, len(rows))
        print(f"  [{done}/{len(rows)}] {done / (time.time() - t0):.3f} img/s", flush=True)
    if fsave:
        fsave.close()
    res = {"adapter": args.adapter, "eval": args.eval, "G": args.G, "temperature": args.temperature,
           **summarize(groups)}
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res["all"], indent=2))


if __name__ == "__main__":
    main()
