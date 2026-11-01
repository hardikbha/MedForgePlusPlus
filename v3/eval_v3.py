"""
eval_v3.py -- batched greedy evaluation of a Qwen3-VL (+LoRA) checkpoint on MF-Clean / legacy eval files.

README
------
GPU script (launch through ~/CIRCE/v3/sched, never by hand on a shared GPU). Mirrors ~/CIRCE/circe/eval_medforge.py
(same model loading: AutoModelForImageTextToText bf16, device_map auto, SDPA, PeftModel adapter; left padding;
batched greedy generate; same flags --base/--adapter/--eval/--out/--save_gen/--batch/--max_new/--limit) with:
  --prompt v3        prompts_v3.SYS_V3 / USER_V3 (default; MF-Clean / CT-GRPO checkpoints), max_new 384
  --prompt legacy    build_sft_data.SYS / USER1 (old CIRCE reasoning-first checkpoints), max_new 448
  --prompt legacy_v2 build_sft_data.SYS_V2 / USER1_V2 (CIRCE-v2 evidence-first checkpoints), max_new 448
  --prompt medforge  MedForge-Reasoner's own prompt, verbatim as in eval_official_medforge_fair.py, max_new 2048
Outputs are parsed with prompts_v3.parse_v3, falling back to circe_reward.parse_output (v2 / legacy / MedForge
syntax) -- the same stats_v3.parse_any used by the CPU scorer.

--save_gen  one JSON row per image: EVERY field of the eval row (image, type, gen, box, organ, group, split, disease,
            twin, orig, ...) + gt_verdict, gt_box, cls, prompt, output (trailing pad tokens stripped), n_tokens,
            truncated (no EOS within max_new), pred_verdict, pred_box, pred_fields, well_formed, format.
            Written incrementally (flushed per batch); --resume skips images already in the file.
--out       metrics JSON from stats_v3.full_report (group-bootstrap CIs, per organ / generator / class, twin
            consistency, rationale fields if --fields is given); a markdown version is written next to it (.md).
--shard i/n evaluate rows i, i+n, i+2n, ... (run n processes on n GPUs, then `cat` the gen files and score them
            with stats_v3.py --gen).
CPU testing only:  --cpu (fp32 on CPU) with a tiny random model, or --dry_run oracle|const_real|const_fake|garbage
                   (no model at all; exercises row bookkeeping, saving, resume and scoring).

Example (scheduler task line):
  cd ${MEDFORGE_WORKSPACE}/CIRCE/v3 && ~/.conda/envs/Qwen3.5/bin/python -u eval_v3.py --base ~/models/Qwen3-VL-8B-Instruct \
     --adapter <ckpt> --eval ~/AAAI/MedForge++/mfclean/select_core.jsonl --prompt v3 \
     --save_gen <dir>/gen_X.jsonl --out <dir>/metrics_X.json
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, _release_path('CIRCE/circe'))
from prompts_v3 import SYS_V3, USER_V3, format_target  # noqa: E402
import stats_v3  # noqa: E402

DEEPFAKE_TYPES = {"remove", "inject", "deepfake", "removal", "injection", "edit", "implant"}

# verbatim from eval_official_medforge_fair.py (itself verbatim from the MedForge-Reasoner HF README "Usage")
MEDFORGE_SYS = (
    "You are an expert in medical image forensics. Analyze the provided image "
    "to determine if it is a deepfake or authentic. First, perform a step-by-step "
    "examination of the image content, looking for artifacts, inconsistencies, or "
    "biological implausibilities. Use <think> tags to articulate your reasoning process. "
    "If you identify manipulated regions, localize them using bounding boxes within your reasoning. "
    "Conclude your analysis with a final classification."
)
MEDFORGE_USER = "Is this image deepfake or real?"


def get_prompt(mode):
    """-> (system, user, default max_new)."""
    if mode == "v3":
        return SYS_V3, USER_V3, 384
    if mode == "medforge":
        return MEDFORGE_SYS, MEDFORGE_USER, 2048
    import build_sft_data as B
    if mode == "legacy":
        return B.SYS, B.USER1.replace("<image>", "").strip(), 448
    if mode == "legacy_v2":
        return B.SYS_V2, B.USER1_V2.replace("<image>", "").strip(), 448
    raise ValueError(mode)


def gt_of(m):
    gt_v = "deepfake" if m["type"] in DEEPFAKE_TYPES else "real"
    return gt_v, (m.get("box") if gt_v == "deepfake" else None)


def make_row(m, out, mode, n_tokens=None, truncated=None):
    gt_v, gt_box = gt_of(m)
    p = stats_v3.parse_any(out)
    row = dict(m)
    row.update({"gt_verdict": gt_v, "gt_box": gt_box, "cls": stats_v3.CLS_OF.get(m["type"], "fake"),
                "prompt": mode, "output": out, "n_tokens": n_tokens, "truncated": truncated,
                "pred_verdict": p["verdict"], "pred_box": p["box"], "pred_fields": p["fields"],
                "well_formed": p["well_formed"], "format": p["format"]})
    return row


def strip_pad(text, pad_tokens=("<|endoftext|>",)):
    t = text
    changed = True
    while changed:
        changed = False
        for pt in pad_tokens:
            if t.endswith(pt):
                t = t[: -len(pt)]
                changed = True
    return t


def dry_output(m, kind):
    gt_v, gt_box = gt_of(m)
    if kind == "oracle" and isinstance(m.get("fields"), dict) and gt_v == "deepfake":
        return format_target(gt_v, gt_box, m["fields"]["type"], m["fields"]["intensity"])
    if kind == "oracle":
        ftype = {"edit": "implant", "inject": "implant", "remove": "removal"}.get(m["type"], "implant")
        return format_target(gt_v, gt_box, ftype, "brighter" if ftype == "implant" else "flattened")
    if kind == "const_real":
        return format_target("real")
    if kind == "const_fake":
        return format_target("deepfake", [100, 100, 300, 300], "implant", "brighter")
    return "I cannot tell."


def run_eval(args):
    rows = [json.loads(l) for l in open(args.eval) if l.strip()]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        rows = rows[i::n]
    if args.limit:
        rows = rows[:args.limit]
    sys_p, user_p, max_new_default = get_prompt(args.prompt)
    max_new = args.max_new or max_new_default
    done = set()
    if args.save_gen and args.resume and os.path.exists(args.save_gen):
        done = {json.loads(l)["image"] for l in open(args.save_gen) if l.strip()}
        print(f"[resume] {len(done)} images already in {args.save_gen}")
    todo = [m for m in rows if m["image"] not in done]
    fh = open(args.save_gen, "a" if (args.resume and done) else "w") if args.save_gen else None
    gens = []
    t0 = time.time()

    def emit(rs):
        for r in rs:
            gens.append(r)
            if fh:
                fh.write(json.dumps(r) + "\n")
        if fh:
            fh.flush()

    if args.dry_run:
        for bi in range(0, len(todo), args.batch):
            emit([make_row(m, dry_output(m, args.dry_run), args.prompt) for m in todo[bi:bi + args.batch]])
    elif todo:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from PIL import Image
        proc = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
        proc.tokenizer.padding_side = "left"   # required for batched generation
        if args.cpu:
            model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.float32,
                                                                device_map={"": "cpu"}, trust_remote_code=True)
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                args.base, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
                attn_implementation="sdpa")
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {proc.tokenizer.convert_tokens_to_ids("<|im_end|>")}
        pad_tok = proc.tokenizer.pad_token or "<|endoftext|>"
        B = args.batch
        for bi in range(0, len(todo), B):
            batch = todo[bi:bi + B]
            texts, imgs = [], []
            for m in batch:
                msgs = [{"role": "system", "content": sys_p},
                        {"role": "user", "content": [{"type": "image", "image": m["image"]},
                                                     {"type": "text", "text": user_p}]}]
                texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                imgs.append([Image.open(m["image"]).convert("RGB")])
            inputs = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            new = gen[:, inputs["input_ids"].shape[1]:]
            outs = proc.batch_decode(new, skip_special_tokens=False)
            rs = []
            for m, out, ids in zip(batch, outs, new.tolist()):
                stop = next((k for k, t in enumerate(ids) if t in eos), None)
                rs.append(make_row(m, strip_pad(out, (pad_tok, "<|endoftext|>")), args.prompt,
                                   n_tokens=(stop + 1) if stop is not None else len(ids), truncated=stop is None))
            emit(rs)
            d = min(bi + B, len(todo))
            print(f"  [{d}/{len(todo)}] {d / (time.time() - t0):.2f} samp/s", flush=True)
    if fh:
        fh.close()
        print(f"[saved gens] {args.save_gen}")
    # score everything that is in the gen file (incl. resumed rows) with the CPU scorer
    if args.save_gen:
        recs = stats_v3.load_gen(args.save_gen, stats_v3.load_fields(args.fields))
    else:
        recs = stats_v3.attach_twins([stats_v3.normalize(r, stats_v3.load_fields(args.fields)) for r in gens])
    rep = stats_v3.full_report(recs, args.B, stats_v3.SEED, os.path.basename(args.save_gen or args.eval))
    P = rep["overall"]["point"]
    print(json.dumps({k: P[k] for k in ("acc", "bal_acc", "f1_macro", "rec_real", "rec_implant", "rec_removal",
                                         "mIoU", "IoU@0.5", "twin", "parse_rate", "well_formed_rate")}, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=1)
        with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
            f.write(stats_v3.md_report(rep) + "\n")
        print(f"[saved] {args.out}")
    return rep


def self_test():
    """CPU: prompt strings match the reference scripts; dry-run end-to-end on a synthetic twin set."""
    import tempfile
    import eval_official_medforge_fair  # noqa: F401  (sets eval_medforge.SYS/USER1 to MedForge's prompt)
    import eval_medforge
    assert (MEDFORGE_SYS, MEDFORGE_USER) == (eval_medforge.SYS, eval_medforge.USER1), "medforge prompt drifted"
    import importlib
    importlib.reload(eval_medforge)  # restore CIRCE legacy prompt globals
    s, u, _ = get_prompt("legacy")
    assert (s, u) == (eval_medforge.SYS, eval_medforge.USER1), "legacy prompt differs from eval_medforge"
    assert get_prompt("v3")[:2] == (SYS_V3, USER_V3)
    d = tempfile.mkdtemp(prefix="eval_v3_selftest_")   # honours $TMPDIR
    ev = os.path.join(d, "ev.jsonl")
    with open(ev, "w") as f:
        for g in range(6):
            real = f"/x/g{g}_real.jpg"
            f.write(json.dumps({"image": real, "type": "real", "gen": "real", "box": None, "organ": "fundus",
                                "group": f"G{g}", "split": "test", "disease": "d", "twin": None, "orig": "/o/r"}) + "\n")
            for t, b in (("edit", [10, 10, 200, 200]), ("remove", [500, 500, 700, 600])):
                f.write(json.dumps({"image": f"/x/g{g}_{t}.jpg", "type": t, "gen": "gx", "box": b, "organ": "fundus",
                                    "group": f"G{g}", "split": "test", "disease": "d", "twin": real,
                                    "orig": "/o/f"}) + "\n")
    res = {}
    for kind in ("oracle", "const_real", "const_fake", "garbage"):
        a = argparse.Namespace(eval=ev, shard=None, limit=0, prompt="v3", max_new=None, save_gen=os.path.join(d, kind + ".jsonl"),
                               resume=False, dry_run=kind, batch=4, fields=None, B=200, out=os.path.join(d, kind + ".json"),
                               base=None, adapter=None, cpu=False)
        res[kind] = run_eval(a)["overall"]["point"]
    assert res["oracle"]["acc"] == 1 and res["oracle"]["twin"] == 1 and res["oracle"]["field_location"] == 1
    for k in ("const_real", "const_fake"):
        assert res[k]["bal_acc"] == 0.5 and res[k]["twin"] == 0
    assert res["garbage"]["parse_rate"] == 0 and res["garbage"]["acc"] == 0
    # resume: truncate the oracle file to 7 rows and resume -> 18 rows, metrics unchanged
    p = os.path.join(d, "oracle.jsonl")
    lines = open(p).read().splitlines()[:7]
    open(p, "w").write("\n".join(lines) + "\n")
    a = argparse.Namespace(eval=ev, shard=None, limit=0, prompt="v3", max_new=None, save_gen=p, resume=True,
                           dry_run="oracle", batch=4, fields=None, B=200, out=None, base=None, adapter=None, cpu=False)
    r2 = run_eval(a)["overall"]
    assert r2["counts"]["n"] == 18 and r2["point"]["acc"] == 1 and len(open(p).read().splitlines()) == 18
    row = json.loads(open(p).readline())
    for k in ("image", "type", "gen", "box", "organ", "group", "split", "disease", "twin", "orig", "gt_verdict",
              "gt_box", "output", "pred_verdict", "well_formed", "format"):
        assert k in row, k
    # shard: 0/2 and 1/2 partition the rows exactly
    got = []
    for sh in ("0/2", "1/2"):
        ps = os.path.join(d, f"sh{sh[0]}.jsonl")
        a = argparse.Namespace(eval=ev, shard=sh, limit=0, prompt="v3", max_new=None, save_gen=ps, resume=False,
                               dry_run="oracle", batch=4, fields=None, B=50, out=None, base=None, adapter=None, cpu=False)
        run_eval(a)
        got += [json.loads(l)["image"] for l in open(ps)]
    assert sorted(got) == sorted(json.loads(l)["image"] for l in open(ev)) and len(got) == len(set(got))
    print("[eval_v3 self_test] PASS: prompts == reference scripts (v3/legacy/medforge); dry-run oracle acc 1 twin 1, "
          "constant predictors bal acc 0.5 twin 0, garbage parse 0; resume appends to 18 rows; all GT fields saved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--eval", help="eval jsonl (MF-Clean rows or legacy {image,type,box})")
    ap.add_argument("--base", default=_release_path('models/Qwen3-VL-8B-Instruct'))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompt", default="v3", choices=["v3", "legacy", "legacy_v2", "medforge"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=None, help="default: v3 384, legacy 448, medforge 2048")
    ap.add_argument("--out", default=None, help="metrics json (+ .md next to it)")
    ap.add_argument("--save_gen", default=None, help="per-image generations jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fields", default=None, help="GT evidence-field jsonl for rationale accuracy")
    ap.add_argument("--B", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--cpu", action="store_true", help="CPU fp32 (tests with a tiny model only)")
    ap.add_argument("--dry_run", default=None, choices=["oracle", "const_real", "const_fake", "garbage"])
    args = ap.parse_args()
    if args.self_test:
        self_test()
    elif args.eval:
        if args.prompt == "medforge" and args.max_new and args.max_new < 2048:
            print("[warn] MedForge-Reasoner's documented budget is 2048 new tokens", file=sys.stderr)
        run_eval(args)
    else:
        ap.error("pass --self_test or --eval <jsonl>")


if __name__ == "__main__":
    main()
