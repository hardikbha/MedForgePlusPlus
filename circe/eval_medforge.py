"""
CIRCE evaluation harness — measures whether we beat MedForge on EVERY metric.

Metrics (all computed side-by-side with the MedForge coverage reference):
  - verdict Accuracy, F1-macro (real vs deepfake)
  - fake-recall (recall of the 'deepfake' class) — the axis where Veritas collapses to 0.065
  - box mIoU (mean IoU over GT-boxed samples)     <- CIRCE's headline localization metric
  - IoU@0.5 (fraction of boxed samples with IoU>0.5)
  - MedForge coverage (mean |pred∩gt|/|gt|)        <- the baseline reward, shown for comparison
  - parse rate (well-formed verdict / box)

`--self_test` validates the metric math on synthetic data (CPU, no GPU/model).
`--adapter <ckpt> --eval <jsonl>` runs the real evaluation on a GPU node.
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, glob, argparse, sys, re
sys.path.insert(0, os.path.dirname(__file__))
from circe_reward import parse_output, iou, medforge_cov

DEEPFAKE_TYPES = {"remove", "inject", "deepfake", "removal", "injection", "edit"}  # 'edit' = MedForge implant


# ---------------- pure metric functions (unit-testable) ----------------
def prf_macro(preds, gts, pos="deepfake", neg="real"):
    """returns dict acc, f1_macro, fake_recall."""
    n = len(gts)
    acc = sum(p == g for p, g in zip(preds, gts)) / n if n else 0.0
    def prf(cls):
        tp = sum(p == cls and g == cls for p, g in zip(preds, gts))
        fp = sum(p == cls and g != cls for p, g in zip(preds, gts))
        fn = sum(p != cls and g == cls for p, g in zip(preds, gts))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return prec, rec, f1
    _, rec_pos, f1_pos = prf(pos)
    _, _, f1_neg = prf(neg)
    return {"acc": acc, "f1_macro": (f1_pos + f1_neg) / 2, "fake_recall": rec_pos}


def loc_metrics(pred_boxes, gt_boxes):
    """box mIoU, IoU@0.5, MedForge coverage — over samples that HAVE a GT box."""
    ious, covs, hit = [], [], 0
    n = 0
    for pb, gb in zip(pred_boxes, gt_boxes):
        if not gb:
            continue
        n += 1
        i = iou(pb, gb) if pb else 0.0
        ious.append(i)
        covs.append(medforge_cov(pb, gb) if pb else 0.0)
        hit += int(i > 0.5)
    return {"n_boxed": n,
            "mIoU": sum(ious) / n if n else 0.0,
            "IoU@0.5": hit / n if n else 0.0,
            "medforge_coverage": sum(covs) / n if n else 0.0}


_CONF_RE = re.compile(r'[Cc]onfidence["\s:=]*\s*(0?\.\d+|1(?:\.0+)?|0)')


def parse_conf(text):
    """extract stated 'Confidence: 0.xx' (None if absent)."""
    m = _CONF_RE.search(text)
    if not m:
        return None
    return min(1.0, max(0.0, float(m.group(1))))


def calibration_metrics(preds, gts, confs, n_bins=10):
    """Brier score + Expected Calibration Error over stated confidences.
    conf is the model's P(its own verdict correct); correctness is 1 if verdict matches GT."""
    ps = [(c, 1.0 if p == g else 0.0) for p, g, c in zip(preds, gts, confs) if c is not None]
    n = len(ps)
    if not n:
        return {"n_conf": 0, "brier": None, "ece": None, "conf_rate": 0.0, "mean_conf": None}
    brier = sum((c - y) ** 2 for c, y in ps) / n
    # ECE: bin by confidence, |acc - conf| weighted by bin size
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        bin_items = [(c, y) for c, y in ps if (lo < c <= hi or (b == 0 and c == 0))]
        if not bin_items:
            continue
        bacc = sum(y for _, y in bin_items) / len(bin_items)
        bconf = sum(c for c, _ in bin_items) / len(bin_items)
        ece += (len(bin_items) / n) * abs(bacc - bconf)
    return {"n_conf": n, "brier": round(brier, 4), "ece": round(ece, 4),
            "conf_rate": round(n / len(preds), 4), "mean_conf": round(sum(c for c, _ in ps) / n, 4)}


def self_test():
    # perfect predictions -> acc 1, mIoU 1 ; MedForge-style lazy-large -> coverage 1 but low mIoU
    gt_v = ["deepfake", "deepfake", "real", "real"]
    gt_b = [[10, 10, 30, 30], [50, 50, 70, 70], None, None]
    perfect = prf_macro(gt_v, gt_v)
    assert abs(perfect["acc"] - 1.0) < 1e-9 and abs(perfect["f1_macro"] - 1.0) < 1e-9
    # a detector that always says 'real' -> fake_recall 0 (the collapse signature)
    collapse = prf_macro(["real"] * 4, gt_v)
    assert collapse["fake_recall"] == 0.0
    lm_perfect = loc_metrics([[10, 10, 30, 30], [50, 50, 70, 70]], [[10, 10, 30, 30], [50, 50, 70, 70]])
    assert abs(lm_perfect["mIoU"] - 1.0) < 1e-9 and lm_perfect["IoU@0.5"] == 1.0
    lazy = [[0, 0, 200, 200], [0, 0, 200, 200]]
    lm_lazy = loc_metrics(lazy, [[10, 10, 30, 30], [50, 50, 70, 70]])
    assert lm_lazy["medforge_coverage"] > 0.99 and lm_lazy["mIoU"] < 0.05
    print("[self_test] PASS — acc/f1/fake_recall + mIoU/IoU@0.5/coverage all correct")
    print(f"  perfect={perfect}  collapse_fake_recall={collapse['fake_recall']}")
    print(f"  lazy-large: coverage={lm_lazy['medforge_coverage']:.3f} but mIoU={lm_lazy['mIoU']:.3f}")


# ---------------- GPU evaluation ----------------
# single source of truth: use the EXACT prompt the model was trained with (train/inference match)
from build_sft_data import SYS, USER1 as _USER1_TRAIN
USER1 = _USER1_TRAIN.replace("<image>", "").strip()
# must match train_grpo.CONF_SUFFIX so the calib model sees the prompt it was trained on
CONF_SUFFIX = " Finally, on a new line output your calibrated confidence as 'Confidence: 0.xx'."


def run_eval(args):
    import torch, time
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    from PIL import Image

    rows = [json.loads(l) for l in open(args.eval)]
    if args.limit:
        rows = rows[:args.limit]
    proc = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"   # required for batched generation
    # SDPA attention -> big speedup on the long (image-token) context vs eager
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        attn_implementation="sdpa")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    pv, gv, pb, gb, pc = [], [], [], [], []
    n_parsed_v, n_parsed_b = 0, 0
    gens = []
    user_text = USER1 + (CONF_SUFFIX if getattr(args, "conf", False) else "")
    B = args.batch
    t0 = time.time()
    for bi in range(0, len(rows), B):
        batch = rows[bi:bi + B]
        texts, imgs = [], []
        for m in batch:
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": [{"type": "image", "image": m["image"]},
                                                 {"type": "text", "text": user_text}]}]
            texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            imgs.append([Image.open(m["image"]).convert("RGB")])
        inputs = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
        outs = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        for m, out in zip(batch, outs):
            gt_v = "deepfake" if m["type"] in DEEPFAKE_TYPES else "real"
            p = parse_output(out)
            n_parsed_v += int(p["verdict"] is not None)
            n_parsed_b += int(p["box"] is not None)
            pv.append(p["verdict"] or "real"); gv.append(gt_v)
            pb.append(p["box"]); gb.append(m.get("box"))
            pc.append(parse_conf(out))
            gens.append({"image": m["image"], "gt_verdict": gt_v, "gt_box": m.get("box"),
                         "subset": m.get("subset"), "type": m["type"], "output": out})
        done = min(bi + B, len(rows))
        print(f"  [{done}/{len(rows)}] {done/(time.time()-t0):.2f} samp/s", flush=True)

    res = {**prf_macro(pv, gv), **loc_metrics(pb, gb),
           **({"calibration": calibration_metrics(pv, gv, pc)} if getattr(args, "conf", False) else {}),
           "n": len(rows),
           "parse_verdict_rate": n_parsed_v / len(rows) if rows else 0.0,
           "parse_box_rate": n_parsed_b / len(rows) if rows else 0.0}
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[saved] {args.out}")
    if args.save_gen:
        with open(args.save_gen, "w") as f:
            for g in gens:
                f.write(json.dumps(g) + "\n")
        print(f"[saved gens] {args.save_gen} ({len(gens)})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--eval", help="eval jsonl with {image,type,box}")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default=_release_path('models/Qwen3-VL-4B-Instruct'))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=448)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_gen", default=None, help="jsonl of raw generations for LLM-judge eval")
    ap.add_argument("--conf", action="store_true", help="append confidence-suffix prompt + compute Brier/ECE")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    elif args.eval:
        run_eval(args)
    else:
        ap.error("pass --self_test or --eval <jsonl>")


if __name__ == "__main__":
    main()
