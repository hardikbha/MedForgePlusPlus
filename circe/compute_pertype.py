"""Per-forgery-type (Real/Implant/Remove) In-Domain metrics from saved generations."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from circe_reward import parse_output, iou
from collections import defaultdict

DF = {"remove", "inject", "edit"}
GRP = {"real": "Real", "remove": "Remove", "inject": "Implant", "edit": "Implant"}


def per_type(gen_path):
    g = defaultdict(lambda: {"n": 0, "correct": 0, "iou": []})
    for l in open(gen_path):
        r = json.loads(l)
        t = r["type"]; gtv = "deepfake" if t in DF else "real"
        p = parse_output(r["output"])
        grp = GRP[t]
        g[grp]["n"] += 1
        g[grp]["correct"] += int((p["verdict"] or "real") == gtv)
        if r.get("gt_box"):
            g[grp]["iou"].append(iou(p["box"], r["gt_box"]) if p["box"] else 0.0)
    out = {}
    for grp, d in g.items():
        out[grp] = {"acc": round(100 * d["correct"] / d["n"], 1) if d["n"] else 0,
                    "n": d["n"],
                    "mIoU": round(sum(d["iou"]) / len(d["iou"]), 3) if d["iou"] else None}
    return out


if __name__ == "__main__":
    for tag in ["4b", "8b"]:
        p = f"${MEDFORGE_WORKSPACE}/AAAI/MedForge++/gen_v0.3_{tag}.jsonl"
        if not os.path.exists(p):
            print(f"{tag}: no gen file"); continue
        print(f"=== {tag.upper()} v0.3 (In-Domain, {os.path.basename(p)}) ===")
        for grp in ["Real", "Implant", "Remove"]:
            d = per_type(p).get(grp, {})
            print(f"  {grp:8s} acc={d.get('acc')}%  n={d.get('n')}  mIoU={d.get('mIoU')}")
