"""
Convert an SFT-record jsonl ({"images":[...], "messages":[...]}) into the eval schema
eval_medforge.py expects ({"image":..., "type":..., "box":...}), using the same box-tag
regex build_medforge_protocol.py::dump_eval uses, and audit_protocol.organ_manip_type for
the type label (path-derived, more reliable than re-deriving from the assistant text).

Needed because mfp_v2_val_*.jsonl / mfp_v2_train_*.jsonl are written in SFT-record schema
(they're meant to feed training), but the strict-replay pipeline needs to first EVALUATE
the baseline model on a validation split to find its failures -- that requires eval schema.
"""
import json, re, sys, argparse
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from audit_protocol import organ_manip_type

_BOX_RE = re.compile(r'x1="(\d+)" y1="(\d+)" x2="(\d+)" y2="(\d+)"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n = 0
    with open(args.out, "w") as f:
        for line in open(args.in_path):
            r = json.loads(line)
            img = r["images"][0]
            manip = organ_manip_type(img)
            typ = "real" if manip in (None, "real") else manip  # 'edit' | 'remove' | 'real'
            asst = " ".join(m["content"] for m in r["messages"] if m["role"] == "assistant")
            m = _BOX_RE.search(asst)
            box = [int(x) for x in m.groups()] if m else None
            f.write(json.dumps({"image": img, "type": typ, "box": box}) + "\n")
            n += 1
    print(f"[out] {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
