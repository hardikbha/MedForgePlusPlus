"""
Continued-SFT control for GRPO (reviewer B1 / P0-6): turn an RL pool (rl_<regime>.jsonl: twin pairs + singletons)
into SFT rows with exactly the v3 targets build_sft_v3.py would write, so "SFT -> more SFT on the RL images" can be
compared with "SFT -> GRPO on the same images". Each image appears once (reals de-duplicated), order shuffled.

  python build_rl_as_sft.py --rl mfclean/v3data_b/rl_cf.jsonl --out mfclean/v3data_b/contsft_cf.jsonl
"""
import argparse, json, random, collections
from build_sft_v3 import sft_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows, seen = [], set()
    for line in open(args.rl):
        u = json.loads(line)
        if u.get("fake"):
            f = u["fake"]
            if f["image"] not in seen:
                seen.add(f["image"])
                rows.append(sft_record({**f, "type": f["type"]}))
        r = u["real"]
        if r and r["image"] not in seen:
            seen.add(r["image"])
            rows.append(sft_record({"type": "real", "image": r["image"]}))
    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    kinds = collections.Counter("real" if "<verdict>real" in r["messages"][2]["content"] else "fake" for r in rows)
    print(args.out, len(rows), dict(kinds))


if __name__ == "__main__":
    main()
