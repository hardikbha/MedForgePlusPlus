"""
Mix a strict replay set with a random sample of the legal original training pool, per
CIRCE-v2 Phase 9's spec: 70-80% original legal training samples, 20-30% replay samples,
never replay-only fine-tuning.
"""
import json, random, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--legal_pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    replay_rows = [json.loads(l) for l in open(args.replay)]
    n_replay = len(replay_rows)
    n_legal = int(n_replay * (1 - args.replay_frac) / args.replay_frac)

    legal_rows = [json.loads(l) for l in open(args.legal_pool)]
    replay_images = {r["images"][0] for r in replay_rows}
    legal_rows = [r for r in legal_rows if r["images"][0] not in replay_images]

    rng = random.Random(args.seed)
    rng.shuffle(legal_rows)
    legal_sample = legal_rows[:n_legal]

    combined = replay_rows + legal_sample
    rng.shuffle(combined)
    # strip replay-only provenance fields before writing -- train_sft.py's collator only
    # looks at "images"/"messages", extra keys are harmless but keep the file clean
    with open(args.out, "w") as f:
        for r in combined:
            r2 = {"images": r["images"], "messages": r["messages"]}
            f.write(json.dumps(r2) + "\n")

    print(f"[mix] replay={n_replay} legal={len(legal_sample)} total={len(combined)} "
          f"(actual replay frac={n_replay/len(combined):.3f}) -> {args.out}")


if __name__ == "__main__":
    main()
