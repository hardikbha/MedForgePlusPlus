"""Bound the number of verdict-disagreement groups per 5-step logging window from logged reward_std.
reward_classification_dominant: correct ~ +4.4 (+<=0.25 loc, -1 contradiction), wrong -4 (-1 contradiction).
std (unbiased, G=8) of a group with c correct / 8-c wrong at +4.4 / -4: 8.4*sqrt(c(8-c)/56) in [2.97, 4.49].
A group with no verdict disagreement has std <= ~0.5 (loc bonus spread <=0.25, one contradiction -1 gives 0.35)."""
import ast, sys, math, json
G = 8
def std_split(c, hi=4.4, lo=-4.0):
    v = [hi]*c + [lo]*(G-c); m = sum(v)/G
    return math.sqrt(sum((x-m)**2 for x in v)/(G-1))
SMIN = min(std_split(c) for c in range(1, G)); SMAX = max(std_split(c) for c in range(1, G))
NONV_MAX = 0.5
def windows(path):
    out = []
    for line in open(path, errors='replace').read().replace('\r', '\n').split('\n'):
        line = line.strip()
        if line.startswith("{'loss'"):
            d = {k: float(v) for k, v in ast.literal_eval(line).items()}
            if 'reward_std' in d: out.append(d)
    return out
print(f"per-group std of a verdict-split group in [{SMIN:.3f},{SMAX:.3f}] (c=7/1 -> {std_split(7):.3f}, c=4 -> {std_split(4):.3f})")
for p in sys.argv[1:]:
    W = windows(p)
    if not W: continue
    steps_per_win = 5
    lo = hi = 0
    for d in W:
        S = steps_per_win * d['reward_std']
        hi += min(steps_per_win, math.floor(S / SMIN + 1e-9))
        lo += max(0, math.ceil((S - steps_per_win * NONV_MAX) / SMAX - 1e-9))
    n_steps = steps_per_win * len(W)
    gn0 = sum(d.get('grad_norm', 1) == 0 for d in W) / len(W)
    R = sum(d['reward'] for d in W) / len(W)
    # implied rollout verdict accuracy (approx): R ~= 4.4*acc - 4*(1-acc) (+ small loc bonus) -> acc ~= (R+4)/8.4
    acc = (R + 4) / 8.4
    print(json.dumps({"run": p.split('/')[-1], "steps": n_steps,
        "verdict_split_steps_lower": lo, "verdict_split_steps_upper": hi,
        "verdict_split_frac_upper": round(hi / n_steps, 3),
        "zero_std_frac_est(logged grad_norm==0)": round(gn0, 3),
        "nonverdict_only_frac_est": round(max(0.0, 1 - gn0 - hi / n_steps), 3),
        "implied_rollout_acc": round(min(1.0, acc), 3)}))
