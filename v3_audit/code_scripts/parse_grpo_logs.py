"""Parse GRPO/SFT logs: tqdm per-step timing + trl metric dicts. READ-ONLY on inputs."""
import re, sys, json, glob, os, ast, statistics as st

TQ = re.compile(r'(\d+)/(\d+) \[(\d+:)?(\d+):(\d+)<')
def elapsed(m):
    h = int(m.group(3)[:-1]) if m.group(3) else 0
    return h*3600 + int(m.group(4))*60 + int(m.group(5))

def parse(path):
    raw = open(path, errors='replace').read().replace('\r', '\n')
    steps = {}   # step -> elapsed seconds (first time seen)
    total = None
    for line in raw.split('\n'):
        for m in TQ.finditer(line):
            n, N = int(m.group(1)), int(m.group(2))
            if N < 20:  # skip weight loading bars etc (750) handled below
                continue
            if 'Loading weights' in line or 'Generating' in line or 'Map' in line:
                continue
            if total is None or N == total:
                total = N
                steps.setdefault(n, elapsed(m))
    metrics = []
    for line in raw.split('\n'):
        line = line.strip()
        if line.startswith("{'loss'") or line.startswith("{'train_runtime'"):
            try:
                d = ast.literal_eval(line)
                metrics.append({k: float(v) for k, v in d.items()})
            except Exception:
                pass
    return steps, total, metrics

def summarize(path):
    steps, total, metrics = parse(path)
    out = {"file": path}
    if steps:
        ks = sorted(steps)
        last = ks[-1]
        out["steps_seen"] = last; out["steps_total"] = total
        out["elapsed_s_at_last"] = steps[last]
        if last > 0:
            out["mean_s_per_step"] = round(steps[last] / last, 2)
        # per-step deltas between consecutive seen steps (whole-second resolution)
        d = [ (steps[b]-steps[a])/(b-a) for a, b in zip(ks, ks[1:]) if b > a ]
        if d:
            out["median_s_per_step"] = round(st.median(d), 2)
            q = sorted(d); out["p10_s"] = round(q[len(q)//10], 2); out["p90_s"] = round(q[9*len(q)//10], 2)
            # first/second half
            half = last // 2
            if half in steps and steps[half] > 0:
                out["s_per_step_first_half"] = round(steps[half]/half, 2)
                out["s_per_step_second_half"] = round((steps[last]-steps[half])/(last-half), 2)
    rt = [m for m in metrics if 'train_runtime' in m]
    if rt:
        out["train_runtime"] = rt[-1]
    lm = [m for m in metrics if 'loss' in m and 'reward_std' in m]
    if lm:
        rs = [m['reward_std'] for m in lm]; rw = [m['reward'] for m in lm]
        gn = [m.get('grad_norm', float('nan')) for m in lm]
        cl = [m.get('completion_length', float('nan')) for m in lm]
        out["n_log_windows"] = len(lm)
        out["reward_std_mean"] = round(st.mean(rs), 4)
        out["reward_std_median"] = round(st.median(rs), 4)
        out["frac_windows_reward_std_lt_0.05"] = round(sum(r < 0.05 for r in rs)/len(rs), 3)
        out["frac_windows_reward_std_gt_0.5"] = round(sum(r > 0.5 for r in rs)/len(rs), 3)
        out["frac_logged_grad_norm_eq_0"] = round(sum(g == 0 for g in gn)/len(gn), 3)
        out["reward_mean"] = round(st.mean(rw), 4)
        out["completion_length_mean"] = round(st.mean(cl), 1)
        out["completion_length_max_window"] = round(max(cl), 1)
    return out

if __name__ == "__main__":
    res = [summarize(p) for p in sys.argv[1:]]
    for r in res:
        print(json.dumps(r))
