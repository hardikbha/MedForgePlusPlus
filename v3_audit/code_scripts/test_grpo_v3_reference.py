"""CPU unit tests for grpo_v3_reference.py, using real saved generations where possible. READ-ONLY on data.
Run: nice ~/.conda/envs/aiface/bin/python test_grpo_v3_reference.py"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import sys, json, random, collections
import numpy as np
import torch
sys.path.insert(0, _release_path('CIRCE/circe'))
sys.path.insert(0, _release_path('CIRCE/v3_audit/code_scripts'))
from circe_reward import parse_output, iou
from circe_grpo_reward import reward_classification_dominant, _cue_of, FORENSIC_CUES
import grpo_v3_reference as R

V = _release_path('AAAI/MedForge++/v3_select/')
rng = random.Random(0)


def load(f):
    return [json.loads(l) for l in open(V + f)]


def clean(o):
    return o.replace("<|endoftext|>", "")


# ---------------- (a) exact decomposition on real completions ----------------
rows = load("gen_sel_cm_s2grpo.jsonl")
bad = 0
for r in rows:
    t = clean(r["output"]).replace("<|im_end|>", "")
    c = R.classdom_components(t, r["gt_verdict"], r["gt_box"], parse_output, iou, _cue_of, FORENSIC_CUES)
    tot = reward_classification_dominant(completions=[t], gt_verdict=[r["gt_verdict"]], gt_box=[r["gt_box"]])[0]
    bad += abs(round(c["verdict"] + c["loc"] + c["const"] + c["contra"], 4) - tot) > 1e-6
print(f"[a] classdom_components == reward_classification_dominant on {len(rows)} real completions: mismatches={bad}")
assert bad == 0

# synthetic group: advantage parts sum to trl advantage
comp = [dict(verdict=4.0, loc=0.25 * x, const=0.4, contra=0.0, correct=True) for x in (0.1, 0.5, 0.52, 0.9, 0.3, 0.3, 0.7)] +\
       [dict(verdict=-4.0, loc=0.0, const=0.0, contra=0.0, correct=False)]
comp += [dict(verdict=4.0, loc=0.0, const=0.4, contra=0.0, correct=True)] * 8          # zero-std real group
pg, summ, A = R.group_diagnostics(comp, G=8)
Rt = torch.tensor([sum(c[k] for k in ("verdict", "loc", "const", "contra")) for c in comp], dtype=torch.float64).view(-1, 8)
A_trl = ((Rt - Rt.mean(1, keepdim=True)) / (Rt.std(1, keepdim=True) + 1e-4))
assert torch.allclose(A, A_trl)
assert pg[1]["zero_std"] and pg[0]["verdict_split"] and not pg[1]["verdict_split"]
print(f"[a] synthetic: advantage == trl formula; summary={ {k: round(v, 3) for k, v in summ.items()} }")

# proxy groups from 7 CF checkpoints (SFT, 4 interpolations, GRPO, base) on the same 3000 selection images.
# NOT policy samples -- only exercises the diagnostics on real text.
ck = ["gen_sel_cf_s1base.jsonl", "gen_sel_cf_s1replay.jsonl", "gen_sel_cf_a0.10.jsonl", "gen_sel_cf_a0.20.jsonl",
      "gen_sel_cf_a0.30.jsonl", "gen_sel_cf_a0.50.jsonl", "gen_sel_cf_s2grpo.jsonl"]
by = [{r["image"]: r for r in load(f)} for f in ck]
imgs = sorted(set.intersection(*[set(b) for b in by]))
comp = []
for im in imgs:
    for b in by:
        r = b[im]; t = clean(r["output"]).replace("<|im_end|>", "")
        comp.append(R.classdom_components(t, r["gt_verdict"], r["gt_box"], parse_output, iou, _cue_of, FORENSIC_CUES))
pg, summ, _ = R.group_diagnostics(comp, G=len(ck))
print(f"[a] proxy groups (7 CF checkpoints x {len(imgs)} images): " + json.dumps({k: round(v, 3) for k, v in summ.items()}))

# ---------------- (c) coupled pair normalisation ----------------
G = 4
r = torch.tensor([4.4] * G + [-4.0] * G + [4.4] * G + [4.4] * G)     # pair1: forged all right, source all wrong; pair2: both right
A_ind = torch.cat([(g - g.mean()) / (g.std() + 1e-4) for g in r.view(-1, G)])
A_pair = R.pair_coupled_advantages(r, G)
assert A_ind.abs().sum() == 0 and A_pair[:2 * G].abs().sum() > 0 and A_pair[2 * G:].abs().sum() == 0
cf = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool); cs = torch.tensor([[1, 0, 0, 0]], dtype=torch.bool)
rf, rs = R.pair_joint_rewards(cf, cs, torch.tensor([[0.1, 0.2, 0.0, 0.0]]), G)
assert rf[0, 0] > rf[0, 3] and rs[0, 0] > rs[0, 1]
print(f"[c] independent groups give |A|={A_ind.abs().sum():.1f}; pair-coupled gives |A|={A_pair.abs().sum():.2f} on the "
      f"constant-but-different pair and 0 on the agreeing pair; joint rewards forged={rf.tolist()} source={rs.tolist()}")

# ---------------- (d) token masks on real completions ----------------
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(_release_path('models/Qwen3-VL-8B-Instruct'))
stats = collections.Counter()
for r in rows:
    body = clean(r["output"]).split("<|im_end|>")[0]
    ids = tok(body, add_special_tokens=False)["input_ids"]
    ts = [tok.decode([i]) for i in ids]
    mv, mb = R.token_masks_legacy(ts)
    p = parse_output(body)
    stats["n"] += 1
    stats["disjoint"] += all(not (a and b) for a, b in zip(mv, mb))
    stats["has_verdict_mask"] += sum(mv) > 0
    if p["box"] is not None:
        stats["boxed"] += 1
        digits = sum(len(str(abs(v))) for v in p["box"])
        stats["box_digit_count_match"] += (sum(mb) == digits)
        stats["box_tok"] += sum(mb)
    else:
        stats["unboxed_mb_empty"] += sum(mb) == 0
    stats["v_tok"] += sum(mv); stats["T"] += len(ids)
print(f"[d] token masks on {stats['n']} real completions: disjoint={stats['disjoint']} verdict-mask-nonempty={stats['has_verdict_mask']} "
      f"boxed={stats['boxed']} box-mask==#digits={stats['box_digit_count_match']} unboxed-with-empty-box-mask={stats['unboxed_mb_empty']} | "
      f"mean tokens/completion={stats['T']/stats['n']:.1f} verdict-mask tokens/completion={stats['v_tok']/stats['n']:.2f} "
      f"box-mask tokens/boxed completion={stats['box_tok']/max(1, stats['boxed']):.2f}")
assert stats["disjoint"] == stats["n"]

# decoupled advantages: equivalence + gating
B, T, G = 8, 10, 8
cm = torch.ones(B, T)
mv = torch.zeros(B, T); mv[:, 2] = 1; mbx = torch.zeros(B, T); mbx[:, 5:8] = 1
rv = torch.tensor([4., 4., 4., 4., 4., 4., 4., -4.]); rb = torch.tensor([.1, .5, .9, .3, .3, .7, .2, .8])
valid = rv > 0
At, Av, Ab = R.decoupled_token_advantages(rv, rb, valid, mv, mbx, cm, G, lam_box=0.0)
ref = (rv - rv.mean()).double()[:, None].expand(B, T) * (1 - mbx.double() * valid.double()[:, None])
assert torch.allclose(At, ref)          # lam=0 -> plain Dr.GRPO advantage everywhere except valid box digits (0)
At, Av, Ab = R.decoupled_token_advantages(rv, rb, valid, mv, mbx, cm, G, lam_box=1.0)
assert Ab[7] == 0 and At[7, 5:8].eq(Av[7]).all()                                  # wrong verdict: no box credit
print(f"[d] decoupled advantages: Dr.GRPO equivalence at lam_box=0 OK; wrong-verdict rollout gets no box credit OK; "
      f"A_v={Av.tolist()} A_b={[round(x, 2) for x in Ab.tolist()]}")

# ---------------- (e) change map ----------------
H = W = 256
base = np.random.RandomState(0).normal(120, 10, (H, W)).astype(np.float32)
forged = base.copy(); forged[100:140, 60:120] += 40          # edit region
C = R.change_map(forged, base, (50, 90, 130, 150))
assert C is not None and C[100:140, 60:120].sum() > 0.95
rt = R.diffmass_reward((60, 100, 120, 140), C, (50, 90, 130, 150), iou_fn=iou)
rl = R.diffmass_reward((0, 0, 256, 256), C, (50, 90, 130, 150), iou_fn=iou)
rgt = R.diffmass_reward((50, 90, 130, 150), C, (50, 90, 130, 150), iou_fn=iou)
print(f"[e] synthetic change map: mass in true edit={C[100:140, 60:120].sum():.3f}; reward tight-edit-box={rt:.3f} "
      f"GT-box={rgt:.3f} whole-image={rl:.3f}")
assert rl < rt
# ---------------- (f) DAPO filter ----------------
rw = torch.tensor([4.4] * 8 + [4.4] * 7 + [-4.0] + [4.4, 4.41] * 4)
corr = rw > 0
k_any = R.dapo_keep_mask(rw, 8); k_v = R.dapo_keep_mask(rw, 8, correct=corr, mode="verdict_split")
assert k_any.tolist() == [False, True, True] and k_v.tolist() == [False, True, False]
print(f"[f] dapo_keep_mask any_variance={k_any.tolist()} verdict_split={k_v.tolist()}")
print("ALL TESTS PASSED")
