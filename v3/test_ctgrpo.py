"""
test_ctgrpo.py -- unit tests for ctgrpo_core (CT-GRPO). CPU only, no model weights.

README
------
Run:  CUDA_VISIBLE_DEVICES= nice ~/.conda/envs/Qwen3.5/bin/python ~/CIRCE/v3/test_ctgrpo.py
(plain asserts, no pytest needed; prints one line per test and "ALL N TESTS PASSED").
Covers:
  * reward: every branch of the v3 reward (GT targets, wrong, unparseable, contradictions, IoU, field bonus);
  * advantages: for a FIXED 6-row reward tensor (one twin pair + one singleton, G=2) every --adv_mode is compared
    with values worked out by hand (arithmetic written next to each expected value); 'image' is also compared with
    the exact trl/vlm_grpo_patch formula; every mode is compared with an independent loop-based reference on random
    rewards with SHUFFLED row order (group ids, not reshapes); singleton == image; std_floor; the solved-pair
    offset that motivates pool_channels='verdict';
  * sampler: twins in the same step and contiguous, every pair once per epoch, singletons coupled, seeded;
  * layout check: detects a split twin / broken run;
  * loader guards: non-train split, regime exclusions, twin mismatch;
  * prompt parity: rendered GRPO prompt is a strict prefix of the rendered SFT record (train_sft collator path).
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json
import math
import os
import random
import statistics
import sys
import tempfile
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, _release_path('CIRCE/circe'))
import torch  # noqa: E402

import ctgrpo_core as core  # noqa: E402
from prompts_v3 import format_target  # noqa: E402

EPS = 1e-4
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def close(a, b, tol=1e-6):
    a = [float(x) for x in (a.tolist() if hasattr(a, "tolist") else a)]
    b = [float(x) for x in b]
    assert len(a) == len(b), (a, b)
    assert all(abs(x - y) <= tol for x, y in zip(a, b)), f"\n got {a}\n exp {b}"


# --------------------------------------------------------------------------------------------------------- reward
GT_FAKE = {"gt_verdict": "deepfake", "gt_box": [100, 100, 300, 300], "gt_type": "implant",
           "gt_location": "upper-left", "gt_size": "medium", "gt_intensity": "brighter"}
GT_REAL = {"gt_verdict": "real", "gt_box": None, "gt_type": "none", "gt_location": "none", "gt_size": "none",
           "gt_intensity": "none"}


@test
def reward_branches():
    S = core.score_completion
    t_fake = format_target("deepfake", [100, 100, 300, 300], "implant", "brighter")
    t_real = format_target("real")
    assert S(t_fake, GT_FAKE)["total"] == 1.5                       # +1 + 0.5*IoU(1.0)
    assert S(t_real, GT_REAL)["total"] == 1.5                       # +1 + 0.5 (real, no box)
    assert S(t_real, GT_FAKE)["total"] == -1.0                      # wrong, nothing else
    assert S(t_fake, GT_REAL)["total"] == -1.0                      # wrong (box ignored when wrong)
    assert S("no tags at all", GT_FAKE)["total"] == -1.5            # unparseable
    no_box = t_fake.replace("<box>100 100 300 300</box>", "<none/>")
    assert S(no_box, GT_FAKE)["total"] == 0.5                       # +1 - 0.5 contradiction
    real_box = t_real.replace("<none/>", "<box>1 1 50 50</box>")
    assert S(real_box, GT_REAL)["total"] == 0.5                     # +1 - 0.5 contradiction (no +0.5)
    half = format_target("deepfake", [100, 100, 300, 200], "implant", "brighter")   # IoU = 20000/40000 = 0.5
    o = S(half, GT_FAKE)
    assert abs(o["iou"] - 0.5) < 1e-9 and abs(o["total"] - 1.25) < 1e-9
    # field bonus: location/size of the half box = upper-left / small?  area 200*100/1024^2 = 0.019 -> 'small'
    o = S(half, GT_FAKE, field_bonus=0.1)
    assert o["n_field_scored"] == 4 and o["n_field"] == 3            # type, location, intensity right; size wrong
    assert abs(o["total"] - (1.25 + 0.3)) < 1e-9
    gt_noint = dict(GT_FAKE, gt_intensity=None)                    # unknown GT intensity -> not scored
    o = S(t_fake, gt_noint, field_bonus=0.1)
    assert o["n_field_scored"] == 3 and abs(o["total"] - 1.8) < 1e-9
    assert S(t_real, GT_REAL, field_bonus=0.1)["total"] == 1.5      # field bonus only on correct fakes
    o = S(t_fake, GT_FAKE)
    assert o["r_verdict"] == 1.0 and o["r_loc"] == 0.5


@test
def reward_callable_matches_components():
    rw = core.CTReward(field_bonus=0.0)
    t = format_target("deepfake", [100, 100, 300, 300], "implant", "brighter")
    kw = {k: [GT_FAKE[k], GT_REAL[k]] for k in GT_FAKE}
    kw.update(pair_id=["p", "p"], role=["fake", "real"])
    out = rw(completions=[[{"role": "assistant", "content": t}], [{"role": "assistant", "content": t}]], **kw)
    assert out == [1.5, -1.0]
    rec = rw.pop()
    assert [c["total"] for c in rec["comps"]] == out and rw.last is None


# ----------------------------------------------------------------------------------------------------- advantages
# Fixed reward tensor, G = 2.  Rows: twin pair P (fake f0 f1, real r0 r1) + singleton S (s0 s1).
#   fake : r_v = [ 1, -1], r_l = [0.4, 0.0]  -> r = [1.4, -1.0]   correct = [T, F]
#   real : r_v = [ 1,  1], r_l = [0.5, 0.5]  -> r = [1.5,  1.5]   correct = [T, T]
#   solo : r_v = [-1,  1], r_l = [0.0, 0.5]  -> r = [-1.0, 1.5]   correct = [F, T]
RV = torch.tensor([1., -1., 1., 1., -1., 1.])
RL = torch.tensor([0.4, 0., 0.5, 0.5, 0., 0.5])
COR = torch.tensor([True, False, True, True, False, True])
PIDS = ["P", "P", "P", "P", "S", "S"]
ROLES = ["fake", "fake", "real", "real", "solo", "solo"]
G2 = 2


def lay2():
    return core.layout_group_ids(PIDS, ROLES, G2)


def adv(mode, **kw):
    L = lay2()
    return core.compute_advantages(RV, RL, L["img_gid"], L["pair_gid"], mode, correct=COR, partner=L["partner"], **kw)


@test
def adv_image_hand():
    a = adv("image", scale_std=False)
    # fake: mean(1.4,-1)=0.2 -> [1.2,-1.2]; real constant -> 0; solo: mean(-1,1.5)=0.25 -> [-1.25,1.25]
    close(a["adv"], [1.2, -1.2, 0, 0, -1.25, 1.25])
    close(a["adv_verdict"], [1, -1, 0, 0, -1, 1])                 # r_v centred per image
    close(a["adv_loc"], [0.2, -0.2, 0, 0, -0.25, 0.25])           # r_l centred per image
    a = adv("image", scale_std=True)
    # std(1.4,-1) = 2.4/sqrt(2); std(-1,1.5) = 2.5/sqrt(2); real std 0 -> den 1e-4, numerator 0
    sf, ss = 2.4 / math.sqrt(2) + EPS, 2.5 / math.sqrt(2) + EPS
    close(a["adv"], [1.2 / sf, -1.2 / sf, 0, 0, -1.25 / ss, 1.25 / ss])


@test
def adv_image_equals_trl_formula():
    torch.manual_seed(0)
    G = 8
    R = torch.randn(6 * G)
    pids = [f"p{i // (2 * G)}" for i in range(6 * G)]
    roles = [("fake" if (i // G) % 2 == 0 else "real") for i in range(6 * G)]
    L = core.layout_group_ids(pids, roles, G)
    a = core.compute_advantages(R, torch.zeros_like(R), L["img_gid"], L["pair_gid"], "image")
    # vlm_grpo_patch / trl 0.16.1: (r - mean_g) / (std_g + 1e-4) with view(-1, G)
    m = R.view(-1, G).mean(1).repeat_interleave(G)
    s = R.view(-1, G).std(1).repeat_interleave(G)
    close(a["adv"], ((R - m) / (s + 1e-4)).tolist(), tol=1e-5)


@test
def adv_twin_pooled_hand():
    a = adv("twin_pooled", scale_std=False)
    # verdict channel centred over the pair: mean(1,-1,1,1)=0.5 -> [0.5,-1.5,0.5,0.5]
    # loc channel centred per image: fake mean 0.2 -> [0.2,-0.2]; real mean 0.5 -> [0,0]
    # singleton: pair group == image group -> identical to 'image'
    close(a["adv"], [0.7, -1.7, 0.5, 0.5, -1.25, 1.25])
    close(a["adv_verdict"], [0.5, -1.5, 0.5, 0.5, -1, 1])
    close(a["adv_loc"], [0.2, -0.2, 0, 0, -0.25, 0.25])
    a = adv("twin_pooled", scale_std=True)
    # pooled std of c=[0.7,-1.7,0.5,0.5] (mean 0): sqrt((0.49+2.89+0.25+0.25)/3) = sqrt(3.88/3)
    sp, ss = math.sqrt(3.88 / 3) + EPS, 2.5 / math.sqrt(2) + EPS
    close(a["adv"], [0.7 / sp, -1.7 / sp, 0.5 / sp, 0.5 / sp, -1.25 / ss, 1.25 / ss])
    a = adv("twin_pooled", scale_std=False, pool_channels="all")
    # literal joint normalisation: mean(1.4,-1,1.5,1.5) = 0.85 -> [0.55,-1.85,0.65,0.65]
    close(a["adv"], [0.55, -1.85, 0.65, 0.65, -1.25, 1.25])


@test
def adv_twin_pooled_contrast_case():
    # fake group all WRONG (r=-1), real group all CORRECT (r=1.5): image-level GRPO gives zero everywhere,
    # twin pooling gives a contrastive signal (verdict channel centred at the pair mean 0).
    rv = torch.tensor([-1., -1., 1., 1.]); rl = torch.tensor([0., 0., 0.5, 0.5])
    L = core.layout_group_ids(["P"] * 4, ["fake", "fake", "real", "real"], 2)
    a_img = core.compute_advantages(rv, rl, L["img_gid"], L["pair_gid"], "image")
    close(a_img["adv"], [0, 0, 0, 0])
    a = core.compute_advantages(rv, rl, L["img_gid"], L["pair_gid"], "twin_pooled", scale_std=False)
    close(a["adv"], [-1, -1, 1, 1])                               # c_v = [-1,-1,1,1], c_l = 0
    a = core.compute_advantages(rv, rl, L["img_gid"], L["pair_gid"], "twin_pooled", scale_std=True)
    s = math.sqrt(4 / 3) + EPS                                    # std of [-1,-1,1,1] (n-1=3)
    close(a["adv"], [-1 / s, -1 / s, 1 / s, 1 / s])


@test
def adv_twin_pooled_solved_pair_offset():
    # both twins solved: fake r_v=[1,1], r_l=[0.4,0.2]; real r_v=[1,1], r_l=[0.5,0.5]
    rv = torch.tensor([1., 1., 1., 1.]); rl = torch.tensor([0.4, 0.2, 0.5, 0.5])
    L = core.layout_group_ids(["P"] * 4, ["fake", "fake", "real", "real"], 2)
    a = core.compute_advantages(rv, rl, L["img_gid"], L["pair_gid"], "twin_pooled", scale_std=False)
    close(a["adv"], [0.1, -0.1, 0, 0])                            # only the IoU spread inside the fake group
    a = core.compute_advantages(rv, rl, L["img_gid"], L["pair_gid"], "twin_pooled", scale_std=False,
                                pool_channels="all")
    # 'all': mean(1.4,1.2,1.5,1.5)=1.4 -> [0,-0.2,0.1,0.1]: correct fake rollouts pushed down, reals up (offset bias)
    close(a["adv"], [0.0, -0.2, 0.1, 0.1])


@test
def adv_cogroup_bonus_hand():
    a = adv("cogroup_bonus", scale_std=False, pair_bonus=0.5)
    # rollout pair (f0,r0): both correct -> +0.5 to f0 and r0; (f1,r1): f1 wrong -> none; singleton: no partner
    # fake r = [1.4+0.5, -1] = [1.9,-1], mean 0.45 -> [1.45,-1.45]
    # real r = [1.5+0.5, 1.5] = [2.0,1.5], mean 1.75 -> [0.25,-0.25]
    close(a["adv"], [1.45, -1.45, 0.25, -0.25, -1.25, 1.25])
    close(a["bonus"], [0.5, 0, 0.5, 0, 0, 0])
    a = adv("cogroup_bonus", scale_std=True, pair_bonus=0.5)
    sf, sr, ss = 2.9 / math.sqrt(2) + EPS, 0.5 / math.sqrt(2) + EPS, 2.5 / math.sqrt(2) + EPS
    close(a["adv"], [1.45 / sf, -1.45 / sf, 0.25 / sr, -0.25 / sr, -1.25 / ss, 1.25 / ss])
    a0 = adv("cogroup_bonus", scale_std=True, pair_bonus=0.0)
    close(a0["adv"], adv("image", scale_std=True)["adv"].tolist())  # lambda=0 == image


@test
def adv_gdpo_hand():
    a = adv("gdpo", scale_std=True)
    # fake: A_v = [1,-1]/(2/sqrt2 + eps); A_l = [0.2,-0.2]/(0.4/sqrt2 + eps)
    # real: both channels constant -> 0
    # solo: A_v = [-1,1]/(2/sqrt2 + eps); A_l = [-0.25,0.25]/(0.5/sqrt2 + eps)
    dv, dlf, dls = 2 / math.sqrt(2) + EPS, 0.4 / math.sqrt(2) + EPS, 0.5 / math.sqrt(2) + EPS
    exp = [1 / dv + 0.2 / dlf, -1 / dv - 0.2 / dlf, 0, 0, -1 / dv - 0.25 / dls, 1 / dv + 0.25 / dls]
    close(a["adv"], exp)
    close(a["adv_verdict"], [1 / dv, -1 / dv, 0, 0, -1 / dv, 1 / dv])
    close(adv("gdpo", scale_std=False)["adv"], adv("image", scale_std=False)["adv"].tolist())


@test
def adv_std_floor():
    rv = torch.tensor([1., 1.]); rl = torch.tensor([0.41, 0.40])
    L = core.layout_group_ids(["S", "S"], ["solo", "solo"], 2)
    a = core.compute_advantages(rv, rl, L["img_gid"], L["pair_gid"], "image", std_floor=0.5)
    close(a["adv"], [0.005 / (0.5 + EPS), -0.005 / (0.5 + EPS)])
    a = core.compute_advantages(rv, rl, L["img_gid"], L["pair_gid"], "image")
    s = 0.01 / math.sqrt(2) + EPS                                # tiny IoU spread -> unit-scale advantage
    close(a["adv"], [0.005 / s, -0.005 / s], tol=1e-4)


# independent loop-based reference (pure Python, dict grouping, statistics.stdev)
def ref_adv(rv, rl, img, pair, correct, partner, mode, scale=True, floor=0.0, lam=0.0, pool="verdict"):
    B = len(rv)
    rv = list(rv)
    bonus = [0.0] * B
    if mode == "cogroup_bonus":
        for b in range(B):
            if partner[b] >= 0 and correct[b] and correct[partner[b]]:
                bonus[b] = lam
        rv = [x + y for x, y in zip(rv, bonus)]

    def groups(ids):
        g = defaultdict(list)
        for b, i in enumerate(ids):
            g[i].append(b)
        return g

    def mean_by(x, ids):
        out = [0.0] * B
        for _, bs in groups(ids).items():
            m = sum(x[b] for b in bs) / len(bs)
            for b in bs:
                out[b] = m
        return out

    def std_by(x, ids):
        out = [0.0] * B
        for _, bs in groups(ids).items():
            s = statistics.stdev([x[b] for b in bs]) if len(bs) > 1 else 0.0
            for b in bs:
                out[b] = s
        return out

    def den(s):
        return max(s, floor) + EPS if scale else 1.0
    if mode in ("image", "cogroup_bonus"):
        r = [a + b for a, b in zip(rv, rl)]
        m, s = mean_by(r, img), std_by(r, img)
        return [(r[b] - m[b]) / den(s[b]) for b in range(B)]
    if mode == "twin_pooled":
        mv = mean_by(rv, pair)
        ml = mean_by(rl, pair) if pool == "all" else mean_by(rl, img)
        c = [rv[b] - mv[b] + rl[b] - ml[b] for b in range(B)]
        s = std_by(c, pair)
        return [c[b] / den(s[b]) for b in range(B)]
    mv, sv, ml, sl = mean_by(rv, img), std_by(rv, img), mean_by(rl, img), std_by(rl, img)
    return [(rv[b] - mv[b]) / den(sv[b]) + (rl[b] - ml[b]) / den(sl[b]) for b in range(B)]


@test
def adv_random_vs_reference_shuffled_rows():
    rng = random.Random(1)
    G = 4
    for trial in range(30):
        pids, roles = [], []
        n_pairs, n_solo = rng.randint(1, 3), rng.randint(0, 2)
        for p in range(n_pairs):
            pids += [f"p{p}"] * (2 * G); roles += ["fake"] * G + ["real"] * G
        for s in range(n_solo):
            pids += [f"s{s}"] * G; roles += ["solo"] * G
        L = core.layout_group_ids(pids, roles, G)
        B = len(pids)
        # discrete-ish rewards so that zero-std groups and ties occur
        rv = torch.tensor([rng.choice([1.0, -1.0, -1.5, 1.0]) for _ in range(B)])
        rl = torch.tensor([rng.choice([0.0, 0.5, 0.5 * rng.random(), -0.5]) for _ in range(B)])
        cor = rv > 0
        perm = torch.randperm(B, generator=torch.Generator().manual_seed(trial))   # rows in ANY order
        args = (rv[perm], rl[perm], L["img_gid"][perm], L["pair_gid"][perm])
        inv_partner = torch.empty_like(perm); inv_partner[perm] = torch.arange(B)
        partner = L["partner"][perm]
        partner = torch.where(partner >= 0, inv_partner[partner.clamp(min=0)], partner)
        for mode in core.ADV_MODES:
            for scale in (True, False):
                for pool in ("verdict", "all"):
                    if pool == "all" and mode != "twin_pooled":
                        continue
                    a = core.compute_advantages(*args, mode, correct=cor[perm], partner=partner, scale_std=scale,
                                                std_floor=0.05 if trial % 2 else 0.0, pair_bonus=0.3,
                                                pool_channels=pool)
                    exp = ref_adv(args[0].tolist(), args[1].tolist(), args[2].tolist(), args[3].tolist(),
                                  cor[perm].tolist(), partner.tolist(), mode, scale,
                                  0.05 if trial % 2 else 0.0, 0.3, pool)
                    close(a["adv"], exp, tol=2e-5)
                    close(a["adv"], (a["adv_verdict"] + a["adv_loc"]).tolist(), tol=1e-6)


@test
def adv_singletons_twin_equals_image():
    torch.manual_seed(3)
    G = 8
    pids = [f"s{i // G}" for i in range(4 * G)]
    roles = ["solo"] * (4 * G)
    L = core.layout_group_ids(pids, roles, G)
    rv, rl = torch.randn(4 * G), torch.rand(4 * G)
    a1 = core.compute_advantages(rv, rl, L["img_gid"], L["pair_gid"], "twin_pooled")
    a2 = core.compute_advantages(rv, rl, L["img_gid"], L["pair_gid"], "image")
    close(a1["adv"], a2["adv"].tolist(), tol=1e-6)


# --------------------------------------------------------------------------------------------- layout + sampler
@test
def layout_detects_violations():
    G = 2
    ok = core.layout_group_ids(["P", "P", "P", "P"], ["fake", "fake", "real", "real"], G)
    assert ok["partner"].tolist() == [2, 3, 0, 1] and ok["rollout_idx"].tolist() == [0, 1, 0, 1]
    for pids, roles in [(["P", "P", "Q", "Q"], ["fake", "fake", "solo", "solo"]),          # twin missing
                        (["P", "P", "P", "P"], ["real", "real", "fake", "fake"]),           # real before fake
                        (["P", "Q", "P", "Q"], ["fake", "solo", "fake", "solo"]),           # interleaved
                        (["P", "P", "P", "Q", "Q", "Q"], ["fake"] * 3 + ["solo"] * 3)]:      # run != G
        try:
            core.layout_group_ids(pids, roles, G)
        except ValueError:
            continue
        raise AssertionError(f"layout violation not detected: {pids} {roles}")


def synth_units(n_pairs, n_solo):
    units, rows = [], []
    for p in range(n_pairs):
        units.append((len(rows), len(rows) + 1))
        rows += [{"pair_id": f"p{p}", "role": "fake"}, {"pair_id": f"p{p}", "role": "real"}]
    for s in range(n_solo):
        units.append((len(rows),))
        rows.append({"pair_id": f"s{s}", "role": "solo"})
    return units, rows


@test
def sampler_twins_same_step_contiguous():
    G, K = 8, 2
    units, rows = synth_units(37, 9)
    smp = core.TwinPairSampler(units, G, K, seed=42)
    assert smp.n_slots == 37 + 4 and smp.steps_per_epoch == 20 and len(smp) == 20 * 2 * K * G
    orders = []
    for ep in range(3):
        idx = list(iter(smp))
        assert len(idx) == len(smp)
        steps = [idx[i:i + 2 * K * G] for i in range(0, len(idx), 2 * K * G)]
        seen = Counter()
        for st in steps:
            L = core.layout_group_ids([rows[i]["pair_id"] for i in st], [rows[i]["role"] for i in st], G)
            assert not L["errors"]                         # strict check would have raised
            for i in st[::G]:
                seen[rows[i]["pair_id"], rows[i]["role"]] += 1
        assert all(v == 1 for v in seen.values())         # no image twice in an epoch
        n_pairs_seen = len({p for p, r in seen if r == "fake"})
        assert n_pairs_seen >= 36                          # 41 slots, 40 used: at most one pair dropped
        orders.append(idx)
    assert orders[0] != orders[1]                          # reshuffled every epoch
    smp2 = core.TwinPairSampler(units, G, K, seed=42)
    assert list(iter(smp2)) == orders[0]                   # seeded / reproducible
    assert list(iter(core.TwinPairSampler(units, G, K, seed=7))) != orders[0]


@test
def sampler_set_epoch_replays():
    # [review] transformers' Trainer calls DataLoader.set_epoch(e) -> sampler.set_epoch(e) before each epoch
    units, rows = synth_units(15, 3)
    smp = core.TwinPairSampler(units, 4, 2, seed=9)
    e0, e1 = list(iter(smp)), list(iter(smp))
    smp.set_epoch(1)
    assert list(iter(smp)) == e1
    smp.set_epoch(0)
    assert list(iter(smp)) == e0 and e0 != e1


@test
def eos_mask_both_ids():
    # [review fix, audit B6] mask = up to and including the first of <|im_end|> 151645 / <|endoftext|> 151643
    import train_ctgrpo as T
    ids = torch.tensor([[7, 151645, 151643, 151643], [7, 151643, 151643, 151643], [7, 8, 9, 10]])
    assert T.completion_mask_multi_eos(ids).tolist() == [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]


@test
def cli_refuses_silent_image_fallbacks():
    # [review fix] cogroup_bonus with lambda 0 / gdpo without std scaling are exactly 'image'; beta>0 is ignored
    import contextlib
    import io
    import train_ctgrpo as T
    for extra in (["--adv_mode", "cogroup_bonus"], ["--adv_mode", "gdpo", "--scale_std", "0"], ["--beta", "0.01"]):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                T.parse_args(["--data", "d", "--out", "o"] + extra)
        except SystemExit:
            continue
        raise AssertionError(f"accepted {extra}")


# -------------------------------------------------------------------------------------------------------- loader
def _write(lines):
    fd, p = tempfile.mkstemp(suffix=".jsonl", dir=os.environ.get("CT_TMP", None))
    with os.fdopen(fd, "w") as fh:
        for l in lines:
            fh.write(json.dumps(l) + "\n")
    return p


@test
def loader_guards():
    img = __file__   # any existing file
    real = {"image": img, "type": "real", "gen": "real", "box": None, "organ": "fundus", "group": "g1",
            "split": "train", "twin": None}
    fake = {"image": img + "#f", "type": "remove", "gen": "flux", "box": [10, 10, 100, 100], "organ": "fundus",
            "group": "g1", "split": "train", "twin": img}
    p = _write([{"fake": fake, "real": real}, {"real": dict(real, group="g2")}, dict(real, group="g3")])
    rows, units, st = core.load_rl_rows(p, check_files=False)
    assert st["pairs"] == 1 and st["solo_real"] == 2 and [len(u) for u in units] == [2, 1, 1]
    assert rows[0]["gt_verdict"] == "deepfake" and rows[0]["gt_type"] == "removal" and rows[1]["gt_box"] is None
    assert rows[0]["gt_location"] == "upper-left" and rows[0]["role"] == "fake" and rows[1]["role"] == "real"
    bad = [
        ([{"fake": dict(fake, split="test"), "real": real}], None),               # leakage guard
        ([{"fake": dict(fake, type="edit"), "real": real}], "CF"),                # CF: no implants
        ([{"fake": dict(fake, gen="gemini"), "real": real}], "CM"),               # CM: held-out generator
        ([{"fake": dict(fake, twin="/other.jpg"), "real": real}], None),          # twin mismatch
        ([{"fake": dict(fake, group="g9"), "real": real}], None),                 # group mismatch
        ([{"fake": dict(fake, box=None), "real": real}], None),                   # fake without box
    ]
    for lines, regime in bad:
        try:
            core.load_rl_rows(_write(lines), regime=regime, check_files=False)
        except ValueError:
            continue
        raise AssertionError(f"guard did not fire: {lines} {regime}")
    core.load_rl_rows(_write([{"fake": dict(fake, gen="gemini"), "real": real}]), regime="CF", check_files=False)


# ------------------------------------------------------------------------------------------------ prompt parity
@test
def prompt_parity_with_sft_collator():
    from transformers import AutoProcessor
    from vlm_grpo_patch import _clean_msgs
    from train_sft import _structure_messages
    from prompts_v3 import SYS_V3, USER_V3
    proc = AutoProcessor.from_pretrained(_release_path('models/Qwen3-VL-8B-Instruct'))
    g = proc.apply_chat_template(_clean_msgs(core.make_prompt()), tokenize=False, add_generation_prompt=True)
    for tgt in (format_target("real"), format_target("deepfake", [1, 2, 300, 400], "removal", "flattened")):
        msgs = [{"role": "system", "content": SYS_V3}, {"role": "user", "content": "<image>" + USER_V3},
                {"role": "assistant", "content": tgt}]
        s = proc.apply_chat_template(_structure_messages(msgs, ["x.jpg"]), tokenize=False, add_generation_prompt=False)
        assert s.startswith(g) and s[len(g):].startswith(tgt), "GRPO prompt is not the SFT prefix"


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    for fn in TESTS:
        fn()
        print(f"[ok] {fn.__name__}")
    print(f"ALL {len(TESTS)} TESTS PASSED")
