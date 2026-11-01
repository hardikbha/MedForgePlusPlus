"""
ctgrpo_core.py -- pure, CPU-testable pieces of Counterfactual-Twin GRPO (CT-GRPO) for MF-Clean v3.

README
------
Imported by train_ctgrpo.py (the trainer), test_ctgrpo.py (unit tests) and dryrun_ctgrpo.py (tiny-model CPU
dry run). Depends only on torch + prompts_v3 (no trl / transformers import), so every function here can be tested
without a model.

  load_rl_rows(path, ...)      rl_<regime>.jsonl -> trl dataset rows + "units" (twin pairs / singletons) + stats.
                               Accepted line formats: the data builder's (build_sft_v3.py)
                               {"kind":"pair","fake":{image,box,type,gen,organ,disease,fields{type,location,size,
                               intensity},fields_src},"real":{image,organ},"group","src_rank","twin_exact"} and
                               {"kind":"single","fake":null,"real":{...},"group"}; also {"fake":row,"real":row},
                               {"real":row} and bare MF-Clean rows {image,type,...}.
                               GT evidence fields = fake.fields when present (checked against type/location_of/
                               size_of of the box), else derived from the row (intensity then unknown -> not scored).
                               Guards: split (if present) must be "train", regime exclusions (CF: no "edit"; CM: no
                               held-out generators), fake.twin == real.image and fake.group == real.group (if
                               present), files exist, no duplicate pair ids. Optional: exact twins only.
  TwinPairSampler              per epoch: seeded shuffle of PAIRS (not images); singletons are coupled two at a time
                               into a 2-image slot; K slots per optimizer step; every image index is yielded G times
                               consecutively, fake block immediately followed by its real twin block. Optional
                               max_pairs_per_real: per epoch keep a fresh random subset of at most c pairs per real
                               twin (MF-Clean reuses one source real for up to 140 fakes in rl_id).
  score_completion(text, gt)   reward components, parsed with prompts_v3.parse_v3:
                                 verdict : +1 correct, -1 wrong, -1.5 unparseable (no <verdict>)
                                 loc     : only if verdict correct. fake: +0.5*IoU(pred, gt); real without box: +0.5
                                 contra  : only if verdict correct. real WITH a box: -0.5; deepfake WITHOUT a box: -0.5
                                 field   : only on correct fakes, +w per evidence field equal to GT (w=--field_bonus)
                               channels used by the advantage: r_verdict = verdict, r_loc = loc + contra + field.
  layout_group_ids(...)        per-row image-group / twin-group ids from the dataset columns pair_id + role, rollout
                               index inside the image group, partner row (fake rollout i <-> real rollout i), and a
                               strict layout check (each image = one contiguous run of G rows, fake run directly
                               followed by its real run).
  compute_advantages(...)      --adv_mode image | twin_pooled | cogroup_bonus | gdpo (definitions in the docstring).
  step_diagnostics(...)        the per-step record written to the diagnostics jsonl.
"""
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts_v3 import SYS_V3, USER_V3, location_of, parse_v3, size_of  # noqa: E402

# ----------------------------------------------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------------------------------------------
R_CORRECT, R_WRONG, R_UNPARSE = 1.0, -1.0, -1.5
W_IOU, R_REAL_NOBOX, R_CONTRA = 0.5, 0.5, -0.5
FIELDS = ("type", "location", "size", "intensity")
TYPE_MAP = {"edit": "implant", "remove": "removal"}
CM_HOLDOUT = {"gemini", "gpt", "stable-diffusion-3.5-medium", "stable-diffusion-xl-1.0-inpainting-0.1"}
REGIMES = {"ID": (set(), set()), "CF": ({"edit"}, set()), "CM": (set(), CM_HOLDOUT),
           "CFxCM": ({"edit"}, CM_HOLDOUT)}
REGIME_ALIASES = {"id": "ID", "cf": "CF", "cm": "CM", "cfcm": "CFxCM", "cfxcm": "CFxCM"}
ADV_MODES = ("image", "twin_pooled", "cogroup_bonus", "gdpo")
ROLE_FAKE, ROLE_REAL, ROLE_SOLO = "fake", "real", "solo"


def make_prompt():
    """system + user[image, text]; same structure as train_grpo.py (prompt parity with the SFT collator is checked
    in test_ctgrpo.py: the rendered GRPO prompt is a strict prefix of the rendered SFT record)."""
    return [{"role": "system", "content": [{"type": "text", "text": SYS_V3}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_V3}]}]


# ----------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------
def _gt_of(r):
    t = r.get("type", "real")
    if t == "real":
        return {"gt_verdict": "real", "gt_box": None, "gt_type": "none", "gt_location": "none",
                "gt_size": "none", "gt_intensity": "none"}
    if t not in TYPE_MAP:
        raise ValueError(f"unknown type {t!r} in {r.get('image')}")
    b = r.get("box")
    if b is None or len(b) != 4:
        raise ValueError(f"fake without a 4-int box: {r.get('image')}")
    b = [int(v) for v in b]
    if not (0 <= b[0] <= b[2] <= 1024 and 0 <= b[1] <= b[3] <= 1024):
        raise ValueError(f"box outside the 1024 frame: {b} {r.get('image')}")
    gt = {"gt_verdict": "deepfake", "gt_box": b, "gt_type": TYPE_MAP[t], "gt_location": location_of(b),
          "gt_size": size_of(b), "gt_intensity": r.get("intensity") or None}
    f = r.get("fields")
    if f:
        for k in ("type", "location", "size"):
            if f.get(k) != gt["gt_" + k]:
                raise ValueError(f"fields.{k}={f.get(k)!r} != derived {gt['gt_' + k]!r} for {r.get('image')}")
        gt["gt_intensity"] = f.get("intensity") or None
    return gt


def _example(r, pair_id, role):
    ex = {"prompt": make_prompt(), "image": r["image"], "pair_id": pair_id, "role": role,
          "ftype": r.get("type", "real"), "gen": r.get("gen") or "", "organ": r.get("organ") or "",
          "group": r.get("group") or ""}
    ex.update(_gt_of(r))
    return ex


def load_rl_rows(path, regime=None, allow_nontrain=False, use_singletons=True, twin_check=True, limit_units=0,
                 check_files=True, max_missing_frac=0.01, exact_twins_only=False):
    """-> rows (list of dataset dicts), units (list of tuples of row indices: (fake, real) or (solo,)), stats.
    stats["unit_real"] (list, per unit: the real image path of a pair, None otherwise) feeds max_pairs_per_real."""
    regime = REGIME_ALIASES.get(regime.lower(), regime) if regime else None
    excl_types, excl_gens = REGIMES[regime] if regime else (set(), set())
    rows, units, unit_real, st = [], [], [], Counter()
    problems = Counter()

    def check(r):
        if not allow_nontrain and r.get("split") not in (None, "train"):
            problems[f"split={r.get('split')}"] += 1
        t = r.get("type", "real")
        if t in excl_types:
            problems[f"regime {regime}: type {t}"] += 1
        if t != "real" and r.get("gen") in excl_gens:
            problems[f"regime {regime}: gen {r.get('gen')}"] += 1

    with open(path) as fh:
        for ln, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pid = str(d.get("pair_id") or f"u{ln}")
            if "image" in d and "type" in d:                       # bare row -> singleton
                members = [(d, ROLE_SOLO)]
            else:
                f, r = d.get("fake"), d.get("real")
                f = dict(f, group=f.get("group") or d.get("group")) if f else None
                r = dict(r, type=r.get("type", "real"), group=r.get("group") or d.get("group")) if r else None
                if f and r:
                    if f.get("type", "real") == "real" or r["type"] != "real":
                        raise ValueError(f"line {ln}: pair roles inconsistent ({f.get('type')}, {r['type']})")
                    if twin_check and f.get("twin") is not None and f["twin"] != r["image"]:
                        problems["fake.twin != real.image"] += 1
                    if twin_check and f.get("group") and r.get("group") and f["group"] != r["group"]:
                        problems["fake.group != real.group"] += 1
                    if d.get("twin_exact") is False:
                        st["pairs_twin_not_exact"] += 1
                        if exact_twins_only:
                            st["pairs_dropped_not_exact"] += 1
                            continue
                    members = [(f, ROLE_FAKE), (r, ROLE_REAL)]
                elif f or r:
                    members = [(f or r, ROLE_SOLO)]
                else:
                    raise ValueError(f"line {ln}: neither a pair nor a row")
            for m, _ in members:
                check(m)
            if len(members) == 1 and not use_singletons:
                st["singletons_skipped"] += 1
                continue
            if check_files and not all(os.path.exists(m["image"]) for m, _ in members):
                st["units_dropped_missing_file"] += 1
                continue
            idx = []
            for m, role in members:
                idx.append(len(rows))
                rows.append(_example(m, pid, role))
            units.append(tuple(idx))
            unit_real.append(members[1][0]["image"] if len(members) == 2 else None)
            st["pairs" if len(members) == 2 else
               ("solo_real" if members[0][0].get("type", "real") == "real" else "solo_fake")] += 1
            if limit_units and len(units) >= limit_units:
                break
    if problems:
        raise ValueError(f"{path}: data guard failed: {dict(problems)}")
    n_seen = len(units) + st["units_dropped_missing_file"]
    if n_seen and st["units_dropped_missing_file"] / n_seen > max_missing_frac:
        raise ValueError(f"{st['units_dropped_missing_file']}/{n_seen} units have missing image files")
    pid_count = Counter(rows[u[0]]["pair_id"] for u in units)
    dup = [p for p, c in pid_count.items() if c > 1]
    if dup:
        raise ValueError(f"duplicate pair_id values (first: {dup[:3]})")
    st["rows"] = len(rows)
    st["fake_types"] = dict(Counter(r["ftype"] for r in rows if r["gt_verdict"] == "deepfake"))
    st["intensity_known_frac_fakes"] = (sum(1 for r in rows if r["gt_verdict"] == "deepfake" and r["gt_intensity"])
                                        / max(1, sum(1 for r in rows if r["gt_verdict"] == "deepfake")))
    reuse = Counter(x for x in unit_real if x)
    st["distinct_twin_reals"] = len(reuse)
    st["max_pairs_per_real"] = max(reuse.values()) if reuse else 0
    st = dict(st)
    st["unit_real"] = unit_real
    return rows, units, st


# ----------------------------------------------------------------------------------------------------------------
# sampler
# ----------------------------------------------------------------------------------------------------------------
class TwinPairSampler(torch.utils.data.Sampler):
    """Yields dataset indices for trl's GRPOTrainer: one optimizer step = K two-image slots = 2K images, each image
    repeated G times consecutively (trl's group contiguity). A slot is either a twin pair (fake block, then its real
    block) or two coupled singletons (each its own group). Pairs/slots are shuffled with a seeded RNG per epoch;
    an incomplete last step and an odd leftover singleton are dropped for that epoch (like trl's sampler)."""

    def __init__(self, units, num_generations, pairs_per_step, seed=42, unit_real=None, max_pairs_per_real=0):
        self.units = [tuple(u) for u in units]
        self.G, self.K, self.seed = int(num_generations), int(pairs_per_step), int(seed)
        self.epoch = 0
        self.cap = int(max_pairs_per_real or 0)
        self.unit_real = list(unit_real) if unit_real is not None else [None] * len(self.units)
        if self.cap and len(self.unit_real) != len(self.units):
            raise ValueError("unit_real must have one entry per unit")
        n_solo = sum(1 for u in self.units if len(u) == 1)
        by_real = Counter(r for u, r in zip(self.units, self.unit_real) if len(u) == 2)
        n_pairs = sum(1 for u in self.units if len(u) == 2) if not self.cap else \
            sum(min(c, self.cap) for c in by_real.values()) + \
            sum(1 for u, r in zip(self.units, self.unit_real) if len(u) == 2 and r is None)
        self.n_pairs_per_epoch = n_pairs
        self.n_slots = n_pairs + n_solo // 2
        if self.n_slots < self.K:
            raise ValueError(f"only {self.n_slots} two-image slots for pairs_per_step={self.K}")

    @property
    def steps_per_epoch(self):
        return self.n_slots // self.K

    def epoch_slots(self, epoch):
        rng = random.Random(self.seed * 1000003 + epoch)
        pairs = [u for u in self.units if len(u) == 2]
        if self.cap:
            by_real, keep = defaultdict(list), []
            for u, r in zip(self.units, self.unit_real):
                if len(u) == 2:
                    (by_real[r].append(u) if r is not None else keep.append(u))
            for r in sorted(by_real):
                us = by_real[r][:]
                rng.shuffle(us)
                keep += us[:self.cap]
            pairs = keep
        solos = [u[0] for u in self.units if len(u) == 1]
        rng.shuffle(solos)
        slots = [("pair", u) for u in pairs] + [("solo2", (solos[i], solos[i + 1])) for i in range(0, len(solos) - 1, 2)]
        rng.shuffle(slots)
        n = (len(slots) // self.K) * self.K
        return [slots[i:i + self.K] for i in range(0, n, self.K)]

    def set_epoch(self, epoch):
        """[review fix] called by transformers' Trainer (via accelerate's DataLoaderShard.set_epoch) before every
        epoch, so the order of epoch e is epoch_slots(e) regardless of how many iterators were created before."""
        self.epoch = int(epoch)

    def __iter__(self):
        steps = self.epoch_slots(self.epoch)
        self.epoch += 1                      # fallback when nobody calls set_epoch (plain DataLoader)
        for step in steps:
            for _, idxs in step:
                for i in idxs:
                    for _ in range(self.G):
                        yield i

    def __len__(self):
        return self.steps_per_epoch * self.K * 2 * self.G


# ----------------------------------------------------------------------------------------------------------------
# reward
# ----------------------------------------------------------------------------------------------------------------
def box_iou(a, b):
    if a is None or b is None:
        return 0.0
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def score_completion(text, gt, field_bonus=0.0):
    p = parse_v3(text)
    v, box = p["verdict"], p["box"]
    o = {"verdict": 0.0, "loc": 0.0, "contra": 0.0, "field": 0.0, "parsed": v is not None, "correct": False,
         "well_formed": bool(p["well_formed"]), "pred": v, "has_box": box is not None, "iou": None,
         "n_field": 0, "n_field_scored": 0}
    if v is None:
        o["verdict"] = R_UNPARSE
    elif v != gt["gt_verdict"]:
        o["verdict"] = R_WRONG
    else:
        o["verdict"], o["correct"] = R_CORRECT, True
        if v == "deepfake":
            if box is None:
                o["contra"], o["iou"] = R_CONTRA, 0.0
            else:
                o["iou"] = box_iou(box, gt["gt_box"])
                o["loc"] = W_IOU * o["iou"]
            for f in FIELDS:                      # scored always (diagnostics); rewarded only if field_bonus > 0
                g = gt.get("gt_" + f)
                if g in (None, ""):
                    continue
                o["n_field_scored"] += 1
                o["n_field"] += int(p["fields"].get(f) == g)
            o["field"] = float(field_bonus) * o["n_field"]
        else:
            if box is None:
                o["loc"] = R_REAL_NOBOX
            else:
                o["contra"] = R_CONTRA
    o["r_verdict"] = o["verdict"]
    o["r_loc"] = o["loc"] + o["contra"] + o["field"]
    o["total"] = o["r_verdict"] + o["r_loc"]
    return o


class CTReward:
    """trl reward callable. Returns the total reward per completion and keeps the component breakdown of the last
    call in .last (the trainer pops it to build the advantages, so trl's logged reward and ours are the same)."""

    def __init__(self, field_bonus=0.0):
        self.field_bonus = float(field_bonus)
        self.__name__ = "ctgrpo_reward"
        self.last = None

    def __call__(self, prompts=None, completions=None, **kw):
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        gts = [{k: kw[k][i] for k in ("gt_verdict", "gt_box", "gt_type", "gt_location", "gt_size", "gt_intensity")}
               for i in range(len(texts))]
        comps = [score_completion(t, g, self.field_bonus) for t, g in zip(texts, gts)]
        self.last = {"comps": comps, "texts": texts, "pair_id": list(kw["pair_id"]), "role": list(kw["role"])}
        return [c["total"] for c in comps]

    def pop(self):
        out, self.last = self.last, None
        if out is None:
            raise RuntimeError("CTReward.pop(): reward function was not called for this batch")
        return out


# ----------------------------------------------------------------------------------------------------------------
# group ids + layout check
# ----------------------------------------------------------------------------------------------------------------
def layout_group_ids(pair_ids, roles, G, strict=True):
    """Group ids come from the dataset columns (pair_id, role), not from reshapes.
    Returns dict(img_gid, pair_gid, rollout_idx, partner (row of the twin's rollout with the same index, -1 if none),
    n_img, n_pair, img_keys (one per image group, step order), pair_keys, is_twin_pair (per pair group), errors)."""
    B = len(pair_ids)
    keys = [f"{p}|{r}" for p, r in zip(pair_ids, roles)]
    runs, i = [], 0
    while i < B:
        j = i
        while j < B and keys[j] == keys[i]:
            j += 1
        runs.append((keys[i], i, j))
        i = j
    errors = []
    if B % G:
        errors.append(f"batch {B} not a multiple of G={G}")
    if any(e - s != G for _, s, e in runs):
        errors.append(f"image runs not of length G: {[e - s for _, s, e in runs]}")
    if len({k for k, _, _ in runs}) != len(runs):
        errors.append("an image group is split into several runs (non-contiguous)")
    img_index = {}
    for k, _, _ in runs:
        img_index.setdefault(k, len(img_index))
    pair_index, pair_runs = {}, defaultdict(list)
    for ri, (k, s, e) in enumerate(runs):
        p = pair_ids[s]
        pair_index.setdefault(p, len(pair_index))
        pair_runs[p].append((ri, roles[s], s, e))
    is_twin = []
    for p, pr in pair_runs.items():
        rl = [r for _, r, _, _ in pr]
        if len(pr) == 2:
            if rl != [ROLE_FAKE, ROLE_REAL] or pr[1][0] != pr[0][0] + 1:
                errors.append(f"pair {p}: runs {rl} not 'fake then real' adjacent")
            is_twin.append(True)
        elif len(pr) == 1:
            if rl[0] != ROLE_SOLO:
                errors.append(f"pair {p}: twin missing from the step (only {rl[0]})")
            is_twin.append(False)
        else:
            errors.append(f"pair {p}: {len(pr)} runs")
            is_twin.append(False)
    if errors and strict:
        raise ValueError("CT-GRPO layout check failed: " + "; ".join(errors))
    img_gid = torch.tensor([img_index[k] for k in keys], dtype=torch.long)
    pair_gid = torch.tensor([pair_index[p] for p in pair_ids], dtype=torch.long)
    rollout_idx = torch.zeros(B, dtype=torch.long)
    seen = Counter()
    for b, k in enumerate(keys):
        rollout_idx[b] = seen[k]
        seen[k] += 1
    partner = torch.full((B,), -1, dtype=torch.long)
    row_of = {(keys[b], int(rollout_idx[b])): b for b in range(B)}
    for b in range(B):
        p, r = pair_ids[b], roles[b]
        if r in (ROLE_FAKE, ROLE_REAL):
            other = f"{p}|{ROLE_REAL if r == ROLE_FAKE else ROLE_FAKE}"
            partner[b] = row_of.get((other, int(rollout_idx[b])), -1)
    return {"img_gid": img_gid, "pair_gid": pair_gid, "rollout_idx": rollout_idx, "partner": partner,
            "n_img": len(img_index), "n_pair": len(pair_index), "img_keys": [k for k, _, _ in runs],
            "pair_keys": list(pair_index), "is_twin_pair": is_twin, "errors": errors}


# ----------------------------------------------------------------------------------------------------------------
# advantages
# ----------------------------------------------------------------------------------------------------------------
def _gmean(x, gid, n):
    s = torch.zeros(n, dtype=x.dtype).index_add_(0, gid, x)
    c = torch.zeros(n, dtype=x.dtype).index_add_(0, gid, torch.ones_like(x))
    return (s / c)[gid]


def _gstd(x, gid, n):
    """unbiased (n-1) std per group, broadcast to rows (torch.std semantics, as trl); 0 for 1-row groups."""
    m = _gmean(x, gid, n)
    ss = torch.zeros(n, dtype=x.dtype).index_add_(0, gid, (x - m) ** 2)
    c = torch.zeros(n, dtype=x.dtype).index_add_(0, gid, torch.ones_like(x))
    return (ss / (c - 1).clamp(min=1)).sqrt()[gid]


def compute_advantages(r_verdict, r_loc, img_gid, pair_gid, mode, correct=None, partner=None, scale_std=True,
                       std_floor=0.0, eps=1e-4, pair_bonus=0.0, pool_channels="verdict"):
    """All inputs are per-row (B,) tensors; group ids are integer ids (any order, contiguity not assumed).
    den(s) = max(s, std_floor) + eps if scale_std else 1   (std_floor=0 -> exactly trl's (std + 1e-4)).

    image         A = (r - mean_img r) / den(std_img r)                                   standard GRPO (baseline)
    twin_pooled   pool_channels='verdict' (default): c = (r_v - mean_pair r_v) + (r_l - mean_img r_l),
                  A = c / den(std_pair c). The verdict channel is centred over the 2G rollouts of the twin pair (a
                  solved twin + an unsolved twin -> contrastive signal); the localisation channel stays centred per
                  image, so the fixed offset between a correct real (+0.5) and a correct fake (+0.5*IoU) does NOT
                  become a spurious 'say real' advantage on solved pairs.
                  pool_channels='all': A = (r - mean_pair r) / den(std_pair r) (literal joint normalisation).
                  Singletons (pair group == image group) reduce exactly to 'image'.
    cogroup_bonus r_v' = r_v + pair_bonus * [fake rollout i correct AND real rollout i correct] (both rows of the
                  rollout pair receive it), then 'image' normalisation (ConsistRoll-style bonus). Singletons: no bonus.
    gdpo          A = (r_v - mean_img r_v)/den(std_img r_v) + (r_l - mean_img r_l)/den(std_img r_l)
                  (decoupled per-channel group normalisation, then summed).
    Returns dict(adv, adv_verdict, adv_loc, reward, bonus, denom): adv == adv_verdict + adv_loc exactly; for the
    single-denominator modes the split is the linear attribution of the advantage to the two reward channels."""
    if mode not in ADV_MODES:
        raise ValueError(f"adv_mode {mode!r} not in {ADV_MODES}")
    rv, rl = r_verdict.double(), r_loc.double()
    img_gid, pair_gid = img_gid.long(), pair_gid.long()
    ni, npair = int(img_gid.max()) + 1, int(pair_gid.max()) + 1
    bonus = torch.zeros_like(rv)
    if mode == "cogroup_bonus":
        if correct is None or partner is None:
            raise ValueError("cogroup_bonus needs correct and partner")
        has = partner >= 0
        pc = correct[partner.clamp(min=0)] & has
        bonus = float(pair_bonus) * (correct.bool() & pc).double()
        rv = rv + bonus

    def den(s):
        return (s.clamp(min=std_floor) + eps) if scale_std else torch.ones_like(s)

    if mode in ("image", "cogroup_bonus"):
        cv, cl = rv - _gmean(rv, img_gid, ni), rl - _gmean(rl, img_gid, ni)
        d = den(_gstd(rv + rl, img_gid, ni))
        av, al = cv / d, cl / d
    elif mode == "twin_pooled":
        cv = rv - _gmean(rv, pair_gid, npair)
        cl = rl - (_gmean(rl, pair_gid, npair) if pool_channels == "all" else _gmean(rl, img_gid, ni))
        if pool_channels not in ("verdict", "all"):
            raise ValueError(pool_channels)
        d = den(_gstd(cv + cl, pair_gid, npair))
        av, al = cv / d, cl / d
    else:  # gdpo
        cv, cl = rv - _gmean(rv, img_gid, ni), rl - _gmean(rl, img_gid, ni)
        dv, dl = den(_gstd(rv, img_gid, ni)), den(_gstd(rl, img_gid, ni))
        av, al = cv / dv, cl / dl
        d = dv
    return {"adv": (av + al).float(), "adv_verdict": av.float(), "adv_loc": al.float(),
            "reward": (rv + rl).float(), "bonus": bonus.float(), "denom": d.float()}


# ----------------------------------------------------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------------------------------------------------
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else None


def step_diagnostics(comps, lay, adv, gt_verdict, ftype, comp_len, has_eos, G, zero_tol=1e-6):
    """comps: list of score_completion dicts (B); lay: layout_group_ids output; adv: compute_advantages output."""
    B = len(comps)
    R = adv["reward"].double()
    rv = torch.tensor([c["r_verdict"] for c in comps], dtype=torch.float64) + adv["bonus"].double()
    correct = torch.tensor([c["correct"] for c in comps])
    A, Av, Al = adv["adv"].double(), adv["adv_verdict"].double(), adv["adv_loc"].double()
    img_gid, pair_gid = lay["img_gid"], lay["pair_gid"]
    img_groups, twin_groups = [], []
    for g in range(lay["n_img"]):
        m = img_gid == g
        rows = m.nonzero().flatten().tolist()
        b0 = rows[0]
        c = correct[m]
        img_groups.append({"key": lay["img_keys"][g], "gt": gt_verdict[b0], "ftype": ftype[b0],
                           "std": float(R[m].std()) if m.sum() > 1 else 0.0,
                           "verdict_std": float(rv[m].std()) if m.sum() > 1 else 0.0,
                           "n_correct": int(c.sum()), "n": int(m.sum()), "mean_reward": float(R[m].mean()),
                           "mean_abs_adv": float(A[m].abs().mean())})
    for g in range(lay["n_pair"]):
        if not lay["is_twin_pair"][g]:
            continue
        m = pair_gid == g
        fake_m = m & torch.tensor([gt_verdict[b] == "deepfake" for b in range(B)])
        real_m = m & ~fake_m
        cf, cr = correct[fake_m], correct[real_m]
        twin_groups.append({"pair": lay["pair_keys"][g], "std": float(R[m].std()), "verdict_std": float(rv[m].std()),
                            "n_correct_fake": int(cf.sum()), "n_correct_real": int(cr.sum()), "n": int(m.sum())})
    ig, tg = img_groups, twin_groups

    def split(n_c, n):
        return 0 < n_c < n

    d = {
        "n_rows": B, "n_img_groups": len(ig), "n_twin_groups": len(tg),
        "n_solo_groups": sum(1 for t in lay["is_twin_pair"] if not t),
        "img_std": [round(x["std"], 6) for x in ig], "twin_std": [round(x["std"], 6) for x in tg],
        "frac_zero_std_img": _mean([x["std"] < zero_tol for x in ig]),
        "frac_zero_std_twin": _mean([x["std"] < zero_tol for x in tg]),
        "frac_verdict_split_img": _mean([split(x["n_correct"], x["n"]) for x in ig]),
        "frac_verdict_split_twin": _mean([split(x["n_correct_fake"] + x["n_correct_real"], x["n"]) for x in tg]),
        # twin groups whose two image groups are each constant in correctness but differ from each other: the case in
        # which image-level GRPO has no verdict gradient and twin pooling does.
        "frac_twin_contrast_only": _mean([
            (x["n_correct_fake"] in (0, x["n"] // 2)) and (x["n_correct_real"] in (0, x["n"] // 2))
            and (x["n_correct_fake"] != x["n_correct_real"]) for x in tg]),
        "frac_zero_adv_img": _mean([x["mean_abs_adv"] < 1e-8 for x in ig]),
        "mean_abs_adv": float(A.abs().mean()), "mean_abs_adv_verdict": float(Av.abs().mean()),
        "mean_abs_adv_loc": float(Al.abs().mean()),
        "adv_share_verdict": float(Av.abs().sum() / (Av.abs().sum() + Al.abs().sum()).clamp(min=1e-12)),
        "acc_all": float(correct.double().mean()),
        "acc_fake": _mean([float(c["correct"]) for c, g in zip(comps, gt_verdict) if g == "deepfake"]),
        "acc_real": _mean([float(c["correct"]) for c, g in zip(comps, gt_verdict) if g == "real"]),
        "acc_edit": _mean([float(c["correct"]) for c, t in zip(comps, ftype) if t == "edit"]),
        "acc_remove": _mean([float(c["correct"]) for c, t in zip(comps, ftype) if t == "remove"]),
        "parse_rate": _mean([float(c["parsed"]) for c in comps]),
        "well_formed_rate": _mean([float(c["well_formed"]) for c in comps]),
        "box_rate_on_real_gt": _mean([float(c["has_box"]) for c, g in zip(comps, gt_verdict) if g == "real"]),
        "mean_iou_correct_fakes": _mean([c["iou"] for c in comps if c["iou"] is not None]),
        "field_acc_correct_fakes": (sum(c["n_field"] for c in comps) / max(1, sum(c["n_field_scored"] for c in comps))
                                    if any(c["n_field_scored"] for c in comps) else None),
        "reward_mean": float(R.mean()),
        "r_verdict_mean": _mean([c["verdict"] for c in comps]), "r_loc_mean": _mean([c["loc"] for c in comps]),
        "r_contra_mean": _mean([c["contra"] for c in comps]), "r_field_mean": _mean([c["field"] for c in comps]),
        "r_bonus_mean": float(adv["bonus"].double().mean()),
        "comp_len_mean": _mean(comp_len), "comp_len_min": min(comp_len), "comp_len_max": max(comp_len),
        "comp_len_mean_fake_gt": _mean([l for l, g in zip(comp_len, gt_verdict) if g == "deepfake"]),
        "comp_len_mean_real_gt": _mean([l for l, g in zip(comp_len, gt_verdict) if g == "real"]),
        "frac_no_eos": _mean([float(not e) for e in has_eos]),
        "img_groups": ig, "twin_groups": tg,
    }
    return d
