"""
trl GRPOTrainer-compatible reward adapter for CIRCE.

trl's GRPOTrainer calls reward functions with signature:
    reward_fn(prompts, completions, **kwargs) -> list[float]
where kwargs carries any extra dataset columns (we pass gt_verdict, gt_box, gt_has_box).

Each verifiable reward from circe_reward is exposed as its own function so the
GRPO run can weight them (trl supports a list of reward_funcs + reward_weights).
This is the V1→V3 reward stack; cue-invariance (V4) needs model rollouts and is
added later.
"""
from circe_reward import parse_output, r_loc, iou, medforge_cov


def _texts(completions):
    # trl passes either list[str] or list[[{"role","content"}]]
    out = []
    for c in completions:
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list) and c and isinstance(c[-1], dict):
            out.append(c[-1].get("content", ""))
        else:
            out.append(str(c))
    return out


def reward_verdict(prompts=None, completions=None, gt_verdict=None, **kw):
    """+1 if predicted verdict matches GT (exact-match, verifiable)."""
    outs = _texts(completions)
    gts = gt_verdict or [None] * len(outs)
    return [1.0 if parse_output(t)["verdict"] == g else 0.0 for t, g in zip(outs, gts)]


def reward_localization(prompts=None, completions=None, gt_box=None, **kw):
    """CIRCE r_loc (gated-IoU + DIoU + area-penalty). Beats MedForge coverage."""
    outs = _texts(completions)
    gts = gt_box or [None] * len(outs)
    return [r_loc(parse_output(t)["box"], g) for t, g in zip(outs, gts)]


def reward_coverage_baseline(prompts=None, completions=None, gt_box=None, **kw):
    """MedForge coverage — provided only for V0 baseline / side-by-side logging."""
    outs = _texts(completions)
    gts = gt_box or [None] * len(outs)
    return [medforge_cov(parse_output(t)["box"], g) for t, g in zip(outs, gts)]


import re, math
_CUE_RE = re.compile(r'<evidence cue="([^"]*)"', re.I)
# running cue->correctness table (persists within a training run) -> PMI transfer signal
_CUE_STATS = {"_all": [0, 0]}  # cue: [n_correct, n_total]


def _cue_of(text):
    m = _CUE_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip().lower()
    # fallback: infer from self_ask affirming an artifact
    t = text.lower()
    if "boundary discontinuity" in t and ("yes" in t or "smoothing" in t or "inpaint" in t):
        return "boundary"
    if "too smooth" in t or "over-smooth" in t or "homogeneous" in t:
        return "oversmoothing"
    return None


def reward_cue_consistency(prompts=None, completions=None, gt_verdict=None, **kw):
    """Cross-cue invariance (tractable): reward a correct verdict that is GROUNDED in a forensic cue,
    scaled by that cue's running PMI (how much the cue predicts correctness above base rate across the
    mixed-organ stream). Cues that transfer to correctness get rewarded; organ-specific flukes decay."""
    outs = _texts(completions); gts = gt_verdict or [None] * len(outs)
    rewards = []
    for t, g in zip(outs, gts):
        p = parse_output(t); correct = (p["verdict"] == g)
        cue = _cue_of(t)
        # base rate
        _CUE_STATS["_all"][1] += 1; _CUE_STATS["_all"][0] += int(correct)
        base = _CUE_STATS["_all"][0] / max(1, _CUE_STATS["_all"][1])
        if cue is None:
            rewards.append(0.0); continue
        st = _CUE_STATS.setdefault(cue, [0, 0]); st[1] += 1; st[0] += int(correct)
        p_cue = st[0] / max(1, st[1])
        pmi = math.log(max(p_cue, 1e-3) / max(base, 1e-3))   # >0 if cue predicts correctness above base
        # only pay a grounded, correct, transfer-positive cue
        rewards.append(round(max(0.0, pmi) if correct else 0.0, 4))
    return rewards


_CUE_WORDS = ("boundary", "blend", "smooth", "inpaint", "texture", "discontinu",
              "artifact", "inconsist", "seam", "edge", "lesion", "tumor")


def reward_forgery_grounded(prompts=None, completions=None, gt_verdict=None, **kw):
    """Forgery-aware reward (our GSPO analogue) targeting the fake-recall gap.
    +1 correct verdict; +0.3 bonus when a correct DEEPFAKE call cites a concrete forgery cue
    (grounded reasoning); -0.3 extra for the specific failure we fix (calling a fake 'real');
    -0.5 format gate for no parseable verdict. Rewards catching fakes with grounded evidence."""
    outs = _texts(completions)
    gts = gt_verdict or [None] * len(outs)
    rewards = []
    for t, g in zip(outs, gts):
        p = parse_output(t)
        if p["verdict"] is None:
            rewards.append(-0.5); continue
        correct = 1.0 if p["verdict"] == g else 0.0
        r = correct
        tl = t.lower()
        if correct and g == "deepfake" and any(w in tl for w in _CUE_WORDS):
            r += 0.3                                   # grounded fake detection
        if g == "deepfake" and p["verdict"] == "real":
            r -= 0.3                                   # penalize the missed-fake failure mode
        rewards.append(round(r, 4))
    return rewards


_CONF_RE = re.compile(r'[Cc]onfidence["\s:=]*\s*(0?\.\d+|1(?:\.0+)?|0)')


def reward_calibration(prompts=None, completions=None, gt_verdict=None, **kw):
    """Hack-proof Stage-2 reward (strictly proper scoring rule): R = G_fmt * [1{correct} - Brier].
    Brier = (q - correct)^2 where q is the model's stated Confidence: 0.xx. Rewards being correct AND
    well-calibrated; the V1 box-suppression hack has nothing to grab (no box/IoU term)."""
    outs = _texts(completions); gts = gt_verdict or [None] * len(outs)
    rewards = []
    for t, g in zip(outs, gts):
        p = parse_output(t)
        if p["verdict"] is None:            # format gate: no parseable verdict -> penalize
            rewards.append(-0.5); continue
        correct = 1.0 if p["verdict"] == g else 0.0
        m = _CONF_RE.search(t)
        q = float(m.group(1)) if m else 0.5   # no confidence stated -> max-uncertainty default
        q = min(1.0, max(0.0, q))
        brier = (q - correct) ** 2
        rewards.append(round(correct - brier, 4))
    return rewards


def reward_balanced_verdict_guard(prompts=None, completions=None, gt_verdict=None, **kw):
    """Verdict reward with explicit guards against both remaining high-cost errors.

    The replay-SFT model already has high fake recall, so Stage-2 RL should not
    learn a cheap "always fake" shortcut. This term rewards correct verdicts,
    mildly penalizes missed fakes, and more strongly penalizes real->deepfake
    false alarms.
    """
    outs = _texts(completions)
    gts = gt_verdict or [None] * len(outs)
    rewards = []
    for t, g in zip(outs, gts):
        pred = parse_output(t)["verdict"]
        if pred is None:
            rewards.append(-0.5)
        elif pred == g:
            rewards.append(1.0)
        elif g == "real" and pred == "deepfake":
            rewards.append(-0.7)
        elif g == "deepfake" and pred == "real":
            rewards.append(-0.4)
        else:
            rewards.append(-0.2)
    return rewards


FORENSIC_CUES = {"boundary_discontinuity", "boundary", "over_smoothing", "oversmoothing",
                 "over-smoothing", "blending", "inpainting", "texture"}


def reward_format(prompts=None, completions=None, **kw):
    """Bootstraps termination: reward a rollout that actually CONCLUDES with </think> + a verdict.
    Creates reward variance the instant some rollouts finish -> gradient toward concluding (fixes the
    non-termination deadlock where every completion runs to the cap with no verdict)."""
    outs = _texts(completions)
    rewards = []
    for t in outs:
        r = 0.0
        if "</think>" in t:
            r += 0.1
        if parse_output(t)["verdict"] is not None:
            r += 0.1
        rewards.append(round(r, 4))
    return rewards


def reward_evidence_consistency(prompts=None, completions=None, gt_verdict=None, **kw):
    """NOVEL forgery-aware term: reward a CORRECT verdict that is GROUNDED in a forensic cue consistent
    with the manipulation (boundary/blending for implant, over-smoothing for removal; consistency for real).
    Anchors reasoning to visual forgery evidence -> transfers to UNSEEN manipulation types (the held-out
    Implant gap). Correct-but-ungrounded gets a small reward; grounded-correct gets the full reward."""
    outs = _texts(completions)
    gts = gt_verdict or [None] * len(outs)
    rewards = []
    for t, g in zip(outs, gts):
        p = parse_output(t)
        v = p["verdict"]
        if v is None or v != g:          # only pay grounded CORRECT answers
            rewards.append(0.0); continue
        cue = _cue_of(t); tl = t.lower()
        if g == "deepfake":
            grounded = (cue in FORENSIC_CUES) or any(k in tl for k in
                       ("boundary", "smooth", "inpaint", "blend", "texture", "homogeneous", "seamless"))
            rewards.append(0.5 if grounded else 0.1)
        else:                            # real
            grounded = any(k in tl for k in ("consistent", "normal", "authentic", "plausible", "matches"))
            rewards.append(0.4 if grounded else 0.15)
    return rewards


# reward stacks
REWARD_SETS_EXPORT = {
    "v1": ([reward_verdict, reward_localization], [1.0, 1.0]),
    "v2": ([reward_verdict, reward_localization, reward_cue_consistency], [1.0, 1.0, 0.5]),
}
# default V1 stack + weights (verdict + localization)
REWARD_FUNCS = [reward_verdict, reward_localization]
REWARD_WEIGHTS = [1.0, 1.0]


import os as _os
_ABLATE_LOC = _os.environ.get("CIRCE_ABLATE_LOC") == "1"
_ABLATE_EVID = _os.environ.get("CIRCE_ABLATE_EVID") == "1"
_ABLATE_FMT = _os.environ.get("CIRCE_ABLATE_FMT") == "1"


def reward_classification_dominant(prompts=None, completions=None, gt_verdict=None, gt_box=None, **kw):
    """CIRCE-v2 classification-dominant reward (Phase 10).

    Every prior reward stack (forgery_guarded, forgery, v1, v2) sums verdict + localization +
    format + evidence terms independently, which means a wrong verdict can still collect
    localization/format/evidence reward -- observed empirically across 5 GRPO runs in this
    project as RL that moves mIoU up but never beats the SFT classification baseline (see
    CIRCE_FULL_APPROACH_AND_RESULTS.md open problem #1). This reward makes that impossible by
    construction: auxiliary bonuses are only added when the verdict is correct.

        correct verdict:    +4.0 + localization_bonus[0,0.25] + evidence_bonus[0,0.20] + format_bonus[0,0.20]
        incorrect verdict:  -4.0 + contradiction penalties (no compensating bonuses)

    Contradiction penalties (apply regardless of verdict correctness, since they're internal
    self-consistency failures, not classification difficulty):
        deepfake verdict with no bbox emitted: -1.0
        real verdict with a nonempty bbox:     -1.0
    """
    outs = _texts(completions)
    gts_v = gt_verdict or [None] * len(outs)
    gts_b = gt_box or [None] * len(outs)
    rewards = []
    for t, g, gb in zip(outs, gts_v, gts_b):
        p = parse_output(t)
        v = p["verdict"]
        if v is None:
            rewards.append(-4.0)  # unparseable -> treated as maximally wrong, no bonuses possible
            continue

        contradiction_pen = 0.0
        if v == "deepfake" and p["box"] is None:
            contradiction_pen -= 1.0
        if v == "real" and p["box"] is not None:
            contradiction_pen -= 1.0

        if v != g:
            rewards.append(round(-4.0 + contradiction_pen, 4))
            continue

        # correct verdict: auxiliary bonuses now allowed, capped as specified.
        # Env-var switches (CIRCE_ABLATE_LOC/EVID/FMT=1) zero out an individual bonus term
        # for the reward-component ablation, leaving verdict gating + contradiction penalties intact.
        loc_bonus = 0.0
        if gb is not None and not _ABLATE_LOC:
            _iou = iou(p["box"], gb) if p["box"] is not None else 0.0
            loc_bonus = min(0.25, 0.25 * _iou)
        cue = _cue_of(t); tl = t.lower()
        grounded = (cue in FORENSIC_CUES) or any(k in tl for k in
                   ("boundary", "smooth", "inpaint", "blend", "texture", "homogeneous",
                    "seamless", "consistent", "normal", "authentic", "plausible"))
        evidence_bonus = (0.20 if grounded else 0.0) if not _ABLATE_EVID else 0.0
        format_bonus = (0.20 if "</think>" in t or "<verdict>" in t else 0.0) if not _ABLATE_FMT else 0.0
        rewards.append(round(4.0 + loc_bonus + evidence_bonus + format_bonus + contradiction_pen, 4))
    return rewards


if __name__ == "__main__":
    # tiny self-check with a synthetic completion
    demo = ['<think>...</think>\nThis image is a deepfake. '
            '<|box_start|>x1="10" y1="10" x2="30" y2="30"<|box_end|>']
    print("verdict:", reward_verdict(completions=demo, gt_verdict=["deepfake"]))
    print("r_loc  :", reward_localization(completions=demo, gt_box=[[10, 10, 30, 30]]))
    print("cov    :", reward_coverage_baseline(completions=demo, gt_box=[[10, 10, 30, 30]]))
