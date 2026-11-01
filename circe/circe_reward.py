"""
CIRCE reward + typed-failure-ledger core.

This is the *novel* part of the CIRCE recipe (see CIRCE_strategy.md), implemented
standalone so it can be smoke-tested without the full GRPO/vLLM stack:

  - parse_output()   : extract verdict + box (+ optional cue tag) from a Qwen3-VL
                       MedForge-format assistant string.
  - iou / diou       : geometry.
  - r_loc()          : CIRCE localization reward  (gated-IoU + DIoU + corner + area penalty)
                       -> strictly penalizes the lazy-large boxes that MedForge REWARDS.
  - medforge_cov()   : MedForge coverage reward (recall) -- the baseline we beat.
  - r_delta()        : zoom-and-fix Δ-IoU process reward (only tightening earns it).
  - classify_failure(): dual-axis typed failure ledger label.

Pure-python (no numpy) so it runs anywhere.
"""
import re
import math

# box tokens exactly as they appear in the MedForge SFT data (legacy "reasoning-first" format)
_BOX_RE = re.compile(
    r'x1="?(-?\d+)"?\s+y1="?(-?\d+)"?\s+x2="?(-?\d+)"?\s+y2="?(-?\d+)"?'
)
# NOTE: build_sft_data.build_record only ever emits the attributed form
# `<evidence cue="...">...</evidence>` (see box_token/self_ask_block) -- match that,
# with a bare `<evidence>` fallback for older/hand-written completions.
_CUE_RE = re.compile(r'<evidence(?:\s+cue="[^"]*")?>(.*?)</evidence>', re.S)

# CIRCE-v2 evidence-first format (see build_sft_data.build_record_v2):
#   <localization><box>x1 y1 x2 y2</box></localization>  OR  <localization><none/></localization>
#   <evidence><boundary>..</boundary><texture>..</texture><intensity>..</intensity><anatomy>..</anatomy></evidence>
#   <verdict>deepfake|real</verdict>
_V2_BOX_RE = re.compile(r'<box>\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*</box>')
_V2_NONE_RE = re.compile(r'<none\s*/?>')
_V2_VERDICT_RE = re.compile(r'<verdict>\s*(deepfake|real)\s*</verdict>', re.I)
_V2_FIELD_RE = re.compile(r'<(boundary|texture|intensity|anatomy)>(.*?)</\1>', re.S)


def _parse_output_v2(text):
    """Parser for the evidence-first CIRCE-v2 format. Returns None if the text doesn't
    look like v2 output at all (no <verdict> tag), so callers can fall back to legacy."""
    vm = _V2_VERDICT_RE.search(text)
    if not vm:
        return None
    verdict = vm.group(1).lower()
    box = None
    if not _V2_NONE_RE.search(text):
        bm = _V2_BOX_RE.search(text)
        if bm:
            x1, y1, x2, y2 = (int(v) for v in bm.groups())
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    fields = {k: v.strip() for k, v in _V2_FIELD_RE.findall(text)}
    cue = fields.get("boundary") or fields.get("texture") or None
    return {"verdict": verdict, "box": box, "cue": cue, "evidence_fields": fields, "format": "v2"}


def parse_output(text):
    """Return dict(verdict in {'real','deepfake',None}, box or None, cue str or None).
    Tries the CIRCE-v2 evidence-first format first, falls back to the legacy
    reasoning-first format so old checkpoints/eval data keep parsing unchanged."""
    v2 = _parse_output_v2(text)
    if v2 is not None:
        return v2

    # verdict sits AFTER the long </think> in every format this project (and MedForge's own
    # released checkpoint) actually emits -- restrict the verdict search to that tail when a
    # </think> is present. This matters for externally-sourced models: MedForge-Reasoner's own
    # boilerplate opens every response with "...analyze whether this image is a deepfake:",
    # which false-triggers a naive whole-string "is a deepfake" search on every sample
    # regardless of the true answer. Our own SFT templates never mention the opposite verdict
    # before the real one, so this is a no-op for every checkpoint trained in this project --
    # verified against the existing smoke tests, which still pass unchanged.
    search_region = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    t = search_region.lower()
    verdict = None
    if "this image is a deepfake" in t or "is a deepfake" in t:
        verdict = "deepfake"
    elif "this image is a real image" in t or "is a real image" in t or "authentic" in t:
        verdict = "real"

    box = None
    m = _BOX_RE.search(text)
    if m:
        x1, y1, x2, y2 = (int(v) for v in m.groups())
        # normalize ordering
        box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    cue = None
    cm = _CUE_RE.search(text)
    if cm:
        cue = cm.group(1).strip()
    return {"verdict": verdict, "box": box, "cue": cue, "format": "legacy"}


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def diou(a, b):
    """Distance-IoU: IoU - rho^2(centers)/c^2. Monotone even when IoU==0."""
    if a is None or b is None:
        return 0.0
    base = iou(a, b)
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    rho2 = (acx - bcx) ** 2 + (acy - bcy) ** 2
    # diagonal of smallest enclosing box
    ex1, ey1 = min(a[0], b[0]), min(a[1], b[1])
    ex2, ey2 = max(a[2], b[2]), max(a[3], b[3])
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2
    return base - (rho2 / c2 if c2 > 0 else 0.0)


def medforge_cov(pred, gt):
    """MedForge coverage = |pred ∩ gt| / |gt|  (recall). Rewards lazy-large boxes."""
    if pred is None or gt is None:
        return 0.0
    ix1, iy1 = max(pred[0], gt[0]), max(pred[1], gt[1])
    ix2, iy2 = min(pred[2], gt[2]), min(pred[3], gt[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ga = _area(gt)
    return inter / ga if ga > 0 else 0.0


def r_loc(pred, gt, w=(1.0, 1.0, 0.3, 0.5), kappa=2.0, corner_scale=10.0):
    """CIRCE localization reward. Higher is better; oversized boxes strictly lose."""
    if gt is None:            # no GT box (normal image): any predicted box is wrong
        return 0.0 if pred is None else -w[3]
    if pred is None:
        return -0.5           # GT HAS a box but model emitted none -> penalize (no free "suppress" escape)
    w1, w2, w3, w4 = w
    _iou = iou(pred, gt)
    gated = w1 * _iou if _iou > 0.5 else 0.0
    dense = w2 * diou(pred, gt)
    corner_l1 = sum(abs(pred[i] - gt[i]) for i in range(4))
    corner = w3 * math.exp(-corner_l1 / corner_scale)
    area_ratio = _area(pred) / _area(gt) if _area(gt) > 0 else 999.0
    # cap the penalty so a single oversized box can't blow up GRPO advantages,
    # while still making oversized boxes strictly lose vs a tight one.
    area_pen = min(2.0, w4 * max(0.0, area_ratio - kappa))
    return gated + dense + corner - area_pen


def r_delta(iou_t1, iou_t2, cap=0.5):
    """Zoom-and-fix Δ-IoU process reward: only *tightening* earns it."""
    return max(0.0, min(cap, iou_t2 - iou_t1))


def classify_failure(pred, gt, gt_has_box, verdict_correct, tau=0.5):
    """
    Dual-axis typed failure ledger label:
      SOLVED, VERDICT, LOCALIZE, BOTH, HALLUCINATE
    """
    box_iou = iou(pred["box"], gt) if (gt_has_box and pred["box"]) else 0.0
    if not gt_has_box:
        # real/normal image: predicting a box on it is a hallucination
        if pred["box"] is not None:
            return "HALLUCINATE"
        return "SOLVED" if verdict_correct else "VERDICT"
    loc_ok = box_iou >= tau
    if verdict_correct and loc_ok:
        return "SOLVED"
    if verdict_correct and not loc_ok:
        return "LOCALIZE"
    if (not verdict_correct) and not loc_ok:
        return "BOTH"
    return "VERDICT"  # verdict wrong but box happened to be tight
