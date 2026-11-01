"""
Single source of truth for the v3 (MF-Clean / CT-GRPO) prompt and output format.
Used by build_sft_v3.py (targets), train_ctgrpo.py (rollout prompts + reward parsing) and eval_v3.py.

Output format (grounded, verifiable rationale; parseable by circe_reward.parse_output's v2 branch because it
contains <verdict> and <box>/<none/>):

<localization>
<box>x1 y1 x2 y2</box>            | <none/>
</localization>
<evidence>
<type>implant|removal|none</type>
<location>upper-left|upper-center|...|none</location>     (3x3 grid of the image, image coordinates)
<size>tiny|small|medium|large|none</size>                 (box area fraction thresholds below)
<intensity>brighter|darker|flattened|none</intensity>     (implant: added structure brighter/darker than its
                                                           surround; removal: local contrast flattened)
</evidence>
<verdict>deepfake|real</verdict>

Box coordinates are in a 1024x1024 frame (same convention as MedForge / previous CIRCE runs).
"""
import re

SYS_V3 = (
    "You are an expert medical-image forensics analyst. Decide whether a medical image is a DEEPFAKE "
    "(a lesion was digitally IMPLANTED or REMOVED by a generative image editor) or an AUTHENTIC scan. "
    "Base your decision only on local forensic evidence in the image. First localize the manipulated region "
    "(or state none), then report verifiable evidence about it, then give your verdict, in exactly this format:\n"
    "<localization>\n<box>x1 y1 x2 y2</box>\n</localization>\n"
    "<evidence>\n<type>implant|removal|none</type>\n<location>upper-left|upper-center|upper-right|"
    "middle-left|center|middle-right|lower-left|lower-center|lower-right|none</location>\n"
    "<size>tiny|small|medium|large|none</size>\n<intensity>brighter|darker|flattened|none</intensity>\n"
    "</evidence>\n<verdict>deepfake|real</verdict>\n"
    "Coordinates are on a 1024x1024 grid. If there is no manipulated region, write <none/> inside "
    "<localization> and 'none' in every evidence field."
)
USER_V3 = "Is this image a deepfake or real? Follow the required output format exactly."

GRID = [["upper-left", "upper-center", "upper-right"],
        ["middle-left", "center", "middle-right"],
        ["lower-left", "lower-center", "lower-right"]]
SIZE_BINS = ((0.005, "tiny"), (0.02, "small"), (0.08, "medium"), (1.01, "large"))  # box area / frame area


def location_of(box, W=1024, H=1024):
    cx, cy = (box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H
    return GRID[min(2, int(cy * 3))][min(2, int(cx * 3))]


def size_of(box, W=1024, H=1024):
    a = max(0, box[2] - box[0]) * max(0, box[3] - box[1]) / float(W * H)
    for t, name in SIZE_BINS:
        if a < t:
            return name
    return "large"


def format_target(verdict, box=None, ftype="none", intensity="none"):
    if verdict == "real" or box is None:
        return ("<localization>\n<none/>\n</localization>\n<evidence>\n<type>none</type>\n<location>none</location>\n"
                "<size>none</size>\n<intensity>none</intensity>\n</evidence>\n<verdict>real</verdict>")
    b = [int(v) for v in box]
    return (f"<localization>\n<box>{b[0]} {b[1]} {b[2]} {b[3]}</box>\n</localization>\n<evidence>\n"
            f"<type>{ftype}</type>\n<location>{location_of(b)}</location>\n<size>{size_of(b)}</size>\n"
            f"<intensity>{intensity}</intensity>\n</evidence>\n<verdict>deepfake</verdict>")


_F = {k: re.compile(rf"<{k}>\s*([a-z\-]+)\s*</{k}>") for k in ("type", "location", "size", "intensity")}
_VERDICT = re.compile(r"<verdict>\s*(deepfake|real)\s*</verdict>", re.I)
_BOX = re.compile(r"<box>\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*</box>")
_NONE = re.compile(r"<none\s*/?>")


def parse_v3(text):
    """-> dict(verdict in {'deepfake','real',None}, box list|None, fields dict, well_formed bool)."""
    vm = _VERDICT.search(text or "")
    verdict = vm.group(1).lower() if vm else None
    bm = _BOX.search(text or "")
    box = None
    if bm:
        x1, y1, x2, y2 = (int(v) for v in bm.groups())
        box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    fields = {k: (r.search(text or "").group(1) if r.search(text or "") else None) for k, r in _F.items()}
    well = verdict is not None and (box is not None or bool(_NONE.search(text or ""))) and all(fields.values())
    return {"verdict": verdict, "box": box, "fields": fields, "well_formed": well}
