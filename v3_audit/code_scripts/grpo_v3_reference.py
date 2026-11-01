"""
Reference (CPU-testable) implementations of the pure pieces of the GRPO-v3 spec in ~/CIRCE/v3_audit/code.md.
NOT wired into training. Every function is side-effect free so it can be unit-tested on CPU and then called
from a GRPOTrainer subclass / the vlm_grpo_patch replacement.

Sections:
  (a) classdom_components, group_diagnostics          per-group diagnostics + advantage attribution
  (c) pair_coupled_advantages, pair_joint_rewards     counterfactual-paired groups
  (d) token_masks_legacy, decoupled_token_advantages  verdict-vs-box token-level advantages
  (e) change_map, diffmass_reward                     difference-map localisation reward (training only)
  (f) dapo_keep_mask                                  dynamic sampling filter
"""
import re
import math
import torch

# --------------------------------------------------------------------------------------------
# (a) diagnostics
# --------------------------------------------------------------------------------------------
_EVID_WORDS = ("boundary", "smooth", "inpaint", "blend", "texture", "homogeneous",
               "seamless", "consistent", "normal", "authentic", "plausible")


def classdom_components(text, gt_verdict, gt_box, parse_output, iou, cue_of, forensic_cues):
    """Split reward_classification_dominant into additive parts. Sum == the original reward (tested).
    verdict: +4 / -4 (unparseable -> -4, nothing else); const: evidence+format bonuses (only on correct);
    loc: 0.25*IoU (only on correct with GT box); contra: contradiction penalty (any parseable verdict)."""
    p = parse_output(text)
    v = p["verdict"]
    out = {"verdict": 0.0, "loc": 0.0, "const": 0.0, "contra": 0.0, "correct": False, "parsed": v is not None,
           "iou": 0.0}
    if v is None:
        out["verdict"] = -4.0
        return out
    if v == "deepfake" and p["box"] is None:
        out["contra"] -= 1.0
    if v == "real" and p["box"] is not None:
        out["contra"] -= 1.0
    if v != gt_verdict:
        out["verdict"] = -4.0
        return out
    out["correct"] = True
    out["verdict"] = 4.0
    if gt_box is not None:
        out["iou"] = iou(p["box"], gt_box) if p["box"] is not None else 0.0
        out["loc"] = min(0.25, 0.25 * out["iou"])
    tl = text.lower()
    grounded = (cue_of(text) in forensic_cues) or any(k in tl for k in _EVID_WORDS)
    out["const"] = (0.20 if grounded else 0.0) + (0.20 if ("</think>" in text or "<verdict>" in text) else 0.0)
    return out


def group_diagnostics(comp_rows, G, scale_rewards=True, eps=1e-4, zero_std_tol=1e-6):
    """comp_rows: list (len B = n_groups*G, group-contiguous) of dicts from classdom_components.
    Returns per-group dicts + batch summary. Advantage is linear in reward given the group mean/std, so the trl
    advantage A = (R-mean)/(std+eps) splits exactly into A_verdict + A_loc + A_const + A_contra."""
    keys = ("verdict", "loc", "const", "contra")
    B = len(comp_rows)
    assert B % G == 0
    comp = {k: torch.tensor([r[k] for r in comp_rows], dtype=torch.float64).view(-1, G) for k in keys}
    R = sum(comp.values())
    mean = R.mean(1, keepdim=True)
    std = R.std(1, keepdim=True)
    denom = (std + eps) if scale_rewards else torch.ones_like(std)
    A = (R - mean) / denom
    parts = {k: (comp[k] - comp[k].mean(1, keepdim=True)) / denom for k in keys}
    correct = torch.tensor([r["correct"] for r in comp_rows]).view(-1, G)
    per_group = []
    for g in range(R.shape[0]):
        absA = A[g].abs().sum().item()
        per_group.append({
            "std": std[g].item(),
            "zero_std": bool(std[g].item() < zero_std_tol),
            "verdict_split": bool(correct[g].any() and not correct[g].all()),
            "all_correct": bool(correct[g].all()),
            "all_wrong": bool((~correct[g]).all()),
            "absA": absA,
            **{f"absA_{k}": parts[k][g].abs().sum().item() for k in keys},
        })
    tot = sum(p["absA"] for p in per_group) or 1.0
    summary = {
        "frac_zero_std": sum(p["zero_std"] for p in per_group) / len(per_group),
        "frac_verdict_split": sum(p["verdict_split"] for p in per_group) / len(per_group),
        "frac_all_correct": sum(p["all_correct"] for p in per_group) / len(per_group),
        "frac_all_wrong": sum(p["all_wrong"] for p in per_group) / len(per_group),
        # share of total |advantage| mass attributable to each reward component (can exceed 1 jointly when
        # components are anti-correlated inside a group; reported as-is)
        **{f"adv_share_{k}": sum(p[f"absA_{k}"] for p in per_group) / tot for k in keys},
        "frac_groups_loc_only": sum((not p["zero_std"]) and (not p["verdict_split"]) for p in per_group) / len(per_group),
    }
    return per_group, summary, A


# --------------------------------------------------------------------------------------------
# (c) counterfactual pairs
# --------------------------------------------------------------------------------------------
def pair_coupled_advantages(rewards, G, scale=True, eps=1e-4):
    """rewards: (n_pairs*2*G,) laid out [forged_1 x G, source_1 x G, forged_2 x G, source_2 x G, ...].
    Baseline and scale are computed over the 2G rollouts of a pair, so a pair whose two groups are each
    constant but DIFFERENT (e.g. forged all-correct, source all-wrong) still yields a gradient."""
    R = rewards.view(-1, 2 * G)
    A = R - R.mean(1, keepdim=True)
    if scale:
        A = A / (R.std(1, keepdim=True) + eps)
    return A.reshape(-1)


def pair_joint_rewards(correct_f, correct_s, loc_f, G):
    """Joint counterfactual reward: rollout i of the forged image is paired with ALL G rollouts of its source
    (expected joint reward under the partner policy, lower variance than i<->i pairing).
    correct_f, correct_s: (n_pairs, G) bool; loc_f: (n_pairs, G) float in [0, 0.25].
    joint(i,j) = 4 + loc_f[i] if forged_i correct AND source_j correct else -4."""
    cf = correct_f.double(); cs = correct_s.double()
    ps = cs.mean(1, keepdim=True)                       # P(source rollout correct)
    pf = cf.mean(1, keepdim=True)
    r_f = cf * ps * (4.0 + loc_f) + (1 - cf * ps) * (-4.0)          # forged rollout i vs all source rollouts
    r_s = cs * (pf * 4.0 + (cf * loc_f).mean(1, keepdim=True)) + (1 - cs * pf) * (-4.0)
    return r_f, r_s


# --------------------------------------------------------------------------------------------
# (d) token masks + decoupled advantages
# --------------------------------------------------------------------------------------------
_LEG_BOX = re.compile(r'x1="?(-?\d+)"?\s+y1="?(-?\d+)"?\s+x2="?(-?\d+)"?\s+y2="?(-?\d+)"?')
_V2_BOX = re.compile(r'<box>\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*</box>')
_VERDICT_LINE = re.compile(r'This image is a (deepfake|real image)\.|<verdict>\s*(deepfake|real)\s*</verdict>', re.I)
_A1 = re.compile(r'A1:\s*\S+(\s+\S+)?')       # first two words of the first self-ask answer = the verdict branch


def token_masks_legacy(token_strs):
    """token_strs: list of decoded strings, one per completion token (tokenizer.decode([id]) each).
    Returns (m_verdict, m_box) lists of 0/1. m_box = digit tokens inside the box span; m_verdict = the A1 branch
    tokens + the final verdict line. Falls back to empty masks (caller then uses sequence-level advantage)."""
    starts, text = [], ""
    for t in token_strs:
        starts.append(len(text)); text += t
    ends = [s + len(t) for s, t in zip(starts, token_strs)]
    n = len(token_strs)
    m_v, m_b = [0] * n, [0] * n

    def mark(mask, a, b, digits_only=False):
        for j in range(n):
            if starts[j] < b and ends[j] > a and (not digits_only or token_strs[j].strip().isdigit()):
                mask[j] = 1
    bm = _LEG_BOX.search(text) or _V2_BOX.search(text)
    if bm:   # only the four coordinate VALUES (not the '1'/'2' of the x1/y2 labels)
        for g in range(1, 5):
            mark(m_b, bm.start(g), bm.end(g), digits_only=True)
    am = _A1.search(text)
    if am:
        mark(m_v, am.start() + len("A1:"), am.end())
    for vm in _VERDICT_LINE.finditer(text):
        mark(m_v, vm.start(), vm.end())
    m_v = [0 if m_b[j] else m_v[j] for j in range(n)]
    return m_v, m_b


def decoupled_token_advantages(r_verdict, r_box, box_valid, m_verdict, m_box, completion_mask, G,
                               lam_box=1.0, rest="verdict", scale_verdict=False, box_std_floor=0.05, eps=1e-4):
    """r_verdict: (B,) e.g. +4/-4 ; r_box: (B,) IoU in [0,1] ; box_valid: (B,) bool (correct verdict AND GT box).
    A_v = groupnorm(r_verdict) (Dr.GRPO style: no std scaling unless scale_verdict).
    A_b = (iou - mean_valid)/max(std_valid, floor), computed only over valid rollouts of the group, 0 elsewhere.
    Per-token: A_t = A_v*m_v + lam*A_b*m_b*valid + A_v*m_b*(1-valid) + (A_v if rest=='verdict' else 0)*m_rest."""
    B, T = completion_mask.shape
    Rv = r_verdict.view(-1, G).double()
    Av = Rv - Rv.mean(1, keepdim=True)
    if scale_verdict:
        Av = Av / (Rv.std(1, keepdim=True) + eps)
    Av = Av.reshape(-1)
    Rb = r_box.view(-1, G).double(); V = box_valid.view(-1, G).double()
    nvalid = V.sum(1, keepdim=True).clamp(min=1)
    mb = (Rb * V).sum(1, keepdim=True) / nvalid
    vb = (((Rb - mb) ** 2) * V).sum(1, keepdim=True) / (nvalid - 1).clamp(min=1)
    Ab = ((Rb - mb) / vb.sqrt().clamp(min=box_std_floor)) * V
    Ab = torch.where(V.sum(1, keepdim=True) >= 2, Ab, torch.zeros_like(Ab)).reshape(-1)
    mv = m_verdict.double(); mbx = m_box.double(); cm = completion_mask.double()
    m_rest = (1 - mv) * (1 - mbx) * cm
    valid = box_valid.double()[:, None]
    # box digits of a VALID rollout (correct verdict, GT box) get the box advantage; box digits of an invalid
    # rollout (e.g. a box emitted with a wrong 'deepfake' verdict) are part of the wrong decision -> verdict adv.
    A_tok = Av[:, None] * mv + lam_box * Ab[:, None] * mbx * valid + Av[:, None] * mbx * (1 - valid)
    if rest == "verdict":
        A_tok = A_tok + Av[:, None] * m_rest
    return A_tok * cm, Av, Ab


# --------------------------------------------------------------------------------------------
# (e) difference-map localisation reward
# --------------------------------------------------------------------------------------------
def change_map(forged, source, gt_box, dilate=0.25, out_pctl=99.0, min_thr=8.0, blur_sigma=3.0):
    """forged/source: HxW float arrays (grayscale, same size). Returns a non-negative map normalised to sum 1
    inside dilate(gt_box) (zeros elsewhere), or None if no change survives the outside-box noise threshold."""
    import numpy as np
    import cv2
    D = forged.astype(np.float32) - source.astype(np.float32)
    D -= np.median(D)
    D = cv2.GaussianBlur(np.abs(D), (0, 0), blur_sigma)
    H, W = D.shape
    x0, y0, x1, y1 = gt_box
    inside = np.zeros_like(D, dtype=bool); inside[y0:y1, x0:x1] = True
    thr = max(min_thr, float(np.percentile(D[~inside], out_pctl))) if (~inside).any() else min_thr
    bw, bh = x1 - x0, y1 - y0
    X0, Y0 = max(0, int(x0 - dilate * bw)), max(0, int(y0 - dilate * bh))
    X1, Y1 = min(W, int(x1 + dilate * bw)), min(H, int(y1 + dilate * bh))
    C = np.zeros_like(D)
    C[Y0:Y1, X0:X1] = np.clip(D[Y0:Y1, X0:X1] - thr, 0, None)
    s = C.sum()
    return (C / s) if s > 0 else None


def diffmass_reward(pred_box, C, gt_box, kappa=0.5, w_iou=0.5, iou_fn=None):
    """r = w_iou*IoU(pred, gt) + (1-w_iou)*[mass of C inside pred - kappa*max(0, area(pred)/area(gt) - 1)].
    Falls back to IoU only when C is None. Training-only; evaluation keeps GT-box IoU."""
    import numpy as np
    if pred_box is None:
        return 0.0
    i = iou_fn(pred_box, gt_box) if iou_fn else 0.0
    if C is None:
        return i
    x0, y0, x1, y1 = [int(v) for v in pred_box]
    H, W = C.shape
    x0, x1 = max(0, min(W, x0)), max(0, min(W, x1)); y0, y1 = max(0, min(H, y0)), max(0, min(H, y1))
    mass = float(C[y0:y1, x0:x1].sum()) if (x1 > x0 and y1 > y0) else 0.0
    ag = max(1, (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]))
    ap = max(0, (x1 - x0) * (y1 - y0))
    return w_iou * i + (1 - w_iou) * (mass - kappa * max(0.0, ap / ag - 1.0))


# --------------------------------------------------------------------------------------------
# (f) DAPO dynamic sampling
# --------------------------------------------------------------------------------------------
def dapo_keep_mask(rewards, G, eps=1e-6, correct=None, mode="any_variance"):
    """Return a (n_groups,) bool mask of groups to KEEP.
    mode='any_variance'  : std > eps (original DAPO)
    mode='verdict_split' : some but not all rollouts correct (needs `correct`, (B,) bool)."""
    R = rewards.view(-1, G)
    if mode == "verdict_split":
        C = correct.view(-1, G)
        return C.any(1) & ~C.all(1)
    return R.std(1) > eps
