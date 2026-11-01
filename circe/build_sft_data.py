"""
S0 SFT data builder — convert local MedForge++ traces into CIRCE's two-turn
SELF-QUESTIONING SFT format.

Input : the existing per-subset `sft_*.jsonl` (full reasoning) + `mapping_*.json` (GT box/type).
Output: `circe_sft.jsonl` — records of {images:[...], messages:[system, user1, asst1, user2, asst2]}.

Two modes:
  --mode template : deterministic self_ask block derived from the existing evidence/verdict
                    (fast, no GPU) — used for the smoke test and for bootstrapping.
  --mode teacher  : (stub) fill self_ask via Qwen3-VL-8B teacher rollouts — GPU, wired later.

Run: python build_sft_data.py --root <naitik_annotated dir> --out circe_sft.jsonl --limit 0
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import json, os, glob, re, argparse

DEEPFAKE_TYPES = {"remove", "inject", "deepfake", "removal", "injection"}
SYS = ("You are an expert medical-image forensics analyst. Decide whether a medical image is a "
       "DEEPFAKE (a lesion was digitally IMPLANTED or REMOVED by an image-editing / generative "
       "model) or an AUTHENTIC scan. Examine the image systematically for manipulation cues: "
       "(1) boundary artifacts — unnatural edges, blending halos, or smoothing where a lesion "
       "meets tissue or where tissue was erased; (2) texture inconsistency — regions whose "
       "noise/granularity differs from the surrounding parenchyma; (3) intensity/density "
       "implausibility — values inconsistent with the tissue type or contrast physics; "
       "(4) anatomical implausibility — structures of impossible shape or location. Ask yourself "
       "organ-neutral diagnostic questions, reason step by step inside <think> tags, and if you "
       "find a manipulated region localize it with a bounding box. Conclude with a final "
       "classification: deepfake or real.")
USER1 = ("<image>Is this image a deepfake or real? Examine it for implantation or removal artifacts, "
         "ask yourself diagnostic questions, reason step by step, localize any manipulated region "
         "with a bounding box, then give your final verdict.")
USER2 = ("Now zoom into the region you localized and re-check at higher resolution: is the box tight "
         "and the verdict correct? Give a correction.")

_EVID = re.compile(r'<evidence>(.*?)</evidence>', re.S)
_DESC = re.compile(r'<description>(.*?)</description>', re.S)


def self_ask_block(ftype, evidence, organ):
    """Forgery-TYPE-aware Q/A. inject vs remove have OPPOSITE tells:
       inject = added object w/ boundary/blending artifacts;
       remove = inpainted region that is over-smoothed / too-homogeneous / missing an expected structure.
    """
    ev = (evidence or "").strip().replace("\n", " ")
    if ftype in ("inject", "injection"):
        a1 = ("Yes — an added focal structure with a subtly artificial edge / blending halo where it "
              "meets surrounding tissue.")
        a2 = ev[:220] if ev else ("The lesion's internal texture does not match the surrounding-tissue "
                                   "noise statistics — a generative-insertion signature.")
        a3 = "The added region's intensity is not fully consistent with the surrounding parenchyma."
    elif ftype in ("remove", "removal"):
        a1 = ("No sharp discontinuity — instead a region that looks TOO smooth/seamless, the signature "
              "of inpainting that erased a structure.")
        a2 = ("The area is unnaturally homogeneous and lacks the fine granular texture of real "
              "parenchyma; an anatomically expected structure appears absent or painted over.")
        a3 = ("Density is locally over-uniform where a lesion or structure should be — consistent with "
              "digital removal, not a real scan.")
    else:  # real
        a1 = "No abnormal boundary discontinuity; contours look anatomically consistent."
        a2 = ev[:220] if ev else "Texture matches surrounding-tissue statistics; normal fine detail present."
        a3 = "The intensity/density is physiologically plausible for this organ."
    return (f"    Q1: Is there a boundary discontinuity or unnatural smoothing around any structure?  A1: {a1}\n"
            f"    Q2: Does local texture match the surrounding-tissue noise statistics?  A2: {a2}\n"
            f"    Q3: Is the density/intensity physiologically plausible for this organ?  A3: {a3}")


def box_token(ref, box):
    if not box:
        return "(no manipulated region identified)"
    return (f'<|object_ref_start|>"{ref}"<|object_ref_end|>'
            f'<|box_start|>x1="{box[0]}" y1="{box[1]}" x2="{box[2]}" y2="{box[3]}"<|box_end|>')


def build_record(img_path, orig_assistant, mp, mode="template"):
    verdict = "deepfake" if mp["type"] in DEEPFAKE_TYPES else "real"
    box = mp.get("box")
    organ, disease = mp["dataset"], mp["disease"]
    ref = "deepfake" if verdict == "deepfake" else disease
    desc = (_DESC.search(orig_assistant).group(1).strip()
            if _DESC.search(orig_assistant) else f"{organ} medical image")
    evid = _EVID.search(orig_assistant).group(1) if _EVID.search(orig_assistant) else ""
    verdict_line = "This image is a deepfake." if verdict == "deepfake" else "This image is a real image."

    asst1 = (f"<think>\n  <description>{desc}</description>\n"
             f"  <self_ask>\n{self_ask_block(mp['type'], evid, organ)}\n  </self_ask>\n"
             f"  <evidence cue=\"boundary_discontinuity\">{evid.strip()}</evidence>\n"
             f"  <conclusion>Findings are consistent with a {ref} region.</conclusion>\n"
             f"  {box_token(ref, box)}\n</think>\n\n{verdict_line}")
    if box:
        asst2 = (f"<sc><repair>Re-examining the crop, the initial box may be loose; the salient "
                 f"region is tightly bounded by the GT extent.</repair>"
                 f"<fix>{verdict_line} {box_token(ref, box)}</fix></sc>")
    else:
        asst2 = (f"<sc><repair>No manipulated region on higher-resolution re-check.</repair>"
                 f"<fix>{verdict_line}</fix></sc>")
    return {"images": [img_path],
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": USER1},
                         {"role": "assistant", "content": asst1},
                         {"role": "user", "content": USER2},
                         {"role": "assistant", "content": asst2}]}


## ---------------------------------------------------------------------------
## CIRCE-v2: evidence-first format (localization before explanation/verdict).
## Kept fully separate from build_record/SYS/USER1 above -- legacy format is
## untouched and remains the default everywhere. Opt in via --format v2.
## ---------------------------------------------------------------------------

SYS_V2 = ("You are an expert medical-image forensics analyst. Decide whether a medical image is a "
          "DEEPFAKE (a lesion was digitally IMPLANTED or REMOVED by an image-editing / generative "
          "model) or an AUTHENTIC scan. First localize any manipulated region (or state none), THEN "
          "give concise forensic evidence, THEN give your final verdict, in exactly this format:\n"
          "<localization>\n<box>x1 y1 x2 y2</box>\n</localization>\n"
          "<evidence>\n<boundary>...</boundary>\n<texture>...</texture>\n<intensity>...</intensity>"
          "\n<anatomy>...</anatomy>\n</evidence>\n<verdict>deepfake|real</verdict>\n"
          "If there is no manipulated region, use <none/> instead of <box>...</box> inside "
          "<localization>. Keep each evidence field to one short sentence.")
USER1_V2 = "<image>Is this image a deepfake or real? Follow the required output format exactly."

_EVID_FIELD_TEXT = {
    "inject": {
        "boundary": "Added focal structure with a subtly artificial edge / blending halo where it meets surrounding tissue.",
        "texture": "The region's internal texture does not match the surrounding-tissue noise statistics.",
        "intensity": "Intensity is not fully consistent with the surrounding parenchyma.",
        "anatomy": "An added structure at this location is not anatomically expected.",
    },
    "remove": {
        "boundary": "No sharp discontinuity, but an unnaturally smooth/seamless area consistent with inpainting.",
        "texture": "The area is unnaturally homogeneous, lacking the fine granular texture of real parenchyma.",
        "intensity": "Density is locally over-uniform where a lesion or structure should be present.",
        "anatomy": "An anatomically expected structure appears absent or painted over.",
    },
    "real": {
        "boundary": "No abnormal boundary discontinuity; contours look anatomically consistent.",
        "texture": "Texture matches surrounding-tissue statistics; normal fine detail present.",
        "intensity": "Intensity/density is physiologically plausible for this organ.",
        "anatomy": "Structures are anatomically plausible for this organ.",
    },
}


def build_record_v2(img_path, orig_assistant, mp):
    """CIRCE-v2 evidence-first single-turn record: <localization> (box-or-none) emitted
    BEFORE <evidence>/<verdict>, per the evidence-first ablation. Parseable by
    circe_reward.parse_output (tries v2 format first, legacy fallback second)."""
    verdict = "deepfake" if mp["type"] in DEEPFAKE_TYPES else "real"
    box = mp.get("box")
    ftype = mp["type"] if mp["type"] in _EVID_FIELD_TEXT else ("inject" if verdict == "deepfake" else "real")
    fields = _EVID_FIELD_TEXT[ftype]
    ev = _EVID.search(orig_assistant).group(1).strip() if _EVID.search(orig_assistant) else ""
    loc_block = (f'<box>{box[0]} {box[1]} {box[2]} {box[3]}</box>' if box else '<none/>')
    asst = ("<localization>\n" + loc_block + "\n</localization>\n"
            "<evidence>\n"
            f"<boundary>{fields['boundary']}</boundary>\n"
            f"<texture>{(ev[:180] if ev else fields['texture'])}</texture>\n"
            f"<intensity>{fields['intensity']}</intensity>\n"
            f"<anatomy>{fields['anatomy']}</anatomy>\n"
            "</evidence>\n"
            f"<verdict>{verdict}</verdict>")
    return {"images": [img_path],
            "messages": [{"role": "system", "content": SYS_V2},
                         {"role": "user", "content": USER1_V2},
                         {"role": "assistant", "content": asst}]}


def convert_subset(subdir, mode="template", limit=0, record_fn=None):
    # merge ALL mapping_*.json + sft_*.jsonl shards (subsets may have leftover
    # partial files like *_0_2 alongside the full *_0_1000); dedupe by image path.
    record_fn = record_fn or (lambda img, asst, mp: build_record(img, asst, mp, mode))
    mp_f = glob.glob(os.path.join(subdir, "mapping_*.json"))
    sf_f = glob.glob(os.path.join(subdir, "sft_*.jsonl"))
    if not mp_f or not sf_f:
        return []
    mapping = {}
    for f in mp_f:
        for m in json.load(open(f)):
            mapping[m["key"]] = m          # later/complete shard overrides
    out, seen = [], set()
    for f in sorted(sf_f):
        for line in open(f):
            rec = json.loads(line)
            img = rec["images"][0]
            key = os.path.basename(img)
            if key in seen or key not in mapping:
                continue
            seen.add(key)
            orig_asst = rec["messages"][-1]["content"]
            out.append(record_fn(img, orig_asst, mapping[key]))
            if limit and len(out) >= limit:
                return out
    return out


# injection manifests (IMPLANT forgeries) — flat {images, box, klass, forgery_type, organ, messages}
INJECT_SOURCES = [
    ("colon_inject", [_release_path('AAAI/MedForge++/colon_injections/annotations.jsonl')]),
    ("kidney_inject", sorted(glob.glob(_release_path('AAAI/kidney/flux_injected/manifest_*.jsonl')))),
]


def convert_injection(paths, mode="template", limit=0, record_fn=None):
    record_fn = record_fn or (lambda img, asst, mp: build_record(img, asst, mp, mode))
    out, seen = [], set()
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            img = r["images"][0]
            if img in seen or not os.path.exists(img):
                continue
            seen.add(img)
            orig_asst = r["messages"][-1]["content"] if r.get("messages") else ""
            mp = {"type": "inject", "box": r.get("box"), "dataset": r.get("organ", "?"),
                  "disease": r.get("klass", "lesion")}
            out.append(record_fn(img, orig_asst, mp))
            if limit and len(out) >= limit:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=_release_path('AAAI/MedForge++/naitik_annotated'))
    ap.add_argument("--out", default="circe_sft.jsonl")
    ap.add_argument("--mode", choices=["template", "teacher"], default="template")
    ap.add_argument("--format", choices=["legacy", "v2"], default="legacy",
                     help="legacy=reasoning-first (default, unchanged); v2=evidence-first (CIRCE-v2 ablation B)")
    ap.add_argument("--limit", type=int, default=0, help="per-subset cap (0=all)")
    ap.add_argument("--balance", action="store_true", help="downsample real to match deepfake count")
    args = ap.parse_args()
    if args.mode == "teacher":
        raise SystemExit("teacher mode (Qwen3-VL-8B rollouts) not wired yet — needs GPU; use --mode template")
    record_fn = (lambda img, asst, mp: build_record_v2(img, asst, mp)) if args.format == "v2" else None
    subsets = sorted(d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d))
    real_recs, fake_recs = [], []
    def bucket(recs):
        for r in recs:
            asst_text = r["messages"][-1]["content"]
            is_fake = ("deepfake" in asst_text.lower())  # matches both legacy verdict line and v2 <verdict>deepfake</verdict>
            (fake_recs if is_fake else real_recs).append(r)
    for sd in subsets:
        recs = convert_subset(sd, args.mode, args.limit, record_fn=record_fn)
        bucket(recs); print(f"  {os.path.basename(sd):18s} -> {len(recs)}")
    for name, paths in INJECT_SOURCES:
        recs = convert_injection(paths, args.mode, args.limit, record_fn=record_fn)
        bucket(recs); print(f"  {name:18s} -> {len(recs)}  (IMPLANT)")
    if args.balance and len(real_recs) > len(fake_recs):
        step = len(real_recs) / len(fake_recs)
        real_recs = [real_recs[int(i * step)] for i in range(len(fake_recs))]
        print(f"  [balance] real downsampled to {len(real_recs)}")
    allr = real_recs + fake_recs
    with open(args.out, "w") as f:
        for r in allr:
            f.write(json.dumps(r) + "\n")
    print(f"TOTAL: {len(allr)} SFT records ({len(fake_recs)} deepfake / {len(real_recs)} real) -> {args.out}")


if __name__ == "__main__":
    main()
