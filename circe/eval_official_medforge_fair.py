"""
Fair re-evaluation of the officially released MedForge-Reasoner checkpoint, using ITS OWN
documented system/user prompt and generation budget (from the model's HF README "Usage"
section) rather than CIRCE's own prompt -- so the apples-to-apples comparison isn't
confounded by feeding an externally-trained model an unfamiliar instruction format.

Reuses eval_medforge.py's run_eval() unchanged, only overriding the module-level SYS/USER1
it closes over and forcing --max_new to the officially recommended 2048.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import eval_medforge

# verbatim from https://huggingface.co/RichardChenZH/MedForge-Reasoner "Usage" section
eval_medforge.SYS = (
    "You are an expert in medical image forensics. Analyze the provided image "
    "to determine if it is a deepfake or authentic. First, perform a step-by-step "
    "examination of the image content, looking for artifacts, inconsistencies, or "
    "biological implausibilities. Use <think> tags to articulate your reasoning process. "
    "If you identify manipulated regions, localize them using bounding boxes within your reasoning. "
    "Conclude your analysis with a final classification."
)
eval_medforge.USER1 = "Is this image deepfake or real?"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_gen", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=2048)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--conf", action="store_true")
    args = ap.parse_args()
    eval_medforge.run_eval(args)
