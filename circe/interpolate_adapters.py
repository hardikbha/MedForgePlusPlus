"""
CIRCE-v2 Phase 13: LoRA adapter interpolation.

Motivation: every GRPO variant tried this project shows the same tradeoff -- RL training
consistently improves localization (mIoU) while eroding classification accuracy relative
to the SFT checkpoint it started from, and this holds across every reward function, beta,
and step count tried. Instead of another RL hyperparameter guess, blend the SFT and RL
LoRA weights directly (no training) at a low alpha, betting the tradeoff is smooth enough
that a small RL contribution keeps most of the SFT accuracy while adding some of the RL
localization gain.

    interpolated_tensor = (1 - alpha) * sft_tensor + alpha * rl_tensor

Only LoRA adapter tensors are touched. Verifies identical keys/shapes before blending.
"""
import argparse, json, os, shutil
import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_adapter", required=True)
    ap.add_argument("--rl_adapter", required=True)
    ap.add_argument("--alpha", type=float, required=True, help="RL weight; 0=pure SFT, 1=pure RL")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sft_path = os.path.join(args.sft_adapter, "adapter_model.safetensors")
    rl_path = os.path.join(args.rl_adapter, "adapter_model.safetensors")
    sft_sd = load_file(sft_path)
    rl_sd = load_file(rl_path)

    assert set(sft_sd.keys()) == set(rl_sd.keys()), (
        f"key mismatch: {set(sft_sd.keys()) ^ set(rl_sd.keys())}")
    for k in sft_sd:
        assert sft_sd[k].shape == rl_sd[k].shape, f"shape mismatch at {k}: {sft_sd[k].shape} vs {rl_sd[k].shape}"

    merged = {}
    for k in sft_sd:
        merged[k] = ((1 - args.alpha) * sft_sd[k].float() + args.alpha * rl_sd[k].float()).to(sft_sd[k].dtype)

    os.makedirs(args.out, exist_ok=True)
    save_file(merged, os.path.join(args.out, "adapter_model.safetensors"))
    # copy the rest of the adapter dir (config, tokenizer, etc.) from the SFT side -- both
    # adapters share the same base model / LoRA config, verified via matching keys above
    for fname in os.listdir(args.sft_adapter):
        if fname == "adapter_model.safetensors":
            continue
        src = os.path.join(args.sft_adapter, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, fname))

    print(f"[interpolate] alpha={args.alpha} ({len(merged)} tensors) -> {args.out}")


if __name__ == "__main__":
    main()
