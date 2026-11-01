"""
dryrun_ctgrpo.py -- MANDATORY CPU dry run of train_ctgrpo.py (CT-GRPO) with a tiny random-weight Qwen3-VL + LoRA.

README
------
Run (CPU only, no GPU):
  CUDA_VISIBLE_DEVICES= NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4 nice ~/.conda/envs/Qwen3.5/bin/python \
      ~/CIRCE/v3/dryrun_ctgrpo.py [--steps 3] [--G 4] [--K 2]
What it does:
  1. make_synth(): writes a tiny MF-Clean-format twin dataset to ~/CIRCE/v3/dryrun/ (256x256 gray JPEG reals,
     paste-back fakes with a bright square inside the GT box, rl_synth.jsonl with 6 twin pairs + 3 singleton reals).
  2. For each --adv_mode (image, twin_pooled, twin_pooled/pool=all, cogroup_bonus, gdpo): builds the tiny model
     (same seed), the real trl 0.16.1 GRPOTrainer + vlm_grpo_patch + CTGRPOTrainer, and trains --steps optimizer
     steps of K pairs x G rollouts. A random tiny model cannot write the v3 format, so model.generate is replaced
     by a SCRIPTED policy that returns tokenised v3 completions (correct / wrong / unparseable / jittered boxes,
     deterministic per (pair, role, rollout, step)); everything downstream (EOS mask, decoding, reward, advantage,
     patched loss forward with pixel_values, backward, optimizer) is the real code path.
  3. Checks: (a) twins in the same step and contiguous, and the step order equals TwinPairSampler.epoch_slots(0)
     (no reordering by Trainer/accelerate); (b) the advantages that reached compute_loss equal an independent
     loop-based reference (test_ctgrpo.ref_adv) recomputed from the per-row rewards in the diagnostics file;
     (c) LoRA weights changed; (d) diagnostics jsonl written with one record per step and all keys.
  4. Loss micro-batching: gradient with --loss_microbatch G equals the one-shot gradient (same batch).
  5. Prints a per-mode table (fraction of rows with non-zero advantage, verdict share of |A|) on IDENTICAL rewards.
Exit code 0 only if every check passes. Outputs: ~/CIRCE/v3/dryrun/ (data, run dirs, *_diag.jsonl).
"""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_paths.py").is_file())))
from release_paths import release_path as _release_path
import argparse
import json
import os
import random
import shutil
import sys
import zlib

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, _release_path('CIRCE/circe'))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import ctgrpo_core as core  # noqa: E402
import train_ctgrpo as T  # noqa: E402
from prompts_v3 import format_target  # noqa: E402
from test_ctgrpo import ref_adv  # noqa: E402

OUT = os.path.join(HERE, "dryrun")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))


# ------------------------------------------------------------------------------------------------- synthetic data
def make_synth(n_pairs=6, n_solo=3, size=256, seed=0):
    rng = np.random.default_rng(seed)
    d = os.path.join(OUT, "img")
    os.makedirs(d, exist_ok=True)
    lines = []

    def real_img(i):
        a = rng.normal(110, 25, (size // 8, size // 8)).clip(0, 255).astype(np.uint8)
        return np.asarray(Image.fromarray(a).resize((size, size), Image.BICUBIC), dtype=np.float32)

    for p in range(n_pairs):
        r = real_img(p)
        x1, y1 = int(rng.integers(10, size - 90)), int(rng.integers(10, size - 90))
        w = int(rng.integers(30, 80))
        box_px = [x1, y1, x1 + w, y1 + w]
        f = r.copy()
        f[y1:y1 + w, x1:x1 + w] += 60 if p % 2 == 0 else -60           # implant brighter / removal-like darker
        rp, fp = f"{d}/real_{p}.jpg", f"{d}/fake_{p}.jpg"
        Image.fromarray(r.clip(0, 255).astype(np.uint8), "L").save(rp, "JPEG", quality=90, subsampling=0)
        Image.fromarray(f.clip(0, 255).astype(np.uint8), "L").save(fp, "JPEG", quality=90, subsampling=0)
        s = 1024 / size
        box = [int(v * s) for v in box_px]
        g = f"gsyn{p}"
        lines.append({"fake": {"image": fp, "type": "edit" if p % 2 == 0 else "remove", "gen": "flux", "box": box,
                               "organ": "brain-mri", "group": g, "split": "train", "disease": "syn", "twin": rp,
                               "intensity": "brighter" if p % 2 == 0 else "flattened"},
                      "real": {"image": rp, "type": "real", "gen": "real", "box": None, "organ": "brain-mri",
                               "group": g, "split": "train", "disease": "syn", "twin": None}})
    for s_ in range(n_solo):
        rp = f"{d}/solo_{s_}.jpg"
        Image.fromarray(real_img(100 + s_).clip(0, 255).astype(np.uint8), "L").save(rp, "JPEG", quality=90)
        lines.append({"real": {"image": rp, "type": "real", "gen": "real", "box": None, "organ": "chest-xray",
                               "group": f"gsolo{s_}", "split": "train", "disease": "syn", "twin": None}})
    path = os.path.join(OUT, "rl_synth.jsonl")
    with open(path, "w") as fh:
        for l in lines:
            fh.write(json.dumps(l) + "\n")
    return path


# ------------------------------------------------------------------------------------------- scripted rollouts
def scripted_text(row, k, step):
    """Deterministic per (pair, role, rollout k, step). Pair archetypes (line index of pair_id 'u<i>' % 4):
       0: fake all WRONG, real all correct (twin-contrast-only case)   1: both mixed
       2: both all correct, fake boxes jittered (loc-only variance)     3: fake mixed, real all WRONG
       singletons: mixed incl. unparseable."""
    h = int(row["pair_id"].lstrip("u")) % 4
    rnd = random.Random(f"{row['pair_id']}|{row['role']}|{k}|{step}")
    fake = row["gt_verdict"] == "deepfake"
    if row["role"] == core.ROLE_SOLO:
        u = rnd.random()
        if u < 0.2:
            return "I am not sure about this one."
        correct = u < 0.65
    elif h == 0:
        correct = not fake
    elif h == 1:
        correct = rnd.random() < 0.5
    elif h == 2:
        correct = True
    else:
        correct = rnd.random() < 0.5 if fake else False
    if fake:
        if not correct:
            return format_target("real")
        b = row["gt_box"]
        j = [int(v + rnd.randint(-40, 40)) for v in b]
        j = [max(0, min(1023, v)) for v in j]
        j = [min(j[0], j[2]), min(j[1], j[3]), max(j[0], j[2]), max(j[1], j[3])]
        return format_target("deepfake", j, row["gt_type"], row["gt_intensity"] or "none")
    if correct:
        return format_target("real")
    b = [rnd.randint(0, 500), rnd.randint(0, 500)]
    return format_target("deepfake", [b[0], b[1], b[0] + 100, b[1] + 100], "implant", "brighter")


class DryTrainer(T.CTGRPOTrainer):
    """Only adds bookkeeping for the checks: stash the batch rows for the scripted generate, record the
    advantages that reach compute_loss."""
    seen_adv = []

    def _generate_and_score_completions(self, inputs):
        self._dry_inputs = inputs
        return super()._generate_and_score_completions(inputs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        DryTrainer.seen_adv.append(inputs["advantages"].detach().clone())
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


def install_scripted_generate(model, trainer_ref, tok, max_new):
    def fake_generate(input_ids, attention_mask=None, generation_config=None, **kw):
        tr = trainer_ref[0]
        rows = tr._dry_inputs
        G = tr.num_generations
        step = int(tr.state.global_step)
        seqs = []
        for b, row in enumerate(rows):
            t = scripted_text(row, b % G, step)
            ids = tok.encode(t, add_special_tokens=False)[: max_new - 1] + [T.IM_END]
            seqs.append(ids)
        L = max(len(s) for s in seqs)
        comp = torch.full((len(seqs), L), T.PAD, dtype=input_ids.dtype)
        for b, s in enumerate(seqs):
            comp[b, :len(s)] = torch.tensor(s, dtype=input_ids.dtype)
        assert input_ids.shape[0] == len(rows)
        return torch.cat([input_ids, comp], 1)
    model.generate = fake_generate


# ------------------------------------------------------------------------------------------------------- checks
def check_layout(diag, smp, K, G, rows_by_idx):
    exp_steps = smp.epoch_slots(0)
    for d, slots in zip(diag, exp_steps):
        keys = d["rows"]
        assert len(keys) == 2 * K, keys
        exp = []
        for _, idxs in slots:
            exp += [f"{rows_by_idx[i]['pair_id']}|{rows_by_idx[i]['role']}" for i in idxs]
        assert keys == exp, f"step order differs from sampler: {keys} vs {exp}"
        for i, k in enumerate(keys):
            p, r = k.split("|")
            if r == "fake":
                assert keys[i + 1] == f"{p}|real", f"twin not contiguous: {keys}"
            if r == "real":
                assert i > 0 and keys[i - 1] == f"{p}|fake", f"real without its fake: {keys}"
        assert not d["layout_errors"]


def run_mode(mode, rl_path, args, pool="verdict", bonus=0.0, steps=None, mb=0):
    tag = mode + ("_poolall" if pool == "all" else "")
    out = os.path.join(OUT, f"run_{tag}")
    shutil.rmtree(out, ignore_errors=True)
    diag_path = out + "_diag.jsonl"
    if os.path.exists(diag_path):
        os.remove(diag_path)
    a = T.parse_args(["--data", rl_path, "--out", out, "--adv_mode", mode, "--num_gen", str(args.G),
                      "--pairs_per_step", str(args.K), "--max_steps", str(steps or args.steps), "--lr", "5e-3",
                      "--pool_channels", pool, "--pair_bonus", str(bonus), "--max_completion", "96",
                      "--grad_ckpt", str(args.grad_ckpt), "--save_steps", "1000", "--loss_microbatch", str(mb),
                      "--regime", "ID"])
    rows, units, st = core.load_rl_rows(a.data, regime=a.regime)
    proc = T.load_processor(a.base)
    model = T.build_tiny_model(a.base, seed=0)
    ref = [None]
    install_scripted_generate(model, ref, proc.tokenizer, a.max_completion)
    DryTrainer.seen_adv = []
    G, K = a.num_gen, a.pairs_per_step
    # [review fix] build through the REAL make_trainer (same GRPOConfig as the GPU run) instead of a hand-copied
    # config; only checkpointing is switched off for the dry run.
    tr = T.make_trainer(model, proc, rows, units, a, use_cpu=True, extra_cfg={"save_strategy": "no"},
                        unit_real=st["unit_real"], trainer_cls=DryTrainer)
    ref[0] = tr
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    tr.train()
    changed = sum(int(not torch.equal(before[n], p.detach())) for n, p in model.named_parameters() if p.requires_grad)
    diag = [json.loads(l) for l in open(diag_path)]
    res = {"mode": tag, "steps": len(diag), "lora_changed": changed, "lora_total": len(before)}
    # (d) diagnostics
    need = ("step", "img_std", "twin_std", "frac_zero_std_img", "frac_zero_std_twin", "frac_verdict_split_img",
            "frac_verdict_split_twin", "mean_abs_adv_verdict", "mean_abs_adv_loc", "acc_fake", "acc_real",
            "comp_len_mean", "rows", "row_r_verdict", "row_adv")
    assert len(diag) == a.max_steps, (len(diag), a.max_steps)
    missing = [k for k in need if k not in diag[0]]
    assert not missing, missing
    # (a) layout vs the sampler's own plan
    smp = core.TwinPairSampler(units, G, K, seed=a.seed)
    check_layout(diag, smp, K, G, rows)
    # (b) advantages reaching compute_loss == independent reference recomputed from the logged per-row rewards
    n_loss_calls = len(DryTrainer.seen_adv)
    per_step = DryTrainer.seen_adv if not mb else [torch.cat(DryTrainer.seen_adv[i:i + 2 * K * G // mb])
                                                    for i in range(0, n_loss_calls, 2 * K * G // mb)]
    maxdiff = 0.0
    for d, seen in zip(diag, per_step):
        keys = [k for k in d["rows"] for _ in range(G)]
        pids = [k.split("|")[0] for k in keys]
        roles = [k.split("|")[1] for k in keys]
        lay = core.layout_group_ids(pids, roles, G)
        exp = ref_adv(d["row_r_verdict"], d["row_r_loc"], lay["img_gid"].tolist(), lay["pair_gid"].tolist(),
                      d["row_correct"], lay["partner"].tolist(), mode, True, 0.0, bonus, pool)
        maxdiff = max(maxdiff, max(abs(x - y) for x, y in zip(seen.tolist(), exp)))
    res["adv_vs_reference_maxdiff"] = maxdiff
    res["frac_rows_nonzero_adv"] = float(np.mean([abs(x) > 1e-8 for d in diag for x in d["row_adv"]]))
    res["adv_share_verdict"] = float(np.mean([d["adv_share_verdict"] for d in diag]))
    res["frac_zero_std_img"] = float(np.mean([d["frac_zero_std_img"] for d in diag]))
    res["frac_twin_contrast_only"] = float(np.mean([d["frac_twin_contrast_only"] for d in diag]))
    res["loss"] = [round(d["loss"], 5) for d in diag]
    res["step_s"] = [d["step_s"] for d in diag]
    res["row_reward_hash"] = zlib.crc32(json.dumps([d["row_r_verdict"] + d["row_r_loc"] for d in diag]).encode())
    res["row_adv_all"] = [x for d in diag for x in d["row_adv"]]
    res["r_bonus_mean"] = float(np.mean([d["r_bonus_mean"] for d in diag]))
    ok = (changed > 0 or res["frac_rows_nonzero_adv"] == 0) and maxdiff < 1e-4
    res["ok"] = bool(ok and changed == len(before))
    return res, tr, model


def microbatch_equivalence(rl_path, args):
    """Same batch, gradient with one loss forward vs --loss_microbatch G chunks (re-weighted by token share)."""
    _, tr, model = run_mode("twin_pooled", rl_path, args, steps=1)
    smp = core.TwinPairSampler(tr._ct_units, tr.num_generations, tr._ct_K, seed=11)
    idx = list(iter(smp))[: 2 * tr._ct_K * tr.num_generations]
    inputs = [tr.train_dataset[i] for i in idx]
    batch = tr._generate_and_score_completions(inputs)
    tr._ct_pending = None
    tr._prepare_inputs = lambda inp: batch
    tr.current_gradient_accumulation_steps = 1
    grads = []
    for mb in (0, tr.num_generations):
        tr._ct_mb = mb
        model.zero_grad(set_to_none=True)
        loss = tr.training_step(model, inputs)
        grads.append(({n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad
                       and p.grad is not None}, float(loss)))
    g0, l0 = grads[0]
    g1, l1 = grads[1]
    assert g0.keys() == g1.keys() and len(g0) > 0
    md = max(float((g0[k] - g1[k]).abs().max()) for k in g0)
    scale = max(float(g0[k].abs().max()) for k in g0)
    return {"n_grad_tensors": len(g0), "max_abs_grad_diff": md, "max_abs_grad": scale, "loss_full": l0,
            "loss_chunked": l1, "ok": md <= 1e-5 * max(1.0, scale) and abs(l0 - l1) < 1e-5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rl = make_synth()
    results = []
    for mode, pool, bonus in [("image", "verdict", 0.0), ("twin_pooled", "verdict", 0.0), ("twin_pooled", "all", 0.0),
                              ("cogroup_bonus", "verdict", 0.5), ("gdpo", "verdict", 0.0)]:
        res, _, _ = run_mode(mode, rl, args, pool=pool, bonus=bonus)
        print("[dryrun]", json.dumps(res), flush=True)
        results.append(res)
    hashes = {r["row_reward_hash"] for r in results}
    same_rewards = len(hashes) == 1
    by = {r["mode"]: r for r in results}
    # the modes must actually differ on identical rewards (otherwise a mode switch is silently ignored)
    differ = {m: max(abs(x - y) for x, y in zip(by[m]["row_adv_all"], by["image"]["row_adv_all"]))
              for m in by if m != "image"}
    bonus_active = by["cogroup_bonus"]["r_bonus_mean"] > 0
    print(f"[dryrun] max |adv(mode) - adv(image)| on identical rewards: {json.dumps(differ)}; "
          f"cogroup bonus mean={by['cogroup_bonus']['r_bonus_mean']:.4f}")
    for r in results:
        r.pop("row_adv_all")
    mbres = microbatch_equivalence(rl, args)
    print("[dryrun] microbatch:", json.dumps(mbres), flush=True)
    print("\n[dryrun] mode               steps  lora_changed  adv==ref(maxdiff)  rows_nonzero_adv  verdict_share  "
          "zero_std_img  twin_contrast_only")
    for r in results:
        print(f"[dryrun] {r['mode']:<18} {r['steps']:>5}  {r['lora_changed']:>4}/{r['lora_total']:<6}  "
              f"{r['adv_vs_reference_maxdiff']:.2e}           {r['frac_rows_nonzero_adv']:.3f}             "
              f"{r['adv_share_verdict']:.3f}          {r['frac_zero_std_img']:.3f}         "
              f"{r['frac_twin_contrast_only']:.3f}")
    print(f"[dryrun] identical per-row raw rewards across all modes: {same_rewards}")
    all_ok = (all(r["ok"] for r in results) and mbres["ok"] and same_rewards and bonus_active
              and all(v > 1e-3 for v in differ.values()))
    print("[dryrun] ALL CHECKS PASSED" if all_ok else "[dryrun] SOME CHECK FAILED")
    json.dump({"results": results, "microbatch": mbres, "same_rewards": same_rewards, "adv_differs_from_image": differ,
               "all_ok": all_ok},
              open(os.path.join(OUT, "dryrun_summary.json"), "w"), indent=1)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
