"""
Monkeypatch trl-0.16 GRPOTrainer for Qwen3-VL (VLM) GRPO.

trl-0.16 is text-only: `_generate_and_score_completions` calls the processor with `text=` only (no
`images=`) and `generate()` / the log-prob forward get no `pixel_values`, so the VLM generates blind
(garbage) and the reward has zero signal. This patch threads `pixel_values` + `image_grid_thw` through:
  (a) prompt processing, (b) generation, (c) the policy log-prob forward in compute_loss.
Assumes beta==0 and num_iterations==1 (our config) so the ref/old-logp forwards are skipped.
Each dataset row must carry an "image" path (build_prompts adds it).
"""
import torch
from PIL import Image
import trl.trainer.grpo_trainer as G
from trl.trainer.grpo_trainer import GRPOTrainer


_BATCH_FIRST_KEYS = (
    "prompt_ids", "prompt_mask", "completion_ids", "completion_mask",
    "old_per_token_logps", "ref_per_token_logps", "advantages",
    "mm_token_type_ids",
)


def _sample_batch_size(tensor_dict):
    for key in _BATCH_FIRST_KEYS:
        tensor = tensor_dict.get(key)
        if tensor is not None:
            return tensor.shape[0]
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    return first_tensor.shape[0]


def _grid_counts(grid):
    return grid.prod(dim=1).detach().cpu().tolist()


def _slice_visual_tensor(pixels, grid, row_start, row_end):
    if pixels is None or grid is None:
        return pixels
    counts = _grid_counts(grid)
    total = sum(counts)
    if pixels.shape[0] != total:
        return pixels[row_start:row_end]
    patch_start = sum(counts[:row_start])
    patch_end = sum(counts[:row_end])
    return pixels[patch_start:patch_end]


def _permute_visual_tensor(pixels, grid, permutation):
    if pixels is None or grid is None:
        return pixels
    counts = _grid_counts(grid)
    total = sum(counts)
    if pixels.shape[0] != total:
        return pixels[permutation]
    chunks = list(torch.split(pixels, counts, dim=0))
    return torch.cat([chunks[i] for i in permutation.detach().cpu().tolist()], dim=0)


def _split_tensor_dict_vlm(tensor_dict, num_chunks):
    batch_size = _sample_batch_size(tensor_dict)
    chunk_size = batch_size // num_chunks
    image_grid = tensor_dict.get("image_grid_thw")
    video_grid = tensor_dict.get("video_grid_thw")
    chunks = []
    for i in range(num_chunks):
        row_start = i * chunk_size
        row_end = (i + 1) * chunk_size
        chunk = {}
        for key, tensor in tensor_dict.items():
            if tensor is None:
                chunk[key] = None
            elif key == "image_grid_thw":
                chunk[key] = tensor[row_start:row_end]
            elif key == "pixel_values":
                chunk[key] = _slice_visual_tensor(tensor, image_grid, row_start, row_end)
            elif key == "video_grid_thw":
                chunk[key] = tensor[row_start:row_end]
            elif key == "pixel_values_videos":
                chunk[key] = _slice_visual_tensor(tensor, video_grid, row_start, row_end)
            else:
                chunk[key] = tensor[row_start:row_end]
        chunks.append(chunk)
    return chunks


def _shuffle_tensor_dict_vlm(tensor_dict):
    batch_size = _sample_batch_size(tensor_dict)
    permutation = torch.randperm(batch_size)
    image_grid = tensor_dict.get("image_grid_thw")
    video_grid = tensor_dict.get("video_grid_thw")
    out = {}
    for key, tensor in tensor_dict.items():
        if tensor is None:
            out[key] = None
        elif key == "image_grid_thw":
            out[key] = tensor[permutation]
        elif key == "pixel_values":
            out[key] = _permute_visual_tensor(tensor, image_grid, permutation)
        elif key == "video_grid_thw":
            out[key] = tensor[permutation]
        elif key == "pixel_values_videos":
            out[key] = _permute_visual_tensor(tensor, video_grid, permutation)
        else:
            out[key] = tensor[permutation]
    return out


def _img_kwargs_from(self, inputs):
    """Build pixel_values/image_grid_thw for the batch's images, on device."""
    images = [Image.open(x["image"]).convert("RGB") for x in inputs]
    proc_out = self.processing_class(images=images, return_tensors="pt")
    kw = {}
    for k in ("pixel_values", "image_grid_thw"):
        if k in proc_out:
            kw[k] = proc_out[k].to(self.accelerator.device)
    return kw


def _clean_msgs(prompt):
    """Drop null 'image'/'text' keys so the chat template doesn't emit a phantom image placeholder
    for text items (build_prompts keeps both keys for pyarrow schema uniformity)."""
    out = []
    for m in prompt:
        content = []
        for it in m["content"]:
            if it.get("type") == "image":
                item = {"type": "image"}
                if it.get("image") is not None:
                    item["image"] = it.get("image")
                content.append(item)
            else:
                content.append({"type": "text", "text": it.get("text")})
        out.append({"role": m["role"], "content": content})
    return out


def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, **vis):
    logits = model(input_ids=input_ids, attention_mask=attention_mask,
                   logits_to_keep=logits_to_keep + 1, **vis).logits
    logits = logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    logits = logits / self.temperature
    return G.selective_log_softmax(logits, input_ids)


def _generate_and_score_completions(self, inputs):
    device = self.accelerator.device
    prompts = [x["prompt"] for x in inputs]
    # mirror the PROVEN eval path: apply_chat_template per prompt + NESTED image list [[img], ...]
    prompts_text = [self.processing_class.apply_chat_template(
        _clean_msgs(x["prompt"]), tokenize=False, add_generation_prompt=True) for x in inputs]
    images = [[Image.open(x["image"]).convert("RGB")] for x in inputs]
    prompt_inputs = self.processing_class(
        text=prompts_text, images=images, return_tensors="pt", padding=True,
        padding_side="left", add_special_tokens=False)
    prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    vis = {k: prompt_inputs[k] for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids")
           if k in prompt_inputs}

    if self.max_prompt_length is not None:
        prompt_ids = prompt_ids[:, -self.max_prompt_length:]
        prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        if "mm_token_type_ids" in vis:
            vis["mm_token_type_ids"] = vis["mm_token_type_ids"][:, -self.max_prompt_length:]

    with G.unwrap_model_for_generation(
            self.model_wrapped, self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation) as unwrapped_model:
        was_training = unwrapped_model.training
        unwrapped_model.eval()
        try:
            with torch.no_grad():
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config, **vis)
        finally:
            if was_training:
                unwrapped_model.train()
    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_completion_ids[:, :prompt_length]
    completion_ids = prompt_completion_ids[:, prompt_length:]

    is_eos = completion_ids == self.processing_class.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)

    # our config: num_iterations==1, beta==0 -> both are None (skip ref/old forwards)
    old_per_token_logps = None
    ref_per_token_logps = None

    completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
    if not getattr(self, "_vlm_completion_logged", False):
        for j, text in enumerate(completions_text[:min(4, len(completions_text))]):
            clen = int(completion_mask[j].sum().item())
            eos_seen = bool(is_eos[j].any().item())
            print(f"[vlm_grpo_patch] completion[{j}] len={clen} eos={eos_seen} "
                  f"head={text[:300]!r} tail={text[-300:]!r}")
        self._vlm_completion_logged = True
    if G.is_conversational(inputs[0]):
        completions = []
        for prompt, completion in zip(prompts, completions_text):
            bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
            completions.append([{"role": "assistant", "content": bootstrap + completion}])
    else:
        completions = completions_text

    rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
    for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)):
        keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
        output_reward_func = [r if r is not None else torch.nan for r in output_reward_func]
        rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

    rewards_per_func = G.gather(rewards_per_func)
    rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
    mean_grouped = rewards.view(-1, self.num_generations).mean(dim=1)
    std_grouped = rewards.view(-1, self.num_generations).std(dim=1)
    mean_grouped = mean_grouped.repeat_interleave(self.num_generations, dim=0)
    std_grouped = std_grouped.repeat_interleave(self.num_generations, dim=0)
    advantages = rewards - mean_grouped
    if self.args.scale_rewards:
        advantages = advantages / (std_grouped + 1e-4)
    process_slice = slice(self.accelerator.process_index * len(prompts),
                          (self.accelerator.process_index + 1) * len(prompts))
    advantages = advantages[process_slice]

    mode = "eval" if self.control.should_evaluate else "train"
    if mode == "train":
        n_tokens = self.accelerator.gather(attention_mask.sum()).sum().item()
        if hasattr(self, "_total_train_tokens"):
            self._total_train_tokens += n_tokens
            total_tokens = self._total_train_tokens
        else:
            self.state.num_input_tokens_seen += n_tokens
            total_tokens = self.state.num_input_tokens_seen
    else:
        total_tokens = getattr(self, "_total_train_tokens", getattr(self.state, "num_input_tokens_seen", 0))
    self._metrics[mode]["num_tokens"] = [total_tokens]
    comp_len = self.accelerator.gather(completion_mask.sum(1)).float().mean().item()
    self._metrics[mode]["completion_length"].append(comp_len)
    for i, reward_func in enumerate(self.reward_funcs):
        name = reward_func.__name__ if not isinstance(reward_func, torch.nn.Module) else reward_func.config._name_or_path.split("/")[-1]
        self._metrics[mode][f"rewards/{name}"].append(torch.nanmean(rewards_per_func[:, i]).item())
    self._metrics[mode]["reward"].append(rewards.mean().item())
    self._metrics[mode]["reward_std"].append(std_grouped.mean().item())

    out = {"prompt_ids": prompt_ids, "prompt_mask": prompt_mask,
           "completion_ids": completion_ids, "completion_mask": completion_mask,
           "old_per_token_logps": old_per_token_logps, "ref_per_token_logps": ref_per_token_logps,
           "advantages": advantages}
    out.update(vis)  # carry pixel_values/image_grid_thw to compute_loss
    return out


def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    if return_outputs:
        raise ValueError("The GRPOTrainer does not support returning outputs")
    prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
    completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)
    vis = {k: inputs[k] for k in ("pixel_values", "image_grid_thw") if k in inputs}
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        if mm_token_type_ids.shape[1] == prompt_ids.shape[1]:
            completion_types = mm_token_type_ids.new_zeros(completion_ids.shape)
            mm_token_type_ids = torch.cat([mm_token_type_ids, completion_types], dim=1)
        elif mm_token_type_ids.shape[1] < input_ids.shape[1]:
            pad = mm_token_type_ids.new_zeros(
                (mm_token_type_ids.shape[0], input_ids.shape[1] - mm_token_type_ids.shape[1])
            )
            mm_token_type_ids = torch.cat([mm_token_type_ids, pad], dim=1)
        elif mm_token_type_ids.shape[1] > input_ids.shape[1]:
            mm_token_type_ids = mm_token_type_ids[:, -input_ids.shape[1]:]
        vis["mm_token_type_ids"] = mm_token_type_ids
    if vis and not getattr(self, "_vlm_shape_logged", False):
        print("[vlm_grpo_patch] loss shapes "
              f"input_ids={tuple(input_ids.shape)} "
              f"pixel_values={tuple(vis['pixel_values'].shape)} "
              f"image_grid_thw={tuple(vis['image_grid_thw'].shape)} "
              f"mm_token_type_ids={tuple(vis['mm_token_type_ids'].shape)}")
        self._vlm_shape_logged = True

    per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, **vis)
    advantages = inputs["advantages"]
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -per_token_loss
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    mode = "eval" if self.control.should_evaluate else "train"
    completion_length = self.accelerator.gather(completion_mask.sum(1)).float().mean().item()
    self._metrics[mode].setdefault("completion_length", []).append(completion_length)
    return loss


def apply_vlm_grpo_patch():
    # TRL's generated-batch split/shuffle helpers assume every tensor's dim 0 is
    # the sample dimension. Qwen VLM `pixel_values` is flattened image patches on
    # dim 0, so handle it using `image_grid_thw` patch counts.
    if hasattr(G, "split_tensor_dict"):
        G.split_tensor_dict = _split_tensor_dict_vlm
    if hasattr(G, "shuffle_tensor_dict"):
        G.shuffle_tensor_dict = _shuffle_tensor_dict_vlm
    GRPOTrainer._get_per_token_logps = _get_per_token_logps
    GRPOTrainer._generate_and_score_completions = _generate_and_score_completions
    GRPOTrainer.compute_loss = compute_loss
    print("[vlm_grpo_patch] applied: images now threaded into generation + loss forward")
