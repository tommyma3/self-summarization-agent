from __future__ import annotations

from dataclasses import dataclass, field
import functools
import inspect
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
import warnings

import torch
from transformers import AutoModel, AutoTokenizer

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    AutoModelForMultimodalLM = AutoModel  # type: ignore[misc,assignment]

from self_summarization_agent.config import ModelConfig, TrainingConfig
from self_summarization_agent.chat_template import configure_tokenizer_chat_template
from self_summarization_agent.prompts import ConversationPrompt, serialize_messages
from self_summarization_agent.trajectory import (
    REFERENCE_LOGPROB_SOURCE_POLICY_RESCORE,
    RLSample,
    TOKEN_CACHE_VERSION,
    tokenize_interval_messages,
)
from self_summarization_agent.value_model import (
    CompactionValueHead,
    capture_lm_head_input,
    expected_binary_value,
    load_value_head,
    model_hidden_size,
    rollout_value_weights,
    rollout_normalized_value_loss,
    save_value_head,
)


def _write_training_progress(path: str | None, stage: str, **payload: Any) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "stage": stage,
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        **payload,
    }
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def _report_progress(completed: int, total: int) -> bool:
    return completed == 1 or completed == total or completed % 10 == 0


@dataclass(slots=True)
class _PolicyBatch:
    flat_samples: list[RLSample]
    advantages: list[float]
    contributing: list[tuple[RLSample, float]]


def compute_group_advantages(samples: list[RLSample]) -> list[float]:
    if not samples:
        return []
    if all(sample.rollout_id is not None for sample in samples):
        rollout_rewards: dict[str, float] = {}
        for sample in samples:
            rollout_id = str(sample.rollout_id)
            existing = rollout_rewards.get(rollout_id)
            if existing is not None and existing != sample.reward:
                raise ValueError(f"Rollout {rollout_id} has inconsistent interval rewards")
            rollout_rewards[rollout_id] = sample.reward
        rewards = torch.tensor(list(rollout_rewards.values()), dtype=torch.float32)
        centered = rewards - rewards.mean()
        std = rewards.std(unbiased=False)
        if std.item() > 0:
            centered = centered / (std + 1e-6)
        advantage_by_rollout = dict(zip(rollout_rewards, centered.tolist()))
        return [advantage_by_rollout[str(sample.rollout_id)] for sample in samples]

    rewards = torch.tensor([sample.reward for sample in samples], dtype=torch.float32)
    centered = rewards - rewards.mean()
    std = rewards.std(unbiased=False)
    if std.item() > 0:
        centered = centered / (std + 1e-6)
    return centered.tolist()


@dataclass(slots=True)
class UpdateMetrics:
    sample_count: int
    mean_reward: float
    mean_advantage: float
    loss: float
    optimizer_step_count: int = 0
    mean_policy_kl: float = 0.0
    clip_fraction: float = 0.0
    extra_metrics: dict[str, Any] = field(default_factory=dict)


def _copy_processor_configs(source_dir: str, target_dir: str) -> None:
    source = Path(source_dir)
    target = Path(target_dir)
    for filename in ("preprocessor_config.json", "video_preprocessor_config.json"):
        src = source / filename
        if src.exists():
            shutil.copy2(src, target / filename)


def _prepare_policy_batch(grouped_samples: dict[str, list[RLSample]]) -> _PolicyBatch:
    flat_samples: list[RLSample] = []
    advantages: list[float] = []
    for samples in grouped_samples.values():
        if not samples:
            continue
        group_advantages = compute_group_advantages(samples)
        flat_samples.extend(samples)
        advantages.extend(group_advantages)
    contributing = [
        (sample, advantage)
        for sample, advantage in zip(flat_samples, advantages)
        if abs(advantage) >= 1e-8
    ]
    return _PolicyBatch(
        flat_samples=flat_samples,
        advantages=advantages,
        contributing=contributing,
    )


def _metrics_without_update(batch: _PolicyBatch) -> UpdateMetrics:
    if not batch.flat_samples:
        return UpdateMetrics(sample_count=0, mean_reward=0.0, mean_advantage=0.0, loss=0.0)
    return UpdateMetrics(
        sample_count=len(batch.flat_samples),
        mean_reward=sum(sample.reward for sample in batch.flat_samples) / len(batch.flat_samples),
        mean_advantage=sum(batch.advantages) / len(batch.advantages),
        loss=0.0,
    )


def _validate_grpo_training_config(training_config: TrainingConfig) -> tuple[int, int | None, int, float]:
    update_epochs = training_config.update_epochs
    if update_epochs < 1:
        raise ValueError(f"training.update_epochs must be at least 1, got {update_epochs}")
    minibatch_size = training_config.minibatch_size
    if minibatch_size is not None and minibatch_size < 1:
        raise ValueError(f"training.minibatch_size must be at least 1, got {minibatch_size}")
    microbatch_size = training_config.gradient_accumulation_microbatch_size
    if microbatch_size < 1:
        raise ValueError(
            "training.gradient_accumulation_microbatch_size must be at least 1, "
            f"got {microbatch_size}"
        )
    clip_range = training_config.clip_range
    if clip_range < 0:
        raise ValueError(f"training.clip_range must be non-negative, got {clip_range}")
    target_kl = training_config.target_kl
    if target_kl is not None and target_kl <= 0:
        raise ValueError(f"training.target_kl must be positive when set, got {target_kl}")
    return update_epochs, minibatch_size, microbatch_size, clip_range


def _minibatches(
    items: list[tuple[RLSample, float]],
    reference_logprobs: list[torch.Tensor],
    minibatch_size: int | None,
):
    effective_size = len(items) if minibatch_size is None else minibatch_size
    for start in range(0, len(items), effective_size):
        end = start + effective_size
        yield items[start:end], reference_logprobs[start:end]


def _minibatch_ranges(item_count: int, minibatch_size: int | None):
    effective_size = item_count if minibatch_size is None else minibatch_size
    for start in range(0, item_count, effective_size):
        yield start, min(start + effective_size, item_count)


def _microbatch_ranges(start: int, end: int, microbatch_size: int):
    for microbatch_start in range(start, end, microbatch_size):
        yield microbatch_start, min(microbatch_start + microbatch_size, end)


class _AllReduceSum(torch.autograd.Function):
    """Differentiable wrapper for a sum reduction across distributed ranks.

    ``torch.distributed.all_reduce`` is not registered with the autograd
    dispatcher, so back-propagating through it triggers a deprecation warning
    (and will eventually error).  For a *sum* reduction the gradient is the
    identity, so we explicitly implement that here.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, reduce_fn) -> torch.Tensor:
        return reduce_fn(tensor.detach())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def _clipped_grpo_loss(
    logprob: torch.Tensor,
    reference_logprob: torch.Tensor,
    advantage: float,
    *,
    clip_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reference_logprob = reference_logprob.to(device=logprob.device, dtype=logprob.dtype)
    advantage_tensor = torch.tensor(advantage, device=logprob.device, dtype=logprob.dtype)
    ratio = torch.exp(logprob - reference_logprob)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    unclipped_objective = ratio * advantage_tensor
    clipped_objective = clipped_ratio * advantage_tensor
    loss = -torch.minimum(unclipped_objective, clipped_objective)
    approx_kl = reference_logprob - logprob.detach()
    clipped = (torch.abs(ratio.detach() - 1.0) > clip_range).to(logprob.dtype)
    return loss, approx_kl, clipped


def _clipped_grpo_losses(
    logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reference_logprobs = reference_logprobs.to(device=logprobs.device, dtype=logprobs.dtype)
    advantages = advantages.to(device=logprobs.device, dtype=logprobs.dtype)
    ratio = torch.exp(logprobs - reference_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages
    losses = -torch.minimum(unclipped_objective, clipped_objective)
    approx_kls = reference_logprobs - logprobs.detach()
    clipped = (torch.abs(ratio.detach() - 1.0) > clip_range).to(logprobs.dtype)
    return losses, approx_kls, clipped


def _pad_1d_tensors(items: list[torch.Tensor], *, pad_value: int | float) -> torch.Tensor:
    if not items:
        raise ValueError("Cannot pad an empty tensor list")
    max_length = max(tensor.shape[0] for tensor in items)
    padded = torch.full(
        (len(items), max_length),
        pad_value,
        dtype=items[0].dtype,
        device=items[0].device,
    )
    for index, tensor in enumerate(items):
        padded[index, : tensor.shape[0]] = tensor
    return padded


def _fit_reference_logprobs_to_shape(reference_logprobs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if reference_logprobs.shape == target.shape:
        return reference_logprobs
    if reference_logprobs.shape[0] != target.shape[0]:
        raise ValueError(
            "Reference logprob batch size does not match current policy logprob batch size: "
            f"{reference_logprobs.shape[0]} != {target.shape[0]}"
        )
    if reference_logprobs.shape[1] > target.shape[1]:
        return reference_logprobs[:, : target.shape[1]]
    pad_width = target.shape[1] - reference_logprobs.shape[1]
    pad = torch.zeros(
        (reference_logprobs.shape[0], pad_width),
        dtype=reference_logprobs.dtype,
        device=reference_logprobs.device,
    )
    return torch.cat([reference_logprobs, pad], dim=1)


def _pad_bool_tensors(items: list[torch.Tensor]) -> torch.Tensor:
    if not items:
        raise ValueError("Cannot pad an empty tensor list")
    max_length = max(tensor.shape[0] for tensor in items)
    padded = torch.zeros(
        (len(items), max_length),
        dtype=torch.bool,
        device=items[0].device,
    )
    for index, tensor in enumerate(items):
        padded[index, : tensor.shape[0]] = tensor
    return padded


def _pad_token_id_from_tokenizer(tokenizer: Any) -> int:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    return int(pad_token_id)


def _encode_shifted_sample_from_text(
    sample: RLSample,
    *,
    tokenizer: Any,
    device: torch.device,
    max_sequence_length: int | None,
    enable_thinking: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_token_id = _pad_token_id_from_tokenizer(tokenizer)
    if sample.has_exact_collection_tokens:
        full_ids = list(sample.collection_full_token_ids or [])
        assistant_mask = list(sample.collection_assistant_token_mask or [])
        if len(full_ids) != len(assistant_mask):
            raise ValueError(f"Sample {sample.turn_id} has mismatched exact collection token lengths")
        if max_sequence_length is not None and len(full_ids) - 1 > max_sequence_length:
            raise ValueError(
                f"Interval {sample.turn_id} exceeds training.max_sequence_length: "
                f"{len(full_ids) - 1} > {max_sequence_length}; exact collection tokens are never truncated"
            )
        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long, device=device)
        labels = torch.tensor(full_ids[1:], dtype=torch.long, device=device)
        completion_mask = torch.tensor(assistant_mask[1:], dtype=torch.bool, device=device)
        if not bool(completion_mask.any()):
            raise ValueError(f"Sample {sample.turn_id} has no exact sampled assistant tokens")
        return input_ids, labels, completion_mask
    if sample.messages is not None:
        input_id_values, label_values, mask_values = tokenize_interval_messages(
            tokenizer,
            sample.messages,
            max_sequence_length=max_sequence_length,
            sample_id=sample.turn_id,
            enable_thinking=enable_thinking,
            tools=sample.tools,
        )
        return (
            torch.tensor(input_id_values, dtype=torch.long, device=device),
            torch.tensor(label_values, dtype=torch.long, device=device),
            torch.tensor(mask_values, dtype=torch.bool, device=device),
        )
    prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(sample.prompt + sample.completion, add_special_tokens=False)
    if max_sequence_length is not None and len(full_ids) > max_sequence_length:
        drop = len(full_ids) - max_sequence_length
        full_ids = full_ids[drop:]
        prompt_ids = prompt_ids[drop:] if len(prompt_ids) > drop else []

    if len(full_ids) <= len(prompt_ids):
        input_ids = torch.tensor([pad_token_id], dtype=torch.long, device=device)
        labels = torch.tensor([pad_token_id], dtype=torch.long, device=device)
        completion_mask = torch.zeros((1,), dtype=torch.bool, device=device)
    else:
        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long, device=device)
        labels = torch.tensor(full_ids[1:], dtype=torch.long, device=device)
        completion_mask = torch.zeros_like(labels, dtype=torch.bool)
        completion_start = max(len(prompt_ids) - 1, 0)
        completion_mask[completion_start:] = True
    return input_ids, labels, completion_mask


def _cached_shifted_sample_tensors(
    sample: RLSample,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not sample.has_training_cache:
        raise ValueError(f"Sample {sample.turn_id} is missing training cache")
    return (
        torch.tensor(sample.input_ids, dtype=torch.long, device=device),
        torch.tensor(sample.labels, dtype=torch.long, device=device),
        torch.tensor(sample.completion_mask, dtype=torch.bool, device=device),
    )


def _all_samples_have_training_cache(samples: list[RLSample]) -> bool:
    return all(sample.has_training_cache for sample in samples)


def _training_cache_payload(
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    completion_mask: torch.Tensor,
    reference_logprob: float,
    reference_logprobs: torch.Tensor,
    state_prefix_length: int,
) -> dict[str, Any]:
    if isinstance(state_prefix_length, bool) or not 1 <= state_prefix_length <= input_ids.numel():
        raise ValueError("Training cache requires an exact value-state prefix anchor")
    return {
        "version": TOKEN_CACHE_VERSION,
        "input_ids": [int(token_id) for token_id in input_ids.detach().cpu().tolist()],
        "labels": [int(token_id) for token_id in labels.detach().cpu().tolist()],
        "completion_mask": [bool(value) for value in completion_mask.detach().cpu().tolist()],
        "reference_logprob": float(reference_logprob),
        "reference_logprobs": [float(value) for value in reference_logprobs.detach().cpu().tolist()],
        "reference_logprob_source": REFERENCE_LOGPROB_SOURCE_POLICY_RESCORE,
        "state_prefix_length": state_prefix_length,
    }


def _exact_state_prefix_length(
    sample: RLSample,
    *,
    tokenizer: Any,
    input_ids: torch.Tensor,
    enable_thinking: bool,
) -> int:
    value = sample.state_prefix_length
    if isinstance(value, int) and not isinstance(value, bool):
        return value

    prefix_ids: list[int] | None = None
    if sample.messages is not None and getattr(tokenizer, "chat_template", None):
        first_assistant = next(
            (index for index, message in enumerate(sample.messages) if message.get("role") == "assistant"),
            None,
        )
        if first_assistant is not None:
            template_kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
            if sample.tools is not None:
                template_kwargs["tools"] = sample.tools
            try:
                rendered = tokenizer.apply_chat_template(
                    sample.messages[:first_assistant],
                    tokenize=True,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(
                    sample.messages[:first_assistant],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            if isinstance(rendered, dict):
                rendered = rendered.get("input_ids")
            if hasattr(rendered, "tolist"):
                rendered = rendered.tolist()
            if isinstance(rendered, list) and rendered and isinstance(rendered[0], list):
                rendered = rendered[0]
            if isinstance(rendered, list):
                prefix_ids = [int(token_id) for token_id in rendered]
    elif sample.prompt:
        prefix_ids = [int(token_id) for token_id in tokenizer.encode(sample.prompt, add_special_tokens=False)]

    encoded_ids = [int(token_id) for token_id in input_ids.detach().cpu().tolist()]
    if (
        not prefix_ids
        or len(prefix_ids) > len(encoded_ids)
        or encoded_ids[: len(prefix_ids)] != prefix_ids
    ):
        raise ValueError(f"Sample {sample.turn_id} is missing an exact value-state anchor")
    return len(prefix_ids)


_CE_SEQUENCE_CHUNK_TOKENS = 2048


def _masked_token_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token logprobs, chunking cross-entropy to bound peak memory.

    ``cross_entropy(reduction="none")`` computes each token's loss independently,
    so chunking along the sequence dimension is numerically identical to a single
    call.  The chunk size caps the log-softmax intermediate allocation at roughly
    ``chunk_size * vocab_size * dtype_bytes``, keeping the cache-overlap scorer
    (single-GPU, no FSDP) under the 80 GiB A100 limit for 40 K‑token no‑compact
    sequences.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    chunk_losses: list[torch.Tensor] = []
    for start in range(0, flat_logits.shape[0], _CE_SEQUENCE_CHUNK_TOKENS):
        end = min(start + _CE_SEQUENCE_CHUNK_TOKENS, flat_logits.shape[0])
        chunk_losses.append(
            torch.nn.functional.cross_entropy(
                flat_logits[start:end],
                flat_labels[start:end],
                reduction="none",
            )
        )
    token_losses = torch.cat(chunk_losses, dim=0).reshape_as(labels)
    return -token_losses * completion_mask.to(token_losses.dtype)


def _mean_masked_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    token_logprobs = _masked_token_logprobs(logits, labels, completion_mask)
    mask = completion_mask.to(token_logprobs.dtype)
    token_counts = mask.sum(dim=1)
    logprob_sums = token_logprobs.sum(dim=1)
    return torch.where(
        token_counts > 0,
        logprob_sums / token_counts.clamp_min(1),
        torch.zeros_like(logprob_sums),
    )


def _mean_from_token_logprobs(token_logprobs: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    mask = completion_mask.to(token_logprobs.dtype)
    token_counts = mask.sum(dim=1)
    logprob_sums = (token_logprobs * mask).sum(dim=1)
    return torch.where(
        token_counts > 0,
        logprob_sums / token_counts.clamp_min(1),
        torch.zeros_like(logprob_sums),
    )


def _clipped_grpo_token_losses(
    logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    clip_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reference_logprobs = reference_logprobs.to(device=logprobs.device, dtype=logprobs.dtype)
    reference_logprobs = _fit_reference_logprobs_to_shape(reference_logprobs, logprobs)
    advantages = advantages.to(device=logprobs.device, dtype=logprobs.dtype).unsqueeze(-1)
    mask = completion_mask.to(device=logprobs.device, dtype=logprobs.dtype)
    ratio = torch.exp(logprobs - reference_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages
    token_losses = -torch.minimum(unclipped_objective, clipped_objective) * mask
    approx_kls = (reference_logprobs - logprobs.detach()) * mask
    clipped = (torch.abs(ratio.detach() - 1.0) > clip_range).to(logprobs.dtype) * mask
    return token_losses, approx_kls, clipped, mask


@dataclass(slots=True)
class TransformersPolicyTrainer:
    model_config: ModelConfig
    training_config: TrainingConfig
    progress_path: str | None = None
    tokenizer: Any = field(init=False)
    model: Any = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)
    value_head: CompactionValueHead | None = field(default=None, init=False)
    value_head_loaded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        print(
            f"[TransformersPolicyTrainer] Loading model from {self.model_config.model_path}",
            flush=True,
        )
        _write_training_progress(self.progress_path, "model_loading")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_path,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        configure_tokenizer_chat_template(self.tokenizer, self.model_config.chat_template_path)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_config.model_path,
            torch_dtype=self._torch_dtype(),
            device_map=self.model_config.device_map,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        if self.training_config.value.enabled:
            forward_parameters = inspect.signature(self.model.forward).parameters
            if "logits_to_keep" not in forward_parameters:
                raise ValueError(
                    "Compaction value training requires a model forward method that supports "
                    "logits_to_keep; refusing the prohibitively expensive full-vocabulary path"
                )
        if self.training_config.activation_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model._set_gradient_checkpointing(
                enable=True,
                gradient_checkpointing_func=functools.partial(
                    torch.utils.checkpoint.checkpoint, use_reentrant=False
                ),
            )
        optimizer_parameters: list[torch.nn.Parameter] = list(self.model.parameters())
        if self.training_config.value.enabled:
            self.value_head, self.value_head_loaded = load_value_head(
                self.model_config.model_path,
                hidden_size=model_hidden_size(self.model),
                zero_initialize=self.training_config.value.zero_initialize_head,
            )
            output_embedding = self.model.get_output_embeddings()
            try:
                output_parameter = next(output_embedding.parameters())
            except (AttributeError, StopIteration):
                output_parameter = next(self.model.parameters())
            self.value_head.to(device=output_parameter.device, dtype=output_parameter.dtype)
            optimizer_parameters.extend(self.value_head.parameters())
        self.optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.training_config.learning_rate)
        self.model.train()
        print("[TransformersPolicyTrainer] Model and optimizer are ready", flush=True)
        _write_training_progress(self.progress_path, "trainer_ready")

    def _torch_dtype(self):
        mapping = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.model_config.dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {self.model_config.dtype}")
        return mapping[self.model_config.dtype]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(self._format_prompt(prompt), add_special_tokens=False))

    def _model_device(self) -> torch.device:
        device = getattr(self.model, "device", None)
        if device is not None:
            return torch.device(device)
        return next(self.model.parameters()).device

    def _format_prompt(self, prompt: str) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt
        if isinstance(prompt, ConversationPrompt):
            messages = serialize_messages(prompt.messages)
        else:
            messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.model_config.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(self, prompt: str) -> str:
        was_training = self.model.training
        self.model.eval()
        try:
            encoded = self.tokenizer(self._format_prompt(prompt), return_tensors="pt")
            encoded = {name: tensor.to(self.model.device) for name, tensor in encoded.items()}
            generation_kwargs = {
                "max_new_tokens": self.model_config.max_new_tokens,
                "do_sample": self.model_config.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if self.model_config.do_sample:
                generation_kwargs["temperature"] = self.model_config.temperature
                generation_kwargs["top_p"] = self.model_config.top_p
            with torch.no_grad():
                output_ids = self.model.generate(
                    **encoded,
                    **generation_kwargs,
                )
        finally:
            if was_training:
                self.model.train()
        generated_ids = output_ids[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _sequence_logprob(self, sample: RLSample) -> torch.Tensor:
        return self._sequence_logprobs([sample])[0]

    def _encode_shifted_samples(self, samples: list[RLSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self._model_device()
        pad_token_id = _pad_token_id_from_tokenizer(self.tokenizer)

        input_tensors: list[torch.Tensor] = []
        label_tensors: list[torch.Tensor] = []
        mask_tensors: list[torch.Tensor] = []
        max_len = getattr(self.training_config, "max_sequence_length", None)
        for sample in samples:
            if sample.has_training_cache:
                input_ids, labels, completion_mask = _cached_shifted_sample_tensors(sample, device=device)
            else:
                input_ids, labels, completion_mask = _encode_shifted_sample_from_text(
                    sample,
                    tokenizer=self.tokenizer,
                    device=device,
                    max_sequence_length=max_len,
                    enable_thinking=self.model_config.enable_thinking,
                )
            input_tensors.append(input_ids)
            label_tensors.append(labels)
            mask_tensors.append(completion_mask)

        return (
            _pad_1d_tensors(input_tensors, pad_value=pad_token_id),
            _pad_1d_tensors(label_tensors, pad_value=pad_token_id),
            _pad_bool_tensors(mask_tensors),
        )

    def _sequence_logprobs(self, samples: list[RLSample]) -> torch.Tensor:
        input_ids, labels, completion_mask = self._encode_shifted_samples(samples)
        outputs = self.model(input_ids=input_ids)
        return _mean_masked_logprobs(outputs.logits, labels, completion_mask)

    def _sequence_token_logprobs_and_mask(self, samples: list[RLSample]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels, completion_mask = self._encode_shifted_samples(samples)
        outputs = self.model(input_ids=input_ids)
        return _masked_token_logprobs(outputs.logits, labels, completion_mask), completion_mask

    def _actor_critic_outputs(
        self,
        samples: list[RLSample],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.value_head is None:
            raise RuntimeError("Compaction value training is enabled without a value head")
        missing = [sample.turn_id for sample in samples if sample.state_prefix_length is None]
        if missing:
            raise ValueError(f"Samples are missing exact value-state anchors: {', '.join(missing)}")
        input_ids, labels, completion_mask = self._encode_shifted_samples(samples)
        anchor_positions = [int(sample.state_prefix_length) - 1 for sample in samples]
        trainable_positions = completion_mask.any(dim=0).nonzero(as_tuple=False).flatten().tolist()
        selected_positions = sorted(set(anchor_positions) | {int(position) for position in trainable_positions})
        if not selected_positions:
            raise ValueError("Compaction value training found no value anchor or policy positions")
        captured, handle = capture_lm_head_input(self.model)
        try:
            outputs = self.model(
                input_ids=input_ids,
                # A Python index list is intentionally used here. Device-map
                # hooks cannot move it onto a GPU that differs from the final
                # hidden-state device before Qwen applies the sequence slice.
                logits_to_keep=selected_positions,
                use_cache=False,
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one language-model head invocation")
        anchor_offsets = torch.tensor(
            [selected_positions.index(position) for position in anchor_positions],
            dtype=torch.long,
            device=captured[0].device,
        )
        rows = torch.arange(len(samples), device=captured[0].device)
        value_hidden_states = captured[0][rows, anchor_offsets]
        value_logits = self.value_head(value_hidden_states)

        position_tensor = torch.tensor(selected_positions, dtype=torch.long, device=outputs.logits.device)
        selected_labels = labels.to(device=outputs.logits.device).index_select(1, position_tensor)
        selected_mask = completion_mask.to(device=outputs.logits.device).index_select(1, position_tensor)
        selected_logprobs = _masked_token_logprobs(outputs.logits, selected_labels, selected_mask)
        full_mask = completion_mask.to(device=outputs.logits.device)
        scatter_indices = position_tensor.unsqueeze(0).expand(len(samples), -1)
        token_logprobs = torch.zeros(full_mask.shape, dtype=selected_logprobs.dtype, device=outputs.logits.device)
        token_logprobs = token_logprobs.scatter(1, scatter_indices, selected_logprobs)
        return token_logprobs, full_mask, value_logits

    def _value_logits(self, samples: list[RLSample]) -> torch.Tensor:
        """Evaluate interval-start values without processing the interval tail."""

        if self.value_head is None:
            raise RuntimeError("Compaction value training is enabled without a value head")
        logits: list[torch.Tensor] = []
        for sample in samples:
            if sample.state_prefix_length is None:
                raise ValueError(f"Sample {sample.turn_id} is missing an exact value-state anchor")
            input_ids, _labels, _completion_mask = self._encode_shifted_samples([sample])
            prefix_length = int(sample.state_prefix_length)
            if not 1 <= prefix_length <= input_ids.shape[1]:
                raise ValueError(f"Sample {sample.turn_id} has an invalid value-state anchor")
            captured, handle = capture_lm_head_input(self.model)
            try:
                self.model(
                    input_ids=input_ids[:, :prefix_length],
                    logits_to_keep=1,
                    use_cache=False,
                )
            finally:
                handle.remove()
            if len(captured) != 1 or captured[0].shape[1] != 1:
                raise RuntimeError("Expected exactly one selected value-anchor hidden state")
            logits.append(self.value_head(captured[0][:, 0]))
        return torch.cat(logits, dim=0)

    def _old_values(self, samples: list[RLSample], *, microbatch_size: int) -> list[float]:
        was_training = self.model.training
        value_was_training = self.value_head.training if self.value_head is not None else False
        self.model.eval()
        if self.value_head is not None:
            self.value_head.eval()
        values: list[float] = []
        try:
            with torch.no_grad():
                ranges = list(_microbatch_ranges(0, len(samples), microbatch_size))
                started = time.monotonic()
                for batch_index, (start, end) in enumerate(ranges, start=1):
                    logits = self._value_logits(samples[start:end])
                    values.extend(expected_binary_value(logits).detach().cpu().tolist())
                    if _report_progress(batch_index, len(ranges)):
                        elapsed = time.monotonic() - started
                        print(
                            f"[TransformersPolicyTrainer] Frozen values: "
                            f"microbatch={batch_index}/{len(ranges)}, "
                            f"states={end}/{len(samples)}, elapsed_seconds={elapsed:.1f}",
                            flush=True,
                        )
                        _write_training_progress(
                            getattr(self, "progress_path", None),
                            "frozen_values",
                            completed_microbatches=batch_index,
                            total_microbatches=len(ranges),
                            completed_states=end,
                            total_states=len(samples),
                            elapsed_seconds=elapsed,
                        )
        finally:
            if was_training:
                self.model.train()
            if self.value_head is not None and value_was_training:
                self.value_head.train()
        return values

    def _step_compaction_value(
        self,
        grouped_samples: dict[str, list[RLSample]],
        *,
        update_epochs: int,
        minibatch_size: int | None,
        microbatch_size: int,
        clip_range: float,
    ) -> UpdateMetrics:
        samples = [sample for group in grouped_samples.values() for sample in group]
        if not samples:
            return UpdateMetrics(sample_count=0, mean_reward=0.0, mean_advantage=0.0, loss=0.0)
        if self.value_head is None:
            raise RuntimeError("Compaction value training is enabled without a value head")
        if any(sample.rollout_id is None for sample in samples):
            raise ValueError("Compaction value training requires rollout_id on every interval")
        if any(sample.reward not in {-1.0, 1.0} for sample in samples):
            raise ValueError("Compaction value training requires binary -1/+1 rollout rewards")

        reward_by_rollout: dict[str, float] = {}
        for sample in samples:
            rollout_id = str(sample.rollout_id)
            previous_reward = reward_by_rollout.setdefault(rollout_id, sample.reward)
            if previous_reward != sample.reward:
                raise ValueError(f"Rollout {rollout_id} has inconsistent interval rewards")

        total_tokens = sum(len(sample.input_ids or []) for sample in samples)
        trainable_tokens = sum(sum(bool(value) for value in (sample.completion_mask or [])) for sample in samples)
        prefix_tokens = sum(int(sample.state_prefix_length or 0) for sample in samples)
        print(
            "[TransformersPolicyTrainer] Starting compaction-value update: "
            f"states={len(samples)}, rollouts={len(reward_by_rollout)}, "
            f"total_tokens={total_tokens}, trainable_tokens={trainable_tokens}, "
            f"prefix_tokens={prefix_tokens}, update_epochs={update_epochs}, "
            f"minibatch_size={minibatch_size or len(samples)}, microbatch_size={microbatch_size}",
            flush=True,
        )
        _write_training_progress(
            getattr(self, "progress_path", None),
            "frozen_values_start",
            total_states=len(samples),
            rollout_count=len(reward_by_rollout),
            total_tokens=total_tokens,
            trainable_tokens=trainable_tokens,
            prefix_tokens=prefix_tokens,
        )
        old_value_started = time.monotonic()
        old_values = self._old_values(samples, microbatch_size=microbatch_size)
        old_value_seconds = time.monotonic() - old_value_started
        advantages = [sample.reward - old_value for sample, old_value in zip(samples, old_values)]
        rollout_ids = [str(sample.rollout_id) for sample in samples]
        value_weights = rollout_value_weights(rollout_ids)
        reference_token_logprobs = self._cached_reference_token_logprobs(samples)
        if reference_token_logprobs is None:
            raise ValueError("Compaction value training requires cached behavior-policy log probabilities")

        detached_policy_losses: list[float] = []
        detached_kls: list[float] = []
        detached_clipped: list[float] = []
        weighted_value_loss_sum = 0.0
        value_loss_count = 0
        optimizer_step_count = 0
        microbatch_ranges = [
            (start, end, micro_start, micro_end)
            for start, end in _minibatch_ranges(len(samples), minibatch_size)
            for micro_start, micro_end in _microbatch_ranges(start, end, microbatch_size)
        ]
        total_update_microbatches = update_epochs * len(microbatch_ranges)
        completed_update_microbatches = 0
        update_started = time.monotonic()
        _write_training_progress(
            getattr(self, "progress_path", None),
            "policy_value_update_start",
            total_microbatches=total_update_microbatches,
            old_value_seconds=old_value_seconds,
        )
        for _epoch_index in range(update_epochs):
            for start, end in _minibatch_ranges(len(samples), minibatch_size):
                self.optimizer.zero_grad(set_to_none=True)
                for micro_start, micro_end in _microbatch_ranges(start, end, microbatch_size):
                    micro_samples = samples[micro_start:micro_end]
                    token_logprobs, completion_mask, value_logits = self._actor_critic_outputs(micro_samples)
                    advantage_tensor = torch.tensor(
                        advantages[micro_start:micro_end],
                        dtype=torch.float32,
                        device=self._model_device(),
                    )
                    policy_losses, approx_kls, clipped, loss_mask = _clipped_grpo_token_losses(
                        token_logprobs,
                        reference_token_logprobs[micro_start:micro_end],
                        advantage_tensor,
                        completion_mask,
                        clip_range=clip_range,
                    )
                    policy_loss = policy_losses.sum() / loss_mask.sum().clamp_min(1.0)
                    rewards = torch.tensor(
                        [sample.reward for sample in micro_samples],
                        dtype=torch.float32,
                        device=value_logits.device,
                    )
                    weights = torch.tensor(
                        value_weights[micro_start:micro_end],
                        dtype=torch.float32,
                        device=value_logits.device,
                    )
                    value_loss = rollout_normalized_value_loss(value_logits, rewards, weights)
                    total_loss = policy_loss + self.training_config.value.loss_coefficient * value_loss.to(
                        device=policy_loss.device
                    )
                    total_loss.backward()

                    valid = loss_mask > 0
                    detached_policy_losses.extend(policy_losses.detach()[valid].cpu().tolist())
                    detached_kls.extend(approx_kls.detach()[valid].cpu().tolist())
                    detached_clipped.extend(clipped.detach()[valid].cpu().tolist())
                    weighted_value_loss_sum += float(value_loss.detach().cpu())
                    value_loss_count += 1
                    completed_update_microbatches += 1
                    if _report_progress(completed_update_microbatches, total_update_microbatches):
                        elapsed = time.monotonic() - update_started
                        print(
                            f"[TransformersPolicyTrainer] Policy/value update: "
                            f"microbatch={completed_update_microbatches}/{total_update_microbatches}, "
                            f"epoch={_epoch_index + 1}/{update_epochs}, "
                            f"states={micro_end}/{len(samples)}, elapsed_seconds={elapsed:.1f}",
                            flush=True,
                        )
                        _write_training_progress(
                            getattr(self, "progress_path", None),
                            "policy_value_update",
                            completed_microbatches=completed_update_microbatches,
                            total_microbatches=total_update_microbatches,
                            epoch=_epoch_index + 1,
                            completed_states=micro_end,
                            total_states=len(samples),
                            elapsed_seconds=elapsed,
                        )
                parameters = list(self.model.parameters()) + list(self.value_head.parameters())
                torch.nn.utils.clip_grad_norm_(parameters, self.training_config.max_grad_norm)
                self.optimizer.step()
                optimizer_step_count += 1

        predicted_classes = [1.0 if value >= 0 else -1.0 for value in old_values]
        accuracy = sum(
            prediction == sample.reward for prediction, sample in zip(predicted_classes, samples)
        ) / len(samples)
        mean_policy_loss = (
            sum(detached_policy_losses) / len(detached_policy_losses)
            if detached_policy_losses
            else 0.0
        )
        mean_value_loss = weighted_value_loss_sum / max(1, value_loss_count)
        return UpdateMetrics(
            sample_count=len(samples),
            mean_reward=sum(sample.reward for sample in samples) / len(samples),
            mean_advantage=sum(advantages) / len(advantages),
            loss=mean_policy_loss + self.training_config.value.loss_coefficient * mean_value_loss,
            optimizer_step_count=optimizer_step_count,
            mean_policy_kl=sum(detached_kls) / len(detached_kls) if detached_kls else 0.0,
            clip_fraction=(
                sum(detached_clipped) / len(detached_clipped) if detached_clipped else 0.0
            ),
            extra_metrics={
                "value/loss": mean_value_loss,
                "value/old_mean": sum(old_values) / len(old_values),
                "value/advantage_std": float(torch.tensor(advantages).std(unbiased=False)),
                "value/classification_accuracy": accuracy,
                "value/state_count": len(samples),
                "value/rollout_count": len(set(rollout_ids)),
                "value/head_loaded": self.value_head_loaded,
                "value/old_value_seconds": old_value_seconds,
                "value/update_seconds": time.monotonic() - update_started,
                "value/total_tokens": total_tokens,
                "value/trainable_tokens": trainable_tokens,
                "value/prefix_tokens": prefix_tokens,
            },
        )

    def _cached_reference_token_logprobs(self, samples: list[RLSample]) -> torch.Tensor | None:
        if not _all_samples_have_training_cache(samples):
            return None
        device = self._model_device()
        tensors: list[torch.Tensor] = []
        for sample in samples:
            if sample.reference_logprobs is not None:
                values = sample.reference_logprobs
            else:
                values = [
                    float(sample.reference_logprob) if is_completion else 0.0
                    for is_completion in (sample.completion_mask or [])
                ]
            tensors.append(torch.tensor(values, dtype=torch.float32, device=device))
        return _pad_1d_tensors(tensors, pad_value=0.0)

    def cache_samples(self, samples: list[RLSample]) -> list[dict[str, Any]]:
        if not samples:
            return []
        device = self._model_device()
        max_len = getattr(self.training_config, "max_sequence_length", None)
        encoded = [
            _encode_shifted_sample_from_text(
                sample,
                tokenizer=self.tokenizer,
                device=device,
                max_sequence_length=max_len,
                enable_thinking=self.model_config.enable_thinking,
            )
            for sample in samples
        ]
        token_logprobs, completion_mask = self._sequence_token_logprobs_and_mask(samples)
        reference_logprobs = _mean_from_token_logprobs(token_logprobs, completion_mask).detach().cpu().tolist()
        reference_token_logprobs = token_logprobs.detach().cpu()
        return [
            _training_cache_payload(
                input_ids=input_ids,
                labels=labels,
                completion_mask=completion_mask,
                reference_logprob=float(reference_logprob),
                reference_logprobs=token_reference_logprobs[: input_ids.numel()],
                state_prefix_length=_exact_state_prefix_length(
                    sample,
                    tokenizer=self.tokenizer,
                    input_ids=input_ids,
                    enable_thinking=self.model_config.enable_thinking,
                ),
            )
            for sample, (input_ids, labels, completion_mask), reference_logprob, token_reference_logprobs in zip(
                samples,
                encoded,
                reference_logprobs,
                reference_token_logprobs,
            )
        ]

    def step(self, grouped_samples: dict[str, list[RLSample]]) -> UpdateMetrics:
        update_epochs, minibatch_size, microbatch_size, clip_range = _validate_grpo_training_config(
            self.training_config
        )
        if self.training_config.value.enabled:
            return self._step_compaction_value(
                grouped_samples,
                update_epochs=update_epochs,
                minibatch_size=minibatch_size,
                microbatch_size=microbatch_size,
                clip_range=clip_range,
            )
        batch = _prepare_policy_batch(grouped_samples)
        if not batch.flat_samples or not batch.contributing:
            return _metrics_without_update(batch)

        is_main = int(os.environ.get("RANK", "0")) == 0
        effective_minibatch_size = minibatch_size or len(batch.contributing)
        if is_main:
            print(
                "[TransformersPolicyTrainer] Starting clipped GRPO update: "
                f"{len(batch.flat_samples)} samples, {len(batch.contributing)} contributing, "
                f"update_epochs={update_epochs}, minibatch_size={effective_minibatch_size}, "
                f"microbatch_size={microbatch_size}"
            )
        with torch.no_grad():
            contributing_samples = [sample for sample, _advantage in batch.contributing]
            reference_token_logprobs = self._cached_reference_token_logprobs(contributing_samples)
            if reference_token_logprobs is None:
                reference_logprob_rows = []
                for start, end in _minibatch_ranges(len(batch.contributing), microbatch_size):
                    reference_samples = [sample for sample, _advantage in batch.contributing[start:end]]
                    token_logprobs, _completion_mask = self._sequence_token_logprobs_and_mask(reference_samples)
                    reference_logprob_rows.extend(token_logprobs.detach())
                reference_token_logprobs = _pad_1d_tensors(reference_logprob_rows, pad_value=0.0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        detached_losses: list[float] = []
        detached_kls: list[float] = []
        detached_clipped: list[float] = []
        optimizer_step_count = 0
        stop_training = False
        for epoch_index in range(update_epochs):
            if stop_training:
                break
            for start, end in _minibatch_ranges(len(batch.contributing), minibatch_size):
                minibatch_size_actual = end - start
                self.optimizer.zero_grad(set_to_none=True)
                minibatch_kl_sum = 0.0
                minibatch_kl_count = 0
                for micro_start, micro_end in _microbatch_ranges(start, end, microbatch_size):
                    microbatch = batch.contributing[micro_start:micro_end]
                    microbatch_reference_logprobs = reference_token_logprobs[micro_start:micro_end]
                    samples = [sample for sample, _advantage in microbatch]
                    advantages = torch.tensor(
                        [advantage for _sample, advantage in microbatch],
                        dtype=torch.float32,
                        device=self._model_device(),
                    )
                    logprobs, completion_mask = self._sequence_token_logprobs_and_mask(samples)
                    losses, approx_kls, clipped, loss_mask = _clipped_grpo_token_losses(
                        logprobs,
                        microbatch_reference_logprobs,
                        advantages,
                        completion_mask,
                        clip_range=clip_range,
                    )
                    valid_mask = loss_mask > 0
                    if valid_mask.any():
                        detached_losses.extend(losses.detach()[valid_mask].cpu().tolist())
                        detached_kls.extend(approx_kls.detach()[valid_mask].cpu().tolist())
                        detached_clipped.extend(clipped.detach()[valid_mask].cpu().tolist())
                        minibatch_kl_sum += float(approx_kls.detach().sum().cpu())
                        minibatch_kl_count += int(valid_mask.sum().item())
                    loss_denominator = loss_mask.sum().clamp_min(1.0)
                    (losses.sum() / loss_denominator).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                self.optimizer.step()
                optimizer_step_count += 1
                if is_main:
                    print(
                        "[TransformersPolicyTrainer]  "
                        f"epoch={epoch_index + 1}/{update_epochs} "
                        f"optimizer_step={optimizer_step_count} minibatch_size={minibatch_size_actual}"
                    )
                target_kl = self.training_config.target_kl
                minibatch_kl = minibatch_kl_sum / minibatch_kl_count if minibatch_kl_count else 0.0
                if target_kl is not None and minibatch_kl_count and minibatch_kl > target_kl:
                    if is_main:
                        print(f"[TransformersPolicyTrainer]  Stopping early: target_kl={target_kl} reached")
                    stop_training = True
                    break
        return UpdateMetrics(
            sample_count=len(batch.flat_samples),
            mean_reward=sum(sample.reward for sample in batch.flat_samples) / len(batch.flat_samples),
            mean_advantage=sum(batch.advantages) / len(batch.advantages),
            loss=sum(detached_losses) / len(detached_losses),
            optimizer_step_count=optimizer_step_count,
            mean_policy_kl=sum(detached_kls) / len(detached_kls),
            clip_fraction=sum(detached_clipped) / len(detached_clipped),
        )

    def save_checkpoint(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        _copy_processor_configs(self.model_config.model_path, path)
        if self.value_head is not None:
            save_value_head(self.value_head, path)


@dataclass(slots=True)
class FSDP2ContextParallelPolicyTrainer:
    model_config: ModelConfig
    training_config: TrainingConfig
    tokenizer: Any = field(init=False)
    model: Any = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)
    accelerator: Any = field(init=False)

    def __post_init__(self) -> None:
        try:
            from accelerate import Accelerator
            try:
                from accelerate.utils import ParallelismConfig
            except ImportError:
                from accelerate.parallelism_config import ParallelismConfig
        except ImportError as exc:
            raise ImportError(
                "training.backend='fsdp2_context_parallel' requires Accelerate in the GPU training environment. "
                "Install a version with FSDP2/context-parallel support and launch with accelerate."
            ) from exc

        parallelism_config = ParallelismConfig(
            dp_shard_size=self.training_config.data_parallel_size,
            tp_size=self.training_config.tensor_parallel_size,
            cp_size=self.training_config.context_parallel_size,
        )
        self.accelerator = Accelerator(parallelism_config=parallelism_config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_path,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        configure_tokenizer_chat_template(self.tokenizer, self.model_config.chat_template_path)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_config.model_path,
            torch_dtype=self._torch_dtype(),
            trust_remote_code=self.model_config.trust_remote_code,
        )
        if self.training_config.activation_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model._set_gradient_checkpointing(
                enable=True,
                gradient_checkpointing_func=functools.partial(
                    torch.utils.checkpoint.checkpoint, use_reentrant=False
                ),
            )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_config.learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.model.train()

    def _torch_dtype(self):
        mapping = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.model_config.dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {self.model_config.dtype}")
        return mapping[self.model_config.dtype]

    def _encode_shifted_sample(self, sample: RLSample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = getattr(self.training_config, "max_sequence_length", None)
        input_ids, labels, completion_mask = _encode_shifted_sample_from_text(
            sample,
            tokenizer=self.tokenizer,
            device=self.accelerator.device,
            max_sequence_length=max_len,
            enable_thinking=self.model_config.enable_thinking,
        )
        input_ids = input_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)
        completion_mask = completion_mask.unsqueeze(0)
        input_ids, labels, completion_mask = self._pad_for_context_parallel(input_ids, labels, completion_mask)
        return input_ids, labels, completion_mask

    def _encode_shifted_samples(self, samples: list[RLSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_token_id = _pad_token_id_from_tokenizer(self.tokenizer)

        input_tensors: list[torch.Tensor] = []
        label_tensors: list[torch.Tensor] = []
        mask_tensors: list[torch.Tensor] = []
        for sample in samples:
            if sample.has_training_cache:
                input_ids, labels, completion_mask = _cached_shifted_sample_tensors(
                    sample,
                    device=self.accelerator.device,
                )
            else:
                input_ids, labels, completion_mask = self._encode_shifted_sample(sample)
                input_ids = input_ids.squeeze(0)
                labels = labels.squeeze(0)
                completion_mask = completion_mask.squeeze(0)
            input_tensors.append(input_ids)
            label_tensors.append(labels)
            mask_tensors.append(completion_mask)

        return (
            _pad_1d_tensors(input_tensors, pad_value=pad_token_id),
            _pad_1d_tensors(label_tensors, pad_value=pad_token_id),
            _pad_bool_tensors(mask_tensors),
        )

    def _pad_for_context_parallel(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        multiple = self.training_config.context_parallel_size * 2
        if multiple <= 1:
            return input_ids, labels, completion_mask
        pad_length = (-input_ids.shape[1]) % multiple
        if pad_length == 0:
            return input_ids, labels, completion_mask

        pad_token_id = _pad_token_id_from_tokenizer(self.tokenizer)

        input_pad = torch.full(
            (input_ids.shape[0], pad_length),
            int(pad_token_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        label_pad = torch.full(
            (labels.shape[0], pad_length),
            int(pad_token_id),
            dtype=labels.dtype,
            device=labels.device,
        )
        mask_pad = torch.zeros(
            (completion_mask.shape[0], pad_length),
            dtype=completion_mask.dtype,
            device=completion_mask.device,
        )
        return (
            torch.cat([input_ids, input_pad], dim=1),
            torch.cat([labels, label_pad], dim=1),
            torch.cat([completion_mask, mask_pad], dim=1),
        )

    def _sequence_logprob(self, sample: RLSample) -> torch.Tensor:
        return self._sequence_logprobs([sample])[0]

    def _sequence_logprobs(self, samples: list[RLSample]) -> torch.Tensor:
        input_ids, labels, completion_mask = self._encode_shifted_samples(samples)
        if input_ids.numel() == 0:
            return torch.zeros((len(samples),), device=self.accelerator.device)

        max_len = getattr(self.training_config, "max_sequence_length", None)
        if max_len is not None and input_ids.shape[1] > max_len:
            raise ValueError(
                "An interval batch exceeds training.max_sequence_length: "
                f"{input_ids.shape[1]} > {max_len}; interval prefixes are never left-truncated"
            )

        if self.training_config.context_parallel_size > 1:
            input_ids, labels, completion_mask = self._pad_for_context_parallel(
                input_ids,
                labels,
                completion_mask,
            )
            buffers = [input_ids, labels, completion_mask]
            with self.accelerator.maybe_context_parallel(
                buffers=buffers,
                buffer_seq_dims=[1, 1, 1],
                no_restore_buffers=set(buffers),
            ):
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits
                local_logprob_sums, local_token_counts = self._masked_logprob_sums_and_counts(
                    logits,
                    labels,
                    completion_mask,
                )
        else:
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
            local_logprob_sums, local_token_counts = self._masked_logprob_sums_and_counts(
                logits,
                labels,
                completion_mask,
            )

        global_logprob_sums = _AllReduceSum.apply(local_logprob_sums, self._accelerator_reduce)
        with torch.no_grad():
            global_token_counts = self._accelerator_reduce(local_token_counts.detach())
        return torch.where(
            global_token_counts > 0,
            global_logprob_sums / global_token_counts.clamp_min(1),
            torch.zeros_like(global_logprob_sums),
        )

    def _cached_reference_logprobs(self, samples: list[RLSample]) -> torch.Tensor | None:
        if not _all_samples_have_training_cache(samples):
            return None
        return torch.tensor(
            [float(sample.reference_logprob) for sample in samples],
            dtype=torch.float32,
            device=self.accelerator.device,
        )

    def cache_samples(self, samples: list[RLSample]) -> list[dict[str, Any]]:
        if not samples:
            return []
        max_len = getattr(self.training_config, "max_sequence_length", None)
        encoded = [
            _encode_shifted_sample_from_text(
                sample,
                tokenizer=self.tokenizer,
                device=self.accelerator.device,
                max_sequence_length=max_len,
                enable_thinking=self.model_config.enable_thinking,
            )
            for sample in samples
        ]
        reference_logprobs = self._sequence_logprobs(samples).detach().cpu().tolist()
        return [
            _training_cache_payload(
                input_ids=input_ids,
                labels=labels,
                completion_mask=completion_mask,
                reference_logprob=float(reference_logprob),
                reference_logprobs=torch.where(
                    completion_mask,
                    torch.full_like(completion_mask, float(reference_logprob), dtype=torch.float32),
                    torch.zeros_like(completion_mask, dtype=torch.float32),
                ),
                state_prefix_length=_exact_state_prefix_length(
                    sample,
                    tokenizer=self.tokenizer,
                    input_ids=input_ids,
                    enable_thinking=self.model_config.enable_thinking,
                ),
            )
            for sample, (input_ids, labels, completion_mask), reference_logprob in zip(
                samples, encoded, reference_logprobs
            )
        ]

    def _masked_logprob_sums_and_counts(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        mask = completion_mask.to(token_losses.dtype)
        return -(token_losses * mask).sum(dim=1), mask.sum(dim=1)

    def _accelerator_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        reduce = getattr(self.accelerator, "reduce", None)
        if reduce is None:
            return tensor
        return reduce(tensor, reduction="sum")

    def step(self, grouped_samples: dict[str, list[RLSample]]) -> UpdateMetrics:
        update_epochs, minibatch_size, microbatch_size, clip_range = _validate_grpo_training_config(
            self.training_config
        )
        batch = _prepare_policy_batch(grouped_samples)
        if not batch.flat_samples or not batch.contributing:
            return _metrics_without_update(batch)

        is_main = getattr(self.accelerator, "is_main_process", True)
        effective_minibatch_size = minibatch_size or len(batch.contributing)
        if is_main:
            print(
                "[FSDP2CPTrainer] Starting clipped GRPO update: "
                f"{len(batch.flat_samples)} samples, {len(batch.contributing)} contributing, "
                f"update_epochs={update_epochs}, minibatch_size={effective_minibatch_size}, "
                f"microbatch_size={microbatch_size}"
            )
        with torch.no_grad():
            contributing_samples = [sample for sample, _advantage in batch.contributing]
            reference_logprobs = self._cached_reference_logprobs(contributing_samples)
            if reference_logprobs is None:
                reference_logprob_chunks = []
                for start, end in _minibatch_ranges(len(batch.contributing), microbatch_size):
                    reference_samples = [sample for sample, _advantage in batch.contributing[start:end]]
                    reference_logprob_chunks.append(self._sequence_logprobs(reference_samples).detach())
                reference_logprobs = torch.cat(reference_logprob_chunks)

        detached_losses: list[float] = []
        detached_kls: list[float] = []
        detached_clipped: list[float] = []
        optimizer_step_count = 0
        stop_training = False
        for epoch_index in range(update_epochs):
            if stop_training:
                break
            for start, end in _minibatch_ranges(len(batch.contributing), minibatch_size):
                minibatch_size_actual = end - start
                self.optimizer.zero_grad(set_to_none=True)
                minibatch_kl_sum = 0.0
                minibatch_kl_count = 0
                for micro_start, micro_end in _microbatch_ranges(start, end, microbatch_size):
                    microbatch = batch.contributing[micro_start:micro_end]
                    microbatch_reference_logprobs = reference_logprobs[micro_start:micro_end]
                    samples = [sample for sample, _advantage in microbatch]
                    advantages = torch.tensor(
                        [advantage for _sample, advantage in microbatch],
                        dtype=torch.float32,
                        device=self.accelerator.device,
                    )
                    logprobs = self._sequence_logprobs(samples)
                    losses, approx_kls, clipped = _clipped_grpo_losses(
                        logprobs,
                        microbatch_reference_logprobs,
                        advantages,
                        clip_range=clip_range,
                    )
                    detached_losses.extend(losses.detach().cpu().tolist())
                    detached_kls.extend(approx_kls.detach().cpu().tolist())
                    detached_clipped.extend(clipped.detach().cpu().tolist())
                    if approx_kls.numel():
                        minibatch_kl_sum += float(approx_kls.detach().sum().cpu())
                        minibatch_kl_count += int(approx_kls.numel())
                    self.accelerator.backward(losses.sum() / minibatch_size_actual)
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                self.optimizer.step()
                optimizer_step_count += 1
                if is_main:
                    print(
                        "[FSDP2CPTrainer]  "
                        f"epoch={epoch_index + 1}/{update_epochs} "
                        f"optimizer_step={optimizer_step_count} minibatch_size={minibatch_size_actual}"
                    )
                target_kl = self.training_config.target_kl
                minibatch_kl = minibatch_kl_sum / minibatch_kl_count if minibatch_kl_count else 0.0
                if target_kl is not None and minibatch_kl_count and minibatch_kl > target_kl:
                    if is_main:
                        print(f"[FSDP2CPTrainer]  Stopping early: target_kl={target_kl} reached")
                    stop_training = True
                    break
        return UpdateMetrics(
            sample_count=len(batch.flat_samples),
            mean_reward=sum(sample.reward for sample in batch.flat_samples) / len(batch.flat_samples),
            mean_advantage=sum(batch.advantages) / len(batch.advantages),
            loss=sum(detached_losses) / len(detached_losses),
            optimizer_step_count=optimizer_step_count,
            mean_policy_kl=sum(detached_kls) / len(detached_kls),
            clip_fraction=sum(detached_clipped) / len(detached_clipped),
        )

    def save_checkpoint(self, path: str) -> None:
        self.accelerator.wait_for_everyone()
        self.accelerator.save_model(self.model, path, safe_serialization=True)
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.config.save_pretrained(path)
            if hasattr(unwrapped_model, "generation_config") and unwrapped_model.generation_config is not None:
                unwrapped_model.generation_config.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            _copy_processor_configs(self.model_config.model_path, path)
        self.accelerator.wait_for_everyone()
