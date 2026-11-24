"""Frozen policy forward pass for caching reference log probabilities."""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    AutoModelForMultimodalLM = AutoModel  # type: ignore[misc,assignment]

from self_summarization_agent.config import ModelConfig
from self_summarization_agent.chat_template import configure_tokenizer_chat_template
from self_summarization_agent.trajectory import (
    REFERENCE_LOGPROB_SOURCE_POLICY_RESCORE,
    RLSample,
    TOKEN_CACHE_VERSION,
    tokenize_interval_messages,
)


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
        raise ValueError(f"Interval {sample.turn_id} exceeds training.max_sequence_length")

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

    if sample.has_exact_collection_tokens:
        raise ValueError(f"Sample {sample.turn_id} is missing its collected value-state anchor")

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


@dataclass(slots=True)
class PolicyCacheScorer:
    model_config: ModelConfig
    max_sequence_length: int | None
    tokenizer: Any = field(init=False)
    model: Any = field(init=False)

    def __post_init__(self) -> None:
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
        self.model.eval()

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

    def _model_device(self) -> torch.device:
        device = getattr(self.model, "device", None)
        if device is not None:
            return torch.device(device)
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def cache_samples(self, samples: list[RLSample]) -> list[dict[str, Any]]:
        if not samples:
            return []
        device = self._model_device()
        forward_params = inspect.signature(self.model.forward).parameters
        payloads = []
        for sample in samples:
            if self.model_config.require_exact_token_ids and not sample.has_exact_collection_tokens:
                raise ValueError(f"Sample {sample.turn_id} is missing exact collection tokens for rescoring")
            ids, labels, mask = _encode_shifted_sample_from_text(
                sample, tokenizer=self.tokenizer, device=device,
                max_sequence_length=self.max_sequence_length,
                enable_thinking=self.model_config.enable_thinking,
            )
            # Keep the complete causal input. Qwen can project just the
            # positions predicting sampled targets, even for a 27k interval.
            positions = mask.nonzero(as_tuple=True)[0]
            kwargs = {"use_cache": False} if "use_cache" in forward_params else {}
            if "logits_to_keep" in forward_params:
                kwargs["logits_to_keep"] = positions.tolist()
            logits = self.model(input_ids=ids.unsqueeze(0), **kwargs).logits.squeeze(0)
            positions = positions.to(logits.device)
            if "logits_to_keep" not in forward_params:
                logits = logits.index_select(0, positions)
            targets = labels.to(logits.device).index_select(0, positions)
            scores = torch.zeros_like(labels, dtype=torch.float32, device=logits.device)
            for start in range(0, len(positions), _CE_SEQUENCE_CHUNK_TOKENS):
                span = slice(start, start + _CE_SEQUENCE_CHUNK_TOKENS)
                scores[positions[span]] = -torch.nn.functional.cross_entropy(
                    logits[span].float(), targets[span], reduction="none",
                )
            del logits
            payloads.append(_training_cache_payload(
                input_ids=ids, labels=labels, completion_mask=mask,
                reference_logprob=float(scores.sum() / max(1, positions.numel())), reference_logprobs=scores,
                state_prefix_length=_exact_state_prefix_length(
                    sample, tokenizer=self.tokenizer, input_ids=ids,
                    enable_thinking=self.model_config.enable_thinking,
                ),
            ))
        return payloads
