from __future__ import annotations

from dataclasses import dataclass, field
import functools
import os
from pathlib import Path
import shutil
from typing import Any
import warnings

import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer

from self_summarization_agent.config import ModelConfig, TrainingConfig
from self_summarization_agent.trajectory import RLSample


@dataclass(slots=True)
class _PolicyBatch:
    flat_samples: list[RLSample]
    advantages: list[float]
    contributing: list[tuple[RLSample, float]]


def compute_group_advantages(samples: list[RLSample]) -> list[float]:
    if not samples:
        return []
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


def _validate_grpo_training_config(training_config: TrainingConfig) -> tuple[int, int | None, float]:
    update_epochs = training_config.update_epochs
    if update_epochs < 1:
        raise ValueError(f"training.update_epochs must be at least 1, got {update_epochs}")
    minibatch_size = training_config.minibatch_size
    if minibatch_size is not None and minibatch_size < 1:
        raise ValueError(f"training.minibatch_size must be at least 1, got {minibatch_size}")
    clip_range = training_config.clip_range
    if clip_range < 0:
        raise ValueError(f"training.clip_range must be non-negative, got {clip_range}")
    target_kl = training_config.target_kl
    if target_kl is not None and target_kl <= 0:
        raise ValueError(f"training.target_kl must be positive when set, got {target_kl}")
    return update_epochs, minibatch_size, clip_range


def _minibatches(
    items: list[tuple[RLSample, float]],
    reference_logprobs: list[torch.Tensor],
    minibatch_size: int | None,
):
    effective_size = len(items) if minibatch_size is None else minibatch_size
    for start in range(0, len(items), effective_size):
        end = start + effective_size
        yield items[start:end], reference_logprobs[start:end]


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


@dataclass(slots=True)
class TransformersPolicyTrainer:
    model_config: ModelConfig
    training_config: TrainingConfig
    tokenizer: Any = field(init=False)
    model: Any = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_path,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_config.model_path,
            torch_dtype=self._torch_dtype(),
            device_map=self.model_config.device_map,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_config.learning_rate)
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

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _format_prompt(self, prompt: str) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt
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
        prompt_ids = self.tokenizer.encode(sample.prompt, add_special_tokens=False)
        full_ids = self.tokenizer.encode(sample.prompt + sample.completion, add_special_tokens=False)
        max_len = getattr(self.training_config, "max_sequence_length", None)
        if max_len is not None and len(full_ids) > max_len:
            drop = len(full_ids) - max_len
            full_ids = full_ids[drop:]
            prompt_ids = prompt_ids[drop:] if len(prompt_ids) > drop else []
        if len(full_ids) <= len(prompt_ids):
            return torch.zeros((), device=self.model.device)
        input_ids = torch.tensor([full_ids], device=self.model.device)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        completion_start = max(len(prompt_ids) - 1, 0)
        completion_log_probs = token_log_probs[:, completion_start:]
        return completion_log_probs.mean()

    def step(self, grouped_samples: dict[str, list[RLSample]]) -> UpdateMetrics:
        update_epochs, minibatch_size, clip_range = _validate_grpo_training_config(self.training_config)
        batch = _prepare_policy_batch(grouped_samples)
        if not batch.flat_samples or not batch.contributing:
            return _metrics_without_update(batch)

        is_main = int(os.environ.get("RANK", "0")) == 0
        if is_main:
            print(
                "[TransformersPolicyTrainer] Starting clipped GRPO update: "
                f"{len(batch.flat_samples)} samples, {len(batch.contributing)} contributing, "
                f"update_epochs={update_epochs}, minibatch_size={minibatch_size or len(batch.contributing)}"
            )
        with torch.no_grad():
            reference_logprobs = [
                self._sequence_logprob(sample).detach()
                for sample, _advantage in batch.contributing
            ]

        detached_losses: list[float] = []
        detached_kls: list[float] = []
        detached_clipped: list[float] = []
        optimizer_step_count = 0
        stop_training = False
        for epoch_index in range(update_epochs):
            if stop_training:
                break
            for minibatch, minibatch_reference_logprobs in _minibatches(
                batch.contributing,
                reference_logprobs,
                minibatch_size,
            ):
                self.optimizer.zero_grad(set_to_none=True)
                scale = 1.0 / len(minibatch)
                minibatch_kls: list[float] = []
                for (sample, advantage), reference_logprob in zip(minibatch, minibatch_reference_logprobs):
                    logprob = self._sequence_logprob(sample)
                    loss, approx_kl, clipped = _clipped_grpo_loss(
                        logprob,
                        reference_logprob,
                        advantage,
                        clip_range=clip_range,
                    )
                    detached_losses.append(float(loss.detach().cpu()))
                    kl_value = float(approx_kl.detach().cpu())
                    detached_kls.append(kl_value)
                    minibatch_kls.append(kl_value)
                    detached_clipped.append(float(clipped.detach().cpu()))
                    (loss * scale).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                self.optimizer.step()
                optimizer_step_count += 1
                if is_main:
                    print(
                        "[TransformersPolicyTrainer]  "
                        f"epoch={epoch_index + 1}/{update_epochs} "
                        f"optimizer_step={optimizer_step_count} minibatch_size={len(minibatch)}"
                    )
                target_kl = self.training_config.target_kl
                if target_kl is not None and minibatch_kls and sum(minibatch_kls) / len(minibatch_kls) > target_kl:
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
        prompt_ids = self.tokenizer.encode(sample.prompt, add_special_tokens=False)
        full_ids = self.tokenizer.encode(sample.prompt + sample.completion, add_special_tokens=False)
        if len(full_ids) <= len(prompt_ids):
            empty = torch.empty((1, 0), dtype=torch.long, device=self.accelerator.device)
            return empty, empty, empty.to(dtype=torch.bool)

        input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=self.accelerator.device)
        labels = torch.tensor([full_ids[1:]], dtype=torch.long, device=self.accelerator.device)
        completion_mask = torch.zeros_like(labels, dtype=torch.bool)
        completion_start = max(len(prompt_ids) - 1, 0)
        completion_mask[:, completion_start:] = True
        input_ids, labels, completion_mask = self._pad_for_context_parallel(input_ids, labels, completion_mask)
        return input_ids, labels, completion_mask

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

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

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
        input_ids, labels, completion_mask = self._encode_shifted_sample(sample)
        if input_ids.numel() == 0:
            return torch.zeros((), device=self.accelerator.device)

        max_len = getattr(self.training_config, "max_sequence_length", None)
        if max_len is not None and input_ids.shape[1] > max_len:
            input_ids = input_ids[:, -max_len:]
            labels = labels[:, -max_len:]
            completion_mask = completion_mask[:, -max_len:]

        if self.training_config.context_parallel_size > 1:
            buffers = [input_ids, labels, completion_mask]
            with self.accelerator.maybe_context_parallel(
                buffers=buffers,
                buffer_seq_dims=[1, 1, 1],
                no_restore_buffers=set(buffers),
            ):
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits
                token_losses = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="none",
                ).reshape_as(labels)
                local_logprob_sum = -(token_losses * completion_mask.to(token_losses.dtype)).sum()
                local_token_count = completion_mask.sum().to(token_losses.dtype)
        else:
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
            token_losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape_as(labels)
            local_logprob_sum = -(token_losses * completion_mask.to(token_losses.dtype)).sum()
            local_token_count = completion_mask.sum().to(token_losses.dtype)

        with torch.no_grad():
            global_token_count = self.accelerator.reduce(local_token_count.detach(), reduction="sum")
        if global_token_count.item() == 0:
            return torch.zeros((), device=self.accelerator.device)
        return local_logprob_sum / global_token_count.clamp_min(1)

    def step(self, grouped_samples: dict[str, list[RLSample]]) -> UpdateMetrics:
        update_epochs, minibatch_size, clip_range = _validate_grpo_training_config(self.training_config)
        batch = _prepare_policy_batch(grouped_samples)
        if not batch.flat_samples or not batch.contributing:
            return _metrics_without_update(batch)

        is_main = getattr(self.accelerator, "is_main_process", True)
        if is_main:
            print(
                "[FSDP2CPTrainer] Starting clipped GRPO update: "
                f"{len(batch.flat_samples)} samples, {len(batch.contributing)} contributing, "
                f"update_epochs={update_epochs}, minibatch_size={minibatch_size or len(batch.contributing)}"
            )
        with torch.no_grad():
            reference_logprobs = [
                self._sequence_logprob(sample).detach()
                for sample, _advantage in batch.contributing
            ]

        detached_losses: list[float] = []
        detached_kls: list[float] = []
        detached_clipped: list[float] = []
        optimizer_step_count = 0
        stop_training = False
        for epoch_index in range(update_epochs):
            if stop_training:
                break
            for minibatch, minibatch_reference_logprobs in _minibatches(
                batch.contributing,
                reference_logprobs,
                minibatch_size,
            ):
                self.optimizer.zero_grad(set_to_none=True)
                scale = 1.0 / len(minibatch)
                minibatch_kls: list[float] = []
                for (sample, advantage), reference_logprob in zip(minibatch, minibatch_reference_logprobs):
                    logprob = self._sequence_logprob(sample)
                    loss, approx_kl, clipped = _clipped_grpo_loss(
                        logprob,
                        reference_logprob,
                        advantage,
                        clip_range=clip_range,
                    )
                    detached_losses.append(float(loss.detach().cpu()))
                    kl_value = float(approx_kl.detach().cpu())
                    detached_kls.append(kl_value)
                    minibatch_kls.append(kl_value)
                    detached_clipped.append(float(clipped.detach().cpu()))
                    self.accelerator.backward(loss * scale)
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                self.optimizer.step()
                optimizer_step_count += 1
                if is_main:
                    print(
                        "[FSDP2CPTrainer]  "
                        f"epoch={epoch_index + 1}/{update_epochs} "
                        f"optimizer_step={optimizer_step_count} minibatch_size={len(minibatch)}"
                    )
                target_kl = self.training_config.target_kl
                if target_kl is not None and minibatch_kls and sum(minibatch_kls) / len(minibatch_kls) > target_kl:
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
