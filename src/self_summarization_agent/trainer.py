from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer

from self_summarization_agent.config import ModelConfig, TrainingConfig
from self_summarization_agent.trajectory import RLSample


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
        flat_samples: list[RLSample] = []
        advantages: list[float] = []
        for samples in grouped_samples.values():
            if not samples:
                continue
            group_advantages = compute_group_advantages(samples)
            flat_samples.extend(samples)
            advantages.extend(group_advantages)
        if not flat_samples:
            return UpdateMetrics(sample_count=0, mean_reward=0.0, mean_advantage=0.0, loss=0.0)

        contributing = [
            (sample, advantage)
            for sample, advantage in zip(flat_samples, advantages)
            if abs(advantage) >= 1e-8
        ]
        if not contributing:
            return UpdateMetrics(
                sample_count=len(flat_samples),
                mean_reward=sum(sample.reward for sample in flat_samples) / len(flat_samples),
                mean_advantage=sum(advantages) / len(advantages),
                loss=0.0,
            )

        self.optimizer.zero_grad(set_to_none=True)
        detached_losses: list[float] = []
        scale = 1.0 / len(contributing)
        for sample, advantage in contributing:
            logprob = self._sequence_logprob(sample)
            loss = -torch.tensor(advantage, device=self.model.device) * logprob
            detached_losses.append(float(loss.detach().cpu()))
            (loss * scale).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
        self.optimizer.step()
        return UpdateMetrics(
            sample_count=len(flat_samples),
            mean_reward=sum(sample.reward for sample in flat_samples) / len(flat_samples),
            mean_advantage=sum(advantages) / len(advantages),
            loss=sum(detached_losses) / len(detached_losses),
        )

    def save_checkpoint(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


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
        if self.training_config.context_parallel_size <= 1:
            raise ValueError("FSDP2 context-parallel training requires context_parallel_size > 1")

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
            self.model.gradient_checkpointing_enable()
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
        return input_ids, labels, completion_mask

    def _sequence_logprob(self, sample: RLSample) -> torch.Tensor:
        input_ids, labels, completion_mask = self._encode_shifted_sample(sample)
        if input_ids.numel() == 0:
            return torch.zeros((), device=self.accelerator.device)

        buffers = [input_ids, labels, completion_mask]
        with self.accelerator.maybe_context_parallel(
            buffers=buffers,
            buffer_seq_dims=[1, 1, 1],
            no_restore_buffers=set(buffers),
        ):
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
            token_losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                labels.reshape(-1),
                reduction="none",
            ).reshape_as(labels)
            local_logprob_sum = -(token_losses * completion_mask.to(token_losses.dtype)).sum()
            local_token_count = completion_mask.sum().to(token_losses.dtype)

        logprob_sum = self.accelerator.reduce(local_logprob_sum, reduction="sum")
        token_count = self.accelerator.reduce(local_token_count, reduction="sum")
        if token_count.item() == 0:
            return torch.zeros((), device=self.accelerator.device)
        return logprob_sum / token_count.clamp_min(1)

    def step(self, grouped_samples: dict[str, list[RLSample]]) -> UpdateMetrics:
        flat_samples: list[RLSample] = []
        advantages: list[float] = []
        for samples in grouped_samples.values():
            if not samples:
                continue
            group_advantages = compute_group_advantages(samples)
            flat_samples.extend(samples)
            advantages.extend(group_advantages)
        if not flat_samples:
            return UpdateMetrics(sample_count=0, mean_reward=0.0, mean_advantage=0.0, loss=0.0)

        contributing = [
            (sample, advantage)
            for sample, advantage in zip(flat_samples, advantages)
            if abs(advantage) >= 1e-8
        ]
        if not contributing:
            return UpdateMetrics(
                sample_count=len(flat_samples),
                mean_reward=sum(sample.reward for sample in flat_samples) / len(flat_samples),
                mean_advantage=sum(advantages) / len(advantages),
                loss=0.0,
            )

        self.optimizer.zero_grad(set_to_none=True)
        detached_losses: list[float] = []
        scale = 1.0 / len(contributing)
        for sample, advantage in contributing:
            logprob = self._sequence_logprob(sample)
            loss = -torch.tensor(advantage, device=self.accelerator.device) * logprob
            detached_losses.append(float(loss.detach().cpu()))
            self.accelerator.backward(loss * scale)
        self.accelerator.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
        self.optimizer.step()
        return UpdateMetrics(
            sample_count=len(flat_samples),
            mean_reward=sum(sample.reward for sample in flat_samples) / len(flat_samples),
            mean_advantage=sum(advantages) / len(advantages),
            loss=sum(detached_losses) / len(detached_losses),
        )

    def save_checkpoint(self, path: str) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(path, safe_serialization=True)
            self.tokenizer.save_pretrained(path)
        self.accelerator.wait_for_everyone()
