from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_path,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
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

        losses: list[torch.Tensor] = []
        for sample, advantage in zip(flat_samples, advantages):
            if abs(advantage) < 1e-8:
                continue
            logprob = self._sequence_logprob(sample)
            losses.append(-torch.tensor(advantage, device=self.model.device) * logprob)
        if not losses:
            return UpdateMetrics(
                sample_count=len(flat_samples),
                mean_reward=sum(sample.reward for sample in flat_samples) / len(flat_samples),
                mean_advantage=sum(advantages) / len(advantages),
                loss=0.0,
            )

        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
        self.optimizer.step()
        return UpdateMetrics(
            sample_count=len(flat_samples),
            mean_reward=sum(sample.reward for sample in flat_samples) / len(flat_samples),
            mean_advantage=sum(advantages) / len(advantages),
            loss=float(loss.detach().cpu()),
        )

    def save_checkpoint(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
