from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from self_summarization_agent.config import TrainingConfig
from self_summarization_agent.token_value import advantage_mean, TokenAdvantages
from self_summarization_agent.trajectory import RLSample


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


@dataclass(slots=True)
class _PolicyBatch:
    flat_samples: list[RLSample]
    advantages: list[float | TokenAdvantages]
    contributing: list[tuple[RLSample, float | TokenAdvantages]]


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


def _prepare_policy_batch(
    grouped_samples: dict[str, list[RLSample]],
    *,
    include_zero_advantage: bool = False,
) -> _PolicyBatch:
    flat_samples: list[RLSample] = []
    advantages: list[float] = []
    for samples in grouped_samples.values():
        if not samples:
            continue
        group_advantages = compute_group_advantages(samples)
        flat_samples.extend(samples)
        advantages.extend(group_advantages)
    if include_zero_advantage:
        # Match the frozen-advantage (separate critic) trace handling: keep
        # every interval with trainable tokens, including zero-variance
        # groups whose group-normalized advantage is exactly zero.
        contributing = [
            (sample, advantage)
            for sample, advantage in zip(flat_samples, advantages)
            if any(sample.completion_mask or [])
        ]
    else:
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
        mean_advantage=advantage_mean(batch.advantages),
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
