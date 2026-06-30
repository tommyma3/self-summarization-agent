from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import time
from pathlib import Path
from typing import Any

import torch

from self_summarization_agent.config import ModelConfig, TrainingConfig
from self_summarization_agent.trainer import TransformersPolicyTrainer, UpdateMetrics
from self_summarization_agent.trajectory import RLSample


def _require_verl_ray():
    try:
        import numpy as np
        import ray
        from verl import DataProto
    except ImportError as exc:
        raise ImportError(
            "training.backend='verl_ray' requires the optional verl dependencies. "
            "Install them with `uv sync --extra verl` or install SkillZero/verl and ray "
            "in the remote GPU environment."
        ) from exc
    return np, ray, DataProto


def _pad_cached_sequences(samples: list[RLSample]) -> dict[str, torch.Tensor]:
    if not samples:
        return {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "labels": torch.empty((0, 0), dtype=torch.long),
            "completion_mask": torch.empty((0, 0), dtype=torch.bool),
            "reference_logprobs": torch.empty((0,), dtype=torch.float32),
            "rewards": torch.empty((0,), dtype=torch.float32),
            "sequence_lengths": torch.empty((0,), dtype=torch.long),
        }
    lengths = [len(sample.input_ids or []) for sample in samples]
    max_length = max(lengths)
    input_ids = torch.zeros((len(samples), max_length), dtype=torch.long)
    labels = torch.zeros((len(samples), max_length), dtype=torch.long)
    completion_mask = torch.zeros((len(samples), max_length), dtype=torch.bool)
    reference_logprobs = torch.empty((len(samples),), dtype=torch.float32)
    rewards = torch.empty((len(samples),), dtype=torch.float32)
    sequence_lengths = torch.tensor(lengths, dtype=torch.long)
    for index, sample in enumerate(samples):
        length = lengths[index]
        input_ids[index, :length] = torch.tensor(sample.input_ids or [], dtype=torch.long)
        labels[index, :length] = torch.tensor(sample.labels or [], dtype=torch.long)
        completion_mask[index, :length] = torch.tensor(sample.completion_mask or [], dtype=torch.bool)
        reference_logprobs[index] = float(sample.reference_logprob)
        rewards[index] = float(sample.reward)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "completion_mask": completion_mask,
        "reference_logprobs": reference_logprobs,
        "rewards": rewards,
        "sequence_lengths": sequence_lengths,
    }


def _flatten_grouped_samples(grouped_samples: dict[str, list[RLSample]]) -> list[RLSample]:
    flat_samples: list[RLSample] = []
    for samples in grouped_samples.values():
        flat_samples.extend(samples)
    return flat_samples


def build_verl_dataproto(grouped_samples: dict[str, list[RLSample]], *, checkpoint_id: str):
    np, _ray, DataProto = _require_verl_ray()
    flat_samples = _flatten_grouped_samples(grouped_samples)
    for sample in flat_samples:
        if not sample.has_training_cache:
            raise ValueError(f"Sample {sample.turn_id} is missing training cache")
    tensors = _pad_cached_sequences(flat_samples)
    total_tokens = int(tensors["sequence_lengths"].sum().item()) if flat_samples else 0
    non_tensors = {
        "query_ids": np.array([sample.query_id for sample in flat_samples], dtype=object),
        "turn_ids": np.array([sample.turn_id for sample in flat_samples], dtype=object),
        "trainable_kinds": np.array([sample.trainable_kind for sample in flat_samples], dtype=object),
    }
    return DataProto.from_single_dict(
        {
            **tensors,
            **non_tensors,
        },
        meta_info={
            "policy_checkpoint_id": checkpoint_id,
            "query_count": len(grouped_samples),
            "sample_count": len(flat_samples),
            "total_tokens": total_tokens,
            "max_sequence_length": max((len(sample.input_ids or []) for sample in flat_samples), default=0),
        },
    )


def grouped_samples_from_verl_dataproto(batch: Any) -> dict[str, list[RLSample]]:
    grouped: dict[str, list[RLSample]] = {}
    tensors = batch.batch
    query_ids = batch.non_tensor_batch["query_ids"]
    turn_ids = batch.non_tensor_batch["turn_ids"]
    kinds = batch.non_tensor_batch["trainable_kinds"]
    for index in range(len(batch)):
        length = int(tensors["sequence_lengths"][index].item())
        sample = RLSample(
            query_id=str(query_ids[index]),
            turn_id=str(turn_ids[index]),
            prompt="",
            completion="",
            reward=float(tensors["rewards"][index].item()),
            trainable_kind=str(kinds[index]),
            input_ids=[int(value) for value in tensors["input_ids"][index, :length].tolist()],
            labels=[int(value) for value in tensors["labels"][index, :length].tolist()],
            completion_mask=[bool(value) for value in tensors["completion_mask"][index, :length].tolist()],
            reference_logprob=float(tensors["reference_logprobs"][index].item()),
        )
        grouped.setdefault(sample.query_id, []).append(sample)
    return grouped


def _metrics_to_dict(metrics: UpdateMetrics) -> dict[str, Any]:
    return asdict(metrics)


def _metrics_from_dict(payload: dict[str, Any]) -> UpdateMetrics:
    known = {
        "sample_count",
        "mean_reward",
        "mean_advantage",
        "loss",
        "optimizer_step_count",
        "mean_policy_kl",
        "clip_fraction",
        "extra_metrics",
    }
    return UpdateMetrics(**{key: value for key, value in payload.items() if key in known})


class _VerlRayTrainerActor:
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        worker_backend = training_config.verl.worker_backend
        if worker_backend != "transformers":
            raise ValueError(
                "training.verl.worker_backend currently supports only 'transformers'. "
                f"Got {worker_backend!r}."
            )
        self.trainer = TransformersPolicyTrainer(
            model_config,
            replace(training_config, backend=worker_backend),
        )

    def step(self, batch: Any) -> dict[str, Any]:
        grouped_samples = grouped_samples_from_verl_dataproto(batch)
        started = time.monotonic()
        metrics = _metrics_to_dict(self.trainer.step(grouped_samples))
        metrics.setdefault("extra_metrics", {})
        metrics["extra_metrics"].update(
            {
                "verl_ray/update_seconds": time.monotonic() - started,
                "verl_ray/worker_backend": "transformers",
            }
        )
        return metrics

    def save_checkpoint(self, path: str) -> None:
        self.trainer.save_checkpoint(path)


@dataclass(slots=True)
class VerlRayPolicyTrainer:
    model_config: ModelConfig
    training_config: TrainingConfig
    checkpoint_id: str
    _ray: Any = field(init=False, repr=False)
    _actor: Any = field(init=False, repr=False)
    _owns_ray: bool = field(init=False, repr=False)
    _last_batch_meta: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        _np, ray, _DataProto = _require_verl_ray()
        self._ray = ray
        init_kwargs: dict[str, Any] = {
            "ignore_reinit_error": self.training_config.verl.ignore_reinit_error,
            "log_to_driver": self.training_config.verl.log_to_driver,
        }
        if self.training_config.verl.address is not None:
            init_kwargs["address"] = self.training_config.verl.address
        if self.training_config.verl.namespace:
            init_kwargs["namespace"] = self.training_config.verl.namespace
        if self.training_config.verl.runtime_env is not None:
            init_kwargs["runtime_env"] = self.training_config.verl.runtime_env
        if self.training_config.verl.address is None and self.training_config.verl.num_cpus is not None:
            init_kwargs["num_cpus"] = self.training_config.verl.num_cpus
        self._owns_ray = not ray.is_initialized()
        if self._owns_ray:
            ray.init(**init_kwargs)

        num_gpus = self.training_config.verl.num_gpus_per_worker
        if num_gpus is None:
            num_gpus = float(len(self.training_config.gpu_ids) or self.training_config.data_parallel_size or 1)
        actor_options: dict[str, Any] = {"num_gpus": num_gpus}
        if self.training_config.verl.num_cpus is not None:
            actor_options["num_cpus"] = self.training_config.verl.num_cpus
        remote_cls = ray.remote(**actor_options)(_VerlRayTrainerActor)
        self._actor = remote_cls.remote(self.model_config, self.training_config)

    def step(self, grouped_samples: dict[str, list[RLSample]]) -> UpdateMetrics:
        batch = build_verl_dataproto(grouped_samples, checkpoint_id=self.checkpoint_id)
        self._last_batch_meta = dict(batch.meta_info)
        started = time.monotonic()
        metrics_payload = self._ray.get(self._actor.step.remote(batch))
        metrics = _metrics_from_dict(metrics_payload)
        metrics.extra_metrics.update(
            {
                "backend": "verl_ray",
                "verl_ray/driver_roundtrip_seconds": time.monotonic() - started,
                "verl_ray/query_count": self._last_batch_meta.get("query_count", 0),
                "verl_ray/sample_count": self._last_batch_meta.get("sample_count", 0),
                "verl_ray/total_tokens": self._last_batch_meta.get("total_tokens", 0),
                "verl_ray/max_sequence_length": self._last_batch_meta.get("max_sequence_length", 0),
            }
        )
        return metrics

    def save_checkpoint(self, path: str) -> None:
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        self._ray.get(self._actor.save_checkpoint.remote(str(output_path)))
        if self.training_config.verl.shutdown_ray and self._owns_ray:
            self._ray.shutdown()
