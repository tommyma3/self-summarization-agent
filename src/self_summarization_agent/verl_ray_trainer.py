from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from self_summarization_agent.config import ModelConfig, TrainingConfig
from self_summarization_agent.trainer import (
    TransformersPolicyTrainer,
    UpdateMetrics,
    _metrics_without_update,
    _prepare_policy_batch,
)
from self_summarization_agent.trajectory import RLSample


SUPPORTED_VERL_WORKER_BACKENDS = {"transformers", "verl_fsdp"}


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


def _require_verl_worker_group():
    try:
        import ray
        from omegaconf import OmegaConf
        from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
        from verl.workers.engine_workers import ActorRolloutRefWorker
    except ImportError as exc:
        raise ImportError(
            "training.verl.worker_backend='verl_fsdp' requires official verl worker-group dependencies. "
            "Install them in the remote GPU environment with `uv sync --extra verl --group dev`."
        ) from exc
    return OmegaConf, RayClassWithInitArgs, RayResourcePool, RayWorkerGroup, ActorRolloutRefWorker, ray


def _coerce_scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _mean_metric(metrics: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, list):
            values = [_coerce_scalar(item) for item in value]
            return sum(values) / len(values) if values else 0.0
        return _coerce_scalar(value)
    return 0.0


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


def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    return torch.clamp(attention_mask.long().cumsum(dim=-1) - 1, min=0)


def _padded_to_nested(padded_2d: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Convert a padded 2D tensor (batch, max_seq_len) to a NestedTensor with
    jagged layout, using the per-sample ``lengths`` to trim each row."""
    if padded_2d.dim() != 2:
        raise ValueError(f"Expected 2D padded tensor, got shape {padded_2d.shape}")
    tensors = [padded_2d[i, : int(lengths[i].item())] for i in range(len(lengths))]
    return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)


def build_verl_actor_dataproto(grouped_samples: dict[str, list[RLSample]], *, checkpoint_id: str):
    np, _ray, DataProto = _require_verl_ray()
    policy_batch = _prepare_policy_batch(grouped_samples)
    for sample in policy_batch.flat_samples:
        if not sample.has_training_cache:
            raise ValueError(f"Sample {sample.turn_id} is missing training cache")
    contributing = policy_batch.contributing
    samples = [sample for sample, _advantage in contributing]
    tensors = _pad_cached_sequences(samples)
    input_ids = tensors["input_ids"]
    completion_mask = tensors["completion_mask"]
    sequence_lengths = tensors["sequence_lengths"]
    attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
    for index, length in enumerate(sequence_lengths.tolist()):
        attention_mask[index, : int(length)] = 1
    advantages = torch.zeros_like(input_ids, dtype=torch.float32)
    old_log_probs = torch.zeros_like(input_ids, dtype=torch.float32)
    for index, (sample, advantage) in enumerate(contributing):
        mask = completion_mask[index]
        advantages[index, mask] = float(advantage)
        if sample.reference_logprobs is not None:
            old_log_probs[index, : len(sample.reference_logprobs)] = torch.tensor(
                sample.reference_logprobs,
                dtype=torch.float32,
            )
        else:
            old_log_probs[index, mask] = float(sample.reference_logprob)
    responses = input_ids.masked_fill(~completion_mask, 0)
    total_tokens = int(sequence_lengths.sum().item()) if samples else 0
    train_tokens = int(completion_mask.sum().item()) if samples else 0
    old_logprob_scope = (
        "token"
        if all(sample.reference_logprobs is not None for sample in samples)
        else "sequence_mean_broadcast_to_completion_tokens"
    )
    non_tensors = {
        "query_ids": np.array([sample.query_id for sample in samples], dtype=object),
        "turn_ids": np.array([sample.turn_id for sample in samples], dtype=object),
        "trainable_kinds": np.array([sample.trainable_kind for sample in samples], dtype=object),
    }
    return DataProto.from_single_dict(
        {
            "input_ids": _padded_to_nested(input_ids, sequence_lengths),
            "attention_mask": _padded_to_nested(attention_mask, sequence_lengths),
            "position_ids": _padded_to_nested(_position_ids(attention_mask), sequence_lengths),
            "responses": _padded_to_nested(responses, sequence_lengths),
            "response_mask": _padded_to_nested(completion_mask, sequence_lengths),
            "loss_mask": _padded_to_nested(completion_mask, sequence_lengths),
            "old_log_probs": _padded_to_nested(old_log_probs, sequence_lengths),
            "advantages": _padded_to_nested(advantages, sequence_lengths),
            "rewards": tensors["rewards"],
            "sequence_lengths": sequence_lengths,
            **non_tensors,
        },
        meta_info={
            "policy_checkpoint_id": checkpoint_id,
            "query_count": len(grouped_samples),
            "sample_count": len(policy_batch.flat_samples),
            "contributing_sample_count": len(contributing),
            "total_tokens": total_tokens,
            "train_tokens": train_tokens,
            "max_sequence_length": max((len(sample.input_ids or []) for sample in samples), default=0),
            "old_logprob_scope": old_logprob_scope,
            "mini_batch_size": None,
            "epochs": 1,
            "seed": 42,
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


def _verl_dtype(model_config: ModelConfig) -> str:
    if model_config.dtype in {"bfloat16", "float16"}:
        return model_config.dtype
    return "bfloat16"


def _effective_verl_token_limit(training_config: TrainingConfig) -> int:
    fsdp_config = training_config.verl.fsdp
    if fsdp_config.ppo_max_token_len_per_gpu is not None:
        return fsdp_config.ppo_max_token_len_per_gpu
    if training_config.max_sequence_length is not None:
        return training_config.max_sequence_length
    return 16384


def build_verl_fsdp_worker_config(model_config: ModelConfig, training_config: TrainingConfig) -> dict[str, Any]:
    fsdp_config = training_config.verl.fsdp
    token_limit = _effective_verl_token_limit(training_config)
    logprob_token_limit = fsdp_config.log_prob_max_token_len_per_gpu or token_limit
    total_training_steps = max(1, int(training_config.steps or 1) * int(training_config.update_epochs or 1))
    mini_batch_size = training_config.minibatch_size or training_config.group_size
    micro_batch_size = fsdp_config.ppo_micro_batch_size_per_gpu or training_config.gradient_accumulation_microbatch_size
    save_contents = ["hf_model"] if fsdp_config.save_hf_model else ["model"]
    return {
        "model": {
            "_target_": "verl.workers.config.HFModelConfig",
            "path": model_config.model_path,
            "trust_remote_code": model_config.trust_remote_code,
            "enable_gradient_checkpointing": training_config.activation_checkpointing,
            "use_remove_padding": fsdp_config.use_remove_padding,
        },
        "actor": {
            "_target_": "verl.workers.config.FSDPActorConfig",
            "strategy": fsdp_config.strategy,
            "ppo_mini_batch_size": mini_batch_size,
            "ppo_micro_batch_size_per_gpu": micro_batch_size,
            "use_dynamic_bsz": fsdp_config.use_dynamic_bsz,
            "ppo_max_token_len_per_gpu": token_limit,
            "clip_ratio": training_config.clip_range,
            "clip_ratio_low": training_config.clip_range,
            "clip_ratio_high": training_config.clip_range,
            "ppo_epochs": training_config.update_epochs,
            "shuffle": False,
            "grad_clip": training_config.max_grad_norm,
            "entropy_coeff": 0.0,
            "calculate_entropy": False,
            "use_kl_loss": False,
            "use_torch_compile": fsdp_config.use_torch_compile,
            "ulysses_sequence_parallel_size": fsdp_config.ulysses_sequence_parallel_size,
            "use_remove_padding": fsdp_config.use_remove_padding,
            "rollout_n": training_config.group_size,
            "loss_agg_mode": "token-mean",
            "optim": {
                "_target_": "verl.workers.config.FSDPOptimizerConfig",
                "lr": training_config.learning_rate,
                "clip_grad": training_config.max_grad_norm,
                "total_training_steps": total_training_steps,
                "weight_decay": 0.0,
                "lr_scheduler_type": "constant",
            },
            "fsdp_config": {
                "_target_": "verl.workers.config.FSDPEngineConfig",
                "strategy": fsdp_config.strategy,
                "dtype": _verl_dtype(model_config),
                "model_dtype": _verl_dtype(model_config),
                "param_offload": fsdp_config.param_offload,
                "optimizer_offload": fsdp_config.optimizer_offload,
                "fsdp_size": fsdp_config.fsdp_size,
                "use_torch_compile": fsdp_config.use_torch_compile,
                "ulysses_sequence_parallel_size": fsdp_config.ulysses_sequence_parallel_size,
            },
            "checkpoint": {
                "_target_": "verl.trainer.config.CheckpointConfig",
                "save_contents": save_contents,
                "load_contents": ["model"],
                "async_save": False,
            },
        },
        "rollout": {
            "_target_": "verl.workers.config.RolloutConfig",
            "name": "hf",
            "mode": "sync",
            "n": training_config.group_size,
            "log_prob_micro_batch_size_per_gpu": fsdp_config.log_prob_micro_batch_size_per_gpu,
            "log_prob_use_dynamic_bsz": fsdp_config.use_dynamic_bsz,
            "log_prob_max_token_len_per_gpu": logprob_token_limit,
            "checkpoint_engine": {
                "_target_": "verl.workers.config.CheckpointEngineConfig",
                "backend": "naive",
                "update_weights_bucket_megabytes": 2048,
                "engine_kwargs": {},
            },
        },
    }


def _extract_worker_metrics(worker_output: Any) -> dict[str, Any]:
    if worker_output is None:
        return {}
    if isinstance(worker_output, list):
        merged: dict[str, list[Any]] = {}
        for item in worker_output:
            for key, value in _extract_worker_metrics(item).items():
                merged.setdefault(key, []).append(value)
        return merged
    if isinstance(worker_output, dict):
        return dict(worker_output.get("metrics", worker_output))
    if hasattr(worker_output, "meta_info") and isinstance(worker_output.meta_info, dict):
        return dict(worker_output.meta_info.get("metrics", worker_output.meta_info))
    try:
        metrics = worker_output.get("metrics")
    except Exception:
        metrics = None
    if metrics is not None:
        return dict(metrics)
    return {}


class VerlFSDPWorkerGroup:
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        OmegaConf, RayClassWithInitArgs, RayResourcePool, RayWorkerGroup, ActorRolloutRefWorker, ray = _require_verl_worker_group()
        self.training_config = training_config
        self.config = OmegaConf.create(build_verl_fsdp_worker_config(model_config, training_config))
        world_size = int(training_config.verl.num_gpus_per_worker or len(training_config.gpu_ids) or training_config.data_parallel_size or 1)
        self._world_size = world_size
        resource_pool = RayResourcePool(
            process_on_nodes=[world_size],
            use_gpu=True,
            name_prefix=f"{training_config.verl.namespace}-actor-",
        )
        worker_cls = RayClassWithInitArgs(ray.remote(ActorRolloutRefWorker), config=self.config, role="actor")
        self.worker_group = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_cls,
            name_prefix=f"{training_config.verl.namespace}-actor",
        )
        self.worker_group.init_model()

    def update_actor(self, batch: Any) -> dict[str, Any]:
        if hasattr(batch, "to_tensordict"):
            batch_td = batch.to_tensordict()
        else:
            batch_td = getattr(batch, "batch", batch)
        # Trim the batch to be evenly divisible across data-parallel workers
        # AND across per-gpu mini-batches.  verl's dispatch splits the batch
        # with chunk_tensordict (requires len(td) % dp_size == 0), then each
        # worker's TrainingWorker.train_mini_batch requires its local shard
        # to be divisible by mini_batch_size_per_gpu.  Using the global
        # mini_batch_size (when set) satisfies both: it is already a multiple
        # of dp_size by assertion in train_mini_batch.
        import math

        from verl.utils.tensordict_utils import get_non_tensor_data

        mini_batch_size = get_non_tensor_data(batch_td, "mini_batch_size", None)
        step_size = math.lcm(self._world_size, mini_batch_size) if mini_batch_size else self._world_size
        batch_len = len(batch_td)
        trimmed_len = (batch_len // step_size) * step_size
        if trimmed_len != batch_len:
            batch_td = batch_td[:trimmed_len]
        output = self.worker_group.update_actor(batch_td)
        return _extract_worker_metrics(output)

    def save_checkpoint(self, path: str) -> None:
        self.worker_group.save_checkpoint(path, None, 0, None)


def _promote_huggingface_checkpoint(output_path: Path) -> None:
    hf_path = output_path / "huggingface"
    if not hf_path.is_dir():
        return
    for child in hf_path.iterdir():
        target = output_path / child.name
        if target.exists():
            continue
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


class _VerlRayTrainerActor:
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        worker_backend = training_config.verl.worker_backend
        if worker_backend != "transformers":
            raise ValueError(
                "_VerlRayTrainerActor supports only training.verl.worker_backend='transformers'. "
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
    _actor: Any = field(default=None, init=False, repr=False)
    _worker_group: VerlFSDPWorkerGroup | None = field(default=None, init=False, repr=False)
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

        worker_backend = self.training_config.verl.worker_backend
        if worker_backend not in SUPPORTED_VERL_WORKER_BACKENDS:
            raise ValueError(
                "training.verl.worker_backend must be one of "
                f"{sorted(SUPPORTED_VERL_WORKER_BACKENDS)}, got {worker_backend!r}."
            )
        if worker_backend == "verl_fsdp":
            self._worker_group = VerlFSDPWorkerGroup(self.model_config, self.training_config)
            return

        num_gpus = self.training_config.verl.num_gpus_per_worker
        if num_gpus is None:
            num_gpus = float(len(self.training_config.gpu_ids) or self.training_config.data_parallel_size or 1)
        actor_options: dict[str, Any] = {"num_gpus": num_gpus}
        if self.training_config.verl.num_cpus is not None:
            actor_options["num_cpus"] = self.training_config.verl.num_cpus
        remote_cls = ray.remote(**actor_options)(_VerlRayTrainerActor)
        self._actor = remote_cls.remote(self.model_config, self.training_config)

    def step(self, grouped_samples: dict[str, list[RLSample]]) -> UpdateMetrics:
        if self.training_config.verl.worker_backend == "verl_fsdp":
            policy_batch = _prepare_policy_batch(grouped_samples)
            if not policy_batch.flat_samples or not policy_batch.contributing:
                return _metrics_without_update(policy_batch)
            batch = build_verl_actor_dataproto(grouped_samples, checkpoint_id=self.checkpoint_id)
            batch.meta_info["mini_batch_size"] = self.training_config.minibatch_size or len(policy_batch.contributing)
            batch.meta_info["epochs"] = self.training_config.update_epochs
            batch.meta_info["seed"] = 42
            self._last_batch_meta = dict(batch.meta_info)
            started = time.monotonic()
            assert self._worker_group is not None
            worker_metrics = self._worker_group.update_actor(batch)
            update_seconds = time.monotonic() - started
            metrics = UpdateMetrics(
                sample_count=int(self._last_batch_meta.get("sample_count", 0)),
                mean_reward=sum(sample.reward for sample in policy_batch.flat_samples) / len(policy_batch.flat_samples),
                mean_advantage=sum(policy_batch.advantages) / len(policy_batch.advantages),
                loss=_mean_metric(worker_metrics, "loss", "actor/pg_loss"),
                optimizer_step_count=int(
                    self.training_config.update_epochs
                    * max(
                        1,
                        (
                            len(policy_batch.contributing)
                            + (self.training_config.minibatch_size or len(policy_batch.contributing))
                            - 1
                        )
                        // (self.training_config.minibatch_size or len(policy_batch.contributing)),
                    )
                ),
                mean_policy_kl=_mean_metric(worker_metrics, "actor/ppo_kl"),
                clip_fraction=_mean_metric(worker_metrics, "actor/pg_clipfrac", "actor/clipfrac"),
            )
            metrics.extra_metrics.update(
                {
                    "backend": "verl_ray",
                    "verl_ray/worker_backend": "verl_fsdp",
                    "verl_ray/driver_roundtrip_seconds": update_seconds,
                    "verl_fsdp/update_seconds": update_seconds,
                    "verl_fsdp/query_count": self._last_batch_meta.get("query_count", 0),
                    "verl_fsdp/sample_count": self._last_batch_meta.get("sample_count", 0),
                    "verl_fsdp/contributing_sample_count": self._last_batch_meta.get("contributing_sample_count", 0),
                    "verl_fsdp/total_tokens": self._last_batch_meta.get("total_tokens", 0),
                    "verl_fsdp/train_tokens": self._last_batch_meta.get("train_tokens", 0),
                    "verl_fsdp/max_sequence_length": self._last_batch_meta.get("max_sequence_length", 0),
                    "verl_fsdp/old_logprob_scope": self._last_batch_meta.get("old_logprob_scope", ""),
                }
            )
            if update_seconds > 0:
                metrics.extra_metrics["verl_fsdp/effective_train_tokens_per_second"] = (
                    float(self._last_batch_meta.get("train_tokens", 0)) / update_seconds
                )
            for key, value in worker_metrics.items():
                metrics.extra_metrics[f"verl_fsdp/raw/{key}"] = value
            return metrics

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
        if self.training_config.verl.worker_backend == "verl_fsdp":
            assert self._worker_group is not None
            started = time.monotonic()
            self._worker_group.save_checkpoint(str(output_path))
            _promote_huggingface_checkpoint(output_path)
            self._last_batch_meta["verl_fsdp/checkpoint_export_seconds"] = time.monotonic() - started
        else:
            self._ray.get(self._actor.save_checkpoint.remote(str(output_path)))
        if self.training_config.verl.shutdown_ray and self._owns_ray:
            self._ray.shutdown()
