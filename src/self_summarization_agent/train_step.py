from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import warnings
from pathlib import Path
from typing import Any

from self_summarization_agent.checkpoints import checkpoint_id_from_path, mark_checkpoint_complete
from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.launcher_utils import append_jsonl, ensure_dir
from self_summarization_agent.train_grpo import group_samples_by_query
from self_summarization_agent.trainer import (
    FSDP2ContextParallelPolicyTrainer,
    TransformersPolicyTrainer,
    _write_training_progress,
)
from self_summarization_agent.trajectory import (
    TOKEN_CACHE_FIELD,
    extract_training_samples,
    is_training_cache_current,
)
from self_summarization_agent.value_model import (
    VALUE_HEAD_FILENAME,
    VALUE_HEAD_MANIFEST_FILENAME,
    migrate_compaction_value_head_sidecar,
)


def _load_rollout_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid rollout JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Rollout row {line_number} must be a JSON object")
            rows.append(row)
    return rows


def _average_summary_tokens(rows: list[dict[str, Any]]) -> float:
    """Return mean summary-body tokens per rollout, excluding thinking tokens."""
    summary_tokens = 0.0
    for row in rows:
        turn_records = row.get("turn_records")
        if not isinstance(turn_records, list):
            continue
        for turn in turn_records:
            if not isinstance(turn, dict) or turn.get("kind") != "summary":
                continue
            value = turn.get("summary_tokens", 0)
            if isinstance(value, int | float):
                summary_tokens += float(value)
    return summary_tokens / len(rows) if rows else 0.0


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def _sample_diagnostics(samples: list[Any]) -> dict[str, int]:
    lengths = [len(sample.input_ids or []) for sample in samples]
    trainable = [sum(bool(value) for value in (sample.completion_mask or [])) for sample in samples]
    prefixes = [int(sample.state_prefix_length or 0) for sample in samples]
    return {
        "total_tokens": sum(lengths),
        "trainable_tokens": sum(trainable),
        "prefix_tokens": sum(prefixes),
        "max_sequence_length": max(lengths, default=0),
        "p50_sequence_length": _percentile(lengths, 0.50),
        "p95_sequence_length": _percentile(lengths, 0.95),
        "max_prefix_length": max(prefixes, default=0),
        "p95_prefix_length": _percentile(prefixes, 0.95),
    }


def samples_from_rollout_rows(
    rows: list[dict[str, Any]],
    *,
    expected_checkpoint_id: str,
    train_compaction_tokens: bool = True,
    retain_critic_only_states: bool = False,
) -> list[Any]:
    samples = []
    for index, row in enumerate(rows, start=1):
        checkpoint_id = row.get("policy_checkpoint_id")
        if checkpoint_id != expected_checkpoint_id:
            raise ValueError(
                f"Rollout row {index} checkpoint mismatch: expected {expected_checkpoint_id!r}, got {checkpoint_id!r}"
            )
        turn_records = row.get("turn_records")
        trajectory_records = row.get("trajectory_records")
        turn_rewards = row.get("turn_rewards")
        if (
            not isinstance(turn_records, list)
            or not isinstance(trajectory_records, list)
            or not isinstance(turn_rewards, dict)
        ):
            raise ValueError(
                f"Rollout row {index} is missing turn_records, trajectory_records, or turn_rewards"
            )
        if row.get("trainable_sample_count") == 0:
            continue
        row_samples = extract_training_samples(
            trajectory_records,
            turn_rewards,
            rollout_id=f"{row.get('query_id')}:{row.get('rollout_index')}",
            train_compaction_tokens=train_compaction_tokens,
            retain_critic_only_states=retain_critic_only_states,
        )
        if not row_samples:
            # Every record is excluded under this policy; nothing to train.
            continue
        sample_turn_ids = {sample.turn_id for sample in row_samples}
        incompatible_cache_turn_ids = [
            str(record.get("turn_id"))
            for record in trajectory_records
            if isinstance(record, dict)
            and record.get(TOKEN_CACHE_FIELD) is not None
            and record.get("turn_id") in sample_turn_ids
            and not is_training_cache_current(
                record.get(TOKEN_CACHE_FIELD),
                train_compaction_tokens=train_compaction_tokens,
            )
        ]
        if incompatible_cache_turn_ids:
            raise ValueError(
                f"Rollout row {index} has caches with the wrong loss-mask policy: "
                f"{', '.join(incompatible_cache_turn_ids)}"
            )
        missing_cache_turn_ids = [sample.turn_id for sample in row_samples if not sample.has_training_cache]
        if missing_cache_turn_ids:
            raise ValueError(
                f"Rollout row {index} has uncached trainable samples: {', '.join(missing_cache_turn_ids)}"
            )
        samples.extend(row_samples)
    return samples


def run_train_step(
    config,
    *,
    checkpoint_path: str | Path,
    rollout_path: str | Path,
    output_checkpoint_path: str | Path,
    output_checkpoint_id: str | None = None,
    metrics_path: str | Path | None = None,
    progress_path: str | Path | None = None,
    trainer: Any | None = None,
) -> Path:
    checkpoint = Path(checkpoint_path).resolve()
    migrate_compaction_value_head_sidecar(checkpoint)
    checkpoint_id = checkpoint_id_from_path(checkpoint)
    if (
        config.training.value.enabled
        and checkpoint_id.startswith("iteration-")
        and checkpoint_id != "iteration-00000"
        and not all(
            (checkpoint / filename).exists()
            for filename in (VALUE_HEAD_FILENAME, VALUE_HEAD_MANIFEST_FILENAME)
        )
    ):
        raise ValueError(
            f"Value-enabled resume checkpoint {checkpoint_id} is missing its value-head sidecar"
        )
    rows = _load_rollout_rows(rollout_path)
    samples = samples_from_rollout_rows(
        rows,
        expected_checkpoint_id=checkpoint_id,
        train_compaction_tokens=config.training.train_compaction_tokens,
        retain_critic_only_states=config.training.value.enabled,
    )
    grouped_samples = group_samples_by_query(samples)
    diagnostics = _sample_diagnostics(samples)
    print(
        f"[train_step] Loaded {len(rows)} rollout rows, {len(samples)} samples, "
        f"{len(grouped_samples)} groups; total_tokens={diagnostics['total_tokens']}, "
        f"trainable_tokens={diagnostics['trainable_tokens']}, "
        f"sequence_length_p50={diagnostics['p50_sequence_length']}, "
        f"sequence_length_p95={diagnostics['p95_sequence_length']}, "
        f"sequence_length_max={diagnostics['max_sequence_length']}, "
        f"prefix_length_p95={diagnostics['p95_prefix_length']}",
        flush=True,
    )
    progress = str(progress_path) if progress_path is not None else None
    _write_training_progress(
        progress,
        "samples_loaded",
        rollout_rows=len(rows),
        sample_count=len(samples),
        group_count=len(grouped_samples),
        **diagnostics,
    )

    if trainer is None:
        model_config = replace(config.model, model_path=str(checkpoint))
        if config.training.backend == "verl_ray":
            from self_summarization_agent.verl_ray_trainer import VerlRayPolicyTrainer

            # Shutdown any stale Ray instance from a previous failed run
            # so orphaned actors don't hold GPU memory.
            try:
                import ray  # type: ignore[import-untyped]
                if ray.is_initialized():
                    ray.shutdown()
            except ImportError:
                pass

            if config.training.gpu_ids:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(gpu_id) for gpu_id in config.training.gpu_ids
                )

            # Release fragmented GPU memory from prior phases (FAISS
            # embedding model, cache step, etc.) before the memory-
            # intensive FSDP training phase begins.
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            trainer = VerlRayPolicyTrainer(
                model_config,
                config.training,
                checkpoint_id=checkpoint_id,
                progress_path=progress,
            )
        elif config.training.backend == "fsdp2_context_parallel":
            if config.training.value.enabled:
                raise NotImplementedError(
                    "Shared-backbone compaction value training is not implemented for "
                    "fsdp2_context_parallel; use transformers or verl_ray/transformers."
                )
            if os.environ.get("RANK") is None:
                warnings.warn(
                    "training.backend='fsdp2_context_parallel' requires accelerate launch. "
                    "Falling back to 'transformers' backend for single-process execution.",
                    stacklevel=2,
                )
                trainer = TransformersPolicyTrainer(model_config, config.training, progress_path=progress)
            else:
                trainer = FSDP2ContextParallelPolicyTrainer(model_config, config.training)
        elif config.training.backend == "transformers":
            trainer = TransformersPolicyTrainer(model_config, config.training, progress_path=progress)
        else:
            raise NotImplementedError(
                "The local environment cannot execute backend="
                f"{config.training.backend!r}. Supported backends are 'transformers', "
                "'fsdp2_context_parallel', and 'verl_ray'."
            )

    _write_training_progress(progress, "update_start", sample_count=len(samples), **diagnostics)
    metrics = trainer.step(grouped_samples)
    _write_training_progress(progress, "update_complete", sample_count=metrics.sample_count)
    print(
        f"[train_step] Done: sample_count={metrics.sample_count}, mean_reward={metrics.mean_reward:.4f}, "
        f"mean_advantage={metrics.mean_advantage:.4f}, loss={metrics.loss:.4f}, "
        f"optimizer_steps={getattr(metrics, 'optimizer_step_count', 0)}, "
        f"mean_policy_kl={getattr(metrics, 'mean_policy_kl', 0.0):.4f}, "
        f"clip_fraction={getattr(metrics, 'clip_fraction', 0.0):.4f}",
        flush=True,
    )
    output_checkpoint = Path(output_checkpoint_path)
    ensure_dir(output_checkpoint)
    print(f"[train_step] Saving checkpoint to {output_checkpoint}", flush=True)
    _write_training_progress(progress, "checkpoint_save_start", output_checkpoint=str(output_checkpoint))
    trainer.save_checkpoint(str(output_checkpoint))
    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main:
        mark_checkpoint_complete(output_checkpoint)
        _write_training_progress(progress, "checkpoint_complete", output_checkpoint=str(output_checkpoint))
        print(f"[train_step] Checkpoint complete: {output_checkpoint}", flush=True)

    if metrics_path is not None and is_main:
        metrics_payload = {
            "policy_checkpoint_id": checkpoint_id,
            "next_checkpoint_id": output_checkpoint_id or checkpoint_id_from_path(output_checkpoint),
            "training_backend": config.training.backend,
            "sample_count": metrics.sample_count,
            "mean_reward": metrics.mean_reward,
            "mean_advantage": metrics.mean_advantage,
            "loss": metrics.loss,
            "optimizer_step_count": getattr(metrics, "optimizer_step_count", 0),
            "mean_policy_kl": getattr(metrics, "mean_policy_kl", 0.0),
            "clip_fraction": getattr(metrics, "clip_fraction", 0.0),
            "avg_summary_tokens": _average_summary_tokens(rows),
        }
        metrics_payload.update(getattr(metrics, "extra_metrics", {}) or {})
        append_jsonl(metrics_path, metrics_payload)
    return output_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one policy update from checkpoint-tagged rollout artifacts.")
    parser.add_argument("--config", required=True, help="Path to the train YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Input policy checkpoint path.")
    parser.add_argument("--rollouts", required=True, help="Rollout JSONL path.")
    parser.add_argument("--output-checkpoint", required=True, help="Output checkpoint directory.")
    parser.add_argument(
        "--output-checkpoint-id",
        default=None,
        help="Logical checkpoint id when writing through an incomplete staging directory.",
    )
    parser.add_argument("--metrics", default=None, help="Optional metrics JSONL path.")
    parser.add_argument("--progress", default=None, help="Optional atomic training progress JSON path.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    # Register SIGTERM handler so Python dumps all-thread stack traces
    # to stderr when the launcher kills a timed-out child. Combined with
    # the 30s SIGTERM grace in _run_timed_phase, this pinpoints the hang
    # site in the tee'd log instead of leaving a silent timeout.
    import faulthandler
    import signal
    faulthandler.register(signal.SIGTERM, all_threads=True)

    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))

    # Reduce CUDA memory fragmentation: tells PyTorch's caching allocator to
    # release memory in expandable segments, avoiding the "reserved but
    # unallocated" fragmentation that can cause OOM during loss.backward().
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    output_checkpoint = run_train_step(
        config,
        checkpoint_path=args.checkpoint,
        rollout_path=args.rollouts,
        output_checkpoint_path=args.output_checkpoint,
        output_checkpoint_id=args.output_checkpoint_id,
        metrics_path=args.metrics,
        progress_path=args.progress,
    )
    print(output_checkpoint)


if __name__ == "__main__":
    main()
