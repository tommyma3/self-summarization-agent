from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any
import warnings

from self_summarization_agent.checkpoints import checkpoint_id_from_path
from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.launcher_utils import append_jsonl, ensure_dir
from self_summarization_agent.trainer import FSDP2ContextParallelPolicyTrainer, TransformersPolicyTrainer
from self_summarization_agent.trajectory import TOKEN_CACHE_FIELD, extract_trainable_samples


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


def _rollout_key(row: dict[str, Any], *, index: int) -> tuple[str, int]:
    query_id = row.get("query_id")
    rollout_index = row.get("rollout_index")
    if not isinstance(query_id, str) or not isinstance(rollout_index, int):
        raise ValueError(f"Rollout row {index} is missing query_id or rollout_index")
    return query_id, rollout_index


def _validate_judged_row(row: dict[str, Any], *, index: int, expected_checkpoint_id: str) -> None:
    if row.get("policy_checkpoint_id") != expected_checkpoint_id:
        raise ValueError(
            f"Rollout row {index} checkpoint mismatch: expected {expected_checkpoint_id!r}, "
            f"got {row.get('policy_checkpoint_id')!r}"
        )
    if not isinstance(row.get("turn_records"), list):
        raise ValueError(f"Rollout row {index} is missing turn_records")
    if not isinstance(row.get("turn_rewards"), dict):
        raise ValueError(f"Rollout row {index} is missing turn_rewards")
    _rollout_key(row, index=index)


def _validate_cached_row(row: dict[str, Any], *, index: int, expected_checkpoint_id: str) -> None:
    _validate_judged_row(row, index=index, expected_checkpoint_id=expected_checkpoint_id)
    if row.get("trainable_sample_count") == 0:
        return
    samples = extract_trainable_samples(row["turn_records"], row["turn_rewards"])
    missing_cache_turn_ids = [sample.turn_id for sample in samples if not sample.has_training_cache]
    if missing_cache_turn_ids:
        raise ValueError(
            f"Cached rollout row {index} has uncached trainable samples: "
            f"{', '.join(missing_cache_turn_ids)}"
        )


def _completed_cached_keys(path: Path, *, expected_checkpoint_id: str) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    rows = _load_rollout_rows(path)
    keys: set[tuple[str, int]] = set()
    for index, row in enumerate(rows, start=1):
        _validate_cached_row(row, index=index, expected_checkpoint_id=expected_checkpoint_id)
        key = _rollout_key(row, index=index)
        if key in keys:
            raise ValueError(f"Cached rollout row {index} duplicates rollout key {key!r}")
        keys.add(key)
    return keys


def _attach_training_caches(
    row: dict[str, Any],
    *,
    cache_payloads: list[dict[str, Any]],
    checkpoint_id: str,
) -> dict[str, Any]:
    row_samples = extract_trainable_samples(row["turn_records"], row["turn_rewards"])
    if len(cache_payloads) != len(row_samples):
        raise ValueError(
            f"Scorer returned {len(cache_payloads)} cache payloads for {len(row_samples)} trainable samples"
        )
    payload_by_turn_id = {
        sample.turn_id: {**payload, "policy_checkpoint_id": checkpoint_id}
        for sample, payload in zip(row_samples, cache_payloads)
    }
    cached_row = dict(row)
    cached_turns: list[dict[str, Any]] = []
    for turn in row["turn_records"]:
        cached_turn = dict(turn)
        turn_id = cached_turn.get("turn_id")
        if isinstance(turn_id, str) and turn_id in payload_by_turn_id:
            cached_turn[TOKEN_CACHE_FIELD] = payload_by_turn_id[turn_id]
        cached_turns.append(cached_turn)
    cached_row["turn_records"] = cached_turns
    return cached_row


def _cache_training_config(config):
    if config.training.backend != "verl_ray":
        return config.training
    worker_backend = config.training.verl.worker_backend
    if worker_backend == "transformers":
        return replace(config.training, backend=worker_backend)
    if worker_backend == "verl_fsdp":
        return replace(config.training, backend="transformers")
    raise NotImplementedError(
        "training.backend='verl_ray' cache scoring currently supports only "
        "training.verl.worker_backend='transformers' or 'verl_fsdp'. "
        f"Got {worker_backend!r}."
    )


def build_cache_scorer(config, *, checkpoint_path: str | Path):
    checkpoint = Path(checkpoint_path).resolve()
    model_config = replace(config.model, model_path=str(checkpoint))
    training_config = _cache_training_config(config)
    if training_config.backend == "fsdp2_context_parallel":
        if os.environ.get("RANK") is None:
            warnings.warn(
                "training.backend='fsdp2_context_parallel' requires accelerate launch. "
                "Falling back to 'transformers' backend for single-process cache scoring.",
                stacklevel=2,
            )
            return TransformersPolicyTrainer(model_config, replace(training_config, backend="transformers"))
        return FSDP2ContextParallelPolicyTrainer(model_config, training_config)
    if training_config.backend == "transformers":
        return TransformersPolicyTrainer(model_config, training_config)
    raise NotImplementedError(
        "The local environment cannot execute backend="
        f"{config.training.backend!r}. Supported backends are 'transformers', "
        "'fsdp2_context_parallel', and 'verl_ray'."
    )


def _is_main_process(scorer: Any) -> bool:
    accelerator = getattr(scorer, "accelerator", None)
    if accelerator is not None:
        return bool(getattr(accelerator, "is_main_process", True))
    return int(os.environ.get("RANK", "0")) == 0


def _wait_for_everyone(scorer: Any) -> None:
    accelerator = getattr(scorer, "accelerator", None)
    if accelerator is not None:
        wait_for_everyone = getattr(accelerator, "wait_for_everyone", None)
        if wait_for_everyone is not None:
            wait_for_everyone()


def run_cache_step(
    config,
    *,
    checkpoint_path: str | Path,
    rollout_path: str | Path,
    output_path: str | Path,
    scorer: Any | None = None,
    resume: bool = False,
) -> Path:
    checkpoint = Path(checkpoint_path).resolve()
    checkpoint_id = checkpoint_id_from_path(checkpoint)
    rows = _load_rollout_rows(rollout_path)
    for index, row in enumerate(rows, start=1):
        _validate_judged_row(row, index=index, expected_checkpoint_id=checkpoint_id)

    output = Path(output_path)
    ensure_dir(output.parent)
    completed_keys: set[tuple[str, int]] = set()
    if resume:
        completed_keys = _completed_cached_keys(output, expected_checkpoint_id=checkpoint_id)
        expected_keys = {_rollout_key(row, index=index) for index, row in enumerate(rows, start=1)}
        unexpected_keys = sorted(completed_keys - expected_keys)
        if unexpected_keys:
            raise ValueError(f"Cached output contains unexpected rollout keys: {unexpected_keys!r}")
    elif output.exists():
        output.unlink()

    pending_rows = [
        row
        for index, row in enumerate(rows, start=1)
        if _rollout_key(row, index=index) not in completed_keys
    ]
    if not pending_rows:
        return output

    scorer = scorer or build_cache_scorer(config, checkpoint_path=checkpoint)
    is_main = _is_main_process(scorer)
    for row in pending_rows:
        samples = extract_trainable_samples(row["turn_records"], row["turn_rewards"])
        cache_payloads = scorer.cache_samples(samples)
        cached_row = _attach_training_caches(
            row,
            cache_payloads=cache_payloads,
            checkpoint_id=checkpoint_id,
        )
        if is_main:
            append_jsonl(output, cached_row)
    _wait_for_everyone(scorer)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize judged rollouts and cache reference logprobs.")
    parser.add_argument("--config", required=True, help="Path to the train YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Input policy checkpoint path.")
    parser.add_argument("--rollouts", required=True, help="Judged rollout JSONL path.")
    parser.add_argument("--output", required=True, help="Cached rollout JSONL output path.")
    parser.add_argument("--resume", action="store_true", help="Append missing cached rows and skip completed rows.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    output = run_cache_step(
        config,
        checkpoint_path=args.checkpoint,
        rollout_path=args.rollouts,
        output_path=args.output,
        resume=args.resume,
    )
    print(output)


if __name__ == "__main__":
    main()
