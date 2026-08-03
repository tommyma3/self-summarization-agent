from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from self_summarization_agent.config import (
    config_to_dict,
    resolved_rollout_sampling_profile,
    sampling_profile_id,
)
from self_summarization_agent.trajectory import (
    TOKEN_CACHE_FIELD,
    extract_trainable_samples,
    is_training_cache_current,
)


_ITERATION_CHECKPOINT_RE = re.compile(r"^iteration-(\d{5,})$")


@dataclass(frozen=True, slots=True)
class IterationArtifacts:
    iteration: int
    checkpoint_id: str
    train_dir: Path
    raw_rollouts: Path
    judged_rollouts: Path
    cached_rollouts: Path
    eval_raw_rollouts: Path
    eval_judged_rollouts: Path
    next_checkpoint: Path
    step_metrics: Path
    eval_metrics: Path
    phase_timings: Path
    manifest: Path

    @classmethod
    def build(cls, train_dir: str | Path, *, iteration: int, checkpoint_id: str) -> "IterationArtifacts":
        if iteration < 1:
            raise ValueError(f"iteration must be at least 1, got {iteration}")
        root = Path(train_dir)
        rollouts_dir = root / "rollouts"
        checkpoints_dir = root / "checkpoints"
        eval_iteration = iteration - 1
        return cls(
            iteration=iteration,
            checkpoint_id=checkpoint_id,
            train_dir=root,
            raw_rollouts=rollouts_dir / f"iteration-{iteration:05d}.raw.jsonl",
            judged_rollouts=rollouts_dir / f"iteration-{iteration:05d}.judged.jsonl",
            cached_rollouts=rollouts_dir / f"iteration-{iteration:05d}.jsonl",
            eval_raw_rollouts=rollouts_dir / f"iteration-{eval_iteration:05d}.eval.raw.jsonl",
            eval_judged_rollouts=rollouts_dir / f"iteration-{eval_iteration:05d}.eval.jsonl",
            next_checkpoint=checkpoints_dir / f"iteration-{iteration:05d}",
            step_metrics=root / "step_metrics.jsonl",
            eval_metrics=root / "eval_metrics.jsonl",
            phase_timings=root / "phase_timings.jsonl",
            manifest=rollouts_dir / f"iteration-{iteration:05d}.manifest.json",
        )


def completed_iteration_from_checkpoint_id(checkpoint_id: str) -> int:
    match = _ITERATION_CHECKPOINT_RE.fullmatch(checkpoint_id)
    if match is None:
        return 0
    return int(match.group(1))


def semantic_config_id(config) -> str:
    payload = config_to_dict(config)
    runtime = payload.get("runtime", {})
    runtime.pop("phase_timeout_seconds", None)
    collection = payload.get("collection", {})
    for key in ("cache_gpu_ids", "worker_queue_size", "worker_stall_timeout_seconds", "judge_batch_size"):
        collection.pop(key, None)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_iteration_manifest(
    path: str | Path,
    *,
    iteration: int,
    checkpoint_id: str,
    config_id: str,
) -> Path:
    manifest_path = Path(path)
    expected = {
        "version": 1,
        "iteration": iteration,
        "policy_checkpoint_id": checkpoint_id,
        "semantic_config_id": config_id,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError(
                f"Cannot resume iteration {iteration}: manifest {manifest_path} does not match "
                "the selected checkpoint and semantic configuration"
            )
        return manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest_path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL row in {input_path} on line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL row in {input_path} on line {line_number} must be an object"
                )
            rows.append(row)
    return rows


def rollout_key(row: dict[str, Any], *, path: Path | None = None, index: int | None = None) -> tuple[str, int]:
    query_id = row.get("query_id")
    rollout_index = row.get("rollout_index")
    if not isinstance(query_id, str) or not isinstance(rollout_index, int):
        location = f" in {path}" if path is not None else ""
        line = f" on row {index}" if index is not None else ""
        raise ValueError(f"Rollout{location}{line} is missing query_id or rollout_index")
    return query_id, rollout_index


def rows_by_key(path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return {}
    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    for index, row in enumerate(load_jsonl(input_path), start=1):
        key = rollout_key(row, path=input_path, index=index)
        if key in keyed:
            raise ValueError(f"Duplicate rollout key {key!r} in {input_path}")
        keyed[key] = row
    return keyed


def validated_rows_by_key(
    path: str | Path,
    *,
    checkpoint_id: str,
    expected_sampling_profile_id: str | None = None,
) -> dict[tuple[str, int], dict[str, Any]]:
    input_path = Path(path)
    keyed = rows_by_key(input_path)
    for index, row in enumerate(keyed.values(), start=1):
        if row.get("policy_checkpoint_id") != checkpoint_id:
            raise ValueError(
                f"Cannot resume from {input_path}: row {index} has checkpoint "
                f"{row.get('policy_checkpoint_id')!r}, expected {checkpoint_id!r}"
            )
        if (
            expected_sampling_profile_id is not None
            and row.get("sampling_profile_id") != expected_sampling_profile_id
        ):
            raise ValueError(
                f"Cannot resume from {input_path}: row {index} has sampling profile "
                f"{row.get('sampling_profile_id')!r}, expected {expected_sampling_profile_id!r}"
            )
    return keyed


def expected_train_rollout_count(config) -> int | None:
    if config.collection.train_task_count is not None:
        return config.collection.train_task_count * config.training.group_size
    if config.training.rollout_query_count is not None:
        return config.training.rollout_query_count * config.training.group_size
    if config.dataset.train_limit is not None:
        query_count = config.dataset.train_limit
        if config.dataset.limit is not None:
            query_count = min(query_count, config.dataset.limit)
        return query_count * config.training.group_size
    if config.dataset.limit is not None:
        return config.dataset.limit * config.training.group_size
    return None


def expected_eval_rollout_count(config) -> int:
    if config.collection.eval_task_count is not None:
        return config.collection.eval_task_count * config.evaluation.samples_per_task
    if config.dataset.train_limit is None:
        return 0
    if config.dataset.limit is None:
        available_after_train = config.dataset.eval_limit
    else:
        available_after_train = max(0, config.dataset.limit - config.dataset.train_limit)
    return min(config.dataset.eval_limit, available_after_train) * config.evaluation.samples_per_task


def has_complete_raw_rollouts(
    path: Path,
    *,
    checkpoint_id: str,
    expected_count: int | None,
    expected_sampling_profile_id: str | None = None,
) -> bool:
    if not path.exists():
        return False
    rows = load_jsonl(path)
    for index, row in enumerate(rows, start=1):
        if row.get("policy_checkpoint_id") != checkpoint_id:
            raise ValueError(
                f"Cannot resume from {path}: row {index} has checkpoint "
                f"{row.get('policy_checkpoint_id')!r}, expected {checkpoint_id!r}"
            )
        if (
            expected_sampling_profile_id is not None
            and row.get("sampling_profile_id") != expected_sampling_profile_id
        ):
            raise ValueError(
                f"Cannot resume from {path}: row {index} has sampling profile "
                f"{row.get('sampling_profile_id')!r}, expected {expected_sampling_profile_id!r}"
            )
        if "turn_rewards" in row:
            raise ValueError(f"Cannot resume from {path}: row {index} is already judged")
        rollout_key(row, path=path, index=index)
        if not isinstance(row.get("turn_records"), list):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing turn_records")
        if not isinstance(row.get("trajectory_records"), list):
            raise ValueError(
                f"Cannot resume from {path}: row {index} is missing trajectory_records; recollect it"
            )
    if expected_count is not None and len(rows) != expected_count:
        return False
    if expected_count is None and not rows:
        return False
    return True


def has_complete_judged_rollouts(
    path: Path,
    *,
    checkpoint_id: str,
    expected_count: int | None,
    require_judge: bool,
    expected_sampling_profile_id: str | None = None,
) -> bool:
    if not path.exists():
        return False
    rows = load_jsonl(path)
    for index, row in enumerate(rows, start=1):
        if row.get("policy_checkpoint_id") != checkpoint_id:
            raise ValueError(
                f"Cannot resume from {path}: row {index} has checkpoint "
                f"{row.get('policy_checkpoint_id')!r}, expected {checkpoint_id!r}"
            )
        if (
            expected_sampling_profile_id is not None
            and row.get("sampling_profile_id") != expected_sampling_profile_id
        ):
            raise ValueError(
                f"Cannot resume from {path}: row {index} has sampling profile "
                f"{row.get('sampling_profile_id')!r}, expected {expected_sampling_profile_id!r}"
            )
        rollout_key(row, path=path, index=index)
        if not isinstance(row.get("turn_records"), list):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing turn_records")
        if not isinstance(row.get("trajectory_records"), list):
            raise ValueError(
                f"Cannot resume from {path}: row {index} is missing trajectory_records; recollect it"
            )
        if not isinstance(row.get("turn_rewards"), dict):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing turn_rewards")
        if require_judge and not isinstance(row.get("judge"), dict):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing judge payload")
    if expected_count is not None and len(rows) != expected_count:
        return False
    if expected_count is None and not rows:
        return False
    return True


def _row_has_current_training_cache(row: dict[str, Any]) -> bool:
    if row.get("trainable_sample_count") == 0:
        return True
    turn_rewards = row.get("turn_rewards")
    trajectory_records = row.get("trajectory_records")
    if not isinstance(turn_rewards, dict) or not isinstance(trajectory_records, list):
        return False
    samples = extract_trainable_samples(
        trajectory_records,
        turn_rewards,
        rollout_id=f"{row.get('query_id')}:{row.get('rollout_index')}",
    )
    return all(sample.has_training_cache for sample in samples)


def has_complete_cached_rollouts(
    path: Path,
    *,
    checkpoint_id: str,
    expected_count: int | None,
    expected_sampling_profile_id: str | None = None,
) -> bool:
    if not path.exists():
        return False
    rows = load_jsonl(path)
    for index, row in enumerate(rows, start=1):
        if row.get("policy_checkpoint_id") != checkpoint_id:
            raise ValueError(
                f"Cannot resume from {path}: row {index} has checkpoint "
                f"{row.get('policy_checkpoint_id')!r}, expected {checkpoint_id!r}"
            )
        if (
            expected_sampling_profile_id is not None
            and row.get("sampling_profile_id") != expected_sampling_profile_id
        ):
            raise ValueError(
                f"Cannot resume from {path}: row {index} has sampling profile "
                f"{row.get('sampling_profile_id')!r}, expected {expected_sampling_profile_id!r}"
            )
        rollout_key(row, path=path, index=index)
        turn_records = row.get("turn_records")
        trajectory_records = row.get("trajectory_records")
        turn_rewards = row.get("turn_rewards")
        if not isinstance(turn_records, list):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing turn_records")
        if not isinstance(trajectory_records, list):
            raise ValueError(
                f"Cannot resume from {path}: row {index} is missing trajectory_records; recollect it"
            )
        if not isinstance(turn_rewards, dict):
            raise ValueError(f"Cannot resume from {path}: row {index} is missing turn_rewards")
        if not _row_has_current_training_cache(row):
            return False
        rewarded_ids = set(turn_rewards)
        current_cache_ids = {
            record.get("turn_id")
            for record in trajectory_records
            if isinstance(record, dict)
            and isinstance(record.get("turn_id"), str)
            and is_training_cache_current(record.get(TOKEN_CACHE_FIELD))
        }
        if not rewarded_ids <= current_cache_ids and row.get("trainable_sample_count") != 0:
            return False
    if expected_count is not None and len(rows) != expected_count:
        return False
    if expected_count is None and not rows:
        return False
    return True


def have_matching_rollout_keys(*paths: str | Path) -> bool:
    keyed_sets: list[set[tuple[str, int]]] = []
    for path in paths:
        input_path = Path(path)
        if not input_path.exists():
            return False
        keyed_sets.append(set(rows_by_key(input_path)))
    return bool(keyed_sets) and all(keys == keyed_sets[0] for keys in keyed_sets[1:])


def current_cached_rows_by_key(
    path: str | Path,
    *,
    checkpoint_id: str,
    expected_sampling_profile_id: str | None = None,
) -> dict[tuple[str, int], dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return {}
    current: dict[tuple[str, int], dict[str, Any]] = {}
    seen: set[tuple[str, int]] = set()
    for index, row in enumerate(load_jsonl(input_path), start=1):
        if row.get("policy_checkpoint_id") != checkpoint_id:
            raise ValueError(
                f"Cannot resume from {input_path}: row {index} has checkpoint "
                f"{row.get('policy_checkpoint_id')!r}, expected {checkpoint_id!r}"
            )
        if (
            expected_sampling_profile_id is not None
            and row.get("sampling_profile_id") != expected_sampling_profile_id
        ):
            raise ValueError(
                f"Cannot resume from {input_path}: row {index} has sampling profile "
                f"{row.get('sampling_profile_id')!r}, expected {expected_sampling_profile_id!r}"
            )
        key = rollout_key(row, path=input_path, index=index)
        if key in seen:
            raise ValueError(f"Duplicate rollout key {key!r} in {input_path}")
        seen.add(key)
        if _row_has_current_training_cache(row):
            current[key] = row
    return current


def replace_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    os.replace(temporary, output)


def has_eval_metrics(
    metrics_path: Path,
    *,
    iteration: int,
    policy_checkpoint_id: str,
    expected_sampling_profile_id: str,
) -> bool:
    if not metrics_path.exists():
        return False
    return any(
        row.get("iteration") == iteration
        and row.get("policy_checkpoint_id") == policy_checkpoint_id
        and row.get("eval_sampling_profile_id") == expected_sampling_profile_id
        for row in load_jsonl(metrics_path)
    )


def eval_sampling_profile_id(config) -> str:
    return sampling_profile_id(resolved_rollout_sampling_profile(config, split="eval"))
