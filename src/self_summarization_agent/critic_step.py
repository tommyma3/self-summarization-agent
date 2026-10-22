"""Process-isolated critic fitting; exits before the policy trainer allocates GPUs."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import re
import time

from self_summarization_agent.config import load_train_config
from self_summarization_agent.separate_value import (
    ADVANTAGES, CRITIC_DIR, atomic_json, digest, sample_fingerprint, sample_key,
    validate_resume_checkpoint,
)

EVALSET_PREFIX = "value/evalset"
TIMINGS_FILE = "critic_phase_timings.json"


def scheduled_iteration(interval, checkpoint_id):
    """Use the output iteration, never the number of critic optimizer steps."""
    if interval == 1:
        return True
    match = re.fullmatch(r"iteration-(\d+)", checkpoint_id.removeprefix(".").removesuffix(".incomplete"))
    return bool(interval and match and int(match[1]) > 0 and int(match[1]) % interval == 0)


def evaluate_on_eval_collection(critic, rollouts_path, checkpoint_name):
    """Score the freshly fitted critic on its own policy's evaluation states.

    A value network corresponds to the behavior checkpoint it was fit against;
    ``iteration-<name>.eval.jsonl`` holds checkpoint-evaluation rollouts generated
    by that same policy. Diagnostic only: rows are deep-copied and native training
    caches are attached in memory, never persisted; records on disk are untouched.
    """
    from self_summarization_agent.train_step import _load_rollout_rows, samples_from_rollout_rows
    from self_summarization_agent.trajectory import build_rollout_native_training_cache
    from self_summarization_agent.value_diagnostics import value_diagnostics

    started = time.monotonic()
    eval_path = Path(rollouts_path).parent / f"{checkpoint_name}.eval.jsonl"
    tags = {f"{EVALSET_PREFIX}/policy_id": checkpoint_name,
            f"{EVALSET_PREFIX}/data_policy_id": checkpoint_name,
            f"{EVALSET_PREFIX}/rollout_path": str(eval_path.resolve())}
    if not eval_path.is_file():
        return {f"{EVALSET_PREFIX}_skipped": "missing_eval_rollouts", **tags}
    rows = [row for row in _load_rollout_rows(eval_path)
            if row.get("policy_checkpoint_id") == checkpoint_name]
    if not rows:
        return {f"{EVALSET_PREFIX}_skipped": "no_rows_for_policy", **tags}
    native_cache_fail_count = 0
    cached_rows = []
    for row in rows:
        row = deepcopy(row)
        kept_records = []
        rewards = dict(row.get("turn_rewards") or {})
        for record in row.get("trajectory_records") or []:
            if isinstance(record, dict) and record.get("training_cache") is None:
                cache = build_rollout_native_training_cache(record.get("collection_tokens"))
                if cache is None:
                    native_cache_fail_count += 1
                    rewards.pop(record.get("turn_id"), None)
                    continue
                record["training_cache"] = cache
            kept_records.append(record)
        row["trajectory_records"] = kept_records
        row["turn_rewards"] = rewards
        if kept_records:
            cached_rows.append(row)
    if not cached_rows:
        return {f"{EVALSET_PREFIX}_skipped": "no_eval_states",
                f"{EVALSET_PREFIX}/native_cache_fail_count": native_cache_fail_count, **tags}
    eval_samples = samples_from_rollout_rows(
        cached_rows, expected_checkpoint_id=checkpoint_name, retain_critic_only_states=True)
    if not eval_samples:
        return {f"{EVALSET_PREFIX}_skipped": "no_eval_states",
                f"{EVALSET_PREFIX}/native_cache_fail_count": native_cache_fail_count, **tags}
    prepare_seconds = time.monotonic() - started
    started = time.monotonic()
    values, losses = critic.evaluate(eval_samples)
    inference_seconds = time.monotonic() - started
    started = time.monotonic()
    metrics = value_diagnostics(eval_samples, values, losses, prefix=EVALSET_PREFIX)
    metrics.update({f"{EVALSET_PREFIX}/eval_scope": "same_policy_eval_collection",
                    f"{EVALSET_PREFIX}/prepare_seconds": prepare_seconds,
                    f"{EVALSET_PREFIX}/inference_seconds": inference_seconds,
                    f"{EVALSET_PREFIX}/diagnostics_seconds": time.monotonic() - started,
                    f"{EVALSET_PREFIX}/native_cache_fail_count": native_cache_fail_count, **tags})
    return metrics


def run(config, checkpoint, rollouts, output, output_id, progress=None):
    total_started = time.monotonic()
    # Imported only after the parent has configured this subprocess's GPU visibility.
    from self_summarization_agent.train_step import _load_rollout_rows, samples_from_rollout_rows
    from self_summarization_agent.separate_value_trainer import SeparateValueTrainer
    from self_summarization_agent.trainer import _write_training_progress
    checkpoint, output = Path(checkpoint), Path(output)
    started = time.monotonic()
    samples = samples_from_rollout_rows(_load_rollout_rows(rollouts), expected_checkpoint_id=checkpoint.name,
                                       retain_critic_only_states=True)
    resume = validate_resume_checkpoint(checkpoint)
    source = checkpoint / CRITIC_DIR if resume else config.training.value.model_path
    phases = {"value/sample_load_seconds": time.monotonic() - started}
    _write_training_progress(progress, "critic_load", source=str(source))
    started = time.monotonic()
    critic = SeparateValueTrainer(source, checkpoint, config.training.value, dtype=config.model.dtype,
        resume=resume, activation_checkpointing=config.training.activation_checkpointing, progress_path=progress)
    synchronize = getattr(critic, "synchronize", lambda: None)
    synchronize()
    phases["value/model_load_seconds"] = time.monotonic() - started
    critic.detailed_diagnostics = scheduled_iteration(config.training.value.diagnostics_interval, output_id)
    values, metrics = critic.fit(samples, max_sequence_length=config.training.max_sequence_length)
    _write_training_progress(progress, "critic_save")
    started = time.monotonic()
    identity = critic.save(output / CRITIC_DIR, output_id)
    synchronize()
    phases["value/checkpoint_save_seconds"] = time.monotonic() - started
    started = time.monotonic()
    if scheduled_iteration(config.training.value.evalset_interval, output_id):
        _write_training_progress(progress, "critic_evalset", policy_id=checkpoint.name)
        metrics.update(evaluate_on_eval_collection(critic, rollouts, checkpoint.name))
    else:
        metrics[f"{EVALSET_PREFIX}_skipped"] = (
            "disabled" if config.training.value.evalset_interval == 0 else "not_scheduled")
    phases["value/evalset_seconds"] = time.monotonic() - started
    started = time.monotonic()
    advantages = [s.reward - v for s, v in zip(samples, values)]
    if advantages:
        mean_advantage = sum(advantages) / len(advantages)
        metrics.update({"value/advantage_mean": mean_advantage,
            "value/advantage_std": (sum((a - mean_advantage) ** 2 for a in advantages) / len(advantages)) ** 0.5,
            "value/advantage_min": min(advantages), "value/advantage_max": max(advantages)})
    metrics["value/advantage_near_zero_fraction"] = sum(abs(a) < 1e-6 for a in advantages) / max(1, len(advantages))
    body = {"version": 1, "behavior_checkpoint_id": checkpoint.name,
        "critic_identity": identity, "sample_fingerprint": sample_fingerprint(samples),
        "config_fingerprint": digest(asdict(config.training.value)),
        "rows": [{"key": sample_key(s), "value": v, "advantage": a}
                 for s, v, a in zip(samples, values, advantages)], "metrics": metrics}
    phases["value/advantage_prepare_seconds"] = time.monotonic() - started
    metrics.update(phases)
    started = time.monotonic()
    atomic_json(output / ADVANTAGES, body)
    phases["value/advantage_export_seconds"] = time.monotonic() - started
    phases["value/critic_total_seconds"] = time.monotonic() - total_started
    # A small sidecar measures the actual artifact write without rewriting it to time itself.
    atomic_json(output / TIMINGS_FILE, phases)
    _write_training_progress(progress, "advantages_ready", state_count=len(samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "rollouts", "output", "output-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--progress")
    args = parser.parse_args()
    run(load_train_config(args.config), args.checkpoint, args.rollouts, args.output, args.output_id, args.progress)


if __name__ == "__main__":
    main()
