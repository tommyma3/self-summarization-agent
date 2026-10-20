"""Process-isolated critic fitting; exits before the policy trainer allocates GPUs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from self_summarization_agent.config import load_train_config
from self_summarization_agent.separate_value import (
    ADVANTAGES, CRITIC_DIR, atomic_json, digest, sample_fingerprint, sample_key,
    validate_resume_checkpoint,
)


def run(config, checkpoint, rollouts, output, output_id, progress=None):
    # Imported only after the parent has configured this subprocess's GPU visibility.
    from self_summarization_agent.train_step import _load_rollout_rows, samples_from_rollout_rows
    from self_summarization_agent.separate_value_trainer import SeparateValueTrainer
    from self_summarization_agent.trainer import _write_training_progress
    checkpoint, output = Path(checkpoint), Path(output)
    samples = samples_from_rollout_rows(_load_rollout_rows(rollouts), expected_checkpoint_id=checkpoint.name,
                                       retain_critic_only_states=True)
    resume = validate_resume_checkpoint(checkpoint)
    source = checkpoint / CRITIC_DIR if resume else config.training.value.model_path
    _write_training_progress(progress, "critic_load", source=str(source))
    critic = SeparateValueTrainer(source, checkpoint, config.training.value, dtype=config.model.dtype,
        resume=resume, activation_checkpointing=config.training.activation_checkpointing, progress_path=progress)
    values, metrics = critic.fit(samples, max_sequence_length=config.training.max_sequence_length)
    _write_training_progress(progress, "critic_save")
    identity = critic.save(output / CRITIC_DIR, output_id)
    advantages = [s.reward - v for s, v in zip(samples, values)]
    if advantages:
        mean_advantage = sum(advantages) / len(advantages)
        metrics.update({"value/advantage_mean": mean_advantage,
            "value/advantage_std": (sum((a - mean_advantage) ** 2 for a in advantages) / len(advantages)) ** 0.5,
            "value/advantage_min": min(advantages), "value/advantage_max": max(advantages)})
    metrics["value/advantage_near_zero_fraction"] = sum(abs(a) < 1e-6 for a in advantages) / max(1, len(advantages))
    atomic_json(output / ADVANTAGES, {"version": 1, "behavior_checkpoint_id": checkpoint.name,
        "critic_identity": identity, "sample_fingerprint": sample_fingerprint(samples),
        "config_fingerprint": digest(asdict(config.training.value)),
        "rows": [{"key": sample_key(s), "value": v, "advantage": a}
                 for s, v, a in zip(samples, values, advantages)], "metrics": metrics})
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
