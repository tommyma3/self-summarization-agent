"""Evaluate a critic checkpoint on fixed, query-disjoint cached rollout records.

This entrypoint never calls trainer.step(), generates trajectories, or saves weights.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

from self_summarization_agent.config import load_train_config
from self_summarization_agent.train_step import _load_rollout_rows, samples_from_rollout_rows
from self_summarization_agent.value_diagnostics import value_diagnostics
from self_summarization_agent.value_model import VALUE_HEAD_FILENAME


def load_query_ids(path: str | Path) -> set[str]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list) or any(not isinstance(x, str) or not x for x in payload):
        raise ValueError("Query manifest must be a JSON array of nonempty query ID strings")
    return set(payload)


def evaluate_value_checkpoint(
    config, *, checkpoint_path, rollout_path, generating_checkpoint_id,
    training_query_ids: set[str], policy_eval_query_ids: set[str], trainer_factory=None,
):
    if not config.training.value.enabled:
        raise ValueError("Critic evaluation requires training.value.enabled")
    rows = _load_rollout_rows(rollout_path)
    queries = {row.get("query_id") for row in rows}
    if not queries or any(not isinstance(q, str) or not q for q in queries):
        raise ValueError("Critic evaluation requires nonempty query IDs")
    if queries & (training_query_ids | policy_eval_query_ids):
        raise ValueError("Critic diagnostic queries overlap training or final policy evaluation")
    for row in rows:
        if any(record.get("query_id") != row["query_id"] for record in row.get("trajectory_records", [])):
            raise ValueError("Rollout and trajectory query identities disagree")
    samples = samples_from_rollout_rows(
        rows, expected_checkpoint_id=generating_checkpoint_id,
        train_compaction_tokens=config.training.train_compaction_tokens, retain_critic_only_states=True,
    )
    if not samples or any(not sample.has_exact_collection_tokens for sample in samples):
        raise ValueError("Critic diagnostics require exact collected tokens and current training caches")
    checkpoint = Path(checkpoint_path)
    if config.training.value.separate:
        from self_summarization_agent.separate_value import CRITIC_DIR, validate_resume_checkpoint
        from self_summarization_agent.separate_value_trainer import SeparateValueTrainer
        if not validate_resume_checkpoint(checkpoint):
            raise ValueError("Separate critic evaluation requires a complete saved pair")
        trainer = SeparateValueTrainer(checkpoint / CRITIC_DIR, checkpoint, config.training.value,
            dtype=config.model.dtype, device=f"cuda:{config.training.value.gpu_id}",
            resume=False, activation_checkpointing=False)
        values, losses = trainer.evaluate(samples)
        return {"evaluated_checkpoint": str(checkpoint.resolve()),
            "generating_checkpoint_id": generating_checkpoint_id, "rollout_path": str(Path(rollout_path).resolve()),
            "query_ids": sorted(queries), "value/eval_scope": "fixed_held_out_records",
            "value/target_policy_may_differ": True,
            **value_diagnostics(samples, values, losses, prefix="value/held_out")}
    if not ((checkpoint / VALUE_HEAD_FILENAME).is_file() or
            (checkpoint / "compaction_value_head.safetensors").is_file()):
        raise ValueError("Critic evaluation requires saved value-head weights")
    model_config = replace(config.model, model_path=str(checkpoint))
    if trainer_factory is not None:
        trainer = trainer_factory(model_config, config.training)
    else:
        if config.training.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.training.gpu_ids))
        if config.training.backend == "transformers":
            from self_summarization_agent.trainer import TransformersPolicyTrainer
            trainer = TransformersPolicyTrainer(model_config, config.training)
        elif config.training.backend == "verl_ray":
            from self_summarization_agent.verl_ray_trainer import VerlRayPolicyTrainer
            trainer = VerlRayPolicyTrainer(model_config, config.training, checkpoint_id=checkpoint.name)
        else:
            raise ValueError("Critic evaluation supports transformers and verl_ray")
    try:
        values, losses = trainer.evaluate_compaction_values(
            samples, microbatch_size=config.training.gradient_accumulation_microbatch_size,
        )
        return {
            "evaluated_checkpoint": str(checkpoint.resolve()),
            "generating_checkpoint_id": generating_checkpoint_id,
            "rollout_path": str(Path(rollout_path).resolve()),
            "query_ids": sorted(queries),
            "value/eval_scope": "fixed_held_out_records",
            "value/target_policy_may_differ": True,
            **value_diagnostics(samples, values, losses, prefix="value/held_out"),
        }
    finally:
        if getattr(trainer, "_owns_ray", False) and config.training.verl.shutdown_ray:
            trainer._ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rollouts", required=True, help="Fixed exact-token, judged, cached rollout JSONL")
    parser.add_argument("--generating-checkpoint-id", required=True)
    parser.add_argument("--training-query-ids", required=True, help="All training query IDs, not just this batch")
    parser.add_argument("--policy-eval-query-ids", required=True, help="Final policy holdout query IDs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metrics = evaluate_value_checkpoint(
        load_train_config(args.config), checkpoint_path=args.checkpoint, rollout_path=args.rollouts,
        generating_checkpoint_id=args.generating_checkpoint_id,
        training_query_ids=load_query_ids(args.training_query_ids),
        policy_eval_query_ids=load_query_ids(args.policy_eval_query_ids),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
