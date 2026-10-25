"""Compare critic execution settings on immutable saved rollouts, without saving weights."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import random
import time

import torch

from self_summarization_agent.config import load_train_config
from self_summarization_agent.separate_value import CRITIC_DIR, validate_resume_checkpoint
from self_summarization_agent.separate_value_trainer import SeparateValueTrainer
from self_summarization_agent.train_step import samples_from_rollout_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "rollouts", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollout-limit", type=int, default=8)
    parser.add_argument("--fitting-epochs", type=int, help="Benchmark-only override for both variants")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.rollout_limit < 1 or args.repeats < 1:
        parser.error("rollout-limit and repeats must be positive")
    if args.output.exists():
        parser.error("output already exists; choose an isolated benchmark output")
    config = load_train_config(args.config)
    if not config.training.value.separate:
        parser.error("a separate-critic configuration is required")
    started = time.monotonic()
    # Sample whole rollouts across the file: its first rows can all be short,
    # early-finishing episodes and miss post-compaction states entirely.
    rows = []
    rng = random.Random(config.training.value.seed)
    with args.rollouts.open() as stream:
        for index, line in enumerate(stream):
            slot = index if index < args.rollout_limit else rng.randrange(index + 1)
            if slot < args.rollout_limit:
                row = json.loads(line)
                if slot == len(rows):
                    rows.append(row)
                else:
                    rows[slot] = row
    samples = samples_from_rollout_rows(rows, expected_checkpoint_id=args.checkpoint.name,
                                       retain_critic_only_states=True)
    if not samples:
        parser.error("selected rollout rows contain no eligible states")
    sample_load_seconds = time.monotonic() - started
    resume = validate_resume_checkpoint(args.checkpoint)
    source = args.checkpoint / CRITIC_DIR if resume else config.training.value.model_path
    value = config.training.value
    if args.fitting_epochs is not None:
        value = replace(value, fitting_epochs=args.fitting_epochs)
    baseline = replace(value, prefix_microbatch_size=1, eval_prefix_microbatch_size=1,
        max_prefix_tokens_per_batch=None, length_bucketing=False, activation_checkpointing=True,
        eval_before_fit=True, eval_every_n_epochs=1)
    report = {"checkpoint": str(args.checkpoint.resolve()), "rollouts": str(args.rollouts.resolve()),
        "rollout_count": len({s.rollout_id for s in samples}), "state_count": len(samples),
        "rollout_ids": sorted({s.rollout_id for s in samples}),
        "post_compaction_states": sum(s.state_kind == "post_compaction" for s in samples),
        "prefix_length_max": max(s.state_prefix_length for s in samples),
        "sample_load_seconds": sample_load_seconds,
        "scope": "model_load_and_fit_only; excludes checkpoint_save and eval_collection", "runs": []}
    for repeat in range(args.repeats):
        reference = None
        for name, settings in (("baseline", baseline), ("optimized", value)):
            torch.manual_seed(value.seed)
            started = time.monotonic()
            trainer = SeparateValueTrainer(source, args.checkpoint, settings, dtype=config.model.dtype,
                device=args.device, resume=resume, activation_checkpointing=config.training.activation_checkpointing,
                advantage_estimator=config.training.advantage_estimator)
            trainer.synchronize()
            loaded = time.monotonic()
            trainer.detailed_diagnostics = name == "baseline"
            fit_kwargs = {}
            if config.training.advantage_estimator == "compaction_gae_value":
                from self_summarization_agent.segment_returns import ordered_rollout_chains, segment_gae
                from self_summarization_agent.separate_value import sample_key
                values, _ = trainer.evaluate(samples)
                returns = segment_gae(ordered_rollout_chains(rows, samples),
                    dict(zip(map(sample_key, samples), values)), settings.gae_lambda)
                fit_kwargs["scalar_targets"] = {key: row["target"] for key, row in returns.items()}
            predictions, metrics = trainer.fit(samples, max_sequence_length=config.training.max_sequence_length,
                                                **fit_kwargs)
            trainer.synchronize()
            run = {"variant": name, "repeat": repeat, "load_seconds": loaded - started,
                   "total_seconds": time.monotonic() - started, "metrics": metrics}
            if reference is None:
                reference = predictions
            else:
                run["max_value_difference"] = max(abs(a - b) for a, b in zip(reference, predictions))
                # Check padding/pooling with identical fitted weights separately
                # from update-rounding differences. Excluded from timed fitting.
                trainer.config = replace(settings, eval_prefix_microbatch_size=1, length_bucketing=False)
                unbatched, _ = trainer.evaluate(samples)
                run["max_frozen_batching_value_difference"] = max(
                    abs(a - b) for a, b in zip(unbatched, predictions))
            report["runs"].append(run)
            print(json.dumps({k: v for k, v in run.items() if k != "metrics"} | {
                "fit_seconds": metrics["value/fit_seconds"],
                "peak_memory_bytes": metrics["value/peak_memory_bytes"],
            }, allow_nan=False), flush=True)
            del trainer
            gc.collect()
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
