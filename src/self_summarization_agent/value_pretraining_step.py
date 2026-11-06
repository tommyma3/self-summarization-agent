"""Collect one fixed-policy batch and pretrain only its separate MC critic."""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from self_summarization_agent.config import config_to_dict, load_train_config, parse_cli_overrides


def run_value_pretraining(config, *, checkpoint_path, output_path, resume=False,
                         command_runner=None, fitter=None, allowed_queries=None):
    from self_summarization_agent.value_pretraining import validate_output_layout

    if (not config.training.value_pretraining.enabled or not config.training.value.separate
            or config.benchmark.name != "browsecomp" or not config.judge.enabled):
        raise ValueError("Value pretraining requires a separate BrowseComp critic and enabled judging")
    output = Path(output_path).resolve()
    validate_output_layout(config, checkpoint_path, output, config.training.value_pretraining.reuse_rollouts_path)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Value pretraining is already running in this output directory") from None
        return _run(config, Path(checkpoint_path).resolve(), output, resume,
                    command_runner, fitter, allowed_queries)


def _run(config, actor, output, resume, command_runner, fitter, allowed_queries):
    from self_summarization_agent.separate_value import atomic_json, validate_resume_checkpoint
    from self_summarization_agent.value_pretraining import completed_pretraining, fit_pretraining, initialization_binding

    settings = config.training.value_pretraining
    if (not settings.enabled or not config.training.value.separate
            or config.benchmark.name != "browsecomp" or not config.judge.enabled):
        raise ValueError("Value pretraining requires a separate BrowseComp critic and enabled judging")
    if validate_resume_checkpoint(actor):
        raise ValueError("Pretraining initializes a fresh run, not an existing actor/critic pair")
    if allowed_queries is None:
        from self_summarization_agent.judge_step import _load_examples_by_query_id
        allowed_queries = {key: example.query for key, example in
                           _load_examples_by_query_id(config, split="train").items()}
    if settings.task_count > len(allowed_queries):
        raise ValueError("Not enough training questions for value pretraining")
    complete = completed_pretraining(config, actor, output)
    if complete is not None:
        return complete
    request = initialization_binding(config, actor)
    request_path = output / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError("Pretraining collection inputs changed; use a fresh output")
        if not resume:
            raise ValueError("Interrupted pretraining exists; use --resume")
    else:
        if not (output / "dataset_manifest.json").exists() and any(p.name != ".lock" for p in output.iterdir()):
            raise ValueError("Refusing to reuse an unmanaged pretraining output directory")
        atomic_json(request_path, request)
    if settings.reuse_rollouts_path is not None:
        rollouts = Path(settings.reuse_rollouts_path).resolve()
        if not rollouts.is_file():
            raise ValueError("The reusable pretraining batch does not exist")
    else:
        collection_config = replace(config, collection=replace(config.collection, train_task_count=settings.task_count),
                                    training=replace(config.training, group_size=settings.samples_per_task))
        collection_path = output / "collection_config.yaml"
        content = yaml.safe_dump(config_to_dict(collection_config), sort_keys=False)
        if collection_path.exists() and config_to_dict(load_train_config(collection_path)) != config_to_dict(collection_config):
            raise ValueError("Pretraining collection configuration changed")
        collection_path.write_text(content)
        rollouts = output / "rollouts.cached.jsonl"
        command = [sys.executable, "-m", "self_summarization_agent.merged_collect_step",
                   "--config", str(collection_path), "--checkpoint", str(actor),
                   "--train-raw-output", str(output / "rollouts.raw.jsonl"),
                   "--train-judged-output", str(output / "rollouts.judged.jsonl"),
                   "--train-cached-output", str(rollouts), "--sample-seed", str(settings.seed)]
        if resume:
            command.append("--resume")
        started, success = time.monotonic(), False
        try:
            result = (command_runner or (lambda cmd: subprocess.run(cmd, check=True)))(command)
            status = result if isinstance(result, int) else getattr(result, "returncode", 0)
            if status:
                raise subprocess.CalledProcessError(status, command)
            success = True
        finally:
            from self_summarization_agent.launcher_utils import append_jsonl
            append_jsonl(output / "phase_timings.jsonl", {
                "phase": "collection", "elapsed_seconds": time.monotonic() - started,
                "exit_code": 0 if success else 1})
    # Only now may this process allocate the critic. Collection owns teardown of
    # policy/retrieval/judge workers; no evaluation output paths were requested.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.training.value.gpu_ids or [config.training.value.gpu_id]))
    return (fitter or fit_pretraining)(config, actor, rollouts, output, allowed_queries, resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    best = run_value_pretraining(config, checkpoint_path=args.checkpoint, output_path=args.output, resume=args.resume)
    print(f"[value_pretraining] selected critic: {best}", flush=True)


if __name__ == "__main__":
    main()
