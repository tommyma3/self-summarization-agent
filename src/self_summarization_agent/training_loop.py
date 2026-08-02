from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import signal
import subprocess
import time
import traceback
from typing import Any, Sequence

from self_summarization_agent.checkpoints import (
    advance_latest_checkpoint,
    is_vllm_loadable_checkpoint,
    publish_checkpoint,
    resolve_latest_checkpoint,
)
from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.iteration_artifacts import (
    IterationArtifacts,
    completed_iteration_from_checkpoint_id,
    ensure_iteration_manifest,
    eval_sampling_profile_id,
    expected_eval_rollout_count,
    expected_train_rollout_count,
    has_complete_cached_rollouts,
    has_complete_judged_rollouts,
    has_complete_raw_rollouts,
    has_eval_metrics,
    have_matching_rollout_keys,
    semantic_config_id,
)
from self_summarization_agent.launcher_utils import append_jsonl, ensure_dir, utc_timestamp


def _phase_entry(
    module_name: str,
    function_name: str,
    kwargs: dict[str, Any],
    result_queue,
) -> None:
    try:
        if os.name == "posix":
            os.setsid()
        module = importlib.import_module(module_name)
        result = getattr(module, function_name)(**kwargs)
        result_queue.put({"result": result})
    except BaseException as exc:
        result_queue.put(
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_python_phase(
    *,
    phase: str,
    module_name: str,
    function_name: str,
    kwargs: dict[str, Any],
    timeout_seconds: float | None,
) -> Any:
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_phase_entry,
        args=(module_name, function_name, kwargs, result_queue),
        name=f"ssa-{phase}",
    )
    process.start()
    process.join(timeout=timeout_seconds)
    if process.is_alive():
        if os.name == "posix" and process.pid is not None:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.join(timeout=30)
        if process.is_alive():
            if os.name == "posix" and process.pid is not None:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.join(timeout=10)
        raise TimeoutError(f"{phase} timed out after {timeout_seconds:.0f}s")
    try:
        payload = result_queue.get(timeout=1)
    except Empty:
        raise RuntimeError(f"{phase} exited with code {process.exitcode} without returning a result")
    if payload.get("error"):
        detail = f"\n{payload['traceback']}" if payload.get("traceback") else ""
        raise RuntimeError(f"{phase} failed: {payload['error']}{detail}")
    if process.exitcode != 0:
        raise RuntimeError(f"{phase} exited with code {process.exitcode}")
    return payload.get("result")


def _record_phase(
    *,
    phase: str,
    iteration: int,
    timings_path: Path,
    started: float,
    skipped: bool = False,
) -> None:
    append_jsonl(
        timings_path,
        {
            "iteration": iteration,
            "timestamp_utc": utc_timestamp(),
            "phase": phase,
            "elapsed_seconds": 0.0 if skipped else time.perf_counter() - started,
            "exit_code": 0,
            **({"skipped": True} if skipped else {}),
        },
    )


def _run_fsdp_train_update(
    config,
    *,
    config_path: str | Path,
    checkpoint_path: Path,
    rollout_path: Path,
    output_checkpoint: Path,
    metrics_path: Path,
    overrides: Sequence[str],
) -> None:
    partial = output_checkpoint.with_name(f".{output_checkpoint.name}.incomplete")
    if partial.exists():
        partial.rename(partial.with_name(f"{partial.name}.stale-{int(time.time())}"))
    command = [
        "accelerate",
        "launch",
        "--num_processes",
        str(len(config.training.gpu_ids) or config.training.context_parallel_size),
        "--num_machines",
        "1",
        "--machine_rank",
        "0",
        "--main_process_port",
        "0",
        "--use-fsdp",
        "--fsdp_version",
        str(config.training.fsdp_version or 2),
        "--fsdp_auto_wrap_policy",
        "transformer_based_wrap",
        "--parallelism-config-cp-size",
        str(config.training.context_parallel_size),
    ]
    if config.training.activation_checkpointing:
        command.append("--fsdp_activation_checkpointing=true")
    command.extend(
        [
            "-m",
            "self_summarization_agent.train_step",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--rollouts",
            str(rollout_path),
            "--output-checkpoint",
            str(partial),
            "--metrics",
            str(metrics_path),
            "--published-checkpoint",
            str(output_checkpoint),
        ]
    )
    for override in overrides:
        command.extend(["--set", override])
    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=config.runtime.phase_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"train_update timed out after {config.runtime.phase_timeout_seconds:.0f}s"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(f"train_update failed with exit code {completed.returncode}")
    publish_checkpoint(partial, output_checkpoint)


def run_training_iteration(
    config,
    *,
    config_path: str | Path,
    iteration: int,
    latest_root: str | Path | None = None,
    resume: bool = True,
    overrides: Sequence[str] = (),
) -> Path:
    train_dir = ensure_dir(
        latest_root
        or Path(config.experiment.output_root) / "artifacts" / "train" / config.experiment.name
    )
    current = resolve_latest_checkpoint(train_dir)
    completed_iteration = completed_iteration_from_checkpoint_id(current.checkpoint_id)
    if completed_iteration == iteration:
        return current.path
    if iteration != completed_iteration + 1:
        raise ValueError(
            f"Cannot run iteration {iteration} from latest checkpoint {current.checkpoint_id!r}; "
            f"expected iteration {completed_iteration + 1}"
        )

    artifacts = IterationArtifacts.build(
        train_dir,
        iteration=iteration,
        checkpoint_id=current.checkpoint_id,
    )
    ensure_dir(artifacts.raw_rollouts.parent)
    ensure_dir(artifacts.next_checkpoint.parent)
    ensure_iteration_manifest(
        artifacts.manifest,
        iteration=iteration,
        checkpoint_id=current.checkpoint_id,
        config_id=semantic_config_id(config),
    )
    eval_profile_id = eval_sampling_profile_id(config)
    from self_summarization_agent.config import resolved_rollout_sampling_profile, sampling_profile_id

    train_profile_id = sampling_profile_id(
        resolved_rollout_sampling_profile(config, split="train")
    )
    expected_train = expected_train_rollout_count(config)
    expected_eval = expected_eval_rollout_count(config)
    collect_complete = (
        has_complete_raw_rollouts(
            artifacts.raw_rollouts,
            checkpoint_id=current.checkpoint_id,
            expected_count=expected_train,
            expected_sampling_profile_id=train_profile_id,
        )
        and has_complete_judged_rollouts(
            artifacts.judged_rollouts,
            checkpoint_id=current.checkpoint_id,
            expected_count=expected_train,
            require_judge=False,
            expected_sampling_profile_id=train_profile_id,
        )
        and has_complete_cached_rollouts(
            artifacts.cached_rollouts,
            checkpoint_id=current.checkpoint_id,
            expected_count=expected_train,
            expected_sampling_profile_id=train_profile_id,
        )
        and have_matching_rollout_keys(
            artifacts.raw_rollouts,
            artifacts.judged_rollouts,
            artifacts.cached_rollouts,
        )
        and (
            config.dataset.eval_limit <= 0
            or (
                has_complete_raw_rollouts(
                    artifacts.eval_raw_rollouts,
                    checkpoint_id=current.checkpoint_id,
                    expected_count=expected_eval,
                    expected_sampling_profile_id=eval_profile_id,
                )
                and has_complete_judged_rollouts(
                    artifacts.eval_judged_rollouts,
                    checkpoint_id=current.checkpoint_id,
                    expected_count=expected_eval,
                    require_judge=True,
                    expected_sampling_profile_id=eval_profile_id,
                )
                and have_matching_rollout_keys(
                    artifacts.eval_raw_rollouts,
                    artifacts.eval_judged_rollouts,
                )
                and has_eval_metrics(
                    artifacts.eval_metrics,
                    iteration=iteration - 1,
                    policy_checkpoint_id=current.checkpoint_id,
                    expected_sampling_profile_id=eval_profile_id,
                )
            )
        )
    ) if resume else False

    started = time.perf_counter()
    if collect_complete:
        print(f"[training_loop] iteration {iteration}: merged_collect already complete", flush=True)
        _record_phase(
            phase="merged_collect",
            iteration=iteration,
            timings_path=artifacts.phase_timings,
            started=started,
            skipped=True,
        )
    else:
        print(f"[training_loop] iteration {iteration}: starting merged_collect", flush=True)
        run_python_phase(
            phase="merged_collect",
            module_name="self_summarization_agent.merged_collect_step",
            function_name="run_merged_collect",
            kwargs={
                "config": config,
                "config_path": config_path,
                "checkpoint_path": current.path,
                "train_raw_output": artifacts.raw_rollouts,
                "train_judged_output": artifacts.judged_rollouts,
                "train_cached_output": artifacts.cached_rollouts,
                "eval_raw_output": artifacts.eval_raw_rollouts if config.dataset.eval_limit > 0 else None,
                "eval_judged_output": artifacts.eval_judged_rollouts if config.dataset.eval_limit > 0 else None,
                "eval_metrics_output": artifacts.eval_metrics if config.dataset.eval_limit > 0 else None,
                "eval_iteration": iteration - 1,
                "sample_seed": config.experiment.seed + iteration,
                "resume": resume,
                "overrides": list(overrides),
            },
            timeout_seconds=config.runtime.phase_timeout_seconds,
        )
        _record_phase(
            phase="merged_collect",
            iteration=iteration,
            timings_path=artifacts.phase_timings,
            started=started,
        )

    if not (
        has_complete_raw_rollouts(
            artifacts.raw_rollouts,
            checkpoint_id=current.checkpoint_id,
            expected_count=expected_train,
            expected_sampling_profile_id=train_profile_id,
        )
        and has_complete_judged_rollouts(
            artifacts.judged_rollouts,
            checkpoint_id=current.checkpoint_id,
            expected_count=expected_train,
            require_judge=False,
            expected_sampling_profile_id=train_profile_id,
        )
        and has_complete_cached_rollouts(
            artifacts.cached_rollouts,
            checkpoint_id=current.checkpoint_id,
            expected_count=expected_train,
            expected_sampling_profile_id=train_profile_id,
        )
        and have_matching_rollout_keys(
            artifacts.raw_rollouts,
            artifacts.judged_rollouts,
            artifacts.cached_rollouts,
        )
    ):
        raise RuntimeError("Merged collection produced incomplete or inconsistent train artifacts")
    if config.dataset.eval_limit > 0 and not (
        has_complete_raw_rollouts(
            artifacts.eval_raw_rollouts,
            checkpoint_id=current.checkpoint_id,
            expected_count=expected_eval,
            expected_sampling_profile_id=eval_profile_id,
        )
        and has_complete_judged_rollouts(
            artifacts.eval_judged_rollouts,
            checkpoint_id=current.checkpoint_id,
            expected_count=expected_eval,
            require_judge=True,
            expected_sampling_profile_id=eval_profile_id,
        )
        and have_matching_rollout_keys(
            artifacts.eval_raw_rollouts,
            artifacts.eval_judged_rollouts,
        )
        and has_eval_metrics(
            artifacts.eval_metrics,
            iteration=iteration - 1,
            policy_checkpoint_id=current.checkpoint_id,
            expected_sampling_profile_id=eval_profile_id,
        )
    ):
        raise RuntimeError("Merged collection produced incomplete eval artifacts")

    checkpoint_complete = is_vllm_loadable_checkpoint(artifacts.next_checkpoint)
    started = time.perf_counter()
    if resume and checkpoint_complete:
        print(f"[training_loop] iteration {iteration}: train_update already complete", flush=True)
        _record_phase(
            phase="train_update",
            iteration=iteration,
            timings_path=artifacts.phase_timings,
            started=started,
            skipped=True,
        )
    else:
        print(f"[training_loop] iteration {iteration}: starting train_update", flush=True)
        if config.training.backend == "fsdp2_context_parallel":
            _run_fsdp_train_update(
                config,
                config_path=config_path,
                checkpoint_path=current.path,
                rollout_path=artifacts.cached_rollouts,
                output_checkpoint=artifacts.next_checkpoint,
                metrics_path=artifacts.step_metrics,
                overrides=overrides,
            )
        else:
            run_python_phase(
                phase="train_update",
                module_name="self_summarization_agent.training_phase",
                function_name="run_train_update",
                kwargs={
                    "config": config,
                    "checkpoint_path": current.path,
                    "rollout_path": artifacts.cached_rollouts,
                    "output_checkpoint_path": artifacts.next_checkpoint,
                    "metrics_path": artifacts.step_metrics,
                },
                timeout_seconds=config.runtime.phase_timeout_seconds,
            )
        _record_phase(
            phase="train_update",
            iteration=iteration,
            timings_path=artifacts.phase_timings,
            started=started,
        )
    return advance_latest_checkpoint(train_dir, artifacts.next_checkpoint).path


def evaluate_latest_checkpoint(
    config,
    *,
    config_path: str | Path,
    latest_root: str | Path | None = None,
    resume: bool = True,
    overrides: Sequence[str] = (),
) -> Path:
    if config.dataset.eval_limit <= 0:
        raise ValueError("dataset.eval_limit must be positive for final checkpoint evaluation")
    train_dir = ensure_dir(
        latest_root
        or Path(config.experiment.output_root) / "artifacts" / "train" / config.experiment.name
    )
    current = resolve_latest_checkpoint(train_dir)
    iteration = completed_iteration_from_checkpoint_id(current.checkpoint_id)
    artifacts = IterationArtifacts.build(
        train_dir,
        iteration=iteration + 1,
        checkpoint_id=current.checkpoint_id,
    )
    profile_id = eval_sampling_profile_id(config)
    if resume:
        expected_eval = expected_eval_rollout_count(config)
        if (
            has_complete_raw_rollouts(
                artifacts.eval_raw_rollouts,
                checkpoint_id=current.checkpoint_id,
                expected_count=expected_eval,
                expected_sampling_profile_id=profile_id,
            )
            and has_complete_judged_rollouts(
                artifacts.eval_judged_rollouts,
                checkpoint_id=current.checkpoint_id,
                expected_count=expected_eval,
                require_judge=True,
                expected_sampling_profile_id=profile_id,
            )
            and have_matching_rollout_keys(
                artifacts.eval_raw_rollouts,
                artifacts.eval_judged_rollouts,
            )
            and has_eval_metrics(
                artifacts.eval_metrics,
                iteration=iteration,
                policy_checkpoint_id=current.checkpoint_id,
                expected_sampling_profile_id=profile_id,
            )
        ):
            return current.path
    run_python_phase(
        phase="final_eval",
        module_name="self_summarization_agent.merged_collect_step",
        function_name="run_merged_collect",
        kwargs={
            "config": config,
            "config_path": config_path,
            "checkpoint_path": current.path,
            "eval_raw_output": artifacts.eval_raw_rollouts,
            "eval_judged_output": artifacts.eval_judged_rollouts,
            "eval_metrics_output": artifacts.eval_metrics,
            "eval_iteration": iteration,
            "resume": resume,
            "overrides": list(overrides),
        },
        timeout_seconds=config.runtime.phase_timeout_seconds,
    )
    expected_eval = expected_eval_rollout_count(config)
    if not (
        has_complete_raw_rollouts(
            artifacts.eval_raw_rollouts,
            checkpoint_id=current.checkpoint_id,
            expected_count=expected_eval,
            expected_sampling_profile_id=profile_id,
        )
        and has_complete_judged_rollouts(
            artifacts.eval_judged_rollouts,
            checkpoint_id=current.checkpoint_id,
            expected_count=expected_eval,
            require_judge=True,
            expected_sampling_profile_id=profile_id,
        )
        and have_matching_rollout_keys(
            artifacts.eval_raw_rollouts,
            artifacts.eval_judged_rollouts,
        )
        and has_eval_metrics(
            artifacts.eval_metrics,
            iteration=iteration,
            policy_checkpoint_id=current.checkpoint_id,
            expected_sampling_profile_id=profile_id,
        )
    ):
        raise RuntimeError("Final checkpoint evaluation did not produce complete artifacts")
    return current.path


def run_training(
    config,
    *,
    config_path: str | Path,
    target_iterations: int,
    latest_root: str | Path | None = None,
    resume: bool = True,
    evaluate_final: bool = False,
    overrides: Sequence[str] = (),
) -> Path:
    if target_iterations < 0:
        raise ValueError(f"target_iterations must be non-negative, got {target_iterations}")
    train_dir = ensure_dir(
        latest_root
        or Path(config.experiment.output_root) / "artifacts" / "train" / config.experiment.name
    )
    current = resolve_latest_checkpoint(train_dir)
    completed = completed_iteration_from_checkpoint_id(current.checkpoint_id)
    if completed > target_iterations:
        raise ValueError(
            f"Latest checkpoint is iteration {completed}, beyond target {target_iterations}"
        )
    for iteration in range(completed + 1, target_iterations + 1):
        current_path = run_training_iteration(
            config,
            config_path=config_path,
            iteration=iteration,
            latest_root=train_dir,
            resume=resume,
            overrides=overrides,
        )
        current = resolve_latest_checkpoint(train_dir)
        if current.path != current_path:
            raise RuntimeError("latest checkpoint pointer did not match the completed update")
    if evaluate_final:
        return evaluate_latest_checkpoint(
            config,
            config_path=config_path,
            latest_root=train_dir,
            resume=resume,
            overrides=overrides,
        )
    return resolve_latest_checkpoint(train_dir).path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training to a target checkpoint iteration.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--iterations", type=int, required=True, help="Target checkpoint iteration.")
    parser.add_argument("--latest-root", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate-final", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    checkpoint = run_training(
        config,
        config_path=args.config,
        target_iterations=args.iterations,
        latest_root=args.latest_root,
        resume=args.resume,
        evaluate_final=args.evaluate_final,
        overrides=args.overrides,
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
