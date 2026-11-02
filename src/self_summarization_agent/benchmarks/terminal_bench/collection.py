"""Harbor trials -> the same exact-token rollout rows used by RL training."""
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import random
import re
import time
from uuid import uuid4

from self_summarization_agent.checkpoints import checkpoint_id_from_path
from self_summarization_agent.collection_contract import collection_profile_id
from self_summarization_agent.config import resolved_rollout_sampling_profile, sampling_profile_id
from self_summarization_agent.launcher_utils import append_jsonl
from self_summarization_agent.token_stream import TITO_CONTRACT
from self_summarization_agent.trajectory import _extract_collection_tokens
from . import HARBOR_VERSION, KIRA_REVISION, SCAFFOLD_VERSION
from ..harbor_common.dataset import task_manifest, selected_tasks
from .verification import apply_verifier_reward


def collect_terminal_rollouts(config, *, checkpoint_path, output_path, generator=None,
                             resume=False, judged_output_path=None, split="eval", sample_seed=None):
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
    from harbor.trial.trial import Trial
    from self_summarization_agent.rollout_collection import _build_rollout_generator, _temporary_sampling_profile
    from .harbor_agent import LockedPolicy, POLICIES, TerminusKiraTito, make_vision

    if not config.rollout.require_exact_token_ids or not config.model.chat_template_path:
        raise ValueError("Terminal-Bench requires exact token IDs and the project Qwen template")
    tasks = task_manifest(config.benchmark, split=split)
    rng = random.Random(config.experiment.seed if sample_seed is None else sample_seed)
    if config.dataset.shuffle and config.benchmark.name != "swe-dev":
        rng.shuffle(tasks)
    tasks = tasks[config.dataset.offset:]
    if config.dataset.limit is not None:
        tasks = tasks[:config.dataset.limit]
    count = getattr(config.collection, f"{split}_task_count")
    if count is not None and config.benchmark.name != "swe-dev":
        if count < 1 or count > len(tasks):
            raise ValueError("Requested task count is outside the selected benchmark")
        tasks = rng.sample(tasks, count)
    if config.benchmark.name == "swe-dev":
        tasks = selected_tasks(config, split=split, sample_seed=sample_seed)
    if not tasks and not (config.benchmark.name == "swe-dev" and split == "eval"):
        raise ValueError("No terminal tasks selected")
    attempts = config.evaluation.samples_per_task if split == "eval" else config.training.group_size
    if attempts < 1:
        raise ValueError("Attempts per task must be positive")
    checkpoint = Path(checkpoint_path).resolve()
    if config.benchmark.name == "swe-dev":
        from ..swe_dev.checkpoint import validate_checkpoint
        validate_checkpoint(checkpoint, config)
    profile = collection_profile_id(config, checkpoint)
    checkpoint_id = checkpoint_id_from_path(checkpoint)
    sampling = resolved_rollout_sampling_profile(config, split=split)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    rows = []
    expected = {(task["name"], i) for task in tasks for i in range(attempts)}
    if output.exists():
        if not resume:
            raise ValueError(f"Output already exists: {output}; use resume or a fresh output path")
        from self_summarization_agent.collection_contract import validate_artifact_lineage
        validate_artifact_lineage([output], config=config, checkpoint=checkpoint)
        for line in output.read_text().splitlines():
            row = json.loads(line)
            if (row.get("collection_profile_id") != profile
                or row.get("sampling_profile_id") != sampling_profile_id(sampling)
                or row.get("policy_checkpoint_id") != checkpoint_id
                or row.get("rollout_split") != split):
                raise ValueError("Cannot resume: terminal collection identity changed")
            for record in row["trajectory_records"]:
                _extract_collection_tokens(record, turn_id=record["turn_id"])
            key = row["query_id"], row["rollout_index"]
            if key not in expected or key in completed:
                raise ValueError("Cannot resume: duplicate or unexpected task/attempt")
            completed.add(key)
            rows.append(row)
    requests = [(t, i) for t in tasks for i in range(attempts) if (t["name"], i) not in completed]
    owns_generator = generator is None
    if requests:
        generator = generator or _build_rollout_generator(config, checkpoint, split=split)
        if not callable(getattr(generator, "generate_token_batch", None)) or not getattr(generator, "require_exact_token_ids", False):
            raise ValueError("Terminal-Bench requires a token-input generator with exact IDs")
        policy_key = uuid4().hex
        POLICIES[policy_key] = (LockedPolicy(generator), make_vision(config.benchmark, checkpoint))

        async def run_all():
            semaphore = asyncio.Semaphore(config.rollout.max_concurrent_episodes)
            async def run_one(task, attempt):
                async with semaphore:
                    # A new trial name always means a fresh environment; interrupted
                    # trials are preserved as artifacts, never replayed in-place.
                    name = f"{task['name'][:40]}-{attempt}-{uuid4().hex[:8]}"
                    if config.benchmark.name == "swe-dev":
                        name = re.sub(r"[^a-zA-Z0-9_-]", "-", name)
                    trial_config = TrialConfig(
                        task=TaskConfig(path=Path(task["path"]), git_url=task.get("git_url"),
                            git_commit_id=task.get("git_commit_id"), source=config.benchmark.dataset),
                        trial_name=name, trials_dir=output.parent / (output.stem + "-trials"),
                        agent=AgentConfig(import_path=TerminusKiraTito.import_path(),
                            model_name=str(checkpoint), override_timeout_sec=config.benchmark.agent_timeout_seconds,
                            kwargs=dict(policy_key=policy_key, query_id=task["name"],
                                        runtime=asdict(config.runtime), benchmark=asdict(config.benchmark))),
                        environment=EnvironmentConfig(type="docker" if config.benchmark.environment == "local-fixture" else config.benchmark.environment))
                    # Task download/build preparation is synchronous in Harbor.
                    if config.benchmark.name == "swe-dev":
                        if config.benchmark.environment == "local-fixture":
                            from ..swe_dev.local_fixture import FixtureTrial as TrialClass
                            if not getattr(generator, "fixture_only", False):
                                raise ValueError("Local fixtures require a deterministic fixture-only generator")
                        else:
                            from ..swe_dev.harbor_trial import SWETrial as TrialClass
                        trial = await asyncio.to_thread(TrialClass, trial_config, swe_task=task)
                    else:
                        trial = await asyncio.to_thread(Trial, trial_config)
                    trial_started = time.monotonic()
                    result = await trial.run()
                    if result.exception_info and result.exception_info.exception_type == "CollectionGenerationError":
                        raise RuntimeError("Terminal policy generation failed: " + result.exception_info.exception_message)
                    artifact = trial.trial_dir / "agent" / "rollout.json"
                    payload = json.loads(artifact.read_text()) if artifact.exists() else dict(
                        query_id=task["name"], query="", status="infrastructure_error",
                        final_answer=None, summary_turns=[], turn_records=[], trajectory_records=[],
                        retrieved_docids=[], tool_call_counts={}, token_usage={})
                    raw = dict(**payload, benchmark=config.benchmark.name, scaffold=SCAFFOLD_VERSION,
                        kira_revision=KIRA_REVISION, harbor_version=HARBOR_VERSION,
                        image_profile=config.benchmark.image_profile, task_source=task,
                        task_checksum=result.task_checksum, trial_path=str(trial.trial_dir),
                        collection_profile_id=profile, collection_contract=TITO_CONTRACT,
                        policy_checkpoint_id=checkpoint_id, policy_checkpoint_path=str(checkpoint),
                        rollout_split=split, rollout_index=attempt, rollout_samples_per_task=attempts,
                        sampling_profile=sampling, sampling_profile_id=sampling_profile_id(sampling),
                        benchmark_verifier=dict(
                            rewards=result.verifier_result.rewards if result.verifier_result else None,
                            exception=result.exception_info.model_dump(mode="json") if result.exception_info else None))
                    if config.benchmark.name == "swe-dev":
                        raw["reward_profile"] = config.benchmark.reward_profile
                        raw["swe_selection"] = dict(version=1, task_ids=sorted(t["name"] for t in tasks),
                            attempts=attempts, sample_seed=sample_seed if split == "train" else None)
                        raw["swe_metadata"] = dict(version=1, fixture_only=task["fixture_only"],
                            task_digest=task["checksum"], manifest_digest=task["manifest_digest"],
                            source=task["source"], image=task["image"], verifier_digest=task["verifier"]["digest"],
                            trial_seconds=time.monotonic() - trial_started,
                            verifier_limits=task["verifier"]["limits"],
                            environment_resource_measurements=None,
                            candidate=getattr(trial, "swe_evidence", {}))
                    row = apply_verifier_reward(raw, reward_profile=config.benchmark.reward_profile)
                    # One event-loop writer commits each completed trial exactly once.
                    append_jsonl(output, row)
                    rows.append(row)
            await asyncio.gather(*(run_one(t, i) for t, i in requests))
        try:
            with _temporary_sampling_profile(generator, sampling):
                asyncio.run(run_all())
        finally:
            POLICIES.pop(policy_key, None)
            if owns_generator:
                core = getattr(getattr(getattr(generator, "llm", None), "llm_engine", None), "engine_core", None)
                if core is not None:
                    core.shutdown()
    if judged_output_path:
        judged = Path(judged_output_path)
        judged.parent.mkdir(parents=True, exist_ok=True)
        temporary = judged.with_suffix(judged.suffix + ".tmp")
        temporary.write_text("".join(json.dumps(apply_verifier_reward(row)) + "\n" for row in rows))
        temporary.replace(judged)
    write_metrics(rows, output.with_suffix(".metrics.json"))
    return output


def write_metrics(rows, path):
    valid = [r for r in rows if r.get("judge", {}).get("outcome") != "infrastructure_error"]
    stats = dict(attempts=len(rows), tasks=len({r["query_id"] for r in rows}),
        valid_attempts=len(valid), infrastructure_errors=len(rows)-len(valid),
        pass_rate=sum(bool(r.get("benchmark_passed")) for r in valid) / len(valid) if valid else None,
        summaries=sum(len(r.get("summary_turns", [])) for r in rows),
        statuses={s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}})
    if any(row.get("benchmark") == "swe-dev" for row in rows):
        stats.update(raw_verifier_passes=sum(bool(r.get("benchmark_passed")) for r in rows),
            all_attempt_pass_rate=sum(bool(r.get("benchmark_passed")) for r in rows) / len(rows) if rows else None,
            training_eligible_attempts=sum(bool(r.get("training_eligible")) for r in rows),
            history_rewrites=sum(r.get("status") == "history_rewrite_detected" for r in rows),
            generated_tokens=sum(r.get("token_usage", {}).get("total_generated_tokens", 0) for r in rows),
            trial_seconds=sum(r.get("swe_metadata", {}).get("trial_seconds", 0) for r in rows),
            resource_measurements="unavailable" if all(r.get("swe_metadata", {}).get("environment_resource_measurements") is None for r in rows) else "available")
    Path(path).write_text(json.dumps(stats, indent=2) + "\n")


def run_terminal_merged(config, *, config_path, checkpoint_path, train_raw_output,
                        train_judged_output, train_cached_output, eval_raw_output,
                        eval_judged_output, eval_metrics_output, eval_iteration,
                        sample_seed, resume, overrides):
    from self_summarization_agent.merged_collect_step import _run_split_collection_process, _run_cache_inline
    from self_summarization_agent.judge_step import judge_rollouts
    from self_summarization_agent.eval_metrics import write_eval_metrics
    from self_summarization_agent.collection_contract import validate_artifact_lineage
    checkpoint = Path(checkpoint_path).resolve()
    overrides = overrides or []
    if resume:
        validate_artifact_lineage([train_raw_output, train_judged_output, train_cached_output,
                                  eval_raw_output, eval_judged_output], config=config, checkpoint=checkpoint)
    outputs = {}
    has_train = bool(config.benchmark.train_manifest_path or config.benchmark.train_task_paths or config.benchmark.allow_benchmark_training)
    if train_cached_output and not has_train:
        raise ValueError("Terminal training requires explicit train_task_paths or allow_benchmark_training")
    for split, raw, judged in (("eval", eval_raw_output, eval_judged_output),
                               ("train", train_raw_output if has_train else None, train_judged_output)):
        if raw is None:
            continue
        _run_split_collection_process(config_path=config_path, overrides=overrides,
            checkpoint_path=checkpoint, split=split, raw_output_path=Path(raw),
            sample_seed=sample_seed if split == "train" else None, resume=resume,
            retrieval_worker_url=None,
            per_split_timeout_seconds=config.rollout.per_split_collection_timeout_seconds)
        outputs[f"{split}_raw"] = Path(raw)
        if judged is not None:
            judge_rollouts(config, rollout_path=raw, output_path=judged,
                           checkpoint_path=checkpoint, split=split)
            outputs[f"{split}_judged"] = Path(judged)
    if eval_metrics_output and eval_judged_output:
        write_eval_metrics(judged_rollout_path=eval_judged_output, metrics_path=eval_metrics_output,
            iteration=eval_iteration or 0, policy_checkpoint_id=checkpoint_id_from_path(checkpoint))
        outputs["eval_metrics"] = Path(eval_metrics_output)
    if train_cached_output:
        if not train_judged_output:
            raise ValueError("Training cache requires train_judged_output")
        _run_cache_inline(config=config, config_path=str(config_path), overrides=overrides,
            checkpoint_path=checkpoint, judged_rollout_path=Path(train_judged_output),
            cached_output_path=Path(train_cached_output), resume=resume)
        outputs["train_cached"] = Path(train_cached_output)
    return outputs
