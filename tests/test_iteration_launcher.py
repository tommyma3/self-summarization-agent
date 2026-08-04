import json
from pathlib import Path

from self_summarization_agent.checkpoints import (
    mark_checkpoint_complete,
    resolve_latest_checkpoint,
    write_latest_checkpoint,
)
from self_summarization_agent.config import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    JudgeConfig,
    ModelConfig,
    RetrievalConfig,
    RolloutConfig,
    RuntimeConfig,
    TrainConfig,
    TrainingConfig,
    sampling_profile_id,
)
from self_summarization_agent.iteration_launcher import (
    _expected_eval_rollout_count,
    _has_complete_cached_rollouts,
    run_checkpoint_evaluation,
    run_training_iteration,
)


TEST_EVAL_PROFILE = {
    "max_new_tokens": 512,
    "temperature": 1.0,
    "top_p": 0.95,
    "do_sample": True,
    "api_extra_body": {},
    "extra_sampling_params": {"top_k": 20},
}
TEST_EVAL_PROFILE_ID = sampling_profile_id(TEST_EVAL_PROFILE)


def write_fake_checkpoint(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_text("weights", encoding="utf-8")
    mark_checkpoint_complete(path)


def write_raw_rollouts(path: Path, checkpoint_id: str, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "policy_checkpoint_id": checkpoint_id,
            "rollout_index": index,
            "query_id": f"q{index}",
            "sampling_profile": TEST_EVAL_PROFILE,
            "sampling_profile_id": TEST_EVAL_PROFILE_ID,
            "turn_records": [],
            "trajectory_records": [],
        }
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_judged_rollouts(path: Path, checkpoint_id: str, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "policy_checkpoint_id": checkpoint_id,
            "rollout_index": index,
            "query_id": f"q{index}",
            "sampling_profile": TEST_EVAL_PROFILE,
            "sampling_profile_id": TEST_EVAL_PROFILE_ID,
            "turn_records": [],
            "trajectory_records": [],
            "turn_rewards": {},
            "judge": {"outcome": "wrong_answer", "parse_error": False},
        }
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def training_cache(*, version: int = 4) -> dict:
    cache = {
        "version": 4,
        "input_ids": [1],
        "labels": [2],
        "completion_mask": [True],
        "reference_logprob": -0.1,
        "reference_logprobs": [-0.1],
    }
    if version != 4:
        cache["version"] = version
        del cache["reference_logprobs"]
    return cache


def write_cached_rollouts(path: Path, checkpoint_id: str, count: int, *, cache_version: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "policy_checkpoint_id": checkpoint_id,
            "rollout_index": index,
            "query_id": f"q{index}",
            "turn_records": [
                {
                    "query_id": f"q{index}",
                    "turn_id": "final-answer",
                    "kind": "final_answer",
                    "prompt": "prompt",
                    "completion": "completion",
                }
            ],
            "trajectory_records": [
                {
                    "query_id": f"q{index}",
                    "turn_id": "trajectory-1",
                    "kind": "trajectory",
                    "termination_kind": "final_answer",
                    "messages": [
                        {"role": "system", "content": "instructions"},
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "answer"},
                    ],
                    "training_cache": training_cache(version=cache_version),
                }
            ],
            "turn_rewards": {"trajectory-1": 1.0},
            "trainable_sample_count": 1,
            "judge": {"outcome": "wrong_answer", "parse_error": False},
        }
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_complete_cached_rollouts_requires_v4_training_cache(tmp_path: Path) -> None:
    cached_path = tmp_path / "cached.jsonl"
    write_cached_rollouts(cached_path, "iteration-00000", count=1, cache_version=2)

    assert not _has_complete_cached_rollouts(
        cached_path,
        checkpoint_id="iteration-00000",
        expected_count=1,
    )


def test_expected_eval_rollout_count_includes_samples_per_task(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    config.collection.eval_task_count = 3
    config.evaluation.samples_per_task = 2

    assert _expected_eval_rollout_count(config) == 6


def write_eval_metric(path: Path, iteration: int, checkpoint_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "iteration": iteration,
                "policy_checkpoint_id": checkpoint_id,
                "eval_sampling_profile_id": TEST_EVAL_PROFILE_ID,
                "eval_accuracy": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def train_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        experiment=ExperimentConfig(name="demo", seed=1, output_root=str(tmp_path), bc_plus_root=str(tmp_path)),
        dataset=DatasetConfig(limit=1),
        retrieval=RetrievalConfig(backend="faiss", index_path="unused"),
        model=ModelConfig(backend="transformers", model_path="unused"),
        rollout=RolloutConfig(backend="vllm_offline"),
        runtime=RuntimeConfig(context_threshold_tokens=1000, max_context_tokens=1024, tool_budget=4),
        judge=JudgeConfig(enabled=True),
        training=TrainingConfig(group_size=2),
        evaluation=EvaluationConfig(
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            extra_sampling_params={"top_k": 20},
        ),
    )


# ---------------------------------------------------------------------------
# Merged-collect tests (overlap judging ON by default)
# ---------------------------------------------------------------------------

def test_iteration_launcher_runs_merged_collect_then_train_and_advances_latest(tmp_path: Path) -> None:
    """With overlap judging on, the pipeline is: merged_collect → train_update (2 phases)."""
    config = train_config(tmp_path)
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    calls = []

    def runner(command):
        calls.append(list(command))
        if "self_summarization_agent.train_step" in command:
            next_checkpoint = latest_root / "checkpoints" / "iteration-00001"
            write_fake_checkpoint(next_checkpoint)
        return 0

    next_checkpoint = run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
    )

    assert next_checkpoint == (latest_root / "checkpoints" / "iteration-00001").resolve()
    assert "self_summarization_agent.merged_collect_step" in calls[0]
    assert "self_summarization_agent.train_step" in calls[1]
    assert str(latest_root / "rollouts" / "iteration-00001.raw.jsonl") in calls[0]
    assert "--train-raw-output" in calls[0]
    assert "--train-judged-output" in calls[0]
    assert "--train-cached-output" in calls[0]
    assert "--sample-seed" in calls[0]
    assert str(config.experiment.seed + 1) in calls[0]
    assert str(latest_root / "rollouts" / "iteration-00001.jsonl") in calls[1]
    assert resolve_latest_checkpoint(latest_root).checkpoint_id == "iteration-00001"
    timing_rows = [
        json.loads(line)
        for line in (latest_root / "phase_timings.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["phase"] for row in timing_rows] == ["merged_collect", "train_update"]
    assert all(row["iteration"] == 1 for row in timing_rows)
    assert all(row["exit_code"] == 0 for row in timing_rows)
    assert all(row["elapsed_seconds"] >= 0 for row in timing_rows)


def test_iteration_launcher_merged_collect_handles_all_outputs_when_overlap_on(tmp_path: Path) -> None:
    """When overlap judging is on, merged_collect produces raw, judged, and cached outputs."""
    config = train_config(tmp_path)
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    calls = []

    def runner(command):
        calls.append(list(command))
        if "self_summarization_agent.merged_collect_step" in command:
            write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.raw.jsonl", "iteration-00000", count=2)
            write_judged_rollouts(
                latest_root / "rollouts" / "iteration-00001.judged.jsonl",
                "iteration-00000",
                count=2,
            )
            write_cached_rollouts(
                latest_root / "rollouts" / "iteration-00001.jsonl",
                "iteration-00000",
                count=2,
            )
        if "self_summarization_agent.train_step" in command:
            write_fake_checkpoint(latest_root / "checkpoints" / "iteration-00001")
        return 0

    run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
    )

    assert "self_summarization_agent.merged_collect_step" in calls[0]
    assert "--train-judged-output" in calls[0]
    assert all("self_summarization_agent.judge_step" not in command for command in calls)
    assert all("self_summarization_agent.cache_step" not in command for command in calls)
    timing_rows = [
        json.loads(line)
        for line in (latest_root / "phase_timings.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["phase"] for row in timing_rows] == ["merged_collect", "train_update"]


def test_iteration_launcher_can_pass_resume_to_merged_collect(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    calls = []

    def runner(command):
        calls.append(list(command))
        if "self_summarization_agent.train_step" in command:
            next_checkpoint = latest_root / "checkpoints" / "iteration-00001"
            write_fake_checkpoint(next_checkpoint)
        return 0

    run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
        resume_rollouts=True,
    )

    assert "--resume" in calls[0]


def test_iteration_launcher_forwards_cli_overrides_to_subprocesses(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    calls = []

    def runner(command):
        calls.append(list(command))
        if "self_summarization_agent.train_step" in command:
            next_checkpoint = latest_root / "checkpoints" / "iteration-00001"
            write_fake_checkpoint(next_checkpoint)
        return 0

    run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
        overrides=["training.update_epochs=2"],
    )

    assert all("training.update_epochs=2" in command for command in calls)


def test_iteration_launcher_merged_collect_includes_both_splits(tmp_path: Path) -> None:
    """When eval_limit > 0, merged_collect includes both eval and train outputs."""
    config = train_config(tmp_path)
    config.dataset.train_limit = 1
    config.dataset.eval_limit = 1
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    calls = []

    def runner(command):
        calls.append(list(command))
        if "self_summarization_agent.train_step" in command:
            next_checkpoint = latest_root / "checkpoints" / "iteration-00001"
            write_fake_checkpoint(next_checkpoint)
        return 0

    run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
    )

    merged_command = calls[0]
    assert "self_summarization_agent.merged_collect_step" in merged_command
    assert str(latest_root / "rollouts" / "iteration-00000.eval.raw.jsonl") in merged_command
    assert "--eval-raw-output" in merged_command
    assert str(latest_root / "rollouts" / "iteration-00001.raw.jsonl") in merged_command
    assert "--train-raw-output" in merged_command
    assert "--eval-metrics-output" in merged_command
    assert "--eval-iteration" in merged_command
    assert "0" in merged_command  # eval_iteration = iteration - 1
    assert "self_summarization_agent.train_step" in calls[-1]


def test_checkpoint_evaluation_runs_final_eval_without_training(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    config.dataset.train_limit = 1
    config.dataset.eval_limit = 1
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    final_checkpoint = latest_root / "checkpoints" / "iteration-00010"
    write_fake_checkpoint(final_checkpoint)
    write_latest_checkpoint(latest_root, final_checkpoint)
    calls = []

    def runner(command):
        calls.append(list(command))
        return 0

    checkpoint = run_checkpoint_evaluation(
        config,
        config_path="train.yaml",
        iteration=10,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
    )

    assert checkpoint == final_checkpoint.resolve()
    assert "self_summarization_agent.rollout_collection" in calls[0]
    assert "--split" in calls[0] and "eval" in calls[0]
    assert str(latest_root / "rollouts" / "iteration-00010.eval.raw.jsonl") in calls[0]
    assert "self_summarization_agent.judge_step" in calls[1]
    assert "self_summarization_agent.eval_metrics" in calls[2]
    assert calls[2][calls[2].index("--iteration") + 1] == "10"
    assert calls[2][calls[2].index("--policy-checkpoint-id") + 1] == "iteration-00010"
    assert all("self_summarization_agent.train_step" not in command for command in calls)


# ---------------------------------------------------------------------------
# Retrieval worker tests
# ---------------------------------------------------------------------------

def test_iteration_launcher_scopes_retrieval_workers_to_merged_collect_phase(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = train_config(tmp_path)
    config.dataset.limit = 2
    config.dataset.train_limit = 1
    config.dataset.eval_limit = 1
    config.training.group_size = 1
    config.retrieval.persistent_worker = True
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    events = []

    class FakeWorkerProcess:
        def __init__(self, worker_id):
            self.worker_id = worker_id

    def fake_start_worker(**kwargs):
        worker_id = len(events) + 1
        events.append(f"start:{worker_id}")
        return FakeWorkerProcess(worker_id), f"http://127.0.0.1:{12344 + worker_id}"

    def fake_stop_worker(process, url):
        if process is None:
            return
        events.append(f"stop:{process.worker_id}")

    monkeypatch.setattr(
        "self_summarization_agent.iteration_launcher._start_retrieval_worker",
        fake_start_worker,
    )
    monkeypatch.setattr(
        "self_summarization_agent.iteration_launcher._stop_retrieval_worker",
        fake_stop_worker,
    )

    call_events = []

    def runner(command):
        call_events.append(f"run:{command[command.index('-m') + 1]}")
        events.append(f"run:{command[command.index('-m') + 1]}")
        if "self_summarization_agent.train_step" in command:
            next_checkpoint = latest_root / "checkpoints" / "iteration-00001"
            write_fake_checkpoint(next_checkpoint)
        return 0

    run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
    )

    # Single retrieval worker start/stop around the merged_collect phase
    merged_commands = [e for e in call_events if "merged_collect_step" in e]
    assert len(merged_commands) == 1
    assert events.index("stop:1") < events.index("run:self_summarization_agent.train_step")


def test_iteration_launcher_stops_retrieval_worker_when_merged_collect_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = train_config(tmp_path)
    config.retrieval.persistent_worker = True
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    worker_process = object()
    worker_stops = []

    monkeypatch.setattr(
        "self_summarization_agent.iteration_launcher._start_retrieval_worker",
        lambda **kwargs: (worker_process, "http://127.0.0.1:12345"),
    )

    def fake_stop_worker(process, url):
        if process is not None:
            worker_stops.append((process, url))

    monkeypatch.setattr(
        "self_summarization_agent.iteration_launcher._stop_retrieval_worker",
        fake_stop_worker,
    )

    try:
        run_training_iteration(
            config,
            config_path="train.yaml",
            iteration=1,
            latest_root=latest_root,
            command_runner=lambda command: 7,
            python_executable="python",
        )
    except RuntimeError as exc:
        assert "Merged collect subprocess failed with exit code 7" in str(exc)
    else:
        raise AssertionError("Expected merged collect failure")

    assert worker_stops == [(worker_process, "http://127.0.0.1:12345")]


def test_iteration_launcher_does_not_advance_latest_when_training_fails(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)

    def runner(command):
        if "self_summarization_agent.train_step" in command:
            return 7
        return 0

    try:
        run_training_iteration(
            config,
            config_path="train.yaml",
            iteration=1,
            latest_root=latest_root,
            command_runner=runner,
            python_executable="python",
        )
    except RuntimeError as exc:
        assert "Training subprocess failed" in str(exc)
    else:
        raise AssertionError("Expected failed training subprocess to stop iteration")

    assert resolve_latest_checkpoint(latest_root).checkpoint_id == "iteration-00000"


# ---------------------------------------------------------------------------
# Resume tests — overlap judging ON
# ---------------------------------------------------------------------------

def test_iteration_launcher_resume_after_train_collection_runs_cache_then_training(tmp_path: Path) -> None:
    """Resume: raw + judged exist, cache missing → merged_collect runs cache inline."""
    config = train_config(tmp_path)
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.raw.jsonl", "iteration-00000", count=2)
    write_judged_rollouts(latest_root / "rollouts" / "iteration-00001.judged.jsonl", "iteration-00000", count=2)
    calls = []

    def runner(command):
        calls.append(list(command))
        if "self_summarization_agent.train_step" in command:
            write_fake_checkpoint(latest_root / "checkpoints" / "iteration-00001")
        return 0

    run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
        resume=True,
    )

    # merged_collect produces cache, then train_update
    assert "self_summarization_agent.merged_collect_step" in calls[0]
    assert "self_summarization_agent.train_step" in calls[1]
    assert all("self_summarization_agent.judge_step" not in command for command in calls)


def test_iteration_launcher_resume_uses_collection_train_task_count(tmp_path: Path) -> None:
    """Resume checks expected count correctly using collection.train_task_count."""
    config = train_config(tmp_path)
    config.dataset.limit = 5
    config.dataset.train_limit = 5
    config.training.group_size = 2
    config.collection.train_task_count = 1
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.raw.jsonl", "iteration-00000", count=2)
    write_judged_rollouts(latest_root / "rollouts" / "iteration-00001.judged.jsonl", "iteration-00000", count=2)
    calls = []

    def runner(command):
        calls.append(list(command))
        if "self_summarization_agent.train_step" in command:
            write_fake_checkpoint(latest_root / "checkpoints" / "iteration-00001")
        return 0

    run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
        resume=True,
    )

    # merged_collect runs (for cache), then train_update
    assert "self_summarization_agent.merged_collect_step" in calls[0]
    assert "self_summarization_agent.train_step" in calls[1]


def test_iteration_launcher_resume_after_train_judge_runs_cache_next(tmp_path: Path) -> None:
    """Resume: raw + judged exist, cache missing → merged_collect skips to cache."""
    config = train_config(tmp_path)
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.raw.jsonl", "iteration-00000", count=2)
    write_judged_rollouts(latest_root / "rollouts" / "iteration-00001.judged.jsonl", "iteration-00000", count=2)
    calls = []

    def runner(command):
        calls.append(list(command))
        if "self_summarization_agent.train_step" in command:
            write_fake_checkpoint(latest_root / "checkpoints" / "iteration-00001")
        return 0

    run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
        resume=True,
    )

    # merged_collect handles cache inline, then train_update
    assert "self_summarization_agent.merged_collect_step" in calls[0]
    assert "self_summarization_agent.train_step" in calls[1]


def test_iteration_launcher_skips_merged_collect_when_all_outputs_exist(tmp_path: Path) -> None:
    """When cached rollouts already exist with inline caches on the judged path,
    the merged_collect phase is skipped and train_update reads from judged path."""
    config = train_config(tmp_path)
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.raw.jsonl", "iteration-00000", count=2)
    write_cached_rollouts(latest_root / "rollouts" / "iteration-00001.judged.jsonl", "iteration-00000", count=2)
    calls = []

    def runner(command):
        calls.append(list(command))
        if "self_summarization_agent.train_step" in command:
            write_fake_checkpoint(latest_root / "checkpoints" / "iteration-00001")
        return 0

    run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
        resume=True,
    )

    # merged_collect skipped (all outputs done), train_update uses judged path
    assert "self_summarization_agent.train_step" in calls[0]
    assert str(latest_root / "rollouts" / "iteration-00001.judged.jsonl") in calls[0]
    assert all("self_summarization_agent.merged_collect_step" not in command for command in calls)


def test_iteration_launcher_resume_after_train_cache_runs_training_next(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.raw.jsonl", "iteration-00000", count=2)
    write_judged_rollouts(latest_root / "rollouts" / "iteration-00001.judged.jsonl", "iteration-00000", count=2)
    write_cached_rollouts(latest_root / "rollouts" / "iteration-00001.jsonl", "iteration-00000", count=2)
    calls = []

    def runner(command):
        calls.append(list(command))
        if "self_summarization_agent.train_step" in command:
            write_fake_checkpoint(latest_root / "checkpoints" / "iteration-00001")
        return 0

    run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
        resume=True,
    )

    # Everything done except train_update
    assert calls == [
        [
            "python",
            "-m",
            "self_summarization_agent.train_step",
            "--config",
            "train.yaml",
            "--checkpoint",
            str(initial_checkpoint.resolve()),
            "--rollouts",
            str(latest_root / "rollouts" / "iteration-00001.jsonl"),
            "--output-checkpoint",
            str(latest_root / "checkpoints" / "iteration-00001"),
            "--metrics",
            str(latest_root / "step_metrics.jsonl"),
        ]
    ]


def test_iteration_launcher_resume_with_completed_update_collects_missing_preupdate_eval(tmp_path: Path) -> None:
    """When training is already advanced but pre-update eval is missing,
    merged_collect runs to fill in the eval data."""
    config = train_config(tmp_path)
    config.dataset.limit = 2
    config.dataset.train_limit = 1
    config.dataset.eval_limit = 1
    config.training.group_size = 1
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    next_checkpoint = latest_root / "checkpoints" / "iteration-00001"
    write_fake_checkpoint(initial_checkpoint)
    write_fake_checkpoint(next_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.raw.jsonl", "iteration-00000", count=1)
    write_judged_rollouts(latest_root / "rollouts" / "iteration-00001.judged.jsonl", "iteration-00000", count=1)
    write_cached_rollouts(latest_root / "rollouts" / "iteration-00001.jsonl", "iteration-00000", count=1)
    calls = []

    def runner(command):
        calls.append(list(command))
        return 0

    run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
        resume=True,
    )

    assert "self_summarization_agent.merged_collect_step" in calls[0]
    assert "--eval-raw-output" in calls[0]
    assert str(latest_root / "rollouts" / "iteration-00000.eval.raw.jsonl") in calls[0]
    assert "--resume" in calls[0]


def test_iteration_launcher_resume_after_eval_rollout_runs_eval_judge_next(tmp_path: Path, monkeypatch) -> None:
    """When train collection is complete but eval judging is not, merged_collect
    finishes the eval path (judging + metrics)."""
    config = train_config(tmp_path)
    config.dataset.limit = 2
    config.dataset.train_limit = 1
    config.dataset.eval_limit = 1
    config.training.group_size = 1
    config.retrieval.persistent_worker = True
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    next_checkpoint = latest_root / "checkpoints" / "iteration-00001"
    write_fake_checkpoint(initial_checkpoint)
    write_fake_checkpoint(next_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.raw.jsonl", "iteration-00000", count=1)
    write_judged_rollouts(latest_root / "rollouts" / "iteration-00001.judged.jsonl", "iteration-00000", count=1)
    write_cached_rollouts(latest_root / "rollouts" / "iteration-00001.jsonl", "iteration-00000", count=1)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00000.eval.raw.jsonl", "iteration-00000", count=1)
    calls = []

    monkeypatch.setattr(
        "self_summarization_agent.iteration_launcher._start_retrieval_worker",
        lambda **kwargs: (object(), "http://127.0.0.1:12345"),
    )
    monkeypatch.setattr(
        "self_summarization_agent.iteration_launcher._stop_retrieval_worker",
        lambda process, url: None,
    )

    def runner(command):
        calls.append(list(command))
        return 0

    run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
        resume=True,
    )

    # merged_collect runs to finish eval (judge + metrics), train_update skipped
    assert "self_summarization_agent.merged_collect_step" in calls[0]
    assert "--eval-raw-output" in calls[0]


def test_iteration_launcher_resume_uses_collection_eval_task_count(tmp_path: Path, monkeypatch) -> None:
    """Resume checks eval task count from collection config."""
    config = train_config(tmp_path)
    config.dataset.limit = 5
    config.dataset.train_limit = 2
    config.dataset.eval_limit = 3
    config.training.group_size = 1
    config.collection.eval_task_count = 2
    config.retrieval.persistent_worker = True
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    next_checkpoint = latest_root / "checkpoints" / "iteration-00001"
    write_fake_checkpoint(initial_checkpoint)
    write_fake_checkpoint(next_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.raw.jsonl", "iteration-00000", count=2)
    write_judged_rollouts(latest_root / "rollouts" / "iteration-00001.judged.jsonl", "iteration-00000", count=2)
    write_cached_rollouts(latest_root / "rollouts" / "iteration-00001.jsonl", "iteration-00000", count=2)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00000.eval.raw.jsonl", "iteration-00000", count=2)
    calls = []

    monkeypatch.setattr(
        "self_summarization_agent.iteration_launcher._start_retrieval_worker",
        lambda **kwargs: (object(), "http://127.0.0.1:12345"),
    )
    monkeypatch.setattr(
        "self_summarization_agent.iteration_launcher._stop_retrieval_worker",
        lambda process, url: None,
    )

    def runner(command):
        calls.append(list(command))
        return 0

    run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
        resume=True,
    )

    assert "self_summarization_agent.merged_collect_step" in calls[0]
    assert "--eval-raw-output" in calls[0]


def test_iteration_launcher_resume_after_eval_judge_runs_eval_metrics_next(tmp_path: Path) -> None:
    """When eval raw + judged exist but metrics missing, merged_collect computes metrics."""
    config = train_config(tmp_path)
    config.dataset.limit = 2
    config.dataset.train_limit = 1
    config.dataset.eval_limit = 1
    config.training.group_size = 1
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    next_checkpoint = latest_root / "checkpoints" / "iteration-00001"
    write_fake_checkpoint(initial_checkpoint)
    write_fake_checkpoint(next_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.raw.jsonl", "iteration-00000", count=1)
    write_judged_rollouts(latest_root / "rollouts" / "iteration-00001.judged.jsonl", "iteration-00000", count=1)
    write_cached_rollouts(latest_root / "rollouts" / "iteration-00001.jsonl", "iteration-00000", count=1)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00000.eval.raw.jsonl", "iteration-00000", count=1)
    write_judged_rollouts(latest_root / "rollouts" / "iteration-00000.eval.jsonl", "iteration-00000", count=1)
    calls = []

    def runner(command):
        calls.append(list(command))
        return 0

    run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
        resume=True,
    )

    # merged_collect runs to produce eval_metrics; train_update skipped (complete)
    assert "self_summarization_agent.merged_collect_step" in calls[0]


def test_iteration_launcher_resume_latest_at_target_returns_without_subprocesses(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    config.dataset.limit = 2
    config.dataset.train_limit = 1
    config.dataset.eval_limit = 1
    config.training.group_size = 1
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    next_checkpoint = latest_root / "checkpoints" / "iteration-00001"
    write_fake_checkpoint(next_checkpoint)
    write_latest_checkpoint(latest_root, next_checkpoint)
    calls = []

    def runner(command):
        calls.append(list(command))
        return 0

    run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
        resume=True,
    )

    assert calls == []


def test_iteration_launcher_resume_after_eval_metrics_advances_without_subprocesses(tmp_path: Path) -> None:
    """When all outputs (including eval metrics) exist, nothing runs — just advance."""
    config = train_config(tmp_path)
    config.dataset.limit = 2
    config.dataset.train_limit = 1
    config.dataset.eval_limit = 1
    config.training.group_size = 1
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    next_checkpoint = latest_root / "checkpoints" / "iteration-00001"
    write_fake_checkpoint(initial_checkpoint)
    write_fake_checkpoint(next_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.raw.jsonl", "iteration-00000", count=1)
    write_judged_rollouts(latest_root / "rollouts" / "iteration-00001.judged.jsonl", "iteration-00000", count=1)
    write_cached_rollouts(latest_root / "rollouts" / "iteration-00001.jsonl", "iteration-00000", count=1)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00000.eval.raw.jsonl", "iteration-00000", count=1)
    write_judged_rollouts(latest_root / "rollouts" / "iteration-00000.eval.jsonl", "iteration-00000", count=1)
    write_eval_metric(latest_root / "eval_metrics.jsonl", iteration=0, checkpoint_id="iteration-00000")
    calls = []

    def runner(command):
        calls.append(list(command))
        return 0

    next_path = run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        command_runner=runner,
        python_executable="python",
        resume=True,
    )

    assert calls == []
    assert next_path == next_checkpoint.resolve()
    assert resolve_latest_checkpoint(latest_root).checkpoint_id == "iteration-00001"


def test_iteration_launcher_resume_rejects_mismatched_artifacts(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.raw.jsonl", "other-checkpoint", count=2)

    try:
        run_training_iteration(
            config,
            config_path="train.yaml",
            iteration=1,
            latest_root=latest_root,
            command_runner=lambda command: 0,
            python_executable="python",
            resume=True,
        )
    except ValueError as exc:
        assert "expected 'iteration-00000'" in str(exc)
    else:
        raise AssertionError("Expected resume to reject mismatched rollout artifacts")
