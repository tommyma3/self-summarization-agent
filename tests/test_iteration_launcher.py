import json
from pathlib import Path

from self_summarization_agent.checkpoints import mark_checkpoint_complete, resolve_latest_checkpoint, write_latest_checkpoint
from self_summarization_agent.config import (
    DatasetConfig,
    ExperimentConfig,
    JudgeConfig,
    ModelConfig,
    RetrievalConfig,
    RolloutConfig,
    RuntimeConfig,
    TrainConfig,
    TrainingConfig,
)
from self_summarization_agent.iteration_launcher import run_training_iteration


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
            "turn_records": [],
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
            "turn_records": [],
            "turn_rewards": {},
            "judge": {"outcome": "wrong_answer", "parse_error": False},
        }
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def training_cache() -> dict:
    return {
        "version": 1,
        "input_ids": [1],
        "labels": [2],
        "completion_mask": [True],
        "reference_logprob": -0.1,
    }


def write_cached_rollouts(path: Path, checkpoint_id: str, count: int) -> None:
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
                    "training_cache": training_cache(),
                }
            ],
            "turn_rewards": {"final-answer": 1.0},
            "trainable_sample_count": 1,
            "judge": {"outcome": "wrong_answer", "parse_error": False},
        }
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_eval_metric(path: Path, iteration: int, checkpoint_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "iteration": iteration,
                "policy_checkpoint_id": checkpoint_id,
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
    )


def test_iteration_launcher_runs_rollout_then_train_and_advances_latest(tmp_path: Path) -> None:
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
    assert "self_summarization_agent.rollout_collection" in calls[0]
    assert "self_summarization_agent.judge_step" in calls[1]
    assert "self_summarization_agent.cache_step" in calls[2]
    assert "self_summarization_agent.train_step" in calls[3]
    assert str(latest_root / "rollouts" / "iteration-00001.raw.jsonl") in calls[0]
    assert "--sample-seed" in calls[0]
    assert str(config.experiment.seed + 1) in calls[0]
    assert str(latest_root / "rollouts" / "iteration-00001.raw.jsonl") in calls[1]
    assert str(latest_root / "rollouts" / "iteration-00001.judged.jsonl") in calls[1]
    assert str(latest_root / "rollouts" / "iteration-00001.judged.jsonl") in calls[2]
    assert str(latest_root / "rollouts" / "iteration-00001.jsonl") in calls[2]
    assert str(latest_root / "rollouts" / "iteration-00001.jsonl") in calls[3]
    assert resolve_latest_checkpoint(latest_root).checkpoint_id == "iteration-00001"
    timing_rows = [
        json.loads(line)
        for line in (latest_root / "phase_timings.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["phase"] for row in timing_rows] == ["train_rollout", "train_judge", "train_cache", "train_update"]
    assert all(row["iteration"] == 1 for row in timing_rows)
    assert all(row["exit_code"] == 0 for row in timing_rows)
    assert all(row["elapsed_seconds"] >= 0 for row in timing_rows)


def test_iteration_launcher_can_pass_resume_to_rollout_collection(tmp_path: Path) -> None:
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

    assert all("training.update_epochs=2" in command for command in calls[:4])


def test_iteration_launcher_runs_eval_after_training_when_eval_split_configured(tmp_path: Path) -> None:
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

    assert "self_summarization_agent.rollout_collection" in calls[4]
    assert "--split" in calls[4]
    assert "eval" in calls[4]
    assert "training.group_size=1" in calls[4]
    assert "self_summarization_agent.judge_step" in calls[5]
    assert "--split" in calls[5]
    assert "eval" in calls[5]
    assert "self_summarization_agent.eval_metrics" in calls[6]
    assert str(latest_root / "eval_metrics.jsonl") in calls[6]


def test_iteration_launcher_reuses_persistent_retrieval_worker_for_rollout_phases(
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
    calls = []
    worker_starts = []
    worker_stops = []

    class FakeWorkerProcess:
        pass

    def fake_start_worker(**kwargs):
        worker_starts.append(kwargs)
        return FakeWorkerProcess(), "http://127.0.0.1:12345"

    def fake_stop_worker(process, url):
        worker_stops.append((process, url))

    monkeypatch.setattr(
        "self_summarization_agent.iteration_launcher._start_retrieval_worker",
        fake_start_worker,
    )
    monkeypatch.setattr(
        "self_summarization_agent.iteration_launcher._stop_retrieval_worker",
        fake_stop_worker,
    )

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

    rollout_calls = [command for command in calls if "self_summarization_agent.rollout_collection" in command]
    assert len(worker_starts) == 1
    assert len(worker_stops) == 1
    assert len(rollout_calls) == 2
    assert all("--retrieval-worker-url" in command for command in rollout_calls)
    assert all("http://127.0.0.1:12345" in command for command in rollout_calls)


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


def test_iteration_launcher_resume_after_train_rollout_runs_judge_next(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial_checkpoint = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial_checkpoint)
    write_latest_checkpoint(latest_root, initial_checkpoint)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.raw.jsonl", "iteration-00000", count=2)
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

    assert "self_summarization_agent.judge_step" in calls[0]
    assert "self_summarization_agent.cache_step" in calls[1]
    assert "self_summarization_agent.train_step" in calls[2]
    assert all("self_summarization_agent.rollout_collection" not in command for command in calls)


def test_iteration_launcher_resume_after_train_judge_runs_cache_next(tmp_path: Path) -> None:
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

    assert calls[0] == [
        "python",
        "-m",
        "self_summarization_agent.cache_step",
        "--config",
        "train.yaml",
        "--checkpoint",
        str(initial_checkpoint.resolve()),
        "--rollouts",
        str(latest_root / "rollouts" / "iteration-00001.judged.jsonl"),
        "--output",
        str(latest_root / "rollouts" / "iteration-00001.jsonl"),
        "--resume",
    ]
    assert calls[1] == [
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


def test_iteration_launcher_resume_after_training_runs_eval_next(tmp_path: Path) -> None:
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

    assert "self_summarization_agent.rollout_collection" in calls[0]
    assert "--split" in calls[0]
    assert "eval" in calls[0]
    assert "--resume" in calls[0]
    assert "self_summarization_agent.judge_step" in calls[1]
    assert "self_summarization_agent.eval_metrics" in calls[2]


def test_iteration_launcher_resume_after_eval_rollout_runs_eval_judge_next(tmp_path: Path) -> None:
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
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.eval.raw.jsonl", "iteration-00001", count=1)
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

    assert "self_summarization_agent.judge_step" in calls[0]
    assert "--split" in calls[0]
    assert "eval" in calls[0]
    assert "self_summarization_agent.eval_metrics" in calls[1]


def test_iteration_launcher_resume_after_eval_judge_runs_eval_metrics_next(tmp_path: Path) -> None:
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
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.eval.raw.jsonl", "iteration-00001", count=1)
    write_judged_rollouts(latest_root / "rollouts" / "iteration-00001.eval.jsonl", "iteration-00001", count=1)
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

    assert calls == [
        [
            "python",
            "-m",
            "self_summarization_agent.eval_metrics",
            "--rollouts",
            str(latest_root / "rollouts" / "iteration-00001.eval.jsonl"),
            "--metrics",
            str(latest_root / "eval_metrics.jsonl"),
            "--iteration",
            "1",
            "--policy-checkpoint-id",
            "iteration-00001",
        ]
    ]


def test_iteration_launcher_resume_latest_at_target_runs_missing_eval_only(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    config.dataset.limit = 2
    config.dataset.train_limit = 1
    config.dataset.eval_limit = 1
    config.training.group_size = 1
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    next_checkpoint = latest_root / "checkpoints" / "iteration-00001"
    write_fake_checkpoint(next_checkpoint)
    write_latest_checkpoint(latest_root, next_checkpoint)
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.eval.raw.jsonl", "iteration-00001", count=1)
    write_judged_rollouts(latest_root / "rollouts" / "iteration-00001.eval.jsonl", "iteration-00001", count=1)
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

    assert calls == [
        [
            "python",
            "-m",
            "self_summarization_agent.eval_metrics",
            "--rollouts",
            str(latest_root / "rollouts" / "iteration-00001.eval.jsonl"),
            "--metrics",
            str(latest_root / "eval_metrics.jsonl"),
            "--iteration",
            "1",
            "--policy-checkpoint-id",
            "iteration-00001",
        ]
    ]


def test_iteration_launcher_resume_after_eval_metrics_advances_without_subprocesses(tmp_path: Path) -> None:
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
    write_raw_rollouts(latest_root / "rollouts" / "iteration-00001.eval.raw.jsonl", "iteration-00001", count=1)
    write_judged_rollouts(latest_root / "rollouts" / "iteration-00001.eval.jsonl", "iteration-00001", count=1)
    write_eval_metric(latest_root / "eval_metrics.jsonl", iteration=1, checkpoint_id="iteration-00001")
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
