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
    assert "self_summarization_agent.train_step" in calls[1]
    assert resolve_latest_checkpoint(latest_root).checkpoint_id == "iteration-00001"


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
