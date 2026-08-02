import json
from pathlib import Path

from self_summarization_agent.checkpoints import (
    mark_checkpoint_complete,
    resolve_latest_checkpoint,
    write_latest_checkpoint,
)
from self_summarization_agent.config import (
    CollectionConfig,
    DatasetConfig,
    ExperimentConfig,
    JudgeConfig,
    ModelConfig,
    RetrievalConfig,
    RolloutConfig,
    RuntimeConfig,
    TrainConfig,
    TrainingConfig,
    resolved_rollout_sampling_profile,
    sampling_profile_id,
)
from self_summarization_agent.iteration_artifacts import (
    IterationArtifacts,
    completed_iteration_from_checkpoint_id,
    has_complete_cached_rollouts,
)
from self_summarization_agent.training_loop import run_training, run_training_iteration


def write_fake_checkpoint(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_text("weights", encoding="utf-8")
    mark_checkpoint_complete(path)


def train_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        experiment=ExperimentConfig(
            name="demo",
            seed=1,
            output_root=str(tmp_path),
            bc_plus_root=str(tmp_path),
        ),
        dataset=DatasetConfig(limit=1, eval_limit=0),
        retrieval=RetrievalConfig(backend="faiss", index_path="unused"),
        model=ModelConfig(backend="transformers", model_path="unused"),
        rollout=RolloutConfig(backend="vllm_offline"),
        runtime=RuntimeConfig(
            context_threshold_tokens=1000,
            max_context_tokens=1024,
            tool_budget=4,
        ),
        judge=JudgeConfig(enabled=True),
        training=TrainingConfig(group_size=2),
        collection=CollectionConfig(cache_gpu_ids=[0]),
    )


def write_phase_rollouts(
    path: Path,
    checkpoint_id: str,
    *,
    cached: bool,
    profile_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(2):
        row = {
            "policy_checkpoint_id": checkpoint_id,
            "rollout_split": "train",
            "query_id": f"q{index}",
            "rollout_index": index,
            "sampling_profile_id": profile_id,
            "turn_records": [],
            "trajectory_records": [],
        }
        if cached:
            row.update(
                {
                    "turn_rewards": {},
                    "trainable_sample_count": 0,
                    "judge": {"outcome": "wrong_answer", "parse_error": False},
                }
            )
        rows.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_iteration_artifacts_preserve_existing_file_names(tmp_path: Path) -> None:
    artifacts = IterationArtifacts.build(
        tmp_path,
        iteration=3,
        checkpoint_id="iteration-00002",
    )
    assert artifacts.raw_rollouts.name == "iteration-00003.raw.jsonl"
    assert artifacts.cached_rollouts.name == "iteration-00003.jsonl"
    assert artifacts.eval_raw_rollouts.name == "iteration-00002.eval.raw.jsonl"
    assert artifacts.next_checkpoint.name == "iteration-00003"
    assert artifacts.manifest.name == "iteration-00003.manifest.json"


def test_completed_iteration_treats_external_base_checkpoint_as_zero() -> None:
    assert completed_iteration_from_checkpoint_id("Qwen3.5-9B") == 0
    assert completed_iteration_from_checkpoint_id("iteration-00012") == 12


def test_cached_rollout_completion_accepts_zero_trainable_samples(tmp_path: Path) -> None:
    path = tmp_path / "cached.jsonl"
    write_phase_rollouts(
        path,
        "iteration-00000",
        cached=True,
        profile_id="train-profile",
    )
    assert has_complete_cached_rollouts(
        path,
        checkpoint_id="iteration-00000",
        expected_count=2,
    )


def test_training_iteration_connects_two_python_phases(tmp_path: Path, monkeypatch) -> None:
    config = train_config(tmp_path)
    profile_id = sampling_profile_id(resolved_rollout_sampling_profile(config, split="train"))
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial)
    write_latest_checkpoint(latest_root, initial)
    phases = []

    def fake_phase(**request):
        phases.append(request["phase"])
        kwargs = request["kwargs"]
        if request["phase"] == "merged_collect":
            write_phase_rollouts(
                Path(kwargs["train_raw_output"]),
                "iteration-00000",
                cached=False,
                profile_id=profile_id,
            )
            write_phase_rollouts(
                Path(kwargs["train_judged_output"]),
                "iteration-00000",
                cached=True,
                profile_id=profile_id,
            )
            write_phase_rollouts(
                Path(kwargs["train_cached_output"]),
                "iteration-00000",
                cached=True,
                profile_id=profile_id,
            )
        else:
            write_fake_checkpoint(Path(kwargs["output_checkpoint_path"]))

    monkeypatch.setattr("self_summarization_agent.training_loop.run_python_phase", fake_phase)
    result = run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        resume=True,
    )
    assert phases == ["merged_collect", "train_update"]
    assert result.name == "iteration-00001"
    assert resolve_latest_checkpoint(latest_root).path == result


def test_resume_advances_complete_checkpoint_without_retraining(tmp_path: Path, monkeypatch) -> None:
    config = train_config(tmp_path)
    profile_id = sampling_profile_id(resolved_rollout_sampling_profile(config, split="train"))
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial = latest_root / "checkpoints" / "iteration-00000"
    next_checkpoint = latest_root / "checkpoints" / "iteration-00001"
    write_fake_checkpoint(initial)
    write_fake_checkpoint(next_checkpoint)
    write_latest_checkpoint(latest_root, initial)
    artifacts = IterationArtifacts.build(
        latest_root,
        iteration=1,
        checkpoint_id="iteration-00000",
    )
    write_phase_rollouts(
        artifacts.raw_rollouts,
        "iteration-00000",
        cached=False,
        profile_id=profile_id,
    )
    write_phase_rollouts(
        artifacts.judged_rollouts,
        "iteration-00000",
        cached=True,
        profile_id=profile_id,
    )
    write_phase_rollouts(
        artifacts.cached_rollouts,
        "iteration-00000",
        cached=True,
        profile_id=profile_id,
    )

    def unexpected_phase(**_request):
        raise AssertionError("No phase should run when cache and checkpoint are complete")

    monkeypatch.setattr("self_summarization_agent.training_loop.run_python_phase", unexpected_phase)
    result = run_training_iteration(
        config,
        config_path="train.yaml",
        iteration=1,
        latest_root=latest_root,
        resume=True,
    )
    assert result == next_checkpoint.resolve()
    assert resolve_latest_checkpoint(latest_root).path == result


def test_target_iteration_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    config = train_config(tmp_path)
    latest_root = tmp_path / "artifacts" / "train" / "demo"
    initial = latest_root / "checkpoints" / "iteration-00000"
    write_fake_checkpoint(initial)
    write_latest_checkpoint(latest_root, initial)
    seen = []

    def fake_iteration(config, *, iteration, latest_root, **_kwargs):
        seen.append(iteration)
        checkpoint = Path(latest_root) / "checkpoints" / f"iteration-{iteration:05d}"
        write_fake_checkpoint(checkpoint)
        write_latest_checkpoint(latest_root, checkpoint)
        return checkpoint.resolve()

    monkeypatch.setattr("self_summarization_agent.training_loop.run_training_iteration", fake_iteration)
    first = run_training(
        config,
        config_path="train.yaml",
        target_iterations=3,
        latest_root=latest_root,
    )
    second = run_training(
        config,
        config_path="train.yaml",
        target_iterations=3,
        latest_root=latest_root,
    )
    assert seen == [1, 2, 3]
    assert first == second
