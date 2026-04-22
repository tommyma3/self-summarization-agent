import json
from dataclasses import dataclass
from pathlib import Path

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
from self_summarization_agent.train_step import run_train_step


@dataclass(slots=True)
class FakeUpdateMetrics:
    sample_count: int
    mean_reward: float
    mean_advantage: float
    loss: float


class FakeTrainer:
    def __init__(self) -> None:
        self.grouped_samples = None
        self.saved_checkpoints: list[str] = []

    def step(self, grouped_samples):
        self.grouped_samples = grouped_samples
        flat = [sample for samples in grouped_samples.values() for sample in samples]
        return FakeUpdateMetrics(
            sample_count=len(flat),
            mean_reward=sum(sample.reward for sample in flat) / len(flat) if flat else 0.0,
            mean_advantage=0.0,
            loss=0.0,
        )

    def save_checkpoint(self, path: str) -> None:
        checkpoint = Path(path)
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_text("weights", encoding="utf-8")
        self.saved_checkpoints.append(path)


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


def write_rollout(path: Path, checkpoint_id: str) -> None:
    row = {
        "policy_checkpoint_id": checkpoint_id,
        "turn_records": [
            {
                "query_id": "q1",
                "turn_id": "final-answer",
                "kind": "final_answer",
                "prompt": "prompt",
                "completion": '{"tool_name": "finish", "arguments": {"answer": "done"}}',
            }
        ],
        "turn_rewards": {"final-answer": 1.0},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_run_train_step_consumes_matching_rollouts_and_saves_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    rollout_path = tmp_path / "rollouts.jsonl"
    write_rollout(rollout_path, "step-00001")
    output_checkpoint = tmp_path / "checkpoints" / "step-00002"
    trainer = FakeTrainer()

    run_train_step(
        train_config(tmp_path),
        checkpoint_path=checkpoint,
        rollout_path=rollout_path,
        output_checkpoint_path=output_checkpoint,
        trainer=trainer,
    )

    assert trainer.grouped_samples is not None
    assert list(trainer.grouped_samples) == ["q1"]
    assert trainer.saved_checkpoints == [str(output_checkpoint)]
    assert (output_checkpoint / ".complete").exists()


def test_run_train_step_rejects_checkpoint_mismatch_before_trainer_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    rollout_path = tmp_path / "rollouts.jsonl"
    write_rollout(rollout_path, "other-step")
    trainer = FakeTrainer()

    try:
        run_train_step(
            train_config(tmp_path),
            checkpoint_path=checkpoint,
            rollout_path=rollout_path,
            output_checkpoint_path=tmp_path / "next",
            trainer=trainer,
        )
    except ValueError as exc:
        assert "checkpoint mismatch" in str(exc)
    else:
        raise AssertionError("Expected checkpoint mismatch to fail")

    assert trainer.grouped_samples is None
    assert trainer.saved_checkpoints == []
