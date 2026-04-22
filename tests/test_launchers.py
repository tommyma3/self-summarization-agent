from dataclasses import dataclass
import json
from pathlib import Path

from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.config import (
    DatasetConfig,
    ExperimentConfig,
    JudgeConfig,
    ModelConfig,
    RetrievalConfig,
    RolloutConfig,
    RunConfig,
    RuntimeConfig,
    TrainConfig,
    TrainingConfig,
)
from self_summarization_agent.dataset import QueryExample
from self_summarization_agent.run_launcher import run_experiment
from self_summarization_agent.train_launcher import split_train_eval_examples, train_experiment


class CyclingGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.cursor = 0

    def generate(self, prompt: str) -> str:
        del prompt
        output = self.outputs[self.cursor % len(self.outputs)]
        self.cursor += 1
        return output

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def tool_output(json_text: str) -> str:
    return f"<think>thinking</think>\n{json_text}"


@dataclass(slots=True)
class FakeJudgeDecision:
    outcome: str
    judge_prompt: str | None
    judge_response: str | None
    parse_error: bool


class FakeJudge:
    def evaluate(self, example: QueryExample, status: str, response: str) -> FakeJudgeDecision:
        del example, response
        outcome = "correct_answer" if status == "completed" else "budget_exhausted"
        return FakeJudgeDecision(
            outcome=outcome,
            judge_prompt="judge prompt",
            judge_response="correct: yes",
            parse_error=False,
        )


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
        self.saved_checkpoints.append(path)


def test_run_launcher_writes_run_and_trajectory_artifacts(tmp_path: Path) -> None:
    config = RunConfig(
        experiment=ExperimentConfig(
            name="demo-run",
            seed=1,
            output_root=str(tmp_path),
            bc_plus_root=str(tmp_path / "bc-plus"),
        ),
        dataset=DatasetConfig(limit=1),
        retrieval=RetrievalConfig(backend="faiss", index_path="unused"),
        model=ModelConfig(backend="transformers", model_path="unused"),
        rollout=RolloutConfig(),
        runtime=RuntimeConfig(context_threshold_tokens=1000, max_context_tokens=1024, tool_budget=4),
    )
    backend = FakeBackend(search_index={"question": ["doc-1"]}, documents={"doc-1": "fact"})
    generator = CyclingGenerator(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "question"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    examples = [QueryExample(query_id="q1", query="question")]

    run_dir = run_experiment(config, examples=examples, backend=backend, generator=generator)

    assert (run_dir / "q1.json").exists()
    assert (run_dir / "trajectories.jsonl").exists()
    assert (run_dir / "manifest.json").exists()
    trajectory_text = (run_dir / "trajectories.jsonl").read_text(encoding="utf-8")
    assert '"query_id": "q1"' in trajectory_text
    assert '"status": "completed"' in trajectory_text


def test_train_launcher_writes_metrics_and_rollouts(tmp_path: Path) -> None:
    config = TrainConfig(
        experiment=ExperimentConfig(
            name="demo-train",
            seed=1,
            output_root=str(tmp_path),
            bc_plus_root=str(tmp_path / "bc-plus"),
        ),
        dataset=DatasetConfig(limit=1),
        retrieval=RetrievalConfig(backend="faiss", index_path="unused"),
        model=ModelConfig(backend="transformers", model_path="unused"),
        runtime=RuntimeConfig(context_threshold_tokens=1000, max_context_tokens=1024, tool_budget=4),
        judge=JudgeConfig(enabled=True),
        training=TrainingConfig(steps=1, batch_size=1, group_size=2, checkpoint_interval=1),
    )
    backend = FakeBackend(search_index={"question": ["doc-1"]}, documents={"doc-1": "fact"})
    generator = CyclingGenerator(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "question"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    examples = [QueryExample(query_id="q1", query="question", answer="done")]
    trainer = FakeTrainer()

    train_dir = train_experiment(
        config,
        examples=examples,
        backend=backend,
        rollout_generator=generator,
        judge=FakeJudge(),
        trainer=trainer,
    )

    assert (train_dir / "metrics.jsonl").exists()
    assert (train_dir / "accuracy_history.jsonl").exists()
    assert (train_dir / "rollouts" / "step-00001.jsonl").exists()
    assert (train_dir / "manifest.json").exists()
    assert trainer.grouped_samples is not None
    assert "q1" in trainer.grouped_samples
    assert trainer.saved_checkpoints
    accuracy_rows = [
        json.loads(line)
        for line in (train_dir / "accuracy_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert accuracy_rows == [
        {
            "epoch": 1,
            "timestamp_utc": accuracy_rows[0]["timestamp_utc"],
            "train_accuracy": 1.0,
            "train_correct": 1,
            "train_total": 1,
            "train_malformed": 0,
            "train_parse_errors": 0,
            "eval_accuracy": 0.0,
            "eval_correct": 0,
            "eval_total": 0,
            "eval_malformed": 0,
            "eval_parse_errors": 0,
        }
    ]


def test_split_train_eval_examples_uses_contiguous_ranges() -> None:
    examples = [
        QueryExample(query_id=f"q{index}", query=f"question {index}")
        for index in range(215)
    ]

    train_examples, eval_examples = split_train_eval_examples(
        examples,
        train_limit=200,
        eval_limit=10,
    )

    assert [example.query_id for example in train_examples[:2]] == ["q0", "q1"]
    assert [example.query_id for example in train_examples[-2:]] == ["q198", "q199"]
    assert [example.query_id for example in eval_examples] == [f"q{index}" for index in range(200, 210)]
