import json
from dataclasses import dataclass
from pathlib import Path

from self_summarization_agent.backend import FakeBackend
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
from self_summarization_agent.dataset import QueryExample
from self_summarization_agent.rollout_collection import collect_rollouts


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


class BatchRecordingGenerator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        self.batch_sizes.append(len(prompts))
        outputs: list[str] = []
        for prompt in prompts:
            if "### TOOL_RESULT" in prompt:
                outputs.append(tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'))
            elif "question one" in prompt:
                outputs.append(tool_output('{"tool_name": "search", "arguments": {"query": "question one"}}'))
            elif "question two" in prompt:
                outputs.append(tool_output('{"tool_name": "search", "arguments": {"query": "question two"}}'))
            else:
                outputs.append("malformed")
        return outputs

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
        return FakeJudgeDecision(
            outcome="correct_answer" if status == "completed" else "budget_exhausted",
            judge_prompt="judge prompt",
            judge_response="correct: yes",
            parse_error=False,
        )


def train_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        experiment=ExperimentConfig(name="demo", seed=1, output_root=str(tmp_path), bc_plus_root=str(tmp_path)),
        dataset=DatasetConfig(limit=1, train_limit=1),
        retrieval=RetrievalConfig(backend="faiss", index_path="unused"),
        model=ModelConfig(backend="transformers", model_path="unused"),
        rollout=RolloutConfig(backend="vllm_offline"),
        runtime=RuntimeConfig(context_threshold_tokens=1000, max_context_tokens=1024, tool_budget=4),
        judge=JudgeConfig(enabled=True),
        training=TrainingConfig(group_size=2),
    )


def test_collect_rollouts_writes_checkpoint_tagged_rows(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    backend = FakeBackend(search_index={"question": ["doc-1"]}, documents={"doc-1": "fact"})
    generator = CyclingGenerator(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "question"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    examples = [QueryExample(query_id="q1", query="question", answer="done")]
    output_path = tmp_path / "rollouts.jsonl"

    collect_rollouts(
        config,
        checkpoint_path=checkpoint,
        output_path=output_path,
        examples=examples,
        backend=backend,
        generator=generator,
        judge=FakeJudge(),
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {row["policy_checkpoint_id"] for row in rows} == {"step-00001"}
    assert all(row["policy_checkpoint_path"] == str(checkpoint.resolve()) for row in rows)
    assert all(row["query_id"] == "q1" for row in rows)
    assert all(row["status"] == "completed" for row in rows)
    assert all(row["trainable_sample_count"] == 1 for row in rows)


def test_collect_rollouts_batches_active_queries(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    config.dataset.train_limit = 2
    config.training.group_size = 1
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    backend = FakeBackend(
        search_index={
            "question one": ["doc-1"],
            "question two": ["doc-2"],
        },
        documents={
            "doc-1": "fact one",
            "doc-2": "fact two",
        },
    )
    generator = BatchRecordingGenerator()
    examples = [
        QueryExample(query_id="q1", query="question one", answer="done"),
        QueryExample(query_id="q2", query="question two", answer="done"),
    ]
    output_path = tmp_path / "rollouts.jsonl"

    collect_rollouts(
        config,
        checkpoint_path=checkpoint,
        output_path=output_path,
        examples=examples,
        backend=backend,
        generator=generator,
        judge=FakeJudge(),
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert generator.batch_sizes == [2, 2]
    assert [row["query_id"] for row in rows] == ["q1", "q2"]
    assert all(row["status"] == "completed" for row in rows)
    assert all(row["trainable_sample_count"] == 1 for row in rows)


def test_collect_rollouts_resume_skips_existing_rows(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    config.dataset.train_limit = 2
    config.training.group_size = 1
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    output_path = tmp_path / "rollouts.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "policy_checkpoint_id": "step-00001",
                "policy_checkpoint_path": str(checkpoint.resolve()),
                "rollout_index": 0,
                "query_id": "q1",
                "query": "question one",
                "status": "completed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    backend = FakeBackend(
        search_index={"question one": ["doc-1"], "question two": ["doc-2"]},
        documents={"doc-1": "fact one", "doc-2": "fact two"},
    )
    generator = BatchRecordingGenerator()
    examples = [
        QueryExample(query_id="q1", query="question one", answer="done"),
        QueryExample(query_id="q2", query="question two", answer="done"),
    ]

    collect_rollouts(
        config,
        checkpoint_path=checkpoint,
        output_path=output_path,
        examples=examples,
        backend=backend,
        generator=generator,
        judge=FakeJudge(),
        resume=True,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert generator.batch_sizes == [1, 1]
    assert [(row["query_id"], row["rollout_index"]) for row in rows] == [("q1", 0), ("q2", 0)]


def test_collect_rollouts_resume_rejects_different_checkpoint(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    checkpoint.mkdir(parents=True)
    output_path = tmp_path / "rollouts.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "policy_checkpoint_id": "step-previous",
                "rollout_index": 0,
                "query_id": "q1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = [QueryExample(query_id="q1", query="question", answer="done")]

    try:
        collect_rollouts(
            config,
            checkpoint_path=checkpoint,
            output_path=output_path,
            examples=examples,
            backend=FakeBackend(search_index={}, documents={}),
            generator=CyclingGenerator([]),
            judge=FakeJudge(),
            resume=True,
        )
    except ValueError as exc:
        assert "expected 'step-00001'" in str(exc)
    else:
        raise AssertionError("Expected resume to reject rollout rows from another checkpoint")
