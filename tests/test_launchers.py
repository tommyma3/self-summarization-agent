from pathlib import Path

from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.config import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    RetrievalConfig,
    RolloutConfig,
    RunConfig,
    RuntimeConfig,
)
from self_summarization_agent.dataset import QueryExample, split_train_eval_examples
from self_summarization_agent.run_launcher import run_experiment


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
        model=ModelConfig(model_path="unused"),
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
