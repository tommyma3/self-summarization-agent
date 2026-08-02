import json
from pathlib import Path

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
)
from self_summarization_agent.dataset import QueryExample
from self_summarization_agent.merged_collect_step import _collect_split_streaming


class RecordingJudge:
    def __init__(self) -> None:
        self.rows = []

    def submit(self, rows, examples) -> None:
        self.rows.extend(rows)

    def drain_available(self):
        return []


class RecordingCache:
    def __init__(self) -> None:
        self.rows = []

    def submit(self, rows) -> None:
        self.rows.extend(rows)

    def drain_available(self):
        return []


def config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        experiment=ExperimentConfig("demo", 1, str(tmp_path), str(tmp_path)),
        dataset=DatasetConfig(limit=1),
        retrieval=RetrievalConfig(),
        model=ModelConfig(model_path="unused"),
        runtime=RuntimeConfig(),
        judge=JudgeConfig(enabled=True),
        training=TrainingConfig(group_size=1),
        collection=CollectionConfig(train_task_count=1),
        rollout=RolloutConfig(backend="transformers"),
    )


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def raw_row() -> dict:
    return {
        "policy_checkpoint_id": "iteration-00000",
        "rollout_split": "train",
        "query_id": "q1",
        "rollout_index": 0,
        "sampling_profile_id": "profile",
        "turn_records": [],
        "trajectory_records": [],
    }


def test_resume_submits_existing_raw_row_when_judged_row_is_missing(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    judged_path = tmp_path / "judged.jsonl"
    write_rows(raw_path, [raw_row()])
    judge = RecordingJudge()

    _collect_split_streaming(
        config=config(tmp_path),
        checkpoint_id="iteration-00000",
        checkpoint_path=tmp_path / "iteration-00000",
        generator=None,
        backend=None,
        judge_client=judge,
        split="train",
        examples=[QueryExample(query_id="q1", query="question", answer="answer")],
        raw_output_path=raw_path,
        judged_output_paths={"train": judged_path},
        judged_keys={"train": set()},
        sampling_profile={},
        profile_id="profile",
        group_size=1,
        sample_seed=1,
        resume=True,
        cache_client=None,
        cached_output_path=None,
        cached_keys=set(),
        cache_submitted_keys=set(),
    )
    assert [row["query_id"] for row in judge.rows] == ["q1"]


def test_resume_submits_existing_judged_row_when_cache_is_missing(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    judged_path = tmp_path / "judged.jsonl"
    row = raw_row()
    judged = {**row, "turn_rewards": {}, "judge": {"outcome": "wrong_answer"}}
    write_rows(raw_path, [row])
    write_rows(judged_path, [judged])
    cache = RecordingCache()

    _collect_split_streaming(
        config=config(tmp_path),
        checkpoint_id="iteration-00000",
        checkpoint_path=tmp_path / "iteration-00000",
        generator=None,
        backend=None,
        judge_client=None,
        split="train",
        examples=[QueryExample(query_id="q1", query="question", answer="answer")],
        raw_output_path=raw_path,
        judged_output_paths={"train": judged_path},
        judged_keys={"train": {("q1", 0)}},
        sampling_profile={},
        profile_id="profile",
        group_size=1,
        sample_seed=1,
        resume=True,
        cache_client=cache,
        cached_output_path=tmp_path / "cached.jsonl",
        cached_keys=set(),
        cache_submitted_keys=set(),
    )
    assert [cached_row["query_id"] for cached_row in cache.rows] == ["q1"]

