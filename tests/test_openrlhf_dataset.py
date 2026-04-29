import json
from pathlib import Path

from self_summarization_agent.config import (
    DatasetConfig,
    ExperimentConfig,
    JudgeConfig,
    ModelConfig,
    RetrievalConfig,
    RuntimeConfig,
    TrainConfig,
    TrainingConfig,
)
from self_summarization_agent.openrlhf_dataset import export_openrlhf_prompt_data


def test_export_openrlhf_prompt_data_writes_prompt_and_json_label(tmp_path: Path) -> None:
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        json.dumps({"query_id": "q1", "query": "question", "answer": "answer"}) + "\n",
        encoding="utf-8",
    )
    config = TrainConfig(
        experiment=ExperimentConfig(name="demo", seed=1, output_root=str(tmp_path), bc_plus_root=str(tmp_path)),
        dataset=DatasetConfig(decrypted_path=str(data_path), train_limit=1),
        retrieval=RetrievalConfig(index_path="unused"),
        model=ModelConfig(model_path="unused"),
        runtime=RuntimeConfig(),
        judge=JudgeConfig(),
        training=TrainingConfig(),
    )

    output_path = export_openrlhf_prompt_data(config, tmp_path / "openrlhf.jsonl")

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "prompt": "question",
            "label": '{"answer": "answer", "query_id": "q1"}',
            "query_id": "q1",
            "answer": "answer",
        }
    ]
