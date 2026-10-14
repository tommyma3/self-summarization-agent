from dataclasses import replace
import json
from pathlib import Path
import random

import pytest

from self_summarization_agent.config import DatasetConfig, load_train_config
from self_summarization_agent.dataset import load_query_examples, split_train_eval_examples


def write_examples(path: Path, ids: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"query_id": qid, "query": f"question {qid}", "answer": "done"}) + "\n"
                for qid in ids),
        encoding="utf-8",
    )


def load_split(config):
    examples = load_query_examples(
        config.experiment.bc_plus_root, config.dataset,
        require_answers=True, seed=config.experiment.seed,
    )
    return tuple(
        [example.query_id for example in split]
        for split in split_train_eval_examples(
            examples, train_limit=config.dataset.train_limit, eval_limit=config.dataset.eval_limit,
        )
    )


def test_split_is_shared_across_methods_runs_and_file_order(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    ids = [str(i) for i in range(60)]
    write_examples(path, ids)
    configs = [load_train_config(Path("configs/train") / name)
               for name in ("default.yaml", "compact_value_mc.yaml")]
    for config in configs:
        config.dataset = replace(
            config.dataset, decrypted_path=str(path), split_seed=42,
            offset=3, limit=50, train_limit=35, eval_limit=15,
        )
    configs[0].dataset.shuffle = False
    configs[1].dataset.shuffle = True
    configs[1].experiment.seed = 987
    configs[1].experiment.name = "independent-value-mc-run"

    before = random.getstate()
    expected = load_split(configs[0])
    write_examples(path, list(reversed(ids)))
    assert load_split(configs[1]) == expected
    assert random.getstate() == before
    assert len(expected[0]) == 35
    assert len(expected[1]) == 15
    assert set(expected[0]).isdisjoint(expected[1])

    configs[1].dataset.split_seed = 43
    changed = load_split(configs[1])
    assert set(changed[1]) != set(expected[1])
    assert set(changed[0]).isdisjoint(changed[1])


def test_partition_rejects_duplicate_query_ids(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_examples(path, ["q1", "q2", "q1"])
    with pytest.raises(ValueError, match="unique query_id"):
        load_query_examples(tmp_path, DatasetConfig(decrypted_path=str(path), train_limit=2, eval_limit=1),
                            require_answers=True, seed=0)


def test_unpartitioned_loading_keeps_existing_shuffle_behavior(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    ids = ["z", "a", "b", "c"]
    write_examples(path, ids)
    config = DatasetConfig(decrypted_path=str(path), split_seed=42)
    loaded = load_query_examples(tmp_path, config, require_answers=True, seed=5)
    assert [example.query_id for example in loaded] == ids
    config.shuffle = True
    expected = ids.copy()
    random.Random(5).shuffle(expected)
    loaded = load_query_examples(tmp_path, config, require_answers=True, seed=5)
    assert [example.query_id for example in loaded] == expected


@pytest.mark.parametrize("seed", [None, True, 1.5, "42"])
def test_split_seed_requires_an_integer(seed) -> None:
    with pytest.raises(ValueError, match="dataset.split_seed must be an integer"):
        DatasetConfig(split_seed=seed)


def test_split_seed_config_default_and_override() -> None:
    assert DatasetConfig().split_seed == 0
    config = load_train_config("configs/train/default.yaml", overrides={"dataset.split_seed": 123})
    assert config.dataset.split_seed == 123
