from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from self_summarization_agent.config import DatasetConfig


@dataclass(slots=True)
class QueryExample:
    query_id: str
    query: str
    answer: str | None = None


def default_decrypted_path(bc_plus_root: str | Path) -> Path:
    return Path(bc_plus_root) / "data" / "browsecomp_plus_decrypted.jsonl"


def default_queries_tsv_path(bc_plus_root: str | Path) -> Path:
    return Path(bc_plus_root) / "topics-qrels" / "queries.tsv"


def load_query_examples(
    bc_plus_root: str | Path,
    dataset_config: DatasetConfig,
    *,
    require_answers: bool,
    seed: int,
) -> list[QueryExample]:
    decrypted_path = (
        Path(dataset_config.decrypted_path)
        if dataset_config.decrypted_path
        else default_decrypted_path(bc_plus_root)
    )
    if decrypted_path.exists():
        examples = _load_decrypted_examples(decrypted_path)
    else:
        if require_answers:
            raise FileNotFoundError(
                f"Decrypted dataset not found at {decrypted_path}. Training and judging require answers."
            )
        queries_tsv_path = (
            Path(dataset_config.queries_tsv_path)
            if dataset_config.queries_tsv_path
            else default_queries_tsv_path(bc_plus_root)
        )
        examples = _load_query_tsv(queries_tsv_path)
    return slice_examples(
        examples,
        offset=dataset_config.offset,
        limit=dataset_config.limit,
        shuffle=dataset_config.shuffle,
        seed=seed,
    )


def _load_decrypted_examples(path: Path) -> list[QueryExample]:
    examples: list[QueryExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            examples.append(
                QueryExample(
                    query_id=str(record["query_id"]),
                    query=str(record["query"]),
                    answer=str(record["answer"]) if record.get("answer") is not None else None,
                )
            )
    return examples


def _load_query_tsv(path: Path) -> list[QueryExample]:
    if not path.exists():
        raise FileNotFoundError(f"Query TSV not found at {path}")
    examples: list[QueryExample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) != 2:
                raise ValueError(f"Malformed query TSV row: {row!r}")
            examples.append(QueryExample(query_id=row[0].strip(), query=row[1].strip()))
    return examples


def slice_examples(
    examples: list[QueryExample],
    *,
    offset: int,
    limit: int | None,
    shuffle: bool,
    seed: int,
) -> list[QueryExample]:
    sliced = list(examples)
    if shuffle:
        random.Random(seed).shuffle(sliced)
    if offset:
        sliced = sliced[offset:]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced
