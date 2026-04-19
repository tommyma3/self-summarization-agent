from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TypeVar


SampleT = TypeVar("SampleT")


def _sample_query_id(sample: object) -> str:
    if isinstance(sample, Mapping):
        query_id = sample.get("query_id")
    else:
        query_id = getattr(sample, "query_id", None)
    if not isinstance(query_id, str):
        raise ValueError(f"RL sample is missing a string query_id: {query_id!r}")
    return query_id


def group_samples_by_query(samples: Iterable[SampleT]) -> dict[str, list[SampleT]]:
    grouped: dict[str, list[SampleT]] = defaultdict(list)
    for sample in samples:
        grouped[_sample_query_id(sample)].append(sample)
    return dict(grouped)
