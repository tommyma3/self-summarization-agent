from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from self_summarization_agent.models import RuntimeResult
from self_summarization_agent.runtime import EpisodeRuntime


def seed_everything(seed: int) -> None:
    random.seed(seed)


def utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def iter_batches(items: list[Any], batch_size: int):
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_jsonl(path: str | Path, payload: Any) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def serialize_runtime_result(result: RuntimeResult, *, query_text: str, judge: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "query_id": result.query_id,
        "query": query_text,
        "status": result.status,
        "final_answer": result.final_answer,
        "summary_turns": list(result.summary_turns),
        "turn_records": list(result.turn_records),
        "turn_rewards": dict(result.turn_rewards),
        "retrieved_docids": list(result.retrieved_docids),
        "tool_call_counts": dict(result.tool_call_counts),
        "judge": judge,
    }


def build_runtime(generator: Any, backend: Any, runtime_config: Any) -> EpisodeRuntime:
    token_counter = getattr(generator, "count_tokens", None)
    if token_counter is None:
        token_counter = lambda text: len(text.split())
    return EpisodeRuntime(
        model=generator,
        backend=backend,
        context_threshold_tokens=runtime_config.context_threshold_tokens,
        max_context_tokens=runtime_config.max_context_tokens,
        max_tool_calls=runtime_config.tool_budget,
        token_counter=token_counter,
    )


def dataclass_to_jsonable(instance: Any) -> Any:
    if is_dataclass(instance):
        return asdict(instance)
    return instance
