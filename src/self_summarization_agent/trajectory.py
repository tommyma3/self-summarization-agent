from dataclasses import dataclass
import math
from collections.abc import Mapping
from typing import Any


TOKEN_CACHE_VERSION = 1
TOKEN_CACHE_FIELD = "training_cache"


@dataclass(slots=True)
class RLSample:
    query_id: str
    turn_id: str
    prompt: str
    completion: str
    reward: float
    trainable_kind: str
    input_ids: list[int] | None = None
    labels: list[int] | None = None
    completion_mask: list[bool] | None = None
    reference_logprob: float | None = None

    @property
    def has_training_cache(self) -> bool:
        return (
            self.input_ids is not None
            and self.labels is not None
            and self.completion_mask is not None
            and self.reference_logprob is not None
        )


def _validate_int_list(value: Any, *, field_name: str, turn_id: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"Trainable turn_id {turn_id} has non-list {field_name}")
    output: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"Trainable turn_id {turn_id} has invalid {field_name}[{index}]")
        output.append(item)
    return output


def _validate_bool_list(value: Any, *, field_name: str, turn_id: str) -> list[bool]:
    if not isinstance(value, list):
        raise ValueError(f"Trainable turn_id {turn_id} has non-list {field_name}")
    output: list[bool] = []
    for index, item in enumerate(value):
        if not isinstance(item, bool):
            raise ValueError(f"Trainable turn_id {turn_id} has invalid {field_name}[{index}]")
        output.append(item)
    return output


def _extract_training_cache(
    turn: Mapping[str, object],
    *,
    turn_id: str,
) -> tuple[list[int] | None, list[int] | None, list[bool] | None, float | None]:
    cache = turn.get(TOKEN_CACHE_FIELD)
    if cache is None:
        return None, None, None, None
    if not isinstance(cache, Mapping):
        raise ValueError(f"Trainable turn_id {turn_id} has non-object {TOKEN_CACHE_FIELD}")
    version = cache.get("version")
    if version != TOKEN_CACHE_VERSION:
        raise ValueError(
            f"Trainable turn_id {turn_id} has unsupported training cache version: {version!r}"
        )
    input_ids = _validate_int_list(cache.get("input_ids"), field_name="input_ids", turn_id=turn_id)
    labels = _validate_int_list(cache.get("labels"), field_name="labels", turn_id=turn_id)
    completion_mask = _validate_bool_list(
        cache.get("completion_mask"),
        field_name="completion_mask",
        turn_id=turn_id,
    )
    if len(input_ids) != len(labels) or len(labels) != len(completion_mask):
        raise ValueError(f"Trainable turn_id {turn_id} has mismatched cached tensor lengths")
    reference_logprob = cache.get("reference_logprob")
    if not isinstance(reference_logprob, (int, float)) or isinstance(reference_logprob, bool):
        raise ValueError(f"Trainable turn_id {turn_id} has non-numeric reference_logprob")
    if not math.isfinite(float(reference_logprob)):
        raise ValueError(f"Trainable turn_id {turn_id} has non-finite reference_logprob")
    return input_ids, labels, completion_mask, float(reference_logprob)


def extract_trainable_samples(turns: list[Mapping[str, object]], rewards: dict[str, float]) -> list[RLSample]:
    samples: list[RLSample] = []
    seen_turn_ids: set[str] = set()
    allowed_trainable_kinds = {"tool", "summary", "final_answer"}
    allowed_non_trainable_kinds: set[str] = set()
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise ValueError(f"Turn record must be a mapping, got {type(turn).__name__}")
        required_keys = {"kind", "turn_id"}
        missing_keys = sorted(required_keys - set(turn))
        if missing_keys:
            raise ValueError(f"Turn record is missing required keys: {', '.join(missing_keys)}")

        turn_kind = turn["kind"]
        turn_id = turn["turn_id"]
        if not isinstance(turn_id, str):
            raise ValueError(f"Turn record has non-string turn_id: {turn_id!r}")
        if turn_id in seen_turn_ids:
            raise ValueError(f"Duplicate turn_id found: {turn_id}")
        seen_turn_ids.add(turn_id)
        if turn_kind not in allowed_trainable_kinds and turn_kind not in allowed_non_trainable_kinds:
            raise ValueError(f"Unknown turn kind: {turn_kind}")
        if turn_kind not in allowed_trainable_kinds:
            continue
        for field_name in ("query_id", "prompt", "completion"):
            if field_name not in turn:
                raise ValueError(f"Trainable turn record is missing required key: {field_name}")
        query_id = turn["query_id"]
        prompt = turn["prompt"]
        completion = turn["completion"]
        if not isinstance(query_id, str):
            raise ValueError(f"Trainable turn_id {turn_id} has non-string query_id")
        if not isinstance(prompt, str):
            raise ValueError(f"Trainable turn_id {turn_id} has non-string prompt")
        if not isinstance(completion, str):
            raise ValueError(f"Trainable turn_id {turn_id} has non-string completion")
        if turn_id not in rewards:
            raise ValueError(f"Missing reward for trainable turn_id: {turn_id}")
        reward = rewards[turn_id]
        if not isinstance(reward, (int, float)) or isinstance(reward, bool):
            raise ValueError(f"Trainable turn_id {turn_id} has non-numeric reward")
        if not math.isfinite(float(reward)):
            raise ValueError(f"Trainable turn_id {turn_id} has non-finite reward")
        input_ids, labels, completion_mask, reference_logprob = _extract_training_cache(
            turn,
            turn_id=turn_id,
        )
        samples.append(
            RLSample(
                query_id=query_id,
                turn_id=turn_id,
                prompt=prompt,
                completion=completion,
                reward=float(reward),
                trainable_kind=turn_kind,
                input_ids=input_ids,
                labels=labels,
                completion_mask=completion_mask,
                reference_logprob=reference_logprob,
            )
        )
    unknown_reward_ids = sorted(set(rewards) - seen_turn_ids)
    if unknown_reward_ids:
        raise ValueError(f"Reward ids do not match any turn: {', '.join(unknown_reward_ids)}")
    return samples
