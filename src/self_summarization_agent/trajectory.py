from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any


TOKEN_CACHE_VERSION = 4
TOKEN_CACHE_FIELD = "training_cache"
TRAJECTORY_SCHEMA_VERSION = 3
COLLECTION_TOKEN_VERSION = 1


@dataclass(slots=True)
class RLSample:
    query_id: str
    turn_id: str
    prompt: str
    completion: str
    reward: float
    trainable_kind: str
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    collection_full_token_ids: list[int] | None = None
    collection_assistant_token_mask: list[bool] | None = None
    rollout_id: str | None = None
    input_ids: list[int] | None = None
    labels: list[int] | None = None
    completion_mask: list[bool] | None = None
    reference_logprob: float | None = None
    reference_logprobs: list[float] | None = None

    @property
    def has_training_cache(self) -> bool:
        return (
            self.input_ids is not None
            and self.labels is not None
            and self.completion_mask is not None
            and self.reference_logprob is not None
        )

    @property
    def has_token_reference_logprobs(self) -> bool:
        return self.has_training_cache and self.reference_logprobs is not None

    @property
    def has_exact_collection_tokens(self) -> bool:
        return (
            self.collection_full_token_ids is not None
            and self.collection_assistant_token_mask is not None
        )


def _validate_int_list(value: Any, *, field_name: str, turn_id: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"Trainable record {turn_id} has non-list {field_name}")
    output: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"Trainable record {turn_id} has invalid {field_name}[{index}]")
        output.append(item)
    return output


def _validate_bool_list(value: Any, *, field_name: str, turn_id: str) -> list[bool]:
    if not isinstance(value, list):
        raise ValueError(f"Trainable record {turn_id} has non-list {field_name}")
    output: list[bool] = []
    for index, item in enumerate(value):
        if not isinstance(item, bool):
            raise ValueError(f"Trainable record {turn_id} has invalid {field_name}[{index}]")
        output.append(item)
    return output


def _validate_float_list(value: Any, *, field_name: str, turn_id: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"Trainable record {turn_id} has non-list {field_name}")
    output: list[float] = []
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"Trainable record {turn_id} has invalid {field_name}[{index}]")
        if not math.isfinite(float(item)):
            raise ValueError(f"Trainable record {turn_id} has non-finite {field_name}[{index}]")
        output.append(float(item))
    return output


def is_training_cache_current(cache: object) -> bool:
    return (
        isinstance(cache, Mapping)
        and cache.get("version") == TOKEN_CACHE_VERSION
        and "reference_logprobs" in cache
    )


def _extract_training_cache(
    record: Mapping[str, object],
    *,
    turn_id: str,
) -> tuple[list[int] | None, list[int] | None, list[bool] | None, float | None, list[float] | None]:
    cache = record.get(TOKEN_CACHE_FIELD)
    if cache is None:
        return None, None, None, None, None
    if not isinstance(cache, Mapping):
        raise ValueError(f"Trainable record {turn_id} has non-object {TOKEN_CACHE_FIELD}")
    version = cache.get("version")
    if version != TOKEN_CACHE_VERSION:
        raise ValueError(
            f"Trainable record {turn_id} has unsupported training cache version: {version!r}; "
            f"expected {TOKEN_CACHE_VERSION}"
        )
    input_ids = _validate_int_list(cache.get("input_ids"), field_name="input_ids", turn_id=turn_id)
    labels = _validate_int_list(cache.get("labels"), field_name="labels", turn_id=turn_id)
    completion_mask = _validate_bool_list(
        cache.get("completion_mask"),
        field_name="completion_mask",
        turn_id=turn_id,
    )
    if len(input_ids) != len(labels) or len(labels) != len(completion_mask):
        raise ValueError(f"Trainable record {turn_id} has mismatched cached tensor lengths")
    if not any(completion_mask):
        raise ValueError(f"Trainable record {turn_id} has no cached completion tokens")
    reference_logprob = cache.get("reference_logprob")
    if not isinstance(reference_logprob, (int, float)) or isinstance(reference_logprob, bool):
        raise ValueError(f"Trainable record {turn_id} has non-numeric reference_logprob")
    if not math.isfinite(float(reference_logprob)):
        raise ValueError(f"Trainable record {turn_id} has non-finite reference_logprob")
    reference_logprobs = _validate_float_list(
        cache.get("reference_logprobs"),
        field_name="reference_logprobs",
        turn_id=turn_id,
    )
    if len(reference_logprobs) != len(completion_mask):
        raise ValueError(f"Trainable record {turn_id} has mismatched cached logprob length")
    return input_ids, labels, completion_mask, float(reference_logprob), reference_logprobs


def _validate_messages(value: object, *, record_id: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Trainable record {record_id} has invalid messages")
    messages: list[dict[str, Any]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise ValueError(f"Trainable record {record_id} has non-object messages[{index}]")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
            raise ValueError(f"Trainable record {record_id} has invalid messages[{index}]")
        copied = deepcopy(dict(message))
        reasoning = copied.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError(f"Trainable record {record_id} has invalid messages[{index}].reasoning_content")
        tool_calls = copied.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            raise ValueError(f"Trainable record {record_id} has invalid messages[{index}].tool_calls")
        if role == "tool" and not isinstance(copied.get("tool_call_id"), str):
            raise ValueError(f"Trainable record {record_id} has unlinked tool message[{index}]")
        messages.append(copied)
    if messages[0]["role"] != "system":
        raise ValueError(f"Trainable record {record_id} must begin with a system message")
    if not any(
        message["role"] == "assistant"
        and (message.get("content") or message.get("tool_calls") or message.get("reasoning_content"))
        for message in messages
    ):
        raise ValueError(f"Trainable record {record_id} has no assistant completion")
    return messages


def validate_trajectory_schema(records: object, *, context: str) -> None:
    if not isinstance(records, list):
        raise ValueError(f"{context} is missing trajectory records")
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"{context} trajectory {index} is not an object")
        version = record.get("schema_version")
        if version != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(
                f"{context} trajectory {index} has schema version {version!r}; "
                f"expected {TRAJECTORY_SCHEMA_VERSION}. Recollect it without --resume."
            )


def _extract_collection_tokens(
    record: Mapping[str, object],
    *,
    turn_id: str,
) -> tuple[list[int] | None, list[bool] | None]:
    payload = record.get("collection_tokens")
    if payload is None:
        return None, None
    if not isinstance(payload, Mapping):
        raise ValueError(f"Trainable record {turn_id} has non-object collection_tokens")
    if payload.get("version") != COLLECTION_TOKEN_VERSION:
        raise ValueError(
            f"Trainable record {turn_id} has unsupported collection token version: "
            f"{payload.get('version')!r}"
        )
    full_token_ids = _validate_int_list(
        payload.get("full_token_ids"),
        field_name="collection_tokens.full_token_ids",
        turn_id=turn_id,
    )
    assistant_token_mask = _validate_bool_list(
        payload.get("assistant_token_mask"),
        field_name="collection_tokens.assistant_token_mask",
        turn_id=turn_id,
    )
    if len(full_token_ids) != len(assistant_token_mask):
        raise ValueError(f"Trainable record {turn_id} has mismatched collection token lengths")
    if len(full_token_ids) < 2:
        raise ValueError(f"Trainable record {turn_id} has fewer than two collection tokens")
    if not any(assistant_token_mask[1:]):
        raise ValueError(f"Trainable record {turn_id} has no sampled collection tokens")
    generations = payload.get("generations")
    if not isinstance(generations, list) or not generations:
        raise ValueError(f"Trainable record {turn_id} has no exact collection generations")
    for generation_index, generation in enumerate(generations):
        if not isinstance(generation, Mapping):
            raise ValueError(
                f"Trainable record {turn_id} has invalid collection generation {generation_index}"
            )
        generation_prompt_ids = _validate_int_list(
            generation.get("prompt_token_ids"),
            field_name=f"collection_tokens.generations[{generation_index}].prompt_token_ids",
            turn_id=turn_id,
        )
        generation_completion_ids = _validate_int_list(
            generation.get("completion_token_ids"),
            field_name=f"collection_tokens.generations[{generation_index}].completion_token_ids",
            turn_id=turn_id,
        )
        generation_full_ids = _validate_int_list(
            generation.get("full_token_ids"),
            field_name=f"collection_tokens.generations[{generation_index}].full_token_ids",
            turn_id=turn_id,
        )
        if generation_full_ids != generation_prompt_ids + generation_completion_ids:
            raise ValueError(
                f"Trainable record {turn_id} has inconsistent collection generation {generation_index}"
            )
    final_generation = generations[-1]
    if list(final_generation["full_token_ids"]) != full_token_ids:
        raise ValueError(f"Trainable record {turn_id} final generation does not match full_token_ids")
    return full_token_ids, assistant_token_mask


def extract_trainable_samples(
    records: list[Mapping[str, object]],
    rewards: dict[str, float],
    *,
    rollout_id: str | None = None,
) -> list[RLSample]:
    validate_trajectory_schema(records, context="Trainable records")
    samples: list[RLSample] = []
    seen_record_ids: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"Trajectory record must be a mapping, got {type(record).__name__}")
        if record.get("kind") != "trajectory":
            raise ValueError(f"Unknown trajectory record kind: {record.get('kind')}")
        record_id = record.get("turn_id")
        if not isinstance(record_id, str):
            raise ValueError(f"Trajectory record has non-string turn_id: {record_id!r}")
        if record_id in seen_record_ids:
            raise ValueError(f"Duplicate trajectory record id found: {record_id}")
        seen_record_ids.add(record_id)
        query_id = record.get("query_id")
        if not isinstance(query_id, str):
            raise ValueError(f"Trainable record {record_id} has non-string query_id")
        messages = _validate_messages(record.get("messages"), record_id=record_id)
        if record_id not in rewards:
            raise ValueError(f"Missing reward for trainable record: {record_id}")
        reward = rewards[record_id]
        if not isinstance(reward, (int, float)) or isinstance(reward, bool):
            raise ValueError(f"Trainable record {record_id} has non-numeric reward")
        if not math.isfinite(float(reward)):
            raise ValueError(f"Trainable record {record_id} has non-finite reward")
        input_ids, labels, completion_mask, reference_logprob, reference_logprobs = _extract_training_cache(
            record,
            turn_id=record_id,
        )
        collection_full_token_ids, collection_assistant_token_mask = _extract_collection_tokens(
            record,
            turn_id=record_id,
        )
        raw_tools = record.get("tools")
        if raw_tools is not None and not isinstance(raw_tools, list):
            raise ValueError(f"Trainable record {record_id} has invalid tools")
        tools = deepcopy(raw_tools) if isinstance(raw_tools, list) else None
        assistant_outputs = [
            str(message.get("content") or "")
            for message in messages
            if message["role"] == "assistant" and message.get("content")
        ]
        samples.append(
            RLSample(
                query_id=query_id,
                turn_id=record_id,
                prompt=str(record.get("prompt") or ""),
                completion=str(record.get("completion") or "\n".join(assistant_outputs)),
                reward=float(reward),
                trainable_kind=str(record.get("termination_kind") or "trajectory"),
                messages=messages,
                tools=tools,
                collection_full_token_ids=collection_full_token_ids,
                collection_assistant_token_mask=collection_assistant_token_mask,
                rollout_id=rollout_id,
                input_ids=input_ids,
                labels=labels,
                completion_mask=completion_mask,
                reference_logprob=reference_logprob,
                reference_logprobs=reference_logprobs,
            )
        )
    unknown_reward_ids = sorted(set(rewards) - seen_record_ids)
    if unknown_reward_ids:
        raise ValueError(f"Reward ids do not match any trajectory record: {', '.join(unknown_reward_ids)}")
    return samples


def _coerce_token_ids(value: object) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise ValueError("Tokenizer did not return a token-id list")
    return [int(token_id) for token_id in value]


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int], *, start: int) -> int | None:
    if not needle:
        return None
    last_start = len(haystack) - len(needle)
    for index in range(max(0, start), last_start + 1):
        if list(haystack[index : index + len(needle)]) == list(needle):
            return index
    return None


def _render_fallback_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"### {message['role'].upper()}\n{message['content']}" for message in messages)


def tokenize_interval_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    max_sequence_length: int | None,
    sample_id: str,
    enable_thinking: bool | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[list[int], list[int], list[bool]]:
    """Tokenize one append-only interval and mask only assistant content tokens."""
    full_ids: list[int]
    assistant_token_mask: list[bool] | None = None
    if getattr(tokenizer, "chat_template", None):
        rendered = None
        template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        if tools:
            template_kwargs["tools"] = tools
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
                **template_kwargs,
            )
        except (TypeError, ValueError):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    **template_kwargs,
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
        if isinstance(rendered, Mapping):
            full_ids = _coerce_token_ids(rendered.get("input_ids"))
            raw_mask = rendered.get("assistant_masks", rendered.get("assistant_mask"))
            if raw_mask is not None:
                candidate_mask = [bool(value) for value in _coerce_token_ids(raw_mask)]
                if len(candidate_mask) == len(full_ids) and any(candidate_mask):
                    assistant_token_mask = candidate_mask
        else:
            full_ids = _coerce_token_ids(rendered)
    else:
        full_ids = list(tokenizer.encode(_render_fallback_messages(messages), add_special_tokens=False))

    if assistant_token_mask is None:
        assistant_token_mask = [False] * len(full_ids)
        search_start = 0
        for message in messages:
            if message["role"] != "assistant" or not message["content"]:
                continue
            content_ids = list(tokenizer.encode(message["content"], add_special_tokens=False))
            content_start = _find_subsequence(full_ids, content_ids, start=search_start)
            if content_start is None:
                raise ValueError(
                    f"Could not align assistant completion tokens for interval {sample_id}; "
                    "the tokenizer chat template must expose an assistant mask"
                )
            for index in range(content_start, content_start + len(content_ids)):
                assistant_token_mask[index] = True
            search_start = content_start + len(content_ids)

    if max_sequence_length is not None and len(full_ids) - 1 > max_sequence_length:
        raise ValueError(
            f"Interval {sample_id} exceeds training.max_sequence_length: "
            f"{len(full_ids) - 1} > {max_sequence_length}; interval prefixes are never left-truncated"
        )
    if len(full_ids) <= 1:
        raise ValueError(f"Interval {sample_id} tokenized to fewer than two tokens")
    input_ids = full_ids[:-1]
    labels = full_ids[1:]
    completion_mask = assistant_token_mask[1:]
    if not any(completion_mask):
        raise ValueError(f"Interval {sample_id} contains no trainable assistant tokens")
    return input_ids, labels, completion_mask
