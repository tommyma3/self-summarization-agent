from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import logging
import math
from typing import Any


LOGGER = logging.getLogger(__name__)


class ProviderHistoryRewriteError(ValueError):
    """Exact collection tokens show the provider re-rendered interval history.

    The chat template normalized an earlier assistant turn (for example an
    unclosed or misspelled think block), so a later server-rendered prompt no
    longer extends the previously sampled tokens. Training on such an interval
    would condition on tokens the policy never saw; the record is excluded
    from trainable samples instead.
    """


TOKEN_CACHE_VERSION = 6
TOKEN_CACHE_FIELD = "training_cache"
TRAJECTORY_SCHEMA_VERSION = 3
COLLECTION_TOKEN_VERSION = 2
LEGACY_COLLECTION_TOKEN_VERSION = 1
REFERENCE_LOGPROB_SOURCE_POLICY_RESCORE = "policy_rescore"
REFERENCE_LOGPROB_SOURCE_VLLM_ROLLOUT = "vllm_raw_rollout"
LOSS_MASK_POLICY_ALL_ASSISTANT = "all_assistant"
LOSS_MASK_POLICY_TOOL_CALLS_ONLY = "tool_calls_only"
_REFERENCE_LOGPROB_SOURCES = {
    REFERENCE_LOGPROB_SOURCE_POLICY_RESCORE,
    REFERENCE_LOGPROB_SOURCE_VLLM_ROLLOUT,
}


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
    state_prefix_length: int | None = None

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


def loss_mask_policy(*, train_compaction_tokens: bool) -> str:
    return (
        LOSS_MASK_POLICY_ALL_ASSISTANT
        if train_compaction_tokens
        else LOSS_MASK_POLICY_TOOL_CALLS_ONLY
    )


def is_training_cache_current(
    cache: object,
    *,
    train_compaction_tokens: bool = True,
) -> bool:
    if not isinstance(cache, Mapping):
        return False
    source = cache.get("reference_logprob_source")
    cached_loss_mask_policy = cache.get(
        "loss_mask_policy",
        LOSS_MASK_POLICY_ALL_ASSISTANT,
    )
    return (
        cache.get("version") == TOKEN_CACHE_VERSION
        and "reference_logprobs" in cache
        and isinstance(cache.get("state_prefix_length"), int)
        and not isinstance(cache.get("state_prefix_length"), bool)
        and source in _REFERENCE_LOGPROB_SOURCES
        and (
            source != REFERENCE_LOGPROB_SOURCE_VLLM_ROLLOUT
            or cache.get("logprobs_mode") == "raw_logprobs"
        )
        and cached_loss_mask_policy
        == loss_mask_policy(train_compaction_tokens=train_compaction_tokens)
    )


def _record_ends_with_summary_generation(record: Mapping[str, object]) -> bool:
    turn_ids = record.get("turn_ids")
    return (
        isinstance(turn_ids, list)
        and bool(turn_ids)
        and isinstance(turn_ids[-1], str)
        and turn_ids[-1].startswith("summary-")
    )


def apply_training_loss_mask(
    record: Mapping[str, object],
    cache: Mapping[str, object],
    *,
    train_compaction_tokens: bool,
    retain_critic_only_state: bool = False,
) -> dict[str, Any] | None:
    """Apply the configured gradient mask without changing collected tokens.

    Returns ``None`` when masking the compaction summary leaves no trainable
    tokens: the record has nothing to optimize under the configured policy
    and must be excluded from training.
    """

    updated = deepcopy(dict(cache))
    policy = loss_mask_policy(train_compaction_tokens=train_compaction_tokens)
    updated["loss_mask_policy"] = policy
    if train_compaction_tokens or not _record_ends_with_summary_generation(record):
        return updated

    collection_tokens = record.get("collection_tokens")
    if not isinstance(collection_tokens, Mapping):
        raise ValueError(
            f"Trajectory {record.get('turn_id')} cannot mask compaction tokens without "
            "authoritative collection tokens"
        )
    generations = collection_tokens.get("generations")
    if not isinstance(generations, list) or not generations:
        raise ValueError(
            f"Trajectory {record.get('turn_id')} cannot identify its summary generation"
        )
    final_generation = generations[-1]
    if not isinstance(final_generation, Mapping):
        raise ValueError(
            f"Trajectory {record.get('turn_id')} has an invalid final generation"
        )
    prompt_ids = final_generation.get("prompt_token_ids")
    completion_ids = final_generation.get("completion_token_ids")
    full_token_ids = collection_tokens.get("full_token_ids")
    if (
        not isinstance(prompt_ids, list)
        or not isinstance(completion_ids, list)
        or not completion_ids
        or not isinstance(full_token_ids, list)
        or prompt_ids + completion_ids != full_token_ids
    ):
        raise ValueError(
            f"Trajectory {record.get('turn_id')} summary generation is not the exact final prefix"
        )

    raw_mask = updated.get("completion_mask")
    raw_reference_logprobs = updated.get("reference_logprobs")
    if (
        not isinstance(raw_mask, list)
        or not isinstance(raw_reference_logprobs, list)
        or len(raw_mask) != len(full_token_ids) - 1
        or len(raw_reference_logprobs) != len(raw_mask)
    ):
        raise ValueError(
            f"Trajectory {record.get('turn_id')} cache does not align with exact collection tokens"
        )

    completion_mask = [bool(value) for value in raw_mask]
    summary_start = len(prompt_ids)
    for full_position in range(summary_start, len(full_token_ids)):
        if full_position <= 0 or not completion_mask[full_position - 1]:
            raise ValueError(
                f"Trajectory {record.get('turn_id')} summary tokens are not fully trainable "
                "in the source mask"
            )
        completion_mask[full_position - 1] = False
    if not any(completion_mask):
        if not retain_critic_only_state:
            return None
        updated["completion_mask"] = completion_mask
        updated["reference_logprob"] = 0.0
        updated["critic_only"] = True
        return updated

    selected_logprobs = [
        float(logprob)
        for logprob, is_trainable in zip(raw_reference_logprobs, completion_mask)
        if is_trainable
    ]
    updated["completion_mask"] = completion_mask
    updated["reference_logprob"] = sum(selected_logprobs) / len(selected_logprobs)
    return updated


def record_has_training_tokens(
    record: Mapping[str, object],
    *,
    train_compaction_tokens: bool,
    retain_critic_only_state: bool = False,
) -> bool:
    """Whether a record keeps trainable tokens under the loss-mask policy.

    Mirrors ``apply_training_loss_mask``: under the tool-calls-only policy a
    summary-ending record keeps trainable content only when its assistant
    mask has a True position before the final (summary) generation. Returns
    True (fail-closed) when the record cannot be classified, so the masking
    path raises its own validation error instead of silently dropping data.
    """

    if retain_critic_only_state:
        return True
    if train_compaction_tokens or not _record_ends_with_summary_generation(record):
        return True
    collection_tokens = record.get("collection_tokens")
    if not isinstance(collection_tokens, Mapping):
        return True
    generations = collection_tokens.get("generations")
    if not isinstance(generations, list) or not generations:
        return True
    final_generation = generations[-1]
    if not isinstance(final_generation, Mapping):
        return True
    prompt_ids = final_generation.get("prompt_token_ids")
    full_token_ids = collection_tokens.get("full_token_ids")
    assistant_token_mask = collection_tokens.get("assistant_token_mask")
    if (
        not isinstance(prompt_ids, list)
        or not prompt_ids
        or not isinstance(full_token_ids, list)
        or not isinstance(assistant_token_mask, list)
        or len(assistant_token_mask) != len(full_token_ids)
        or len(prompt_ids) > len(full_token_ids)
    ):
        return True
    return any(assistant_token_mask[1 : len(prompt_ids)])


def _extract_training_cache(
    record: Mapping[str, object],
    *,
    turn_id: str,
) -> tuple[
    list[int] | None,
    list[int] | None,
    list[bool] | None,
    float | None,
    list[float] | None,
    int | None,
]:
    cache = record.get(TOKEN_CACHE_FIELD)
    if cache is None:
        return None, None, None, None, None, None
    if not isinstance(cache, Mapping):
        raise ValueError(f"Trainable record {turn_id} has non-object {TOKEN_CACHE_FIELD}")
    version = cache.get("version")
    if version != TOKEN_CACHE_VERSION:
        raise ValueError(
            f"Trainable record {turn_id} has unsupported training cache version: {version!r}; "
            f"expected {TOKEN_CACHE_VERSION}"
        )
    reference_logprob_source = cache.get("reference_logprob_source")
    if reference_logprob_source not in _REFERENCE_LOGPROB_SOURCES:
        raise ValueError(
            f"Trainable record {turn_id} has unsupported reference_logprob_source: "
            f"{reference_logprob_source!r}"
        )
    if (
        reference_logprob_source == REFERENCE_LOGPROB_SOURCE_VLLM_ROLLOUT
        and cache.get("logprobs_mode") != "raw_logprobs"
    ):
        raise ValueError(
            f"Trainable record {turn_id} has non-raw rollout reference logprobs"
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
    if not any(completion_mask) and cache.get("critic_only") is not True:
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
    state_prefix_length = cache.get("state_prefix_length")
    if (
        not isinstance(state_prefix_length, int)
        or isinstance(state_prefix_length, bool)
        or state_prefix_length < 1
        or state_prefix_length > len(input_ids)
    ):
        raise ValueError(f"Trainable record {turn_id} has invalid state_prefix_length")
    return (
        input_ids,
        labels,
        completion_mask,
        float(reference_logprob),
        reference_logprobs,
        state_prefix_length,
    )


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
) -> tuple[list[int] | None, list[bool] | None, int | None]:
    payload = record.get("collection_tokens")
    if payload is None:
        return None, None, None
    if not isinstance(payload, Mapping):
        raise ValueError(f"Trainable record {turn_id} has non-object collection_tokens")
    if payload.get("version") not in {
        LEGACY_COLLECTION_TOKEN_VERSION,
        COLLECTION_TOKEN_VERSION,
    }:
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
        if generation_index > 0:
            previous_full_ids = generations[generation_index - 1]["full_token_ids"]
            if generation_prompt_ids[: len(previous_full_ids)] != previous_full_ids:
                raise ProviderHistoryRewriteError(
                    f"Trainable record {turn_id} collection generation {generation_index} rewrote "
                    "the interval history instead of extending it"
                )
    final_generation = generations[-1]
    if list(final_generation["full_token_ids"]) != full_token_ids:
        raise ValueError(f"Trainable record {turn_id} final generation does not match full_token_ids")
    first_prompt_ids = list(generations[0]["prompt_token_ids"])
    if not first_prompt_ids or full_token_ids[: len(first_prompt_ids)] != first_prompt_ids:
        raise ProviderHistoryRewriteError(
            f"Trainable record {turn_id} first state prompt is not an exact prefix"
        )
    if len(first_prompt_ids) > len(full_token_ids) - 1:
        raise ValueError(f"Trainable record {turn_id} state prefix has no following sampled token")
    return full_token_ids, assistant_token_mask, len(first_prompt_ids)


def build_rollout_native_training_cache(
    collection_tokens: object,
) -> dict[str, Any] | None:
    """Build a cache from authoritative raw vLLM decode logprobs.

    This is deliberately fail-closed. The fast path is valid only when every
    generated completion carries finite raw-policy logprobs and every complete
    generation token sequence is an exact prefix of the finalized interval.
    Later appended tokens then cannot change any earlier autoregressive
    probability. Callers must fall back to policy rescoring when this returns
    ``None``.
    """

    if not isinstance(collection_tokens, Mapping):
        return None
    if collection_tokens.get("version") != COLLECTION_TOKEN_VERSION:
        return None
    full_token_ids = collection_tokens.get("full_token_ids")
    assistant_token_mask = collection_tokens.get("assistant_token_mask")
    generations = collection_tokens.get("generations")
    if (
        not isinstance(full_token_ids, list)
        or not isinstance(assistant_token_mask, list)
        or not isinstance(generations, list)
        or not generations
        or len(full_token_ids) < 2
        or len(full_token_ids) != len(assistant_token_mask)
    ):
        return None
    if not all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in full_token_ids):
        return None
    if not all(isinstance(mask_value, bool) for mask_value in assistant_token_mask):
        return None

    expected_assistant_positions = {
        index
        for index, is_assistant in enumerate(assistant_token_mask)
        if index > 0 and is_assistant
    }
    if not expected_assistant_positions:
        return None

    reference_logprobs = [0.0] * (len(full_token_ids) - 1)
    covered_assistant_positions: set[int] = set()
    for generation in generations:
        if not isinstance(generation, Mapping):
            return None
        prompt_ids = generation.get("prompt_token_ids")
        completion_ids = generation.get("completion_token_ids")
        completion_logprobs = generation.get("completion_token_logprobs")
        if generation.get("logprobs_mode") != "raw_logprobs":
            return None
        if (
            not isinstance(prompt_ids, list)
            or not isinstance(completion_ids, list)
            or not isinstance(completion_logprobs, list)
            or len(completion_ids) != len(completion_logprobs)
            or not completion_ids
        ):
            return None
        if not all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in prompt_ids):
            return None
        if not all(
            isinstance(token_id, int) and not isinstance(token_id, bool)
            for token_id in completion_ids
        ):
            return None
        numeric_logprobs: list[float] = []
        for logprob in completion_logprobs:
            if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
                return None
            numeric_logprob = float(logprob)
            if not math.isfinite(numeric_logprob):
                return None
            numeric_logprobs.append(numeric_logprob)

        generation_full_ids = prompt_ids + completion_ids
        if full_token_ids[: len(generation_full_ids)] != generation_full_ids:
            return None
        completion_start = len(prompt_ids)
        for offset, logprob in enumerate(numeric_logprobs):
            full_position = completion_start + offset
            if (
                full_position <= 0
                or full_position >= len(full_token_ids)
                or not assistant_token_mask[full_position]
                or full_position in covered_assistant_positions
            ):
                return None
            reference_logprobs[full_position - 1] = logprob
            covered_assistant_positions.add(full_position)

    if covered_assistant_positions != expected_assistant_positions:
        return None
    first_generation = generations[0]
    first_prompt_ids = first_generation.get("prompt_token_ids")
    if (
        not isinstance(first_prompt_ids, list)
        or not first_prompt_ids
        or full_token_ids[: len(first_prompt_ids)] != first_prompt_ids
        or len(first_prompt_ids) > len(full_token_ids) - 1
    ):
        return None
    completion_mask = list(assistant_token_mask[1:])
    masked_logprobs = [
        reference_logprobs[index]
        for index, is_assistant in enumerate(completion_mask)
        if is_assistant
    ]
    return {
        "version": TOKEN_CACHE_VERSION,
        "input_ids": list(full_token_ids[:-1]),
        "labels": list(full_token_ids[1:]),
        "completion_mask": completion_mask,
        "reference_logprob": sum(masked_logprobs) / len(masked_logprobs),
        "reference_logprobs": reference_logprobs,
        "reference_logprob_source": REFERENCE_LOGPROB_SOURCE_VLLM_ROLLOUT,
        "logprobs_mode": "raw_logprobs",
        "state_prefix_length": len(first_prompt_ids),
    }


def extract_trainable_samples(
    records: list[Mapping[str, object]],
    rewards: dict[str, float],
    *,
    rollout_id: str | None = None,
) -> list[RLSample]:
    validate_trajectory_schema(records, context="Trainable records")
    samples: list[RLSample] = []
    seen_record_ids: set[str] = set()
    skipped_history_rewrite_ids: set[str] = set()
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
        (
            input_ids,
            labels,
            completion_mask,
            reference_logprob,
            reference_logprobs,
            cached_state_prefix_length,
        ) = _extract_training_cache(
            record,
            turn_id=record_id,
        )
        try:
            (
                collection_full_token_ids,
                collection_assistant_token_mask,
                collection_state_prefix_length,
            ) = _extract_collection_tokens(
                record,
                turn_id=record_id,
            )
        except ProviderHistoryRewriteError as exc:
            LOGGER.warning("Skipping trainable record %s: %s", record_id, exc)
            skipped_history_rewrite_ids.add(record_id)
            continue
        state_prefix_length = cached_state_prefix_length or collection_state_prefix_length
        if (
            cached_state_prefix_length is not None
            and collection_state_prefix_length is not None
            and cached_state_prefix_length != collection_state_prefix_length
        ):
            raise ValueError(f"Trainable record {record_id} has mismatched value state anchors")
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
                state_prefix_length=state_prefix_length,
            )
        )
    unknown_reward_ids = sorted(set(rewards) - seen_record_ids - skipped_history_rewrite_ids)
    if unknown_reward_ids:
        raise ValueError(f"Reward ids do not match any trajectory record: {', '.join(unknown_reward_ids)}")
    return samples


def extract_training_samples(
    records: list[Mapping[str, object]],
    rewards: dict[str, float],
    *,
    rollout_id: str | None = None,
    train_compaction_tokens: bool = True,
    retain_critic_only_states: bool = False,
) -> list[RLSample]:
    """Extract the samples that are trainable under the loss-mask policy.

    Identical to :func:`extract_trainable_samples` when training compaction
    tokens; under the tool-calls-only policy it drops records whose mask
    would empty out (summary-only intervals with no tool-call content).
    """

    samples = extract_trainable_samples(records, rewards, rollout_id=rollout_id)
    if train_compaction_tokens or retain_critic_only_states:
        return samples
    record_by_turn_id = {
        record.get("turn_id"): record
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("turn_id"), str)
    }
    return [
        sample
        for sample in samples
        if record_by_turn_id.get(sample.turn_id) is None
        or record_has_training_tokens(
            record_by_turn_id[sample.turn_id],
            train_compaction_tokens=train_compaction_tokens,
        )
    ]


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

    if max_sequence_length is not None and len(full_ids) - 1 > max_sequence_length:
        raise ValueError(
            f"Interval {sample_id} exceeds training.max_sequence_length: "
            f"{len(full_ids) - 1} > {max_sequence_length}; interval prefixes are never left-truncated"
        )

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

    if len(full_ids) <= 1:
        raise ValueError(f"Interval {sample_id} tokenized to fewer than two tokens")
    input_ids = full_ids[:-1]
    labels = full_ids[1:]
    completion_mask = assistant_token_mask[1:]
    if not any(completion_mask):
        raise ValueError(f"Interval {sample_id} contains no trainable assistant tokens")
    return input_ids, labels, completion_mask
