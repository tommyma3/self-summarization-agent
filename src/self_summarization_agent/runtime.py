import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

LOGGER = logging.getLogger(__name__)

from self_summarization_agent.backend import BrowseCompBackend, SearchResult
from self_summarization_agent.context import ContextManager
from self_summarization_agent.models import EpisodeState, Message, RuntimeResult
from self_summarization_agent.prompts import (
    ConversationPrompt,
    build_compacted_messages,
    build_forced_answer_prompt,
    build_initial_messages,
    format_tool_result,
    serialize_messages,
)
from self_summarization_agent.rewards import (
    apply_malformed_tool_penalty,
    apply_terminal_reward,
    trainable_turn_ids_from_records,
)


_JSON_DECODER = json.JSONDecoder()
_THINK_END_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)
_THINK_START_RE = re.compile(r"^\s*<think\b[^>]*>", flags=re.IGNORECASE)
_ACTION_TAGS = ("search", "document", "answer")
_ACTION_OPEN_RE = {
    tag: re.compile(rf"<\s*{tag}\s*>", flags=re.IGNORECASE) for tag in _ACTION_TAGS
}
_ACTION_BLOCK_RE = {
    tag: re.compile(rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>", flags=re.IGNORECASE | re.DOTALL)
    for tag in _ACTION_TAGS
}
_ACTION_CLOSE_RE = {
    tag: re.compile(rf"<\s*/\s*{tag}\s*>", flags=re.IGNORECASE) for tag in _ACTION_TAGS
}


@dataclass(frozen=True, slots=True)
class SummaryExtraction:
    thinking: str
    summary: str


@dataclass(frozen=True, slots=True)
class ThinkingExtraction:
    thinking: str
    remainder: str


class RuntimeModel(Protocol):
    def generate(self, prompt: str) -> str:
        ...


def _extract_completed_thinking(raw_output: str) -> ThinkingExtraction | None:
    think_end = _THINK_END_RE.search(raw_output)
    if think_end is None:
        return None
    thinking = raw_output[: think_end.start()]
    thinking = _THINK_START_RE.sub("", thinking).strip()
    remainder = raw_output[think_end.end() :].strip()
    return ThinkingExtraction(thinking=thinking, remainder=remainder)


def _iter_json_objects(text: str):
    cleaned = text.strip()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = _JSON_DECODER.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        yield parsed


def _action_text(raw_output: str) -> str:
    extracted = _extract_completed_thinking(raw_output)
    return extracted.remainder if extracted is not None else raw_output.strip()


def _trim_after_first_action_close(text: str) -> str:
    first_end: int | None = None
    for close_re in _ACTION_CLOSE_RE.values():
        match = close_re.search(text)
        if match is None:
            continue
        if first_end is None or match.end() < first_end:
            first_end = match.end()
    return text[:first_end] if first_end is not None else text


def _action_tag_counts(text: str) -> dict[str, int]:
    return {tag: len(open_re.findall(text)) for tag, open_re in _ACTION_OPEN_RE.items()}


def _contains_action_tag(text: str) -> bool:
    return any(_action_tag_counts(text).values())


def _parse_tag_tool_call(raw_output: str) -> tuple[dict[str, object], str] | None:
    action_text = _action_text(raw_output)
    counts = _action_tag_counts(action_text)
    if not any(counts.values()):
        return None
    if sum(counts.values()) != 1:
        return None

    trimmed = _trim_after_first_action_close(action_text)
    for tag, block_re in _ACTION_BLOCK_RE.items():
        match = block_re.search(trimmed)
        if match is None:
            continue
        value = match.group(1).strip()
        if tag == "search":
            normalized = {"tool_name": "search", "arguments": {"query": value}}
        elif tag == "document":
            normalized = {"tool_name": "get_document", "arguments": {"doc_id": value}}
        else:
            normalized = {"tool_name": "finish", "arguments": {"answer": value}}
        return normalized, json.dumps(normalized, ensure_ascii=False)
    return None


def _parse_json_tool_call(raw_output: str) -> tuple[dict[str, object], str] | None:
    extracted = _extract_completed_thinking(raw_output)
    if extracted is None:
        return None
    for candidate in _iter_json_objects(extracted.remainder):
        if not isinstance(candidate, dict):
            return None
        tool_name = candidate.get("tool_name")
        arguments = candidate.get("arguments")
        if isinstance(tool_name, str) and isinstance(arguments, dict):
            normalized = {"tool_name": tool_name, "arguments": arguments}
            return normalized, json.dumps(normalized, ensure_ascii=False)
        return None
    return None


def parse_model_tool_call(raw_output: str) -> tuple[dict[str, object], str] | None:
    tag_result = _parse_tag_tool_call(raw_output)
    if tag_result is not None:
        return tag_result
    if _contains_action_tag(_action_text(raw_output)):
        return None
    return _parse_json_tool_call(raw_output)


def extract_summary_output(raw_output: str) -> SummaryExtraction:
    extracted = _extract_completed_thinking(raw_output)
    if extracted is None:
        return SummaryExtraction(thinking="", summary=raw_output.strip())
    return SummaryExtraction(thinking=extracted.thinking, summary=extracted.remainder)


@dataclass(slots=True)
class ScriptedModel:
    outputs: list[str]
    cursor: int = 0

    def generate(self, prompt: str) -> str:
        del prompt
        output = self.outputs[self.cursor]
        self.cursor += 1
        return output

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return [self.generate(prompt) for prompt in prompts]


@dataclass(slots=True)
class _TokenUsage:
    reasoning_generated_tokens: int = 0
    summary_generated_tokens: int = 0
    forced_answer_generated_tokens: int = 0
    tool_result_tokens: int = 0
    max_prompt_tokens_seen: int = 0
    retired_round_count: int = 0
    forced_answer_reasons: list[str] = field(default_factory=list)

    def as_dict(self, *, summary_count: int, turn_records: list[dict[str, Any]]) -> dict[str, Any]:
        total_generated_tokens = (
            self.reasoning_generated_tokens
            + self.summary_generated_tokens
            + self.forced_answer_generated_tokens
        )
        return {
            "reasoning_generated_tokens": self.reasoning_generated_tokens,
            "summary_generated_tokens": self.summary_generated_tokens,
            "forced_answer_generated_tokens": self.forced_answer_generated_tokens,
            "tool_result_tokens": self.tool_result_tokens,
            "total_generated_tokens": total_generated_tokens,
            "prompt_tokens_by_turn": [
                {
                    "turn_id": record["turn_id"],
                    "kind": record["kind"],
                    "generation_kind": record.get("generation_kind", record["kind"]),
                    "prompt_tokens": record["prompt_tokens"],
                }
                for record in turn_records
                if "turn_id" in record and "prompt_tokens" in record
            ],
            "max_prompt_tokens_seen": self.max_prompt_tokens_seen,
            "summary_count": summary_count,
            "retired_round_count": self.retired_round_count,
            "forced_answer_reasons": list(dict.fromkeys(self.forced_answer_reasons)),
        }


@dataclass(slots=True)
class _ActiveEpisode:
    state: EpisodeState
    context_manager: ContextManager
    summary_turns: list[str] = field(default_factory=list)
    retrieved_docids: list[str] = field(default_factory=list)
    tool_call_counts: dict[str, int] = field(default_factory=lambda: {"search": 0, "get_document": 0})
    turn_records: list[dict[str, Any]] = field(default_factory=list)
    trajectory_records: list[dict[str, Any]] = field(default_factory=list)
    interval_turn_ids: list[str] = field(default_factory=list)
    interval_round_count: int = 0
    token_usage: _TokenUsage = field(default_factory=_TokenUsage)
    result: RuntimeResult | None = None


@dataclass(slots=True)
class _GeneratedOutput:
    text: str
    completion_tokens: int


@dataclass(slots=True)
class _PendingToolAction:
    active: _ActiveEpisode
    prompt: str
    raw_output: str
    normalized_output: str
    tool_name: str
    arguments: dict[str, object]
    prompt_tokens: int
    completion_tokens: int


@dataclass(slots=True)
class EpisodeRuntime:
    model: RuntimeModel
    backend: BrowseCompBackend
    context_threshold_tokens: int
    max_context_tokens: int
    max_summary_tokens: int = 2048
    max_tool_calls: int | None = None
    generated_token_budget: int | None = None
    token_counter: Callable[[str], int] = field(default=lambda text: len(text.split()))

    def __post_init__(self) -> None:
        if self.max_summary_tokens < 1:
            raise ValueError(f"max_summary_tokens must be at least 1, got {self.max_summary_tokens}")

    def _tool_calls_used(self, active: _ActiveEpisode) -> int:
        return active.tool_call_counts.get("search", 0) + active.tool_call_counts.get("get_document", 0)

    def _remaining_tool_calls(self, active: _ActiveEpisode) -> int | None:
        if self.max_tool_calls is None:
            return None
        return max(0, self.max_tool_calls - self._tool_calls_used(active))

    def _generated_token_budget_exhausted(self, active: _ActiveEpisode) -> bool:
        return (
            self.generated_token_budget is not None
            and active.token_usage.reasoning_generated_tokens >= self.generated_token_budget
        )

    def _prompt_token_count(self, active: _ActiveEpisode, prompt: str) -> int:
        prompt_counter = getattr(self.model, "count_prompt_tokens", None)
        prompt_tokens = prompt_counter(prompt) if prompt_counter is not None else self.token_counter(prompt)
        active.token_usage.max_prompt_tokens_seen = max(
            active.token_usage.max_prompt_tokens_seen,
            prompt_tokens,
        )
        return prompt_tokens

    def _completion_token_count(self, text: str) -> int:
        return self.token_counter(text)

    def _token_usage_payload(self, active: _ActiveEpisode) -> dict[str, Any]:
        return active.token_usage.as_dict(
            summary_count=len(active.summary_turns),
            turn_records=active.turn_records,
        )

    def _build_runtime_prompt(self, state: EpisodeState) -> ConversationPrompt:
        return ConversationPrompt(state.messages)

    def _build_forced_answer_prompt(self, active: _ActiveEpisode) -> ConversationPrompt:
        messages = list(active.state.messages)
        messages.append(Message(role="user", content=build_forced_answer_prompt()))
        return ConversationPrompt(messages)

    def _next_tool_turn_id(self, state: EpisodeState) -> str:
        return f"tool-{state.tool_turn_count + 1}"

    def _finalize_trajectory(
        self,
        active: _ActiveEpisode,
        messages: list[Message] | tuple[Message, ...],
        *,
        termination_kind: str,
    ) -> str:
        trajectory_id = f"trajectory-{len(active.trajectory_records) + 1}"
        serialized_messages = serialize_messages(messages)
        assistant_outputs = [message.content for message in messages if message.role == "assistant"]
        if not assistant_outputs:
            return trajectory_id
        active.trajectory_records.append(
            {
                "schema_version": 1,
                "query_id": active.state.query_id,
                "turn_id": trajectory_id,
                "kind": "trajectory",
                "termination_kind": termination_kind,
                "messages": serialized_messages,
                "prompt": str(ConversationPrompt(messages)),
                "completion": "\n".join(assistant_outputs),
                "turn_ids": list(active.interval_turn_ids),
                "assistant_completion_count": len(assistant_outputs),
            }
        )
        active.interval_turn_ids.clear()
        active.interval_round_count = 0
        return trajectory_id

    def _malformed_result(
        self,
        active: _ActiveEpisode,
        prompt: str,
        completion: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        generation_kind: str,
    ) -> RuntimeResult:
        state = active.state
        query_id = state.query_id
        malformed_turn_id = self._next_tool_turn_id(state)
        recorded_turns = list(active.turn_records)
        turn_record: dict[str, Any] = {
            "query_id": query_id,
            "turn_id": malformed_turn_id,
            "kind": "tool",
            "prompt": prompt,
            "completion": completion,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generation_kind": generation_kind,
        }
        recorded_turns.append(turn_record)
        active.interval_turn_ids.append(malformed_turn_id)
        if isinstance(prompt, ConversationPrompt):
            interval_messages = list(prompt.messages)
        else:
            interval_messages = list(active.state.messages)
        interval_messages.append(Message(role="assistant", content=completion))
        self._finalize_trajectory(active, interval_messages, termination_kind="malformed")
        return self._penalized_result(active, status="malformed_tool_call", turn_records=recorded_turns)

    def _penalized_result(
        self,
        active: _ActiveEpisode,
        *,
        status: str,
        turn_records: list[dict[str, Any]] | None = None,
    ) -> RuntimeResult:
        state = active.state
        recorded_turns = list(active.turn_records if turn_records is None else turn_records)
        return RuntimeResult(
            query_id=state.query_id,
            status=status,
            final_answer=None,
            summary_turns=list(active.summary_turns),
            turn_rewards=apply_malformed_tool_penalty(
                trainable_turn_ids_from_records(active.trajectory_records)
            ),
            retrieved_docids=list(active.retrieved_docids),
            tool_call_counts=dict(active.tool_call_counts),
            turn_records=recorded_turns,
            trajectory_records=list(active.trajectory_records),
            token_usage=active.token_usage.as_dict(
                summary_count=len(active.summary_turns),
                turn_records=recorded_turns,
            ),
        )

    def _budget_exhausted_result(
        self,
        query_id: str,
        summary_turns: list[str],
        retrieved_docids: list[str],
        tool_call_counts: dict[str, int],
        turn_records: list[dict[str, Any]],
        trajectory_records: list[dict[str, Any]] | None = None,
    ) -> RuntimeResult:
        return RuntimeResult(
            query_id=query_id,
            status="budget_exhausted",
            final_answer=None,
            summary_turns=list(summary_turns),
            turn_rewards=apply_terminal_reward(
                outcome="budget_exhausted",
                trainable_turn_ids=trainable_turn_ids_from_records(trajectory_records or []),
            ),
            retrieved_docids=list(retrieved_docids),
            tool_call_counts=dict(tool_call_counts),
            turn_records=list(turn_records),
            trajectory_records=list(trajectory_records or []),
            token_usage={},
        )

    def _record_retrieved_docids(self, retrieved_docids: list[str], doc_ids: list[str]) -> None:
        seen = set(retrieved_docids)
        for doc_id in doc_ids:
            if doc_id not in seen:
                retrieved_docids.append(doc_id)
                seen.add(doc_id)

    def _record_search_result_docids(
        self,
        retrieved_docids: list[str],
        search_results: list[SearchResult],
    ) -> None:
        doc_ids = [str(result["docid"]) for result in search_results if result.get("docid") is not None]
        self._record_retrieved_docids(retrieved_docids, doc_ids)

    def _generate_batch(self, prompts: list[str]) -> list[_GeneratedOutput]:
        generate_batch = getattr(self.model, "generate_batch", None)
        if generate_batch is None:
            outputs = [self.model.generate(prompt) for prompt in prompts]
            return [
                _GeneratedOutput(text=output, completion_tokens=self._completion_token_count(output))
                for output in outputs
            ]
        outputs = generate_batch(prompts)
        if len(outputs) != len(prompts):
            raise ValueError(f"Batch generator returned {len(outputs)} outputs for {len(prompts)} prompts")
        return [
            _GeneratedOutput(text=output, completion_tokens=self._completion_token_count(output))
            for output in outputs
        ]

    def _new_active_episode(self, query_id: str, user_prompt: str) -> _ActiveEpisode:
        state = EpisodeState(
            query_id=query_id,
            user_prompt=user_prompt,
            context_threshold_tokens=self.context_threshold_tokens,
            messages=build_initial_messages(user_prompt),
        )
        return _ActiveEpisode(
            state=state,
            context_manager=ContextManager(
                token_counter=self.token_counter,
                max_context_tokens=self.max_context_tokens,
                safety_margin_tokens=0,
                prompt_token_counter=getattr(self.model, "count_prompt_tokens", None),
            ),
        )

    def _completed_result(
        self,
        active: _ActiveEpisode,
        answer: str,
    ) -> RuntimeResult:
        return RuntimeResult(
            query_id=active.state.query_id,
            status="completed",
            final_answer=answer,
            summary_turns=list(active.summary_turns),
            turn_rewards=apply_terminal_reward(
                outcome="correct_answer",
                trainable_turn_ids=trainable_turn_ids_from_records(active.trajectory_records),
            ),
            retrieved_docids=list(active.retrieved_docids),
            tool_call_counts=dict(active.tool_call_counts),
            turn_records=list(active.turn_records),
            trajectory_records=list(active.trajectory_records),
            token_usage=self._token_usage_payload(active),
        )

    def _prepare_action_output(
        self,
        active: _ActiveEpisode,
        generated_output: _GeneratedOutput,
        prompt: str | None = None,
        prompt_tokens: int | None = None,
        generation_kind: str = "action",
    ) -> _PendingToolAction | None:
        state = active.state
        query_id = state.query_id
        prompt = prompt if prompt is not None else self._build_runtime_prompt(state)
        prompt_tokens = prompt_tokens if prompt_tokens is not None else self._prompt_token_count(active, prompt)
        raw_output = generated_output.text
        if generation_kind == "forced_answer":
            active.token_usage.forced_answer_generated_tokens += generated_output.completion_tokens
        else:
            active.token_usage.reasoning_generated_tokens += generated_output.completion_tokens
        parsed_tool_call = parse_model_tool_call(raw_output)
        if parsed_tool_call is None:
            active.result = self._malformed_result(
                active,
                prompt,
                raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_output.completion_tokens,
                generation_kind=generation_kind,
            )
            return
        payload, normalized_output = parsed_tool_call
        tool_name = payload["tool_name"]
        arguments = payload["arguments"]

        if tool_name == "finish":
            answer = arguments.get("answer")
            if not isinstance(answer, str):
                active.result = self._malformed_result(
                    active,
                    prompt,
                    raw_output,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=generated_output.completion_tokens,
                    generation_kind=generation_kind,
                )
                return
            turn_record: dict[str, Any] = {
                "query_id": query_id,
                "turn_id": "final-answer",
                "kind": "final_answer",
                "prompt": prompt,
                "completion": raw_output,
                "normalized_completion": normalized_output,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": generated_output.completion_tokens,
                "generation_kind": generation_kind,
            }
            active.turn_records.append(turn_record)
            active.interval_turn_ids.append("final-answer")
            interval_messages = list(prompt.messages) if isinstance(prompt, ConversationPrompt) else list(state.messages)
            interval_messages.append(Message(role="assistant", content=raw_output))
            self._finalize_trajectory(active, interval_messages, termination_kind="final_answer")
            active.result = self._completed_result(active, answer)
            return None

        if tool_name == "search":
            query = arguments.get("query")
            if not isinstance(query, str):
                active.result = self._malformed_result(
                    active,
                    prompt,
                    raw_output,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=generated_output.completion_tokens,
                    generation_kind=generation_kind,
                )
                return None
        elif tool_name == "get_document":
            doc_id = arguments.get("doc_id")
            if not isinstance(doc_id, str):
                active.result = self._malformed_result(
                    active,
                    prompt,
                    raw_output,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=generated_output.completion_tokens,
                    generation_kind=generation_kind,
                )
                return None
        else:
            active.result = self._malformed_result(
                active,
                prompt,
                raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_output.completion_tokens,
                generation_kind=generation_kind,
            )
            return None

        return _PendingToolAction(
            active=active,
            prompt=prompt,
            raw_output=raw_output,
            normalized_output=normalized_output,
            tool_name=tool_name,
            arguments=arguments,
            prompt_tokens=prompt_tokens,
            completion_tokens=generated_output.completion_tokens,
        )

    def _search_many(self, queries: list[str]) -> list[list[SearchResult]]:
        search_many = getattr(self.backend, "search_many", None)
        if search_many is None:
            return [self.backend.search(query) for query in queries]
        results = search_many(queries)
        if len(results) != len(queries):
            raise ValueError(f"search_many returned {len(results)} result batches for {len(queries)} queries")
        return results

    def _apply_tool_result(self, action: _PendingToolAction, tool_result: str) -> None:
        active = action.active
        state = active.state
        query_id = state.query_id
        tool_result_tokens = self._completion_token_count(tool_result)
        active.token_usage.tool_result_tokens += tool_result_tokens
        active.token_usage.reasoning_generated_tokens += tool_result_tokens

        tool_turn_id = self._next_tool_turn_id(state)
        turn_record: dict[str, Any] = {
            "query_id": query_id,
            "turn_id": tool_turn_id,
            "kind": "tool",
            "prompt": action.prompt,
            "completion": action.raw_output,
            "normalized_completion": action.normalized_output,
            "prompt_tokens": action.prompt_tokens,
            "completion_tokens": action.completion_tokens,
            "generation_kind": "action",
        }
        active.turn_records.append(turn_record)
        active.interval_turn_ids.append(tool_turn_id)
        state.tool_turn_count += 1
        active.interval_round_count += 1
        state.messages.append(Message(role="assistant", content=action.raw_output))
        state.messages.append(Message(role="user", content=format_tool_result(tool_result)))

    def _execute_pending_tool_actions(self, actions: list[_PendingToolAction]) -> None:
        search_actions = [action for action in actions if action.tool_name == "search"]
        if search_actions:
            queries = [str(action.arguments["query"]) for action in search_actions]
            for action, search_results in zip(search_actions, self._search_many(queries)):
                action.active.tool_call_counts["search"] += 1
                self._record_search_result_docids(action.active.retrieved_docids, search_results)
                self._apply_tool_result(action, json.dumps(search_results, ensure_ascii=False))

        for action in actions:
            if action.tool_name != "get_document":
                continue
            doc_id = str(action.arguments["doc_id"])
            self._record_retrieved_docids(action.active.retrieved_docids, [doc_id])
            try:
                tool_result = self.backend.get_document(doc_id)
            except Exception:
                LOGGER.warning("Failed to retrieve document %s", doc_id, exc_info=True)
                tool_result = f"Error: Document '{doc_id}' not found in retrieval index."
            action.active.tool_call_counts["get_document"] += 1
            self._apply_tool_result(action, tool_result)

    def _apply_action_output(self, active: _ActiveEpisode, raw_output: str, prompt: str | None = None) -> None:
        action = self._prepare_action_output(
            active,
            _GeneratedOutput(text=raw_output, completion_tokens=self._completion_token_count(raw_output)),
            prompt,
        )
        if action is not None:
            self._execute_pending_tool_actions([action])

    def _apply_forced_answer_output(
        self,
        active: _ActiveEpisode,
        generated_output: _GeneratedOutput,
        prompt: str,
        prompt_tokens: int,
    ) -> None:
        state = active.state
        query_id = state.query_id
        raw_output = generated_output.text
        active.token_usage.forced_answer_generated_tokens += generated_output.completion_tokens
        parsed_tool_call = parse_model_tool_call(raw_output)
        if parsed_tool_call is None:
            active.result = self._malformed_result(
                active,
                prompt,
                raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_output.completion_tokens,
                generation_kind="forced_answer",
            )
            return

        payload, normalized_output = parsed_tool_call
        tool_name = payload["tool_name"]
        arguments = payload["arguments"]
        if tool_name != "finish":
            active.result = self._malformed_result(
                active,
                prompt,
                raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_output.completion_tokens,
                generation_kind="forced_answer",
            )
            return

        answer = arguments.get("answer")
        if not isinstance(answer, str):
            active.result = self._malformed_result(
                active,
                prompt,
                raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_output.completion_tokens,
                generation_kind="forced_answer",
            )
            return

        turn_record: dict[str, Any] = {
            "query_id": query_id,
            "turn_id": "final-answer",
            "kind": "final_answer",
            "prompt": prompt,
            "completion": raw_output,
            "normalized_completion": normalized_output,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generated_output.completion_tokens,
            "generation_kind": "forced_answer",
        }
        active.turn_records.append(turn_record)
        active.interval_turn_ids.append("final-answer")
        interval_messages = list(prompt.messages) if isinstance(prompt, ConversationPrompt) else list(state.messages)
        interval_messages.append(Message(role="assistant", content=raw_output))
        self._finalize_trajectory(active, interval_messages, termination_kind="forced_answer")
        active.result = self._completed_result(active, answer)

    def _build_summary_prompt_for_active(self, active: _ActiveEpisode) -> ConversationPrompt | None:
        if not active.context_manager.should_summarize(active.state):
            return None
        if active.interval_round_count == 0:
            return None
        prompt = active.context_manager.build_summary_context(
            active.state,
            max_summary_tokens=self.max_summary_tokens,
        )
        active.context_manager.assert_fits(prompt, reserved_tokens=self.max_summary_tokens)
        return prompt

    def _apply_summary_output(
        self,
        active: _ActiveEpisode,
        prompt: str,
        prompt_tokens: int,
        generated_output: _GeneratedOutput,
    ) -> None:
        generated_summary = generated_output.text
        active.token_usage.summary_generated_tokens += generated_output.completion_tokens
        summary_extraction = extract_summary_output(generated_summary)
        summary_tokens = self._completion_token_count(summary_extraction.summary)
        state = active.state
        state.summary_count += 1
        summary_turn_id = f"summary-{state.summary_count}"
        turn_record: dict[str, Any] = {
            "query_id": state.query_id,
            "turn_id": summary_turn_id,
            "kind": "summary",
            "prompt": prompt,
            "completion": generated_summary,
            "thinking": summary_extraction.thinking,
            "summary": summary_extraction.summary,
            "summary_tokens": summary_tokens,
            "max_summary_tokens": self.max_summary_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generated_output.completion_tokens,
            "generation_kind": "summary",
        }
        active.turn_records.append(turn_record)
        active.interval_turn_ids.append(summary_turn_id)
        retired_count = active.interval_round_count
        interval_messages = list(prompt.messages) if isinstance(prompt, ConversationPrompt) else list(state.messages)
        interval_messages.append(Message(role="assistant", content=generated_summary))
        self._finalize_trajectory(active, interval_messages, termination_kind="compaction")
        if summary_tokens > self.max_summary_tokens:
            active.result = self._penalized_result(active, status="summary_length_exceeded")
            return
        if not summary_extraction.summary:
            active.result = self._penalized_result(active, status="empty_summary")
            return
        state.latest_summary = summary_extraction.summary
        state.messages = build_compacted_messages(summary_extraction.summary)
        active.token_usage.retired_round_count += retired_count
        active.summary_turns.append(summary_turn_id)

    def run_many(self, episodes: Iterable[tuple[str, str]]) -> list[RuntimeResult]:
        active_episodes = [self._new_active_episode(query_id, user_prompt) for query_id, user_prompt in episodes]
        while any(active.result is None for active in active_episodes):
            action_items: list[tuple[_ActiveEpisode, str, int, bool]] = []
            for active in active_episodes:
                if active.result is not None:
                    continue
                remaining_tool_calls = self._remaining_tool_calls(active)
                forced_reasons: list[str] = []
                if remaining_tool_calls == 0:
                    forced_reasons.append("tool_budget")
                if self._generated_token_budget_exhausted(active):
                    forced_reasons.append("generated_token_budget")

                if forced_reasons:
                    active.token_usage.forced_answer_reasons.extend(forced_reasons)
                    acting_prompt = self._build_forced_answer_prompt(active)
                    active.context_manager.assert_fits(acting_prompt)
                    prompt_tokens = self._prompt_token_count(active, acting_prompt)
                    action_items.append((active, acting_prompt, prompt_tokens, True))
                    continue
                acting_prompt = self._build_runtime_prompt(active.state)
                active.context_manager.assert_fits(acting_prompt)
                prompt_tokens = self._prompt_token_count(active, acting_prompt)
                action_items.append((active, acting_prompt, prompt_tokens, False))

            if action_items:
                action_outputs = self._generate_batch([prompt for _, prompt, _, _ in action_items])
                pending_tool_actions: list[_PendingToolAction] = []
                for (active, prompt, prompt_tokens, forced_answer), generated_output in zip(action_items, action_outputs):
                    if forced_answer:
                        self._apply_forced_answer_output(active, generated_output, prompt, prompt_tokens)
                    else:
                        pending_action = self._prepare_action_output(
                            active,
                            generated_output,
                            prompt,
                            prompt_tokens,
                        )
                        if pending_action is not None:
                            pending_tool_actions.append(pending_action)
                if pending_tool_actions:
                    self._execute_pending_tool_actions(pending_tool_actions)

            summary_items: list[tuple[_ActiveEpisode, str, int]] = []
            for active in active_episodes:
                if active.result is not None:
                    continue
                summary_request = self._build_summary_prompt_for_active(active)
                if summary_request is None:
                    continue
                summary_prompt = summary_request
                prompt_tokens = self._prompt_token_count(active, summary_prompt)
                summary_items.append((active, summary_prompt, prompt_tokens))

            if summary_items:
                summary_outputs = self._generate_batch([prompt for _, prompt, _ in summary_items])
                for (active, prompt, prompt_tokens), generated_output in zip(summary_items, summary_outputs):
                    self._apply_summary_output(active, prompt, prompt_tokens, generated_output)

        return [active.result for active in active_episodes if active.result is not None]

    def run(self, query_id: str, user_prompt: str) -> RuntimeResult:
        return self.run_many([(query_id, user_prompt)])[0]
