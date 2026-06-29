import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from self_summarization_agent.backend import BrowseCompBackend, SearchResult
from self_summarization_agent.context import ContextManager
from self_summarization_agent.models import EpisodeState, Message, RuntimeResult, ToolCallRecord, ToolRound
from self_summarization_agent.prompts import (
    build_forced_answer_system_prompt,
    build_system_prompt,
    format_history_round,
)
from self_summarization_agent.rewards import (
    apply_malformed_tool_penalty,
    apply_terminal_reward,
    trainable_turn_ids_from_records,
)
from self_summarization_agent.trajectory import build_training_cache_from_token_ids


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
    token_usage: _TokenUsage = field(default_factory=_TokenUsage)
    result: RuntimeResult | None = None


@dataclass(slots=True)
class _GeneratedOutput:
    text: str
    completion_tokens: int
    training_cache: dict[str, Any] | None = None


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
    training_cache: dict[str, Any] | None = None


@dataclass(slots=True)
class EpisodeRuntime:
    model: RuntimeModel
    backend: BrowseCompBackend
    context_threshold_tokens: int
    max_context_tokens: int
    max_tool_calls: int | None = None
    generated_token_budget: int | None = None
    token_counter: Callable[[str], int] = field(default=lambda text: len(text.split()))
    cache_policy_checkpoint_id: str | None = None

    def _build_transcript_block(self, label: str, content: str) -> str:
        return f"### {label}\n{content}"

    def _raw_tail_rounds(self, state: EpisodeState) -> list[ToolRound]:
        return state.rounds[state.summarized_round_count :]

    def _summary_retirement_count(self, state: EpisodeState) -> int:
        raw_tail_count = len(self._raw_tail_rounds(state))
        if raw_tail_count <= 1:
            return 0
        return raw_tail_count - 1

    def _build_summary_state(self, state: EpisodeState) -> tuple[EpisodeState, int]:
        retired_count = self._summary_retirement_count(state)
        summary_state = EpisodeState(
            query_id=state.query_id,
            user_prompt=state.user_prompt,
            context_threshold_tokens=state.context_threshold_tokens,
            latest_summary=state.latest_summary,
            summary_count=state.summary_count,
            summarized_round_count=0,
            rounds=list(self._raw_tail_rounds(state)[:retired_count]),
        )
        return summary_state, retired_count

    def _compacted_state(self, state: EpisodeState) -> EpisodeState:
        return EpisodeState(
            query_id=state.query_id,
            user_prompt=state.user_prompt,
            context_threshold_tokens=state.context_threshold_tokens,
            latest_summary=state.latest_summary,
            summary_count=state.summary_count,
            rounds=list(self._raw_tail_rounds(state)),
        )

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
        prompt_tokens = self.token_counter(prompt)
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

    def _build_runtime_prompt(self, state: EpisodeState) -> str:
        pieces = [
            self._build_transcript_block("SYSTEM", build_system_prompt()),
            self._build_transcript_block("USER", state.user_prompt),
        ]
        if state.latest_summary:
            pieces.append(self._build_transcript_block("SUMMARY", state.latest_summary))
        history = [
            format_history_round(
                round_record.tool_call.tool_name,
                round_record.tool_call.arguments,
                round_record.tool_result.content,
            )
            for round_record in self._raw_tail_rounds(state)
        ]
        if history:
            pieces.append(self._build_transcript_block("HISTORY", "\n".join(history)))
        pieces.append(
            "### NEXT_ACTION\n"
            "Think first, then output one action tag: "
            "<search>...</search>, <document>...</document>, or <answer>...</answer>."
        )
        return "\n".join(pieces)

    def _build_forced_answer_prompt(self, active: _ActiveEpisode) -> str:
        state = active.state
        pieces = [
            self._build_transcript_block("SYSTEM", build_forced_answer_system_prompt()),
            self._build_transcript_block("USER", state.user_prompt),
        ]
        if state.latest_summary:
            pieces.append(self._build_transcript_block("SUMMARY", state.latest_summary))
        history = [
            format_history_round(
                round_record.tool_call.tool_name,
                round_record.tool_call.arguments,
                round_record.tool_result.content,
            )
            for round_record in self._raw_tail_rounds(state)
        ]
        if history:
            pieces.append(self._build_transcript_block("HISTORY", "\n".join(history)))
        pieces.append(
            "### NEXT_ACTION\n"
            "Search and document actions are no longer available. "
            "Think first, then output one <answer>...</answer> tag."
        )
        return "\n".join(pieces)

    def _next_tool_turn_id(self, state: EpisodeState) -> str:
        return f"tool-{len(state.rounds) + 1}"

    def _malformed_result(
        self,
        active: _ActiveEpisode,
        prompt: str,
        completion: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        generation_kind: str,
        training_cache: dict[str, Any] | None = None,
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
        if training_cache is not None:
            turn_record["training_cache"] = training_cache
        recorded_turns.append(turn_record)
        return RuntimeResult(
            query_id=query_id,
            status="malformed_tool_call",
            final_answer=None,
            summary_turns=list(active.summary_turns),
            turn_rewards=apply_malformed_tool_penalty(trainable_turn_ids_from_records(recorded_turns)),
            retrieved_docids=list(active.retrieved_docids),
            tool_call_counts=dict(active.tool_call_counts),
            turn_records=recorded_turns,
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
    ) -> RuntimeResult:
        return RuntimeResult(
            query_id=query_id,
            status="budget_exhausted",
            final_answer=None,
            summary_turns=list(summary_turns),
            turn_rewards=apply_terminal_reward(
                outcome="budget_exhausted",
                trainable_turn_ids=trainable_turn_ids_from_records(turn_records),
            ),
            retrieved_docids=list(retrieved_docids),
            tool_call_counts=dict(tool_call_counts),
            turn_records=list(turn_records),
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

    def _training_cache_from_generation(self, result: Any) -> dict[str, Any] | None:
        prompt_token_ids = getattr(result, "prompt_token_ids", None)
        completion_token_ids = getattr(result, "completion_token_ids", None)
        cumulative_logprob = getattr(result, "cumulative_logprob", None)
        if prompt_token_ids is None or completion_token_ids is None:
            return None
        return build_training_cache_from_token_ids(
            prompt_token_ids=list(prompt_token_ids),
            completion_token_ids=list(completion_token_ids),
            cumulative_logprob=cumulative_logprob,
            policy_checkpoint_id=self.cache_policy_checkpoint_id,
        )

    def _generate_batch(self, prompts: list[str]) -> list[_GeneratedOutput]:
        generate_batch_with_metadata = getattr(self.model, "generate_batch_with_metadata", None)
        if self.cache_policy_checkpoint_id is not None and generate_batch_with_metadata is not None:
            metadata_outputs = generate_batch_with_metadata(prompts)
            if len(metadata_outputs) != len(prompts):
                raise ValueError(
                    f"Batch generator returned {len(metadata_outputs)} outputs for {len(prompts)} prompts"
                )
            generated_outputs: list[_GeneratedOutput] = []
            for output in metadata_outputs:
                completion_token_ids = getattr(output, "completion_token_ids", None)
                completion_tokens = (
                    len(completion_token_ids)
                    if completion_token_ids is not None
                    else self._completion_token_count(output.text)
                )
                generated_outputs.append(
                    _GeneratedOutput(
                        text=output.text,
                        completion_tokens=completion_tokens,
                        training_cache=self._training_cache_from_generation(output),
                    )
                )
            return generated_outputs
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
        return _ActiveEpisode(
            state=EpisodeState(
                query_id=query_id,
                user_prompt=user_prompt,
                context_threshold_tokens=self.context_threshold_tokens,
            ),
            context_manager=ContextManager(
                token_counter=self.token_counter,
                max_context_tokens=self.max_context_tokens,
                safety_margin_tokens=0,
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
                trainable_turn_ids=trainable_turn_ids_from_records(active.turn_records),
            ),
            retrieved_docids=list(active.retrieved_docids),
            tool_call_counts=dict(active.tool_call_counts),
            turn_records=list(active.turn_records),
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
                training_cache=generated_output.training_cache,
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
                    normalized_output,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=generated_output.completion_tokens,
                    generation_kind=generation_kind,
                    training_cache=generated_output.training_cache,
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
            if generated_output.training_cache is not None:
                turn_record["training_cache"] = generated_output.training_cache
            active.turn_records.append(turn_record)
            active.result = self._completed_result(active, answer)
            return None

        if tool_name == "search":
            query = arguments.get("query")
            if not isinstance(query, str):
                active.result = self._malformed_result(
                    active,
                    prompt,
                    normalized_output,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=generated_output.completion_tokens,
                    generation_kind=generation_kind,
                    training_cache=generated_output.training_cache,
                )
                return None
        elif tool_name == "get_document":
            doc_id = arguments.get("doc_id")
            if not isinstance(doc_id, str):
                active.result = self._malformed_result(
                    active,
                    prompt,
                    normalized_output,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=generated_output.completion_tokens,
                    generation_kind=generation_kind,
                    training_cache=generated_output.training_cache,
                )
                return None
        else:
            active.result = self._malformed_result(
                active,
                prompt,
                normalized_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_output.completion_tokens,
                generation_kind=generation_kind,
                training_cache=generated_output.training_cache,
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
            training_cache=generated_output.training_cache,
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
        if action.training_cache is not None:
            turn_record["training_cache"] = action.training_cache
        active.turn_records.append(turn_record)
        state.rounds.append(
            ToolRound(
                assistant_message=Message(role="assistant", content=action.normalized_output),
                tool_call=ToolCallRecord(
                    tool_name=action.tool_name,
                    arguments=action.arguments,
                    raw_output=action.normalized_output,
                ),
                tool_result=Message(role="tool", content=tool_result),
            )
        )

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
            tool_result = self.backend.get_document(doc_id)
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
                training_cache=generated_output.training_cache,
            )
            return

        payload, normalized_output = parsed_tool_call
        tool_name = payload["tool_name"]
        arguments = payload["arguments"]
        if tool_name != "finish":
            active.result = self._malformed_result(
                active,
                prompt,
                normalized_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_output.completion_tokens,
                generation_kind="forced_answer",
                training_cache=generated_output.training_cache,
            )
            return

        answer = arguments.get("answer")
        if not isinstance(answer, str):
            active.result = self._malformed_result(
                active,
                prompt,
                normalized_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_output.completion_tokens,
                generation_kind="forced_answer",
                training_cache=generated_output.training_cache,
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
        if generated_output.training_cache is not None:
            turn_record["training_cache"] = generated_output.training_cache
        active.turn_records.append(turn_record)
        active.result = self._completed_result(active, answer)

    def _build_summary_prompt_for_active(self, active: _ActiveEpisode) -> tuple[str, int] | None:
        compacted_state = self._compacted_state(active.state)
        if not active.context_manager.should_summarize(compacted_state):
            return None
        summary_state, retired_count = self._build_summary_state(active.state)
        if retired_count == 0:
            return None
        prompt = active.context_manager.build_summary_context(summary_state)
        active.context_manager.assert_fits(prompt)
        return prompt, retired_count

    def _apply_summary_output(
        self,
        active: _ActiveEpisode,
        prompt: str,
        prompt_tokens: int,
        retired_count: int,
        generated_output: _GeneratedOutput,
    ) -> None:
        generated_summary = generated_output.text
        active.token_usage.summary_generated_tokens += generated_output.completion_tokens
        summary_extraction = extract_summary_output(generated_summary)
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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generated_output.completion_tokens,
            "generation_kind": "summary",
        }
        if generated_output.training_cache is not None:
            turn_record["training_cache"] = generated_output.training_cache
        active.turn_records.append(turn_record)
        if not summary_extraction.summary:
            return
        state.latest_summary = summary_extraction.summary
        state.summarized_round_count += retired_count
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

            summary_items: list[tuple[_ActiveEpisode, str, int, int]] = []
            for active in active_episodes:
                if active.result is not None:
                    continue
                summary_request = self._build_summary_prompt_for_active(active)
                if summary_request is None:
                    continue
                summary_prompt, retired_count = summary_request
                prompt_tokens = self._prompt_token_count(active, summary_prompt)
                summary_items.append((active, summary_prompt, prompt_tokens, retired_count))

            if summary_items:
                summary_outputs = self._generate_batch([prompt for _, prompt, _, _ in summary_items])
                for (active, prompt, prompt_tokens, retired_count), generated_output in zip(summary_items, summary_outputs):
                    self._apply_summary_output(active, prompt, prompt_tokens, retired_count, generated_output)

        return [active.result for active in active_episodes if active.result is not None]

    def run(self, query_id: str, user_prompt: str) -> RuntimeResult:
        return self.run_many([(query_id, user_prompt)])[0]
