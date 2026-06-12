import json
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

from self_summarization_agent.backend import BrowseCompBackend, SearchResult
from self_summarization_agent.context import ContextManager
from self_summarization_agent.models import EpisodeState, Message, RuntimeResult, ToolCallRecord, ToolRound
from self_summarization_agent.prompts import build_forced_answer_system_prompt, build_system_prompt
from self_summarization_agent.rewards import (
    apply_malformed_tool_penalty,
    apply_terminal_reward,
    trainable_turn_ids_from_records,
)


_JSON_DECODER = json.JSONDecoder()
_THINK_END_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)
_THINK_START_RE = re.compile(r"^\s*<think\b[^>]*>", flags=re.IGNORECASE)


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


def parse_model_tool_call(raw_output: str) -> tuple[dict[str, object], str] | None:
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
class _ActiveEpisode:
    state: EpisodeState
    context_manager: ContextManager
    summary_turns: list[str] = field(default_factory=list)
    retrieved_docids: list[str] = field(default_factory=list)
    tool_call_counts: dict[str, int] = field(default_factory=lambda: {"search": 0, "get_document": 0})
    turn_records: list[dict[str, str]] = field(default_factory=list)
    result: RuntimeResult | None = None


@dataclass(slots=True)
class EpisodeRuntime:
    model: RuntimeModel
    backend: BrowseCompBackend
    context_threshold_tokens: int
    max_context_tokens: int
    max_tool_calls: int | None = None
    token_counter: Callable[[str], int] = field(default=lambda text: len(text.split()))

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

    def _build_runtime_prompt(self, state: EpisodeState, remaining_tool_calls: int | None = None) -> str:
        pieces = [
            self._build_transcript_block("SYSTEM", build_system_prompt(remaining_tool_calls)),
            self._build_transcript_block("USER", state.user_prompt),
        ]
        if state.latest_summary:
            pieces.append(self._build_transcript_block("SUMMARY", state.latest_summary))
        for round_record in self._raw_tail_rounds(state):
            pieces.extend(
                [
                    self._build_transcript_block("ASSISTANT_TOOL_CALL", round_record.assistant_message.content),
                    self._build_transcript_block("TOOL_RESULT", round_record.tool_result.content),
                ]
            )
        pieces.append(
            "### NEXT_ACTION\n"
            "Return exactly one JSON object for the next tool call. "
            "After any thinking, the final visible action must be only the JSON object. "
            "Do not include labels, markdown, code fences, explanations, or any text before or after the final JSON object. "
            "Return one action only."
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
        for round_record in self._raw_tail_rounds(state):
            pieces.extend(
                [
                    self._build_transcript_block("ASSISTANT_TOOL_CALL", round_record.assistant_message.content),
                    self._build_transcript_block("TOOL_RESULT", round_record.tool_result.content),
                ]
            )
        pieces.append(
            "### NEXT_ACTION\n"
            "The search/get_document budget is exhausted. "
            "Return exactly one JSON object using finish only. "
            "After any thinking, the final visible action must be only the JSON object. "
            "Do not include labels, markdown, code fences, explanations, or any text before or after the final JSON object."
        )
        return "\n".join(pieces)

    def _next_tool_turn_id(self, state: EpisodeState) -> str:
        return f"tool-{len(state.rounds) + 1}"

    def _malformed_result(
        self,
        state: EpisodeState,
        query_id: str,
        prompt: str,
        completion: str,
        summary_turns: list[str],
        retrieved_docids: list[str],
        tool_call_counts: dict[str, int],
        turn_records: list[dict[str, str]],
    ) -> RuntimeResult:
        malformed_turn_id = self._next_tool_turn_id(state)
        recorded_turns = list(turn_records)
        recorded_turns.append(
            {
                "query_id": query_id,
                "turn_id": malformed_turn_id,
                "kind": "tool",
                "prompt": prompt,
                "completion": completion,
            }
        )
        return RuntimeResult(
            query_id=query_id,
            status="malformed_tool_call",
            final_answer=None,
            summary_turns=list(summary_turns),
            turn_rewards=apply_malformed_tool_penalty(trainable_turn_ids_from_records(recorded_turns)),
            retrieved_docids=list(retrieved_docids),
            tool_call_counts=dict(tool_call_counts),
            turn_records=recorded_turns,
        )

    def _budget_exhausted_result(
        self,
        query_id: str,
        summary_turns: list[str],
        retrieved_docids: list[str],
        tool_call_counts: dict[str, int],
        turn_records: list[dict[str, str]],
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

    def _generate_batch(self, prompts: list[str]) -> list[str]:
        generate_batch = getattr(self.model, "generate_batch", None)
        if generate_batch is None:
            return [self.model.generate(prompt) for prompt in prompts]
        outputs = generate_batch(prompts)
        if len(outputs) != len(prompts):
            raise ValueError(f"Batch generator returned {len(outputs)} outputs for {len(prompts)} prompts")
        return outputs

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
        )

    def _apply_action_output(self, active: _ActiveEpisode, raw_output: str, prompt: str | None = None) -> None:
        state = active.state
        query_id = state.query_id
        prompt = prompt if prompt is not None else self._build_runtime_prompt(state, self._remaining_tool_calls(active))
        parsed_tool_call = parse_model_tool_call(raw_output)
        if parsed_tool_call is None:
            active.result = self._malformed_result(
                state,
                query_id,
                prompt,
                raw_output,
                active.summary_turns,
                active.retrieved_docids,
                active.tool_call_counts,
                active.turn_records,
            )
            return
        payload, normalized_output = parsed_tool_call
        tool_name = payload["tool_name"]
        arguments = payload["arguments"]

        if tool_name == "finish":
            answer = arguments.get("answer")
            if not isinstance(answer, str):
                active.result = self._malformed_result(
                    state,
                    query_id,
                    prompt,
                    normalized_output,
                    active.summary_turns,
                    active.retrieved_docids,
                    active.tool_call_counts,
                    active.turn_records,
                )
                return
            active.turn_records.append(
                {
                    "query_id": query_id,
                    "turn_id": "final-answer",
                    "kind": "final_answer",
                    "prompt": prompt,
                    "completion": normalized_output,
                }
            )
            active.result = self._completed_result(active, answer)
            return

        if tool_name == "search":
            query = arguments.get("query")
            if not isinstance(query, str):
                active.result = self._malformed_result(
                    state,
                    query_id,
                    prompt,
                    normalized_output,
                    active.summary_turns,
                    active.retrieved_docids,
                    active.tool_call_counts,
                    active.turn_records,
                )
                return
            search_results = self.backend.search(query)
            active.tool_call_counts["search"] += 1
            self._record_search_result_docids(active.retrieved_docids, search_results)
            tool_result = json.dumps(search_results, ensure_ascii=False)
        elif tool_name == "get_document":
            doc_id = arguments.get("doc_id")
            if not isinstance(doc_id, str):
                active.result = self._malformed_result(
                    state,
                    query_id,
                    prompt,
                    normalized_output,
                    active.summary_turns,
                    active.retrieved_docids,
                    active.tool_call_counts,
                    active.turn_records,
                )
                return
            self._record_retrieved_docids(active.retrieved_docids, [doc_id])
            tool_result = self.backend.get_document(doc_id)
            active.tool_call_counts["get_document"] += 1
        else:
            active.result = self._malformed_result(
                state,
                query_id,
                prompt,
                normalized_output,
                active.summary_turns,
                active.retrieved_docids,
                active.tool_call_counts,
                active.turn_records,
            )
            return

        tool_turn_id = self._next_tool_turn_id(state)
        active.turn_records.append(
            {
                "query_id": query_id,
                "turn_id": tool_turn_id,
                "kind": "tool",
                "prompt": prompt,
                "completion": normalized_output,
            }
        )
        state.rounds.append(
            ToolRound(
                assistant_message=Message(role="assistant", content=normalized_output),
                tool_call=ToolCallRecord(tool_name=tool_name, arguments=arguments, raw_output=normalized_output),
                tool_result=Message(role="tool", content=tool_result),
            )
        )

    def _apply_forced_answer_output(self, active: _ActiveEpisode, raw_output: str, prompt: str) -> None:
        state = active.state
        query_id = state.query_id
        parsed_tool_call = parse_model_tool_call(raw_output)
        if parsed_tool_call is None:
            active.result = self._malformed_result(
                state,
                query_id,
                prompt,
                raw_output,
                active.summary_turns,
                active.retrieved_docids,
                active.tool_call_counts,
                active.turn_records,
            )
            return

        payload, normalized_output = parsed_tool_call
        tool_name = payload["tool_name"]
        arguments = payload["arguments"]
        if tool_name != "finish":
            active.result = self._malformed_result(
                state,
                query_id,
                prompt,
                normalized_output,
                active.summary_turns,
                active.retrieved_docids,
                active.tool_call_counts,
                active.turn_records,
            )
            return

        answer = arguments.get("answer")
        if not isinstance(answer, str):
            active.result = self._malformed_result(
                state,
                query_id,
                prompt,
                normalized_output,
                active.summary_turns,
                active.retrieved_docids,
                active.tool_call_counts,
                active.turn_records,
            )
            return

        active.turn_records.append(
            {
                "query_id": query_id,
                "turn_id": "final-answer",
                "kind": "final_answer",
                "prompt": prompt,
                "completion": normalized_output,
            }
        )
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
        retired_count: int,
        generated_summary: str,
    ) -> None:
        summary_extraction = extract_summary_output(generated_summary)
        state = active.state
        state.summary_count += 1
        summary_turn_id = f"summary-{state.summary_count}"
        active.turn_records.append(
            {
                "query_id": state.query_id,
                "turn_id": summary_turn_id,
                "kind": "summary",
                "prompt": prompt,
                "completion": generated_summary,
                "thinking": summary_extraction.thinking,
                "summary": summary_extraction.summary,
            }
        )
        if not summary_extraction.summary:
            return
        state.latest_summary = summary_extraction.summary
        state.summarized_round_count += retired_count
        active.summary_turns.append(summary_turn_id)

    def run_many(self, episodes: Iterable[tuple[str, str]]) -> list[RuntimeResult]:
        active_episodes = [self._new_active_episode(query_id, user_prompt) for query_id, user_prompt in episodes]
        while any(active.result is None for active in active_episodes):
            action_items: list[tuple[_ActiveEpisode, str, bool]] = []
            for active in active_episodes:
                if active.result is not None:
                    continue
                remaining_tool_calls = self._remaining_tool_calls(active)
                if remaining_tool_calls == 0:
                    acting_prompt = self._build_forced_answer_prompt(active)
                    active.context_manager.assert_fits(acting_prompt)
                    action_items.append((active, acting_prompt, True))
                    continue
                acting_prompt = self._build_runtime_prompt(active.state, remaining_tool_calls)
                active.context_manager.assert_fits(acting_prompt)
                action_items.append((active, acting_prompt, False))

            if action_items:
                action_outputs = self._generate_batch([prompt for _, prompt, _ in action_items])
                for (active, prompt, forced_answer), raw_output in zip(action_items, action_outputs):
                    if forced_answer:
                        self._apply_forced_answer_output(active, raw_output, prompt)
                    else:
                        self._apply_action_output(active, raw_output, prompt)

            summary_items: list[tuple[_ActiveEpisode, str, int]] = []
            for active in active_episodes:
                if active.result is not None:
                    continue
                summary_request = self._build_summary_prompt_for_active(active)
                if summary_request is None:
                    continue
                summary_prompt, retired_count = summary_request
                summary_items.append((active, summary_prompt, retired_count))

            if summary_items:
                summary_outputs = self._generate_batch([prompt for _, prompt, _ in summary_items])
                for (active, prompt, retired_count), generated_summary in zip(summary_items, summary_outputs):
                    self._apply_summary_output(active, prompt, retired_count, generated_summary)

        return [active.result for active in active_episodes if active.result is not None]

    def run(self, query_id: str, user_prompt: str) -> RuntimeResult:
        return self.run_many([(query_id, user_prompt)])[0]
