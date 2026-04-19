import json
from dataclasses import dataclass, field
from typing import Callable

from self_summarization_agent.backend import BrowseCompBackend
from self_summarization_agent.context import ContextManager
from self_summarization_agent.models import EpisodeState, Message, RuntimeResult, ToolCallRecord, ToolRound
from self_summarization_agent.prompts import build_system_prompt
from self_summarization_agent.rewards import apply_malformed_tool_penalty, apply_terminal_reward


@dataclass(slots=True)
class ScriptedModel:
    outputs: list[str]
    cursor: int = 0

    def generate(self, prompt: str) -> str:
        del prompt
        output = self.outputs[self.cursor]
        self.cursor += 1
        return output


@dataclass(slots=True)
class EpisodeRuntime:
    model: ScriptedModel
    backend: BrowseCompBackend
    context_threshold_tokens: int
    max_context_tokens: int
    max_tool_calls: int | None = None
    token_counter: Callable[[str], int] = field(default=lambda text: len(text.split()))

    def _build_transcript_block(self, label: str, content: str) -> str:
        return json.dumps({"label": label, "content": content})

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

    def _build_runtime_prompt(self, state: EpisodeState) -> str:
        pieces = [
            self._build_transcript_block("SYSTEM", build_system_prompt()),
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
        return "\n".join(pieces)

    def _next_tool_turn_id(self, state: EpisodeState) -> str:
        return f"tool-{len(state.rounds) + 1}"

    def _malformed_result(
        self,
        state: EpisodeState,
        query_id: str,
        summary_turns: list[str],
        retrieved_docids: list[str],
        tool_call_counts: dict[str, int],
        turn_records: list[dict[str, str]],
    ) -> RuntimeResult:
        malformed_turn_id = self._next_tool_turn_id(state)
        recorded_turns = list(turn_records)
        recorded_turns.append({"turn_id": malformed_turn_id, "kind": "tool"})
        return RuntimeResult(
            query_id=query_id,
            status="malformed_tool_call",
            final_answer=None,
            summary_turns=list(summary_turns),
            turn_rewards=apply_malformed_tool_penalty(malformed_turn_id),
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
                summary_turn_ids=summary_turns,
                final_answer_turn_id=None,
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

    def run(self, query_id: str, user_prompt: str) -> RuntimeResult:
        state = EpisodeState(
            query_id=query_id,
            user_prompt=user_prompt,
            context_threshold_tokens=self.context_threshold_tokens,
        )
        context_manager = ContextManager(
            token_counter=self.token_counter,
            max_context_tokens=self.max_context_tokens,
            safety_margin_tokens=0,
        )
        summary_turns: list[str] = []
        retrieved_docids: list[str] = []
        tool_call_counts = {"search": 0, "get_document": 0}
        turn_records: list[dict[str, str]] = []

        while True:
            if self.max_tool_calls is not None and sum(tool_call_counts.values()) >= self.max_tool_calls:
                return self._budget_exhausted_result(
                    query_id,
                    summary_turns,
                    retrieved_docids,
                    tool_call_counts,
                    turn_records,
                )
            acting_prompt = self._build_runtime_prompt(state)
            context_manager.assert_fits(acting_prompt)
            raw_output = self.model.generate(acting_prompt)
            try:
                payload = json.loads(raw_output)
                tool_name = payload["tool_name"]
                arguments = payload["arguments"]
            except (json.JSONDecodeError, KeyError, TypeError):
                return self._malformed_result(
                    state,
                    query_id,
                    summary_turns,
                    retrieved_docids,
                    tool_call_counts,
                    turn_records,
                )

            if not isinstance(tool_name, str) or not isinstance(arguments, dict):
                return self._malformed_result(
                    state,
                    query_id,
                    summary_turns,
                    retrieved_docids,
                    tool_call_counts,
                    turn_records,
                )

            if tool_name == "finish":
                answer = arguments.get("answer")
                if not isinstance(answer, str):
                    return self._malformed_result(
                        state,
                        query_id,
                        summary_turns,
                        retrieved_docids,
                        tool_call_counts,
                        turn_records,
                    )
                turn_records.append(
                    {
                        "query_id": query_id,
                        "turn_id": "final-answer",
                        "kind": "final_answer",
                        "prompt": acting_prompt,
                        "completion": raw_output,
                    }
                )
                return RuntimeResult(
                    query_id=query_id,
                    status="completed",
                    final_answer=answer,
                    summary_turns=list(summary_turns),
                    turn_rewards=apply_terminal_reward(
                        outcome="correct_answer",
                        summary_turn_ids=summary_turns,
                        final_answer_turn_id="final-answer",
                    ),
                    retrieved_docids=list(retrieved_docids),
                    tool_call_counts=dict(tool_call_counts),
                    turn_records=list(turn_records),
                )

            if tool_name == "search":
                query = arguments.get("query")
                if not isinstance(query, str):
                    return self._malformed_result(
                        state,
                        query_id,
                        summary_turns,
                        retrieved_docids,
                        tool_call_counts,
                        turn_records,
                )
                doc_ids = self.backend.search(query)
                tool_call_counts["search"] += 1
                self._record_retrieved_docids(retrieved_docids, doc_ids)
                tool_result = json.dumps(doc_ids)
            elif tool_name == "get_document":
                doc_id = arguments.get("doc_id")
                if not isinstance(doc_id, str):
                    return self._malformed_result(
                        state,
                        query_id,
                        summary_turns,
                        retrieved_docids,
                        tool_call_counts,
                        turn_records,
                    )
                self._record_retrieved_docids(retrieved_docids, [doc_id])
                tool_result = self.backend.get_document(doc_id)
                tool_call_counts["get_document"] += 1
            else:
                return self._malformed_result(
                    state,
                    query_id,
                    summary_turns,
                    retrieved_docids,
                    tool_call_counts,
                    turn_records,
                )

            state.rounds.append(
                ToolRound(
                    assistant_message=Message(role="assistant", content=raw_output),
                    tool_call=ToolCallRecord(tool_name=tool_name, arguments=arguments, raw_output=raw_output),
                    tool_result=Message(role="tool", content=tool_result),
                )
            )

            compacted_state = self._compacted_state(state)
            if context_manager.should_summarize(compacted_state):
                summary_state, retired_count = self._build_summary_state(state)
                if retired_count == 0:
                    continue
                summary_generation_prompt = context_manager.build_summary_context(summary_state)
                context_manager.assert_fits(summary_generation_prompt)
                generated_summary = self.model.generate(summary_generation_prompt)
                if not generated_summary.strip():
                    continue
                state.latest_summary = generated_summary
                state.summarized_round_count += retired_count
                state.summary_count += 1
                summary_turn_id = f"summary-{state.summary_count}"
                summary_turns.append(summary_turn_id)
                turn_records.append(
                    {
                        "query_id": query_id,
                        "turn_id": summary_turn_id,
                        "kind": "summary",
                        "prompt": summary_generation_prompt,
                        "completion": generated_summary,
                    }
                )
