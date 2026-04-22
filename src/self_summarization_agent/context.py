import json
from dataclasses import dataclass
from typing import Callable

from self_summarization_agent.models import EpisodeState, ToolCallRecord
from self_summarization_agent.prompts import build_summary_prompt, build_summary_system_prompt, build_system_prompt


@dataclass(slots=True)
class ContextManager:
    token_counter: Callable[[str], int]
    max_context_tokens: int
    safety_margin_tokens: int = 256

    def _serialize_tool_call(self, tool_call: ToolCallRecord) -> str:
        return json.dumps(
            {
                "tool_name": tool_call.tool_name,
                "arguments": tool_call.arguments,
                "raw_output": tool_call.raw_output,
                "is_valid": tool_call.is_valid,
            },
            sort_keys=True,
        )

    def current_token_count(self, state: EpisodeState) -> int:
        pieces = [build_system_prompt(), state.user_prompt]
        if state.latest_summary:
            pieces.append(state.latest_summary)
        for round_record in state.rounds:
            pieces.extend(
                [
                    round_record.assistant_message.content,
                    self._serialize_tool_call(round_record.tool_call),
                    round_record.tool_result.content,
                ]
            )
        return self.token_counter("\n".join(pieces))

    def should_summarize(self, state: EpisodeState) -> bool:
        return self.current_token_count(state) >= state.context_threshold_tokens

    def build_summary_context(self, state: EpisodeState) -> str:
        pieces = [build_summary_system_prompt(), state.user_prompt]
        if state.latest_summary:
            pieces.append(state.latest_summary)
        for round_record in state.rounds:
            pieces.extend(
                [
                    round_record.assistant_message.content,
                    self._serialize_tool_call(round_record.tool_call),
                    round_record.tool_result.content,
                ]
            )
        pieces.append(build_summary_prompt())
        return "\n".join(pieces)

    def assert_fits(self, packed_prompt: str) -> None:
        packed_tokens = self.token_counter(packed_prompt)
        limit = max(1, self.max_context_tokens - self.safety_margin_tokens)
        if packed_tokens > limit:
            raise ValueError(f"Packed prompt exceeds safe limit: {packed_tokens} > {limit}")
