from dataclasses import dataclass
from typing import Callable

from self_summarization_agent.models import EpisodeState
from self_summarization_agent.prompts import (
    build_summary_prompt,
    build_summary_system_prompt,
    build_system_prompt,
    format_history_round,
)


@dataclass(slots=True)
class ContextManager:
    token_counter: Callable[[str], int]
    max_context_tokens: int
    safety_margin_tokens: int = 256

    def current_token_count(self, state: EpisodeState) -> int:
        pieces = [build_system_prompt(), state.user_prompt]
        if state.latest_summary:
            pieces.append(state.latest_summary)
        for round_record in state.rounds:
            pieces.append(
                format_history_round(
                    round_record.tool_call.tool_name,
                    round_record.tool_call.arguments,
                    round_record.tool_result.content,
                )
            )
        return self.token_counter("\n".join(pieces))

    def should_summarize(self, state: EpisodeState) -> bool:
        return self.current_token_count(state) >= state.context_threshold_tokens

    def build_summary_context(self, state: EpisodeState) -> str:
        pieces = [build_summary_system_prompt(), state.user_prompt]
        if state.latest_summary:
            pieces.append(state.latest_summary)
        for round_record in state.rounds:
            pieces.append(
                format_history_round(
                    round_record.tool_call.tool_name,
                    round_record.tool_call.arguments,
                    round_record.tool_result.content,
                )
            )
        pieces.append(build_summary_prompt())
        return "\n".join(pieces)

    def assert_fits(self, packed_prompt: str) -> None:
        packed_tokens = self.token_counter(packed_prompt)
        limit = max(1, self.max_context_tokens - self.safety_margin_tokens)
        if packed_tokens > limit:
            raise ValueError(f"Packed prompt exceeds safe limit: {packed_tokens} > {limit}")
