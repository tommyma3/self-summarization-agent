from dataclasses import dataclass
from typing import Callable

from self_summarization_agent.models import EpisodeState, Message
from self_summarization_agent.prompts import ConversationPrompt, build_summary_prompt


@dataclass(slots=True)
class ContextManager:
    token_counter: Callable[[str], int]
    max_context_tokens: int
    safety_margin_tokens: int = 256
    prompt_token_counter: Callable[[str], int] | None = None

    def _count_prompt(self, prompt: str) -> int:
        if self.prompt_token_counter is not None:
            return self.prompt_token_counter(prompt)
        return self.token_counter(prompt)

    def current_token_count(self, state: EpisodeState) -> int:
        return self._count_prompt(ConversationPrompt(state.messages))

    def should_summarize(self, state: EpisodeState) -> bool:
        return self.current_token_count(state) >= state.context_threshold_tokens

    def build_summary_context(self, state: EpisodeState, *, max_summary_tokens: int | None = None) -> ConversationPrompt:
        messages = list(state.messages)
        messages.append(Message(role="system", content=build_summary_prompt(max_summary_tokens=max_summary_tokens)))
        return ConversationPrompt(messages)

    def assert_fits(self, packed_prompt: str, *, reserved_tokens: int = 0) -> None:
        packed_tokens = self._count_prompt(packed_prompt)
        limit = max(1, self.max_context_tokens - self.safety_margin_tokens - reserved_tokens)
        if packed_tokens > limit:
            raise ValueError(f"Packed prompt exceeds safe limit: {packed_tokens} > {limit}")
