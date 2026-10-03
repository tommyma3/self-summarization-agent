from typing import Any, Protocol

from self_summarization_agent.models import Message


class AgentScaffold(Protocol):
    """Routing views must never be used to reconstruct sampled token history."""

    tools: list[dict[str, Any]]
    system_prompt: str
    finish_tool: str
    forced_control: str
    summary_control: str
    forced_body_regex: str
    fingerprint: str

    def parse(self, text: str, *, call_id: str, thinking: bool) -> Message | None: ...

    def validate(self, name: str, arguments: dict[str, Any]) -> bool: ...

    def is_complete(self, name: str, *, query_id: str, forced: bool) -> bool: ...

    def execute(self, name: str, arguments: dict[str, Any], *, query_id: str) -> str: ...
