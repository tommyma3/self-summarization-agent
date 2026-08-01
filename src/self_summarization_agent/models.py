from dataclasses import dataclass, field
from typing import Any
from typing import Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["function"] = "function"


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass(slots=True)
class EpisodeState:
    query_id: str
    user_prompt: str
    context_threshold_tokens: int
    latest_summary: str | None = None
    summary_count: int = 0
    tool_turn_count: int = 0
    messages: list[Message] = field(default_factory=list)


@dataclass(slots=True)
class RuntimeResult:
    query_id: str
    status: str
    final_answer: str | None
    summary_turns: list[str]
    turn_rewards: dict[str, float]
    retrieved_docids: list[str]
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    turn_records: list[dict[str, Any]] = field(default_factory=list)
    trajectory_records: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, Any] = field(default_factory=dict)
