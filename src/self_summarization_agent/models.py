from dataclasses import dataclass, field
from typing import Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class Message:
    role: Role
    content: str


@dataclass(slots=True)
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, str]
    raw_output: str
    is_valid: bool = True


@dataclass(slots=True)
class ToolRound:
    assistant_message: Message
    tool_call: ToolCallRecord
    tool_result: Message


@dataclass(slots=True)
class EpisodeState:
    query_id: str
    user_prompt: str
    context_threshold_tokens: int
    latest_summary: str | None = None
    summary_count: int = 0
    summarized_round_count: int = 0
    rounds: list[ToolRound] = field(default_factory=list)


@dataclass(slots=True)
class RuntimeResult:
    query_id: str
    status: str
    final_answer: str | None
    summary_turns: list[str]
    turn_rewards: dict[str, float]
    retrieved_docids: list[str]
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    turn_records: list[dict[str, str]] = field(default_factory=list)
