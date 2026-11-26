import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Iterator, Protocol

LOGGER = logging.getLogger(__name__)

from self_summarization_agent.backend import BrowseCompBackend, SearchResult
from self_summarization_agent.benchmarks.base import AgentScaffold
from self_summarization_agent.context import ContextManager
from self_summarization_agent.models import EpisodeState, Message, RuntimeResult, ToolCall
from self_summarization_agent.prompts import (
    ACTION_TOOLS,
    DOCUMENT_ID_ARGUMENT,
    ConversationPrompt,
    FORCED_FINISH_TOOL_CHOICE,
    build_compacted_messages,
    build_forced_answer_prompt,
    build_initial_messages,
    format_tool_response,
    format_compacted_summary,
    serialize_messages,
)
from self_summarization_agent.rewards import (
    BINARY_REWARD,
    Outcome,
    apply_malformed_tool_penalty,
    apply_terminal_reward,
    trainable_turn_ids_from_records,
)
from self_summarization_agent.trajectory import (
    COLLECTION_TOKEN_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
)
from self_summarization_agent.token_stream import IntervalTokenLedger, TokenRequest
from self_summarization_agent.token_renderer import UnsupportedTokenBoundary


_JSON_DECODER = json.JSONDecoder()
_THINK_END_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)
_THINK_START_RE = re.compile(r"^\s*<think\b[^>]*>", flags=re.IGNORECASE)
_THINK_CLOSE_LIKE_RE = re.compile(r"</\s*think[^>]*>", flags=re.IGNORECASE)
THINK_CLOSE_TOKEN = "</think>"
_SUMMARY_BLOCK_RE = re.compile(
    r"<\s*summary\s*>(.*?)<\s*/\s*summary\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
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
_FINAL_ANSWER_FIELD_RE = re.compile(
    r"^[ \t]*(?:(?:Explanation|Exact Answer|Confidence):|\*\*(?:Explanation|Exact Answer|Confidence):\*\*)",
    re.IGNORECASE | re.MULTILINE,
)
_FINAL_ANSWER_RE = re.compile(
    r"(?:Explanation:|\*\*Explanation:\*\*)[ \t]*(?P<explanation>.+?)\r?\n[ \t]*"
    r"(?:Exact Answer:|\*\*Exact Answer:\*\*)[ \t]*(?P<answer>.+?)\r?\n[ \t]*"
    r"(?:Confidence:|\*\*Confidence:\*\*)[ \t]*(?P<confidence>\d+(?:\.\d+)?)[ \t]*%?",
    re.IGNORECASE | re.DOTALL,
)


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


def classify_thinking_closure(raw_output: str, *, enable_thinking: bool) -> str | None:
    """Return a malformed reason for unclosed/wrongly-closed thinking, else None.

    Only the byte-exact ``</think>`` closes the block; known typos such as
    ``</thinking>`` and case/whitespace variants are classified, never repaired.
    """
    if not enable_thinking:
        return None
    close_like = list(_THINK_CLOSE_LIKE_RE.finditer(raw_output))
    exact = [match for match in close_like if match.group(0) == THINK_CLOSE_TOKEN]
    if exact and not any(match.group(0) != THINK_CLOSE_TOKEN for match in close_like):
        return None
    return "wrongly_closed_thinking" if close_like else "unclosed_thinking"


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


def _find_token_subsequence(haystack: list[int], needle: list[int], *, start: int = 0) -> int | None:
    if not needle:
        return start
    last_start = len(haystack) - len(needle)
    for index in range(max(0, start), last_start + 1):
        if haystack[index : index + len(needle)] == needle:
            return index
    return None


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
            normalized = {"tool_name": "get_document", "arguments": {DOCUMENT_ID_ARGUMENT: value}}
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
            if tool_name == "get_document" and "doc_id" in arguments and DOCUMENT_ID_ARGUMENT not in arguments:
                # Older JSON actions used doc_id. Canonicalize only the routing view.
                arguments[DOCUMENT_ID_ARGUMENT] = arguments.pop("doc_id")
            normalized = {"tool_name": tool_name, "arguments": arguments}
            return normalized, json.dumps(normalized, ensure_ascii=False)
        return None
    return None


def parse_model_tool_call(raw_output: str) -> tuple[dict[str, object], str] | None:
    # A plain final answer is a routing view, never a replacement for sampled
    # output. Strip the sampled ChatML terminator only from this parsed view.
    action_text = _action_text(raw_output).removesuffix("<|im_end|>").strip()
    if _FINAL_ANSWER_FIELD_RE.match(action_text):
        answer = _FINAL_ANSWER_RE.fullmatch(action_text)
        if (
            answer is None
            or not answer.group("explanation").strip()
            or not answer.group("answer").strip()
            or not 0 <= float(answer.group("confidence")) <= 100
            or len(_FINAL_ANSWER_FIELD_RE.findall(action_text)) != 3
        ):
            return None
        normalized = {"tool_name": "finish", "arguments": {"answer": action_text}}
        return normalized, json.dumps(normalized, ensure_ascii=False)
    tag_result = _parse_tag_tool_call(raw_output)
    if tag_result is not None:
        return tag_result
    if _contains_action_tag(_action_text(raw_output)):
        return None
    return _parse_json_tool_call(raw_output)


def extract_summary_output(raw_output: str) -> SummaryExtraction | None:
    extracted = _extract_completed_thinking(raw_output)
    if extracted is None:
        return None
    summary_match = _SUMMARY_BLOCK_RE.search(extracted.remainder)
    if summary_match is None:
        return None
    return SummaryExtraction(thinking=extracted.thinking, summary=summary_match.group(1).strip())


def extract_structured_summary(message: Message) -> SummaryExtraction | None:
    summary_match = _SUMMARY_BLOCK_RE.search(message.content)
    if summary_match is None:
        return None
    return SummaryExtraction(
        thinking=(message.reasoning_content or "").strip(),
        summary=summary_match.group(1).strip(),
    )


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

    @property
    def total_generated_tokens(self) -> int:
        """All model-generated completion tokens; excludes external tool results."""
        return (
            self.reasoning_generated_tokens
            + self.summary_generated_tokens
            + self.forced_answer_generated_tokens
        )

    @property
    def budget_consumed_tokens(self) -> int:
        """Episode budget usage from generated output plus appended tool results."""
        return self.total_generated_tokens + self.tool_result_tokens

    def as_dict(self, *, summary_count: int, turn_records: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "reasoning_generated_tokens": self.reasoning_generated_tokens,
            "summary_generated_tokens": self.summary_generated_tokens,
            "forced_answer_generated_tokens": self.forced_answer_generated_tokens,
            "tool_result_tokens": self.tool_result_tokens,
            "total_generated_tokens": self.total_generated_tokens,
            "budget_consumed_tokens": self.budget_consumed_tokens,
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
    interval_outputs: list["_GeneratedOutput"] = field(default_factory=list)
    interval_round_count: int = 0
    token_usage: _TokenUsage = field(default_factory=_TokenUsage)
    result: RuntimeResult | None = None
    token_ledger: IntervalTokenLedger | None = None
    token_request: TokenRequest | None = None
    pending_summary: ConversationPrompt | None = None


@dataclass(slots=True)
class _GeneratedOutput:
    text: str
    completion_tokens: int
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    cumulative_logprob: float | None = None
    token_logprobs: list[float] | None = None
    token_logprobs_mode: str | None = None
    message: Message | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


@dataclass(slots=True)
class _ToolInvocation:
    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass(slots=True)
class _PendingToolRound:
    active: _ActiveEpisode
    prompt: str
    raw_output: str
    normalized_output: str
    prompt_tokens: int
    completion_tokens: int
    assistant_message: Message | None
    calls: list[_ToolInvocation]


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
    # Maximum wall-clock seconds for all tool calls (search + get_document)
    # within a single round.  Remaining calls are skipped with an error string
    # so the episode can still complete.  None = no limit.
    tool_execution_timeout_seconds: float | None = 600
    reward_profile: str = BINARY_REWARD
    scaffold: AgentScaffold | None = None
    max_compactions: int | None = None
    force_answer_at_compaction_limit: bool = False
    token_renderer: Any = field(init=False, default=None)
    collection_metrics: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_summary_tokens < 1:
            raise ValueError(f"max_summary_tokens must be at least 1, got {self.max_summary_tokens}")
        if self.max_compactions is not None and (
            type(self.max_compactions) is not int or self.max_compactions < 0
        ):
            raise ValueError("max_compactions must be a nonnegative integer or None")
        create_renderer = getattr(self.model, "create_token_renderer", None)
        if create_renderer is not None:
            if not callable(getattr(self.model, "generate_token_batch", None)):
                raise ValueError("This collector does not implement token-input generation")
            self.token_renderer = create_renderer()
        if self.scaffold is not None:
            if self.token_renderer is None or not self._requires_exact_token_ids():
                raise ValueError("Benchmark scaffolds require exact token-in/token-out generation")
            from self_summarization_agent.token_renderer import QwenAgentTokenRenderer
            self.token_renderer = QwenAgentTokenRenderer(
                self.token_renderer.tokenizer, enable_thinking=self.token_renderer.enable_thinking,
                native_tools=True, scaffold_fingerprint=self.scaffold.fingerprint,
                forced_control=self.scaffold.forced_control, summary_control=self.scaffold.summary_control)

    def _initialize_ledger(self, active: _ActiveEpisode) -> None:
        if self.token_renderer is not None:
            if active.token_ledger is not None and not active.token_ledger.finalized:
                raise RuntimeError("Cannot replace an active interval before finalization")
            active.token_ledger = IntervalTokenLedger(
                self.token_renderer.render_initial_state(self._build_runtime_prompt(active.state)),
                fingerprint=self.token_renderer.fingerprint)
            active.token_request = None

    def _token_context_limit(self) -> int:
        return min(self.max_context_tokens, getattr(self.model, "max_model_len", None) or self.max_context_tokens)

    def _token_input(self, active: _ActiveEpisode, prompt: ConversationPrompt) -> TokenRequest:
        assert active.token_ledger is not None
        request = active.token_ledger.request(self.token_renderer.header(prompt.generation_kind),
                                              generation_kind=prompt.generation_kind)
        available = self._token_context_limit() - len(request.prompt_token_ids)
        return replace(request, max_new_tokens=max(0, min(getattr(self.model, "max_new_tokens", available), available)),
                       # Prose-only reasoning in constrained control turns prevents
                       # a wildcard from consuming a closing think tag and tools.
                       response_regex=((r"[^<]*</think>\s*" if self.token_renderer.enable_thinking else "")
                                       + (r"<summary>[\s\S]+?</summary>" if prompt.generation_kind == "summary"
                                          else self.scaffold.forced_body_regex) + r"\s*")
                       if self.scaffold is not None and prompt.generation_kind in {"summary", "forced_answer"} else None)

    def _commit_token_input(self, active: _ActiveEpisode, prompt: ConversationPrompt):
        if active.token_ledger is None:
            return prompt
        request = self._token_input(active, prompt)
        active.token_ledger.commit(request)
        active.token_request = request
        return request

    def _append_output(self, active: _ActiveEpisode, output: "_GeneratedOutput", kind: str) -> None:
        if active.token_ledger is not None:
            if output.prompt_token_ids is None or output.completion_token_ids is None:
                raise RuntimeError("TITO requires exact prompt and completion IDs")
            active.token_ledger.append_sampled(output.completion_token_ids,
                                               expected_prompt=output.prompt_token_ids, kind=kind)
        active.interval_outputs.append(output)

    @property
    def _uses_native_tools(self) -> bool:
        return self.scaffold is not None or bool(getattr(self.model, "supports_native_tools", False))

    def _tool_calls_used(self, active: _ActiveEpisode) -> int:
        if self.scaffold is not None:
            return sum(active.tool_call_counts.values())
        return active.tool_call_counts.get("search", 0) + active.tool_call_counts.get("get_document", 0)

    def _remaining_tool_calls(self, active: _ActiveEpisode) -> int | None:
        if self.max_compactions is not None or self.max_tool_calls is None:
            return None
        return max(0, self.max_tool_calls - self._tool_calls_used(active))

    def _generated_token_budget_exhausted(self, active: _ActiveEpisode) -> bool:
        return (
            self.max_compactions is None
            and self.generated_token_budget is not None
            and active.token_usage.budget_consumed_tokens >= self.generated_token_budget
        )

    def _prompt_token_count(self, active: _ActiveEpisode, prompt: str) -> int:
        prompt_counter = getattr(self.model, "count_prompt_tokens", None)
        prompt_tokens = (len(active.token_ledger) + len(self.token_renderer.header(prompt.generation_kind))
                         if active.token_ledger is not None else
                         prompt_counter(prompt) if prompt_counter is not None else self.token_counter(prompt))
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
        if self._uses_native_tools:
            return ConversationPrompt(
                state.messages,
                tools=self.scaffold.tools if self.scaffold is not None else ACTION_TOOLS,
                tool_choice="auto",
                parallel_tool_calls=self.scaffold is None,
                generation_kind="action",
            )
        return ConversationPrompt(state.messages, generation_kind="action")

    def _build_forced_answer_prompt(self, active: _ActiveEpisode) -> ConversationPrompt:
        messages = list(active.state.messages)
        messages.append(Message(role="system", content=self.scaffold.forced_control if self.scaffold is not None else build_forced_answer_prompt()))
        if self._uses_native_tools:
            return ConversationPrompt(
                messages,
                tools=self.scaffold.tools if self.scaffold is not None else ACTION_TOOLS,
                tool_choice={"type": "function", "function": {"name": self.scaffold.finish_tool}} if self.scaffold is not None else FORCED_FINISH_TOOL_CHOICE,
                parallel_tool_calls=False,
                generation_kind="forced_answer",
            )
        return ConversationPrompt(messages, generation_kind="forced_answer")

    def _terminate_on_context_overflow(self, active: _ActiveEpisode, prompt: ConversationPrompt) -> bool:
        """Penalize and finish an episode whose next prompt exceeds the safe context limit.

        Tool results are appended to the episode state after the round-level
        budget checks, so a large retrieval payload can push the following
        round's prompt past the safe limit before the generated-token budget
        trips.  Letting ContextManager.assert_fits raise here would abort the
        whole collection process; terminate just this episode instead.
        Returns True when the episode was terminated rather than queued for
        generation.
        """
        try:
            if active.token_ledger is not None:
                if self._prompt_token_count(active, prompt) >= self._token_context_limit():
                    raise ValueError("TITO prompt exceeds the context limit")
            else:
                active.context_manager.assert_fits(prompt)
        except ValueError as exc:
            self._finish_context_overflow(active, prompt, str(exc))
            return True
        return False

    def _finish_context_overflow(self, active: _ActiveEpisode, prompt: ConversationPrompt | None, reason: str) -> None:
        LOGGER.warning("Terminating episode %s: %s", active.state.query_id, reason)
        self._finalize_trajectory(active, active.state.messages, termination_kind="context_overflow", prompt=prompt)
        active.result = self._penalized_result(active, status="context_length_exceeded")

    def _next_tool_turn_id(self, state: EpisodeState) -> str:
        return f"tool-{state.tool_turn_count + 1}"

    def _requires_exact_token_ids(self) -> bool:
        return bool(getattr(self.model, "require_exact_token_ids", False))

    def _history_rewrite_detected(self, active: _ActiveEpisode, generated_output: "_GeneratedOutput") -> bool:
        """Whether the server-rendered prompt rewrote instead of extended the interval.

        The provider's chat template can normalize a malformed assistant turn
        (for example an unclosed or misspelled think block), so the next
        request's prompt no longer starts with the previously sampled tokens.
        Training on a continuation of that prompt would condition on tokens the
        policy never saw, so the generation must be discarded.
        """
        if active.token_ledger is not None:
            return generated_output.prompt_token_ids is None or tuple(generated_output.prompt_token_ids) != active.token_ledger.ids
        if not self._requires_exact_token_ids():
            return False
        if generated_output.prompt_token_ids is None or not active.interval_outputs:
            return False
        previous_output = active.interval_outputs[-1]
        if previous_output.prompt_token_ids is None or previous_output.completion_token_ids is None:
            return False
        previous_full_ids = list(previous_output.prompt_token_ids) + list(previous_output.completion_token_ids)
        current_prompt_ids = list(generated_output.prompt_token_ids)
        return current_prompt_ids[: len(previous_full_ids)] != previous_full_ids

    def _history_rewritten_result(
        self,
        active: _ActiveEpisode,
        prompt: str,
        generated_output: "_GeneratedOutput",
        *,
        prompt_tokens: int,
        generation_kind: str,
    ) -> RuntimeResult:
        """Penalize an episode whose interval history was rewritten by the provider.

        The discarded generation is kept only as a diagnostic turn record; the
        interval is finalized with the generations sampled before the rewrite.
        """
        recorded_turns = list(active.turn_records)
        turn_record: dict[str, Any] = {
            "query_id": active.state.query_id,
            "turn_id": self._next_tool_turn_id(active.state),
            "kind": "tool",
            "prompt": prompt,
            "completion": generated_output.text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generated_output.completion_tokens,
            "generation_kind": generation_kind,
            "history_rewrite": True,
        }
        recorded_turns.append(turn_record)
        interval_messages = (
            list(prompt.messages) if isinstance(prompt, ConversationPrompt) else list(active.state.messages)
        )
        self._finalize_trajectory(
            active,
            interval_messages,
            termination_kind="malformed",
            prompt=prompt if isinstance(prompt, ConversationPrompt) else None,
        )
        return self._penalized_result(active, status="history_rewrite_detected", turn_records=recorded_turns)

    def _collection_token_payload(self, active: _ActiveEpisode) -> dict[str, Any] | None:
        if not active.interval_outputs:
            return None
        final_output = active.interval_outputs[-1]
        require_exact = bool(getattr(self.model, "require_exact_token_ids", False))
        if final_output.prompt_token_ids is None or final_output.completion_token_ids is None:
            if require_exact:
                raise RuntimeError(
                    "Exact collection token IDs are required, but the final generation did not return them"
                )
            return None

        prompt_token_ids = list(final_output.prompt_token_ids)
        completion_token_ids = list(final_output.completion_token_ids)
        full_token_ids = prompt_token_ids + completion_token_ids
        assistant_token_mask = [False] * len(full_token_ids)
        generations: list[dict[str, Any]] = []
        for index, output in enumerate(active.interval_outputs):
            if output.prompt_token_ids is None or output.completion_token_ids is None:
                if require_exact:
                    raise RuntimeError(
                        f"Exact collection token IDs are missing for interval output {index + 1}"
                    )
                return None
            generation_prompt_ids = list(output.prompt_token_ids)
            generation_completion_ids = list(output.completion_token_ids)
            generations.append(
                {
                    "index": index,
                    "prompt_token_ids": generation_prompt_ids,
                    "completion_token_ids": generation_completion_ids,
                    "full_token_ids": generation_prompt_ids + generation_completion_ids,
                    "cumulative_logprob": output.cumulative_logprob,
                    "completion_token_logprobs": (
                        list(output.token_logprobs) if output.token_logprobs is not None else None
                    ),
                    "logprobs_mode": output.token_logprobs_mode,
                    "finish_reason": output.finish_reason,
                }
            )
        for index in range(1, len(generations)):
            previous_full_ids = generations[index - 1]["full_token_ids"]
            if generations[index]["prompt_token_ids"][: len(previous_full_ids)] != previous_full_ids:
                raise RuntimeError(
                    f"Collection generation {index + 1} rewrote the interval history instead of "
                    "extending it; the provider chat template changed the sampled tokens"
                )
        if active.token_ledger is not None:
            payload = active.token_ledger.payload()
            payload.update(prompt_token_ids=prompt_token_ids, completion_token_ids=completion_token_ids,
                           generations=generations)
            return payload
        search_start = 0
        for index, output in enumerate(active.interval_outputs[:-1]):
            token_ids = output.completion_token_ids
            assert token_ids is not None
            token_start = _find_token_subsequence(prompt_token_ids, token_ids, start=search_start)
            if token_start is None:
                if require_exact:
                    raise RuntimeError(
                        "A sampled assistant completion is not an exact token subsequence of the final "
                        "collection prompt; the server chat template or structured history changed"
                    )
                # Rebuild a degraded mask that only covers the final completion.
                assistant_token_mask = [False] * len(full_token_ids)
                break
            for token_index in range(token_start, token_start + len(token_ids)):
                assistant_token_mask[token_index] = True
            search_start = token_start + len(token_ids)
        completion_start = len(prompt_token_ids)
        for token_index in range(completion_start, len(full_token_ids)):
            assistant_token_mask[token_index] = True
        return {
            "version": COLLECTION_TOKEN_VERSION,
            "prompt_token_ids": prompt_token_ids,
            "completion_token_ids": completion_token_ids,
            "full_token_ids": full_token_ids,
            "assistant_token_mask": assistant_token_mask,
            "generations": generations,
        }

    def _finalize_trajectory(
        self,
        active: _ActiveEpisode,
        messages: list[Message] | tuple[Message, ...],
        *,
        termination_kind: str,
        prompt: ConversationPrompt | None = None,
    ) -> str:
        trajectory_id = f"trajectory-{len(active.trajectory_records) + 1}"
        serialized_messages = serialize_messages(messages)
        assistant_outputs = [output.text for output in active.interval_outputs]
        if not assistant_outputs:
            assistant_outputs = [message.content for message in messages if message.role == "assistant"]
        if not assistant_outputs:
            return trajectory_id
        record: dict[str, Any] = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "query_id": active.state.query_id,
            "turn_id": trajectory_id,
            "segment_index": len(active.trajectory_records) + 1,
            "prefix_summary_turn_id": active.summary_turns[-1] if active.summary_turns else None,
            "kind": "trajectory",
            "termination_kind": termination_kind,
            "messages": serialized_messages,
            "prompt": str(prompt) if prompt is not None else str(ConversationPrompt(messages)),
            "completion": "\n".join(assistant_outputs),
            "turn_ids": list(active.interval_turn_ids),
            "assistant_completion_count": len(assistant_outputs),
        }
        if prompt is not None and prompt.tools:
            record["tools"] = list(prompt.tools)
        collection_tokens = self._collection_token_payload(active)
        if collection_tokens is not None:
            record["collection_tokens"] = collection_tokens
        active.trajectory_records.append(record)
        if active.token_ledger is not None:
            active.token_ledger.finalize()
        active.interval_turn_ids.clear()
        active.interval_outputs.clear()
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
        generated_message = active.interval_outputs[-1].message if active.interval_outputs else None
        interval_messages.append(generated_message or Message(role="assistant", content=completion))
        self._finalize_trajectory(
            active,
            interval_messages,
            termination_kind="malformed",
            prompt=prompt if isinstance(prompt, ConversationPrompt) else None,
        )
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
                trainable_turn_ids_from_records(active.trajectory_records),
                reward_profile=self.reward_profile,
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
                reward_profile=self.reward_profile,
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

    def _generated_output(self, result) -> _GeneratedOutput:
        text = str(getattr(result, "text", ""))
        completion_token_ids = getattr(result, "completion_token_ids", None)
        completion_tokens = (
            len(completion_token_ids)
            if completion_token_ids is not None
            else self._completion_token_count(text)
        )
        return _GeneratedOutput(
            text=text,
            completion_tokens=completion_tokens,
            prompt_token_ids=(
                list(result.prompt_token_ids)
                if getattr(result, "prompt_token_ids", None) is not None
                else None
            ),
            completion_token_ids=(
                list(completion_token_ids) if completion_token_ids is not None else None
            ),
            cumulative_logprob=(
                float(result.cumulative_logprob)
                if getattr(result, "cumulative_logprob", None) is not None
                else None
            ),
            token_logprobs=(
                list(result.token_logprobs)
                if getattr(result, "token_logprobs", None) is not None
                else None
            ),
            token_logprobs_mode=getattr(result, "token_logprobs_mode", None),
            message=getattr(result, "message", None),
            finish_reason=getattr(result, "finish_reason", None),
            usage=getattr(result, "usage", None),
        )

    def _generate_batch(self, prompts: list[str]) -> list[_GeneratedOutput]:
        generate_batch_with_metadata = (getattr(self.model, "generate_token_batch", None)
            if prompts and isinstance(prompts[0], TokenRequest)
            else getattr(self.model, "generate_batch_with_metadata", None))
        if generate_batch_with_metadata is not None:
            started = time.monotonic()
            results = generate_batch_with_metadata(prompts)
            self.collection_metrics["generation_wait_seconds"] = (
                self.collection_metrics.get("generation_wait_seconds", 0.0) + time.monotonic() - started
            )
            if len(results) != len(prompts):
                raise ValueError(
                    f"Batch generator returned {len(results)} metadata results for {len(prompts)} prompts"
                )
            return [self._generated_output(result) for result in results]
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

    def _initial_messages(self, user_prompt: str) -> list[Message]:
        if self.scaffold is not None:
            return [Message(role="system", content=self.scaffold.system_prompt),
                    Message(role="user", content=user_prompt)]
        return build_initial_messages(user_prompt, native_tools=self._uses_native_tools)

    def _new_active_episode(self, query_id: str, user_prompt: str) -> _ActiveEpisode:
        state = EpisodeState(
            query_id=query_id,
            user_prompt=user_prompt,
            context_threshold_tokens=self.context_threshold_tokens,
            messages=self._initial_messages(user_prompt),
        )
        active = _ActiveEpisode(
            state=state,
            context_manager=ContextManager(
                token_counter=self.token_counter,
                max_context_tokens=self.max_context_tokens,
                safety_margin_tokens=0,
                prompt_token_counter=getattr(self.model, "count_prompt_tokens", None),
            ),
        )
        self._initialize_ledger(active)
        return active

    def _terminal_result(
        self,
        active: _ActiveEpisode,
        *,
        status: str,
        outcome: Outcome,
        answer: str | None = None,
    ) -> RuntimeResult:
        return RuntimeResult(
            query_id=active.state.query_id,
            status=status,
            final_answer=answer,
            summary_turns=list(active.summary_turns),
            turn_rewards=apply_terminal_reward(
                outcome=outcome,
                reward_profile=self.reward_profile,
                trainable_turn_ids=trainable_turn_ids_from_records(active.trajectory_records),
            ),
            retrieved_docids=list(active.retrieved_docids),
            tool_call_counts=dict(active.tool_call_counts),
            turn_records=list(active.turn_records),
            trajectory_records=list(active.trajectory_records),
            token_usage=self._token_usage_payload(active),
        )

    def _parse_action_output(
        self,
        generated_output: _GeneratedOutput,
    ) -> tuple[list[_ToolInvocation], str, Message | None] | None:
        if not self._uses_native_tools:
            parsed = parse_model_tool_call(generated_output.text)
            if parsed is None:
                return None
            payload, normalized_output = parsed
            return [_ToolInvocation(payload["tool_name"], payload["arguments"])], normalized_output, None
        message = generated_output.message
        if self.scaffold is not None:
            from hashlib import sha256
            call_id = "call_" + sha256(json.dumps([
                generated_output.prompt_token_ids, generated_output.completion_token_ids
            ]).encode()).hexdigest()[:24]
            message = self.scaffold.parse(generated_output.text, call_id=call_id,
                                          thinking=self.token_renderer.enable_thinking)
        if message is None or not message.tool_calls:
            return None
        calls = [_ToolInvocation(call.name, call.arguments, call.id) for call in message.tool_calls]
        ids = {call.id for call in calls}
        if len(ids) != len(calls) or not all(ids):
            return None
        payloads = [{"tool_name": call.name, "arguments": call.arguments} for call in calls]
        return calls, json.dumps(payloads[0] if len(calls) == 1 else payloads, ensure_ascii=False), message

    def _prepare_action_output(
        self,
        active: _ActiveEpisode,
        generated_output: _GeneratedOutput,
        prompt: str | None = None,
        prompt_tokens: int | None = None,
        generation_kind: str = "action",
    ) -> _PendingToolRound | None:
        state = active.state
        query_id = state.query_id
        prompt = prompt if prompt is not None else self._build_runtime_prompt(state)
        prompt_tokens = prompt_tokens if prompt_tokens is not None else self._prompt_token_count(active, prompt)
        if generated_output.prompt_token_ids is not None:
            prompt_tokens = len(generated_output.prompt_token_ids)
            active.token_usage.max_prompt_tokens_seen = max(
                active.token_usage.max_prompt_tokens_seen,
                prompt_tokens,
            )
        if self._history_rewrite_detected(active, generated_output):
            active.result = self._history_rewritten_result(
                active,
                prompt,
                generated_output,
                prompt_tokens=prompt_tokens,
                generation_kind=generation_kind,
            )
            return None
        self._append_output(active, generated_output, generation_kind)
        raw_output = generated_output.text
        if generation_kind == "forced_answer":
            active.token_usage.forced_answer_generated_tokens += generated_output.completion_tokens
        else:
            active.token_usage.reasoning_generated_tokens += generated_output.completion_tokens
        if (
            active.token_ledger is not None
            and self.token_renderer is not None
            and self.token_renderer.enable_thinking
        ):
            closure_reason = classify_thinking_closure(raw_output, enable_thinking=True)
            if closure_reason is not None:
                active.turn_records.append({
                    "kind": "format_anomaly",
                    "query_id": state.query_id,
                    "anomaly": closure_reason,
                })
                active.result = self._malformed_result(
                    active,
                    prompt,
                    raw_output,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=generated_output.completion_tokens,
                    generation_kind=generation_kind,
                )
                return None
        parsed_action = self._parse_action_output(generated_output)
        if parsed_action is None:
            active.result = self._malformed_result(
                active,
                prompt,
                raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_output.completion_tokens,
                generation_kind=generation_kind,
            )
            return
        calls, normalized_output, assistant_message = parsed_action
        first_call = calls[0]
        if len(calls) > 1 and (self.scaffold is not None or any(call.name == "finish" for call in calls)):
            active.result = self._malformed_result(
                active, prompt, raw_output, prompt_tokens=prompt_tokens,
                completion_tokens=generated_output.completion_tokens, generation_kind=generation_kind,
            )
            return None

        if (
            any(call.name in {"search", "get_document"} for call in calls)
            and self._requires_exact_token_ids()
            and active.token_ledger is None
            and _extract_completed_thinking(raw_output) is None
        ):
            # The provider cannot re-render this completion losslessly on the
            # next request (its think block never closes), so continuing would
            # rewrite the interval history; treat it like a malformed action.
            active.result = self._malformed_result(
                active,
                prompt,
                raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_output.completion_tokens,
                generation_kind=generation_kind,
            )
            return None

        if active.token_ledger is not None:
            try:
                self.token_renderer.validate_completion_boundary(
                    generated_output.completion_token_ids, generated_output.finish_reason)
            except UnsupportedTokenBoundary:
                active.result = self._malformed_result(active, prompt, raw_output,
                    prompt_tokens=prompt_tokens, completion_tokens=generated_output.completion_tokens,
                    generation_kind=generation_kind)
                return None

        tool_name, arguments = first_call.name, first_call.arguments
        scaffold_complete = self.scaffold is not None and self.scaffold.is_complete(
            tool_name, query_id=query_id, forced=False
        )

        if scaffold_complete or (self.scaffold is None and tool_name == "finish"):
            answer = "" if scaffold_complete else arguments.get("answer")
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
            if assistant_message is not None:
                turn_record["assistant_message"] = serialize_messages([assistant_message])[0]
                turn_record["reasoning"] = assistant_message.reasoning_content or ""
            if first_call.id is not None:
                turn_record["tool_call_id"] = first_call.id
            active.turn_records.append(turn_record)
            active.interval_turn_ids.append("final-answer")
            interval_messages = list(prompt.messages) if isinstance(prompt, ConversationPrompt) else list(state.messages)
            interval_messages.append(
                assistant_message or Message(role="assistant", content=raw_output)
            )
            self._finalize_trajectory(
                active,
                interval_messages,
                termination_kind="final_answer",
                prompt=prompt if isinstance(prompt, ConversationPrompt) else None,
            )
            active.result = self._terminal_result(
                active, status="completed", outcome="correct_answer", answer=answer
            )
            return None

        retrieval_arguments = {"search": "query", "get_document": DOCUMENT_ID_ARGUMENT}
        if self.scaffold is None and any(
            call.name not in retrieval_arguments
            or not isinstance(call.arguments.get(retrieval_arguments[call.name]), str)
            for call in calls
        ):
            active.result = self._malformed_result(
                active,
                prompt,
                raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_output.completion_tokens,
                generation_kind=generation_kind,
            )
            return None

        action = _PendingToolRound(
            active=active,
            prompt=prompt,
            raw_output=raw_output,
            normalized_output=normalized_output,
            prompt_tokens=prompt_tokens,
            completion_tokens=generated_output.completion_tokens,
            assistant_message=assistant_message,
            calls=calls,
        )
        if active.token_ledger is not None:
            empty_responses = self._tool_result_messages(action, [""] * len(calls))
            minimum_length = len(active.token_ledger) + len(self._render_tool_results(active, empty_responses))
            if minimum_length > self._token_context_limit():
                # There is no room even for linked empty responses. Stop before
                # executing the calls, with the entire sampled action retained.
                state.messages.append(assistant_message or Message(role="assistant", content=raw_output))
                self._finish_context_overflow(
                    active,
                    prompt if isinstance(prompt, ConversationPrompt) else None,
                    "Tool response framing exceeds the context limit",
                )
                return None
        return action

    def _search_many(self, queries: list[str]) -> list[list[SearchResult]]:
        search_many = getattr(self.backend, "search_many", None)
        if search_many is None:
            return [self.backend.search(query) for query in queries]
        results = search_many(queries)
        if len(results) != len(queries):
            raise ValueError(f"search_many returned {len(results)} result batches for {len(queries)} queries")
        return results

    def _tool_result_messages(self, action: _PendingToolRound, tool_results: list[str]) -> list[Message]:
        if self._uses_native_tools:
            return [
                Message(role="tool", content=result, tool_call_id=call.id)
                for call, result in zip(action.calls, tool_results, strict=True)
            ]
        tool_result, = tool_results
        return [Message(role="user", content=format_tool_response(tool_result))]

    def _render_tool_results(self, active: _ActiveEpisode, messages: list[Message]) -> tuple[int, ...]:
        output = active.interval_outputs[-1]
        return self.token_renderer.render_tool_results(
            messages, completion_ids=output.completion_token_ids, finish_reason=output.finish_reason
        )

    def _fit_tool_results(
        self, action: _PendingToolRound, tool_results: list[str], budget: int
    ) -> tuple[list[str], list[Message], tuple[int, ...]]:
        # Fit the entire linked response block before its first append. A
        # terminal ledger can include this block even if no later request fits.
        chosen_results = [""] * len(tool_results)
        messages = self._tool_result_messages(action, chosen_results)
        ids = self._render_tool_results(action.active, messages)
        budget = max(len(ids), budget)
        note = "\n[Observation truncated to fit the context.]"
        low, high = 0, max(map(len, tool_results))
        while low <= high:
            cap = (low + high) // 2
            candidates = [result[:cap] + note if len(result) > cap else result for result in tool_results]
            candidate_messages = self._tool_result_messages(action, candidates)
            candidate_ids = self._render_tool_results(action.active, candidate_messages)
            if len(candidate_ids) <= budget:
                chosen_results, messages, ids = candidates, candidate_messages, candidate_ids
                low = cap + 1
            else:
                high = cap - 1
        return chosen_results, messages, ids

    def _apply_tool_results(self, action: _PendingToolRound, tool_results: list[str]) -> None:
        active = action.active
        state = active.state
        query_id = state.query_id
        if self._uses_native_tools and action.assistant_message is None:
            raise RuntimeError("Native tool round is missing its assistant message")
        assistant_message = action.assistant_message or Message(role="assistant", content=action.raw_output)
        result_messages = self._tool_result_messages(action, tool_results)
        if active.token_ledger is not None:
            result_ids = self._render_tool_results(active, result_messages)
            available = self._token_context_limit() - len(active.token_ledger)
            result_budget = available
            if self.scaffold is not None or len(result_ids) > available:
                result_budget -= max(
                    len(self.token_renderer.header("summary")) + self.max_summary_tokens,
                    len(self.token_renderer.header("forced_answer")) + getattr(self.model, "max_new_tokens", 1024),
                )
            if len(result_ids) > result_budget:
                tool_results, result_messages, result_ids = self._fit_tool_results(action, tool_results, result_budget)
            active.token_ledger.append_external(result_ids, kind="tool_result")
        state.messages.extend([assistant_message, *result_messages])
        active.token_usage.tool_result_tokens += sum(self._completion_token_count(result) for result in tool_results)

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
        if action.assistant_message is not None:
            turn_record["assistant_message"] = serialize_messages([action.assistant_message])[0]
            turn_record["reasoning"] = action.assistant_message.reasoning_content or ""
            if len(action.calls) == 1:
                turn_record["tool_call_id"] = action.calls[0].id
        active.turn_records.append(turn_record)
        active.interval_turn_ids.append(tool_turn_id)
        state.tool_turn_count += 1
        active.interval_round_count += 1

    def _execute_pending_tool_actions(self, actions: list[_PendingToolRound]) -> None:
        if self.scaffold is not None:
            for action in actions:
                call, = action.calls
                output = self.scaffold.execute(call.name, call.arguments,
                                               query_id=action.active.state.query_id)
                counts = action.active.tool_call_counts
                counts[call.name] = counts.get(call.name, 0) + 1
                self._apply_tool_results(action, [output])
            return
        tool_started = time.monotonic()

        allowed_calls = [action.calls[:self._remaining_tool_calls(action.active)] for action in actions]
        queries = [call.arguments["query"] for calls in allowed_calls for call in calls if call.name == "search"]
        search_batches = iter(self._search_many(queries) if queries else [])

        for action, calls in zip(actions, allowed_calls, strict=True):
            results = []
            for call in calls:
                action.active.tool_call_counts[call.name] += 1
                if call.name == "search":
                    hits = next(search_batches)
                    self._record_search_result_docids(action.active.retrieved_docids, hits)
                    results.append(json.dumps(hits, ensure_ascii=False))
                else:
                    results.append(self._retrieve_document(
                        action.active, call.arguments[DOCUMENT_ID_ARGUMENT], tool_started
                    ))
            # Every sampled call needs a linked response, including calls skipped by the budget.
            results.extend(["Error: Tool call budget exhausted."] * (len(action.calls) - len(calls)))
            self._apply_tool_results(action, results)
        self.collection_metrics["tool_seconds"] = (
            self.collection_metrics.get("tool_seconds", 0.0) + time.monotonic() - tool_started
        )

    def _retrieve_document(self, active: _ActiveEpisode, doc_id: str, tool_started: float) -> str:
        if (
            self.tool_execution_timeout_seconds is not None
            and time.monotonic() - tool_started >= self.tool_execution_timeout_seconds
        ):
            LOGGER.warning("Tool execution timeout (%.0fs) reached; skipping get_document for %s",
                           self.tool_execution_timeout_seconds, doc_id)
            return "Error: Tool execution timed out during document retrieval."
        self._record_retrieved_docids(active.retrieved_docids, [doc_id])
        try:
            return self.backend.get_document(doc_id)
        except Exception:
            LOGGER.warning("Failed to retrieve document %s", doc_id, exc_info=True)
            return f"Error: Document '{doc_id}' not found in retrieval index."

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
        if generated_output.prompt_token_ids is not None:
            prompt_tokens = len(generated_output.prompt_token_ids)
            active.token_usage.max_prompt_tokens_seen = max(
                active.token_usage.max_prompt_tokens_seen,
                prompt_tokens,
            )
        if self._history_rewrite_detected(active, generated_output):
            active.result = self._history_rewritten_result(
                active,
                prompt,
                generated_output,
                prompt_tokens=prompt_tokens,
                generation_kind="forced_answer",
            )
            return
        self._append_output(active, generated_output, "forced_answer")
        raw_output = generated_output.text
        active.token_usage.forced_answer_generated_tokens += generated_output.completion_tokens
        if (
            active.token_ledger is not None
            and self.token_renderer is not None
            and self.token_renderer.enable_thinking
        ):
            closure_reason = classify_thinking_closure(raw_output, enable_thinking=True)
            if closure_reason is not None:
                active.turn_records.append({
                    "kind": "format_anomaly",
                    "query_id": state.query_id,
                    "anomaly": closure_reason,
                })
                active.result = self._malformed_result(
                    active,
                    prompt,
                    raw_output,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=generated_output.completion_tokens,
                    generation_kind="forced_answer",
                )
                return
        parsed_action = self._parse_action_output(generated_output)
        if parsed_action is None or len(parsed_action[0]) != 1:
            active.result = self._malformed_result(
                active,
                prompt,
                raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_output.completion_tokens,
                generation_kind="forced_answer",
            )
            return

        (call,), normalized_output, assistant_message = parsed_action
        tool_name, arguments = call.name, call.arguments
        if active.token_ledger is not None:
            try:
                self.token_renderer.validate_completion_boundary(
                    generated_output.completion_token_ids, generated_output.finish_reason)
            except UnsupportedTokenBoundary:
                active.result = self._malformed_result(active, prompt, raw_output,
                    prompt_tokens=prompt_tokens, completion_tokens=generated_output.completion_tokens,
                    generation_kind="forced_answer")
                return
        if self.scaffold is not None:
            if self.scaffold.is_complete(tool_name, query_id=query_id, forced=True):
                tool_name, arguments = "finish", {"answer": ""}
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
        if assistant_message is not None:
            turn_record["assistant_message"] = serialize_messages([assistant_message])[0]
            turn_record["reasoning"] = assistant_message.reasoning_content or ""
        if call.id is not None:
            turn_record["tool_call_id"] = call.id
        active.turn_records.append(turn_record)
        active.interval_turn_ids.append("final-answer")
        interval_messages = list(prompt.messages) if isinstance(prompt, ConversationPrompt) else list(state.messages)
        interval_messages.append(assistant_message or Message(role="assistant", content=raw_output))
        self._finalize_trajectory(
            active,
            interval_messages,
            termination_kind="forced_answer",
            prompt=prompt if isinstance(prompt, ConversationPrompt) else None,
        )
        active.result = self._terminal_result(
            active, status="completed", outcome="correct_answer", answer=answer
        )

    def _build_summary_prompt_for_active(self, active: _ActiveEpisode) -> ConversationPrompt | None:
        if self._generated_token_budget_exhausted(active):
            return None
        current_prompt = self._build_runtime_prompt(active.state)
        current_prompt_tokens = self._prompt_token_count(active, current_prompt)
        if current_prompt_tokens < active.state.context_threshold_tokens:
            return None
        if active.interval_round_count == 0:
            return None
        if self.max_compactions is not None and len(active.summary_turns) >= self.max_compactions:
            if self.force_answer_at_compaction_limit:
                active.token_usage.forced_answer_reasons.append("max_compactions")
                return None
            self._finalize_trajectory(
                active, active.state.messages, termination_kind="overlong", prompt=current_prompt
            )
            active.result = self._terminal_result(active, status="overlong", outcome="overlong")
            return None
        prompt = active.context_manager.build_summary_context(
            active.state,
            max_summary_tokens=self.max_summary_tokens,
        )
        if self.scaffold is not None:
            prompt = ConversationPrompt([*active.state.messages,
                Message(role="user", content=self.scaffold.summary_control)], generation_kind="summary")
        if self._uses_native_tools:
            prompt = ConversationPrompt(
                prompt.messages,
                tools=self.scaffold.tools if self.scaffold is not None else ACTION_TOOLS,
                tool_choice="none",
                parallel_tool_calls=False,
                generation_kind="summary",
            )
        if active.token_ledger is not None:
            if self._terminate_on_context_overflow(active, prompt):
                return None
        else:
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
        if generated_output.prompt_token_ids is not None:
            prompt_tokens = len(generated_output.prompt_token_ids)
            active.token_usage.max_prompt_tokens_seen = max(
                active.token_usage.max_prompt_tokens_seen,
                prompt_tokens,
            )
        if self._history_rewrite_detected(active, generated_output):
            active.result = self._history_rewritten_result(
                active,
                prompt,
                generated_output,
                prompt_tokens=prompt_tokens,
                generation_kind="summary",
            )
            return
        self._append_output(active, generated_output, "summary")
        active.token_usage.summary_generated_tokens += generated_output.completion_tokens
        invalid_boundary = False
        if active.token_ledger is not None:
            closure_reason = classify_thinking_closure(
                generated_summary, enable_thinking=self.token_renderer.enable_thinking)
            if closure_reason is not None:
                invalid_boundary = True
                active.turn_records.append({
                    "kind": "format_anomaly",
                    "query_id": active.state.query_id,
                    "anomaly": closure_reason,
                })
            try:
                self.token_renderer.validate_completion_boundary(
                    generated_output.completion_token_ids, generated_output.finish_reason)
            except UnsupportedTokenBoundary:
                invalid_boundary = True
            if self.scaffold is None and self._uses_native_tools:
                if generated_output.message is None or generated_output.message.tool_calls:
                    invalid_boundary = True
        if self.scaffold is not None:
            parsed_summary = self.scaffold.parse(generated_summary, call_id="summary-diagnostic",
                                                thinking=self.token_renderer.enable_thinking)
            invalid_boundary = invalid_boundary or parsed_summary is None or bool(parsed_summary.tool_calls)
        summary_extraction = (
            None if invalid_boundary else
            extract_structured_summary(generated_output.message)
            if self._uses_native_tools and generated_output.message is not None
            else extract_structured_summary(Message(role="assistant", content=generated_summary))
            if self.token_renderer is not None and not self.token_renderer.enable_thinking
            else extract_summary_output(generated_summary)
        )
        summary_tokens = self._completion_token_count(summary_extraction.summary) if summary_extraction is not None else 0
        state = active.state
        state.summary_count += 1
        summary_turn_id = f"summary-{state.summary_count}"
        turn_record: dict[str, Any] = {
            "query_id": state.query_id,
            "turn_id": summary_turn_id,
            "kind": "summary",
            "prompt": prompt,
            "completion": generated_summary,
            "thinking": summary_extraction.thinking if summary_extraction is not None else "",
            "summary": summary_extraction.summary if summary_extraction is not None else "",
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
        interval_messages.append(
            generated_output.message or Message(role="assistant", content=generated_summary)
        )
        self._finalize_trajectory(
            active,
            interval_messages,
            termination_kind="compaction" if summary_extraction is not None and summary_extraction.summary and summary_tokens <= self.max_summary_tokens else "malformed",
            prompt=prompt if isinstance(prompt, ConversationPrompt) else None,
        )
        if summary_extraction is None:
            active.result = self._penalized_result(active, status="malformed_tool_call")
            return
        if summary_tokens > self.max_summary_tokens:
            active.result = self._penalized_result(active, status="summary_length_exceeded")
            return
        if not summary_extraction.summary:
            active.result = self._penalized_result(active, status="empty_summary")
            return
        state.latest_summary = summary_extraction.summary
        state.messages = (self._initial_messages(state.user_prompt) + [
            Message(role="user", content=format_compacted_summary(summary_extraction.summary))]
            if self.scaffold is not None else build_compacted_messages(
                state.user_prompt, summary_extraction.summary, native_tools=self._uses_native_tools))
        active.token_usage.retired_round_count += retired_count
        active.summary_turns.append(summary_turn_id)
        self._initialize_ledger(active)

    def _next_prompt(self, active: _ActiveEpisode) -> ConversationPrompt | None:
        if active.pending_summary is not None:
            prompt = active.pending_summary
            active.pending_summary = None
            return prompt
        reasons = active.token_usage.forced_answer_reasons
        if self._remaining_tool_calls(active) == 0:
            reasons.append("tool_budget")
        if self._generated_token_budget_exhausted(active):
            reasons.append("generated_token_budget")
        prompt = (self._build_forced_answer_prompt(active) if reasons
                  else self._build_runtime_prompt(active.state))
        if self._terminate_on_context_overflow(active, prompt):
            return None
        return prompt

    def _route_generation(
        self, active: _ActiveEpisode, prompt: ConversationPrompt, prompt_tokens: int, output: _GeneratedOutput,
    ) -> _PendingToolRound | None:
        if prompt.generation_kind == "summary":
            self._apply_summary_output(active, prompt, prompt_tokens, output)
        elif prompt.generation_kind == "forced_answer":
            self._apply_forced_answer_output(active, output, prompt, prompt_tokens)
        else:
            return self._prepare_action_output(active, output, prompt, prompt_tokens)
        return None

    def _tool_round_boundary(self, active: _ActiveEpisode) -> ConversationPrompt | None:
        if active.result is None and active.interval_round_count:
            return self._build_summary_prompt_for_active(active)
        return None

    def _advance_active_episodes(self, active_episodes: list[_ActiveEpisode]) -> None:
        generation_items = []
        for active in active_episodes:
            if active.result is None and (prompt := self._next_prompt(active)) is not None:
                generation_items.append((active, prompt, self._prompt_token_count(active, prompt)))

        if generation_items:
            outputs = self._generate_batch([
                self._commit_token_input(active, prompt) for active, prompt, _ in generation_items
            ])
            actions = []
            for (active, prompt, count), output in zip(generation_items, outputs, strict=True):
                action = self._route_generation(active, prompt, count, output)
                if action is not None:
                    actions.append(action)
            if actions:
                self._execute_pending_tool_actions(actions)

        summary_items = []
        for active in active_episodes:
            summary_prompt = self._tool_round_boundary(active)
            if summary_prompt is None:
                continue
            if len(active_episodes) > 1 and getattr(self.model, "supports_mixed_control_batches", False):
                active.pending_summary = summary_prompt
            else:
                summary_items.append((active, summary_prompt, self._prompt_token_count(active, summary_prompt)))

        if summary_items:
            summary_outputs = self._generate_batch([self._commit_token_input(active, prompt) for active, prompt, _ in summary_items])
            for (active, prompt, prompt_tokens), generated_output in zip(summary_items, summary_outputs):
                self._apply_summary_output(active, prompt, prompt_tokens, generated_output)

    def _run_completion_driven(
        self, episodes: Iterable[tuple[str, str]], max_active: int,
    ) -> Iterator[list[tuple[int, RuntimeResult]]]:
        """Admit one committed request per episode and route replies by request ID."""
        inputs = iter(enumerate(episodes))
        inflight = {}
        ready = []
        serial = 0
        latencies = []
        completed_batches = 0
        peak_inflight = 0
        try:
            while True:
                while len(ready) + len(inflight) < max_active:
                    item = next(inputs, None)
                    if item is None:
                        break
                    index, (query_id, query) = item
                    ready.append((index, self._new_active_episode(query_id, query)))

                finished = []
                for index, active in ready:
                    prompt = self._next_prompt(active) if active.result is None else None
                    if prompt is None:
                        assert active.result is not None
                        finished.append((index, active.result))
                        continue
                    count = self._prompt_token_count(active, prompt)
                    request = self._commit_token_input(active, prompt)
                    request_id = str(serial)
                    serial += 1
                    self.model.submit_token_request(request_id, request)
                    inflight[request_id] = (index, active, prompt, count, time.monotonic())
                peak_inflight = max(peak_inflight, len(inflight))
                ready.clear()
                if finished:
                    yield finished
                    continue
                if not inflight:
                    return

                actions = []
                started = time.monotonic()
                completions = self.model.completed_token_requests()
                self.collection_metrics["generation_wait_seconds"] = (
                    self.collection_metrics.get("generation_wait_seconds", 0.0) + time.monotonic() - started
                )
                if not completions:
                    raise RuntimeError("No completions for outstanding rollout requests")
                completed_batches += 1
                received = time.monotonic()
                for request_id, result in completions:
                    index, active, prompt, count, submitted = inflight.pop(request_id)
                    latencies.append(received - submitted)
                    action = self._route_generation(active, prompt, count, self._generated_output(result))
                    if action is not None:
                        actions.append(action)
                    ready.append((index, active))
                if actions:
                    self._execute_pending_tool_actions(actions)
                    for action in actions:
                        action.active.pending_summary = self._tool_round_boundary(action.active)
        finally:
            self.model.abort_token_requests()
            if latencies:
                latencies.sort()
                self.collection_metrics.update({
                    "generation_requests": len(latencies),
                    "generation_completion_batches": completed_batches,
                    "peak_inflight_requests": peak_inflight,
                    "request_latency_p50_seconds": latencies[(len(latencies) - 1) // 2],
                    "request_latency_p90_seconds": latencies[(len(latencies) - 1) * 9 // 10],
                    "request_latency_max_seconds": latencies[-1],
                })

    def run_many_stream(
        self,
        episodes: Iterable[tuple[str, str]],
        *,
        max_active_episodes: int,
    ) -> Iterator[list[tuple[int, RuntimeResult]]]:
        """Yield completed episode batches while continuously refilling slots.

        Result indices refer to the input order. A yielded batch contains all
        episodes that completed in the same runtime round, which gives
        downstream judge/cache workers useful micro-batches without waiting for
        the slowest episode in the original collection window.
        """

        if max_active_episodes < 1:
            raise ValueError(
                f"max_active_episodes must be at least 1, got {max_active_episodes}"
            )
        if getattr(self.model, "completion_driven", False):
            yield from self._run_completion_driven(episodes, max_active_episodes)
            return
        pending = iter(enumerate(episodes))
        active_slots: list[tuple[int, _ActiveEpisode]] = []

        def refill() -> None:
            while len(active_slots) < max_active_episodes:
                try:
                    episode_index, (query_id, user_prompt) = next(pending)
                except StopIteration:
                    break
                active_slots.append(
                    (episode_index, self._new_active_episode(query_id, user_prompt))
                )

        refill()
        while active_slots:
            self._advance_active_episodes([active for _, active in active_slots])
            completed = [
                (episode_index, active.result)
                for episode_index, active in active_slots
                if active.result is not None
            ]
            if not completed:
                continue
            active_slots = [
                (episode_index, active)
                for episode_index, active in active_slots
                if active.result is None
            ]
            yield [
                (episode_index, result)
                for episode_index, result in completed
                if result is not None
            ]
            refill()

    def run_many(self, episodes: Iterable[tuple[str, str]]) -> list[RuntimeResult]:
        episode_list = list(episodes)
        if not episode_list:
            return []
        results_by_index: dict[int, RuntimeResult] = {}
        for completed in self.run_many_stream(
            episode_list,
            max_active_episodes=len(episode_list),
        ):
            results_by_index.update(completed)
        return [results_by_index[index] for index in range(len(episode_list))]

    def run(self, query_id: str, user_prompt: str) -> RuntimeResult:
        return self.run_many([(query_id, user_prompt)])[0]
