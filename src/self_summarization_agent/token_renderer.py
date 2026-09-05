"""Explicit continuation renderer for the repository-owned Qwen agent template."""
from hashlib import sha256
import json
import re
from typing import Any

from self_summarization_agent.chat_template import load_chat_template
from self_summarization_agent.models import Message, ToolCall
from self_summarization_agent.prompts import ConversationPrompt, serialize_messages, build_summary_prompt, build_forced_answer_prompt
from self_summarization_agent.token_stream import TITO_CONTRACT, token_ids


TEMPLATE_PATH = "src/self_summarization_agent/chat_templates/qwen3_5_agent.jinja"


def parse_native_completion(text: str, *, require_thinking_close: bool = False,
                            call_id: str | None = None) -> Message | None:
    """Routing view only; never used to rebuild inference tokens."""
    body = text.removesuffix("<|im_end|>").strip()
    reasoning = None
    if "</think>" in body:
        reasoning, body = body.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip()
        body = body.strip()
    elif require_thinking_close or body.startswith("<think>"):
        return None
    if "<tool_call>" not in body:
        return Message(role="assistant", content=body, reasoning_content=reasoning)
    match = re.fullmatch(r"<tool_call>\s*<function=(search|get_document|finish)>\s*(.*?)\s*</function>\s*</tool_call>", body, re.DOTALL)
    if match is None:
        return None
    name, parameters = match.groups()
    argument = {"search": "query", "get_document": "doc_id", "finish": "answer"}[name]
    value = re.fullmatch(r"<parameter=" + argument + r">\s*([\s\S]*?)\s*</parameter>", parameters)
    if value is None or "<parameter=" in value[1]:
        return None
    call_id = call_id or "call_" + sha256(text.encode()).hexdigest()[:24]
    return Message(role="assistant", content="", reasoning_content=reasoning,
                   tool_calls=[ToolCall(id=call_id, name=name, arguments={argument: value[1]})])


class UnsupportedTokenBoundary(ValueError):
    pass


class QwenAgentTokenRenderer:
    def __init__(self, tokenizer: Any, *, enable_thinking: bool, native_tools: bool = False) -> None:
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.native_tools = native_tools
        approved = load_chat_template(TEMPLATE_PATH)
        if tokenizer.chat_template != approved:
            raise ValueError("TITO requires the repository qwen3_5_agent.jinja template")
        self.im_start = self._special("<|im_start|>")
        self.im_end = self._special("<|im_end|>")
        # Include the actual vocabulary/segmentation rules, not just a model path.
        backend = getattr(tokenizer, "backend_tokenizer", None)
        tokenizer_spec = backend.to_str() if backend is not None else json.dumps(
            tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False)
        identity = [TITO_CONTRACT, approved, tokenizer_spec, enable_thinking, native_tools]
        self.fingerprint = sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()

    def _encode(self, text: str) -> tuple[int, ...]:
        return token_ids(self.tokenizer.encode(text, add_special_tokens=False))

    def _special(self, text: str) -> int:
        ids = self._encode(text)
        if len(ids) != 1 or self.tokenizer.convert_ids_to_tokens(ids[0]) != text:
            raise ValueError(f"TITO requires atomic special token {text}")
        return ids[0]

    def render_initial_state(self, prompt: ConversationPrompt) -> tuple[int, ...]:
        if any(m.role == "assistant" or m.role == "tool" for m in prompt.messages):
            raise ValueError("Initial state cannot contain sampled history")
        return token_ids(self.tokenizer.apply_chat_template(
            serialize_messages(prompt.messages), tools=list(prompt.tools) or None,
            tokenize=True, return_dict=False, add_generation_prompt=False, enable_thinking=self.enable_thinking,
        ))

    def header(self, generation_kind: str) -> tuple[int, ...]:
        if generation_kind not in {"action", "summary", "forced_answer"}:
            raise ValueError(f"Unsupported generation kind: {generation_kind}")
        prefix = ""
        if generation_kind == "summary":
            prefix = "<|im_start|>user\n" + build_summary_prompt().strip() + "<|im_end|>\n"
        elif generation_kind == "forced_answer":
            prefix = "<|im_start|>system\n" + build_forced_answer_prompt().strip() + "<|im_end|>\n"
        return self._encode(prefix + "<|im_start|>assistant\n" + (
            "<think>\n" if self.enable_thinking else "<think>\n\n</think>\n\n"))

    def validate_completion_boundary(self, completion_ids: list[int] | tuple[int, ...],
                                     finish_reason: str | None) -> None:
        if finish_reason == "length" or not completion_ids or completion_ids[-1] != self.im_end:
            raise UnsupportedTokenBoundary("Tool continuation requires a sampled <|im_end|> terminator")

    def render_tool_result(self, message: Message, *, completion_ids: list[int] | tuple[int, ...],
                           finish_reason: str | None) -> tuple[int, ...]:
        self.validate_completion_boundary(completion_ids, finish_reason)
        if self.native_tools:
            if message.role != "tool" or not message.tool_call_id:
                raise ValueError("Native tool result must retain its linked call ID")
            body = "<tool_response>\n" + message.content.strip() + "\n</tool_response>"
        else:
            if message.role != "user" or not message.content.startswith("<tool_response>"):
                raise ValueError("Tagged tool result must use the existing user wrapper")
            body = message.content.strip()
        # The model sampled im_end, but its following template newline is external.
        return self._encode("\n<|im_start|>user\n" + body + "<|im_end|>\n")
