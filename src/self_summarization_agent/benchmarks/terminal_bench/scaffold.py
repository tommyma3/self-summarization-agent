"""KIRA routing and completion semantics, independent of Harbor and inference.

Prompt and tool descriptions adapted from krafton-ai/KIRA at KIRA_REVISION.
See resources/KIRA-LICENSE and resources/NOTICE for attribution.
"""
from hashlib import sha256
from importlib.resources import files
import json
import math
import re

from self_summarization_agent.models import Message, ToolCall
from . import KIRA_REVISION, SCAFFOLD_VERSION


RESOURCE_ROOT = files(__package__).joinpath("resources")
TOOLS = json.loads(RESOURCE_ROOT.joinpath("tools.json").read_text())
SYSTEM_PROMPT = RESOURCE_ROOT.joinpath("system.txt").read_text()
FORCED_CONTROL = """<forced_answer_request>
The execution budget has ended. Do not execute more commands. End this attempt with exactly one task_complete tool call, with no parameters. This is a forced completion; the current filesystem will be verified.
</forced_answer_request>"""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON parameter: {key}")
        result[key] = value
    return result


class TerminusKiraScaffold:
    finish_tool = "task_complete"
    forced_control = FORCED_CONTROL
    summary_control = """<summary_request>
Pause task execution now to compact the history. Do not call any tool, including task_complete. The next response must contain your completed thinking followed by exactly one <summary>...</summary> block. Preserve files, shell state, evidence, conclusions, unresolved work, and next steps. Do not restate the original task, which will remain available. Tools resume after this summary.
</summary_request>"""
    forced_body_regex = r"<tool_call>\s*<function=task_complete>\s*</function>\s*</tool_call>"

    def __init__(self, backend, *, image_enabled: bool = False):
        self.backend = backend
        self.tools = [t for t in TOOLS if image_enabled or t["function"]["name"] != "image_read"]
        self.system_prompt = SYSTEM_PROMPT
        self.fingerprint = sha256(json.dumps(
            [SCAFFOLD_VERSION, KIRA_REVISION, self.tools, self.system_prompt,
             self.forced_control, self.summary_control, self.forced_body_regex],
            sort_keys=True).encode()).hexdigest()
        self._confirmation_pending: set[str] = set()

    def parse(self, text: str, *, call_id: str, thinking: bool) -> Message | None:
        body = text.removesuffix("<|im_end|>").strip()
        reasoning = None
        if "</think>" in body:
            reasoning, body = body.split("</think>", 1)
            reasoning = reasoning.removeprefix("<think>").strip()
        elif thinking:
            return None
        if "<tool_call>" not in body:
            return Message(role="assistant", content=body.strip(), reasoning_content=reasoning)
        match = re.fullmatch(r"\s*<tool_call>\s*<function=([a-z_]+)>\s*(.*?)\s*</function>\s*</tool_call>\s*", body, re.S)
        if match is None:
            return None
        name, params = match.groups()
        arguments = {}
        cursor = 0
        # Qwen serializes arrays as JSON, string parameters as literal text.
        # Remove only the template's one framing newline, never string whitespace.
        for parameter in re.finditer(r"<parameter=([a-z_]+)>(.*?)</parameter>", params, re.S):
            if params[cursor:parameter.start()].strip():
                return None
            key, value = parameter.groups()
            if key in arguments:
                return None
            value = value.removeprefix("\n").removesuffix("\n")
            if key == "commands":
                try:
                    value = json.loads(value, object_pairs_hook=_unique_object)
                except (ValueError, TypeError):
                    return None
            arguments[key] = value
            cursor = parameter.end()
        if params[cursor:].strip() or not self.validate(name, arguments):
            return None
        return Message(role="assistant", reasoning_content=reasoning,
                       tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])

    def validate(self, name, arguments):
        if name not in {t["function"]["name"] for t in self.tools}:
            return False
        if name == "task_complete":
            return arguments == {}
        if name == "image_read":
            return set(arguments) == {"file_path", "image_read_instruction"} and all(
                isinstance(v, str) and bool(v) for v in arguments.values())
        if set(arguments) != {"analysis", "plan", "commands"} or not all(
            isinstance(arguments[k], str) for k in ("analysis", "plan")
        ) or not isinstance(arguments["commands"], list):
            return False
        for command in arguments["commands"]:
            if not isinstance(command, dict) or set(command) - {"keystrokes", "duration"}:
                return False
            duration = command.get("duration", 1.0)
            if (not isinstance(command.get("keystrokes"), str)
                or isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration < 0):
                return False
        return True

    def is_complete(self, name, *, query_id, forced):
        return name == self.finish_tool and (forced or query_id in self._confirmation_pending)

    def execute(self, name, arguments, *, query_id):
        if name == self.finish_tool:
            self._confirmation_pending.add(query_id)
            return (
                "Before marking the task complete, re-read the original task and verify your solution "
                "from the perspectives of a test engineer, QA engineer, and the user. Check edge cases "
                "and ensure only required files changed. Call task_complete again when satisfied. "
                "Verification begins then; no further edits will be possible."
            )
        # Any further work requires a fresh confirmation afterwards.
        self._confirmation_pending.discard(query_id)
        return self.backend.execute(name, arguments, query_id=query_id)
