from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_summarization_agent.generation import OpenAICompatibleGenerator
from self_summarization_agent.models import Message
from self_summarization_agent.prompts import (
    ConversationPrompt,
    FORCED_ANSWER_TOOLS,
    FORCED_FINISH_TOOL_CHOICE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the native-tool and exact-token-ID contract required by the "
            "OpenAI-compatible Qwen3.5 rollout runtime."
        )
    )
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible /v1 base URL")
    parser.add_argument("--model", required=True, help="Model name exposed by the API server")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Local Qwen3.5 tokenizer path used only to decode the returned token IDs",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generator = OpenAICompatibleGenerator(
        model_path=args.tokenizer,
        api_model=args.model,
        api_base_url=args.base_url,
        api_key_env=args.api_key_env,
        api_timeout_seconds=args.timeout_seconds,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        enable_thinking=args.enable_thinking,
        require_exact_token_ids=True,
        api_max_concurrency=1,
    )
    prompt = ConversationPrompt(
        [
            Message(
                role="system",
                content="Call the required finish function with the answer contract-ok.",
            ),
            Message(role="user", content="Complete the compatibility probe."),
        ],
        tools=FORCED_ANSWER_TOOLS,
        tool_choice=FORCED_FINISH_TOOL_CHOICE,
        parallel_tool_calls=False,
        generation_kind="forced_answer",
    )
    result = generator.generate_batch_with_metadata([prompt])[0]
    if result.message is None or len(result.message.tool_calls) != 1:
        raise RuntimeError("Server did not return exactly one structured tool call")
    tool_call = result.message.tool_calls[0]
    if tool_call.name != "finish" or not isinstance(tool_call.arguments.get("answer"), str):
        raise RuntimeError("Server did not honor the required finish tool schema")
    if result.prompt_token_ids is None or result.completion_token_ids is None:
        raise RuntimeError("Server did not return exact prompt and completion token IDs")
    print(
        json.dumps(
            {
                "status": "ok",
                "model": args.model,
                "tool_name": tool_call.name,
                "finish_reason": result.finish_reason,
                "prompt_token_count": len(result.prompt_token_ids),
                "completion_token_count": len(result.completion_token_ids),
                "reasoning_present": bool(result.message.reasoning_content),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
