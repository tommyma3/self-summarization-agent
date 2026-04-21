from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.dataset import load_query_examples
from self_summarization_agent.generation import build_generator
from self_summarization_agent.launcher_utils import build_runtime
from self_summarization_agent.models import EpisodeState
from self_summarization_agent.runtime import parse_model_tool_call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the first rollout prompt and raw model output without executing tools or training."
    )
    parser.add_argument("--config", default="configs/train/default.yaml", help="Path to the train YAML config.")
    parser.add_argument("--sample-index", type=int, default=None, help="Use a fixed index after dataset slicing.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random query sampling. Defaults to config seed.")
    parser.add_argument("--hide-prompt", action="store_true", help="Only print metadata and model output.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Additional dotted config overrides, e.g. model.max_new_tokens=256",
    )
    return parser.parse_args()


def choose_example(examples, *, sample_index: int | None, seed: int):
    if not examples:
        raise ValueError("No examples available after dataset slicing")
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(examples):
            raise IndexError(f"--sample-index must be in [0, {len(examples) - 1}], got {sample_index}")
        return examples[sample_index], sample_index
    selected_index = random.Random(seed).randrange(len(examples))
    return examples[selected_index], selected_index


def format_prompt_for_display(generator, prompt: str) -> str:
    formatter = getattr(generator, "_format_prompt", None)
    if callable(formatter):
        return formatter(prompt)
    return prompt


def print_thinking_diagnostics(generator, raw_output: str) -> None:
    tokenizer = getattr(generator, "tokenizer", None)
    chat_template = getattr(tokenizer, "chat_template", None)
    chat_template_text = chat_template if isinstance(chat_template, str) else ""
    output_lower = raw_output.lower()
    uses_chat_template = bool(chat_template_text)
    enable_thinking = bool(getattr(generator, "enable_thinking", False))

    print("\n=== Thinking Check ===")
    print(f"uses_chat_template: {uses_chat_template}")
    print(f"enable_thinking_config: {enable_thinking}")
    print(f"tokenizer_template_mentions_enable_thinking: {'enable_thinking' in chat_template_text}")
    print(f"output_contains_think_tags: {'<think' in output_lower or '</think>' in output_lower}")
    print("note: when the tokenizer has a chat template, this code applies it and passes enable_thinking from the model config.")


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    seed = config.experiment.seed if args.seed is None else args.seed
    examples = load_query_examples(
        config.experiment.bc_plus_root,
        config.dataset,
        require_answers=True,
        seed=config.experiment.seed,
    )
    example, sample_index = choose_example(examples, sample_index=args.sample_index, seed=seed)

    generator = build_generator(config.model)
    runtime = build_runtime(
        generator,
        FakeBackend(search_index={}, documents={}),
        config.runtime,
    )
    state = EpisodeState(
        query_id=example.query_id,
        user_prompt=example.query,
        context_threshold_tokens=config.runtime.context_threshold_tokens,
    )
    prompt = runtime._build_runtime_prompt(state)
    model_prompt = format_prompt_for_display(generator, prompt)
    raw_output = generator.generate(prompt)

    print("=== Sample ===")
    print(f"query_id: {example.query_id}")
    print(f"sample_index: {sample_index}")
    print(f"query: {example.query}")
    if example.answer is not None:
        print(f"answer: {example.answer}")

    if not args.hide_prompt:
        print("\n=== Prompt Sent To Model ===")
        print(model_prompt)

    print("\n=== Raw Model Output ===")
    print(raw_output)

    print_thinking_diagnostics(generator, raw_output)

    print("\n=== JSON Format Check ===")
    parsed_tool_call = parse_model_tool_call(raw_output)
    if parsed_tool_call is None:
        try:
            json.loads(raw_output)
        except json.JSONDecodeError as exc:
            print(f"invalid_json: {exc}")
        else:
            print("invalid_tool_call: parsed JSON does not contain tool_name and object arguments")
        return
    parsed, normalized_output = parsed_tool_call
    if raw_output.strip() != normalized_output:
        print("accepted_after_cleanup: True")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    tool_name = parsed.get("tool_name") if isinstance(parsed, dict) else None
    arguments = parsed.get("arguments") if isinstance(parsed, dict) else None
    print(f"tool_name_ok: {tool_name in {'search', 'get_document', 'finish'}}")
    print(f"arguments_is_object: {isinstance(arguments, dict)}")


if __name__ == "__main__":
    main()
