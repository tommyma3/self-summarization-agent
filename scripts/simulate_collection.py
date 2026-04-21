from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_summarization_agent.bcplus_backend import build_backend
from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.context import ContextManager
from self_summarization_agent.dataset import QueryExample, load_query_examples
from self_summarization_agent.generation import build_generator
from self_summarization_agent.launcher_utils import build_runtime, ensure_dir, utc_timestamp
from self_summarization_agent.models import EpisodeState, Message, ToolCallRecord, ToolRound
from self_summarization_agent.runtime import EpisodeRuntime, parse_model_tool_call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate one training collection rollout and write every packed context, "
            "tool result, and summary output to a text trace."
        )
    )
    parser.add_argument("--config", default="configs/train/default.yaml", help="Path to the train YAML config.")
    parser.add_argument("--sample-index", type=int, default=None, help="Use a fixed index after dataset slicing.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random query sampling. Defaults to config seed.")
    parser.add_argument("--output", default=None, help="Trace text file path. Defaults under experiment.output_root.")
    parser.add_argument("--model-path", default=None, help="Override model.model_path.")
    parser.add_argument("--retrieval-backend", default=None, help="Override retrieval.backend.")
    parser.add_argument("--include-formatted-prompt", action="store_true", help="Also write the tokenizer chat-template prompt.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Additional dotted config overrides, e.g. runtime.tool_budget=8",
    )
    return parser.parse_args()


def merge_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides = parse_cli_overrides(args.overrides)
    if args.model_path is not None:
        overrides["model.model_path"] = args.model_path
    if args.retrieval_backend is not None:
        overrides["retrieval.backend"] = args.retrieval_backend
    return overrides


def choose_example(examples: list[QueryExample], *, sample_index: int | None, seed: int) -> tuple[QueryExample, int]:
    if not examples:
        raise ValueError("No examples available after dataset slicing")
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(examples):
            raise IndexError(f"--sample-index must be in [0, {len(examples) - 1}], got {sample_index}")
        return examples[sample_index], sample_index
    selected_index = random.Random(seed).randrange(len(examples))
    return examples[selected_index], selected_index


def default_output_path(config: Any, query_id: str) -> Path:
    trace_dir = ensure_dir(Path(config.experiment.output_root) / "artifacts" / "collection_probe" / config.experiment.name)
    safe_query_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in query_id)
    return trace_dir / f"{safe_query_id}.txt"


def write_section(handle: TextIO, title: str, body: str = "") -> None:
    handle.write("\n")
    handle.write("=" * 88)
    handle.write(f"\n{title}\n")
    handle.write("=" * 88)
    handle.write("\n")
    if body:
        handle.write(body)
        if not body.endswith("\n"):
            handle.write("\n")


def write_key_values(handle: TextIO, values: dict[str, Any]) -> None:
    for key, value in values.items():
        handle.write(f"{key}: {value}\n")


def write_prompt(
    handle: TextIO,
    *,
    runtime: EpisodeRuntime,
    generator: Any,
    title: str,
    prompt: str,
    include_formatted_prompt: bool,
) -> None:
    write_section(
        handle,
        title,
        json.dumps(
            {
                "token_count": runtime.token_counter(prompt),
                "character_count": len(prompt),
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    handle.write("\n--- Runtime prompt ---\n")
    handle.write(prompt)
    handle.write("\n")
    if not include_formatted_prompt:
        return
    formatter = getattr(generator, "_format_prompt", None)
    if callable(formatter):
        formatted = formatter(prompt)
        if formatted != prompt:
            handle.write("\n--- Prompt after tokenizer chat template ---\n")
            handle.write(formatted)
            handle.write("\n")


def build_context_manager(runtime: EpisodeRuntime) -> ContextManager:
    return ContextManager(
        token_counter=runtime.token_counter,
        max_context_tokens=runtime.max_context_tokens,
        safety_margin_tokens=0,
    )


def trace_collection(
    *,
    runtime: EpisodeRuntime,
    generator: Any,
    example: QueryExample,
    sample_index: int,
    output_path: Path,
    include_formatted_prompt: bool,
) -> None:
    state = EpisodeState(
        query_id=example.query_id,
        user_prompt=example.query,
        context_threshold_tokens=runtime.context_threshold_tokens,
    )
    context_manager = build_context_manager(runtime)
    tool_call_counts = {"search": 0, "get_document": 0}
    retrieved_docids: list[str] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        write_section(handle, "Collection Trace")
        write_key_values(
            handle,
            {
                "timestamp_utc": utc_timestamp(),
                "query_id": example.query_id,
                "sample_index": sample_index,
                "query": example.query,
                "answer": example.answer,
                "context_threshold_tokens": runtime.context_threshold_tokens,
                "max_context_tokens": runtime.max_context_tokens,
                "tool_budget": runtime.max_tool_calls,
            },
        )

        while True:
            used_tools = sum(tool_call_counts.values())
            if runtime.max_tool_calls is not None and used_tools >= runtime.max_tool_calls:
                write_prompt(
                    handle,
                    runtime=runtime,
                    generator=generator,
                    title="Budget Exhausted Context",
                    prompt=runtime._build_runtime_prompt(state),
                    include_formatted_prompt=include_formatted_prompt,
                )
                write_section(handle, "Terminal Status", "status: budget_exhausted\n")
                return

            round_number = len(state.rounds) + 1
            acting_prompt = runtime._build_runtime_prompt(state)
            context_manager.assert_fits(acting_prompt)
            write_prompt(
                handle,
                runtime=runtime,
                generator=generator,
                title=f"Round {round_number} Acting Context",
                prompt=acting_prompt,
                include_formatted_prompt=include_formatted_prompt,
            )

            raw_output = runtime.model.generate(acting_prompt)
            write_section(handle, f"Round {round_number} Model Output", raw_output)
            parsed_tool_call = parse_model_tool_call(raw_output)
            if parsed_tool_call is None:
                write_section(handle, "Terminal Status", "status: malformed_tool_call\n")
                return

            payload, normalized_output = parsed_tool_call
            tool_name = payload["tool_name"]
            arguments = payload["arguments"]
            write_section(
                handle,
                f"Round {round_number} Parsed Action",
                json.dumps(payload, indent=2, ensure_ascii=False),
            )

            if tool_name == "finish":
                answer = arguments.get("answer") if isinstance(arguments, dict) else None
                if not isinstance(answer, str):
                    write_section(handle, "Terminal Status", "status: malformed_tool_call\nreason: finish.answer is not a string\n")
                    return
                write_section(handle, "Terminal Status", f"status: completed\nfinal_answer: {answer}\n")
                return

            if tool_name == "search":
                query = arguments.get("query") if isinstance(arguments, dict) else None
                if not isinstance(query, str):
                    write_section(handle, "Terminal Status", "status: malformed_tool_call\nreason: search.query is not a string\n")
                    return
                doc_ids = runtime.backend.search(query)
                tool_call_counts["search"] += 1
                runtime._record_retrieved_docids(retrieved_docids, doc_ids)
                tool_result = json.dumps(doc_ids, ensure_ascii=False)
            elif tool_name == "get_document":
                doc_id = arguments.get("doc_id") if isinstance(arguments, dict) else None
                if not isinstance(doc_id, str):
                    write_section(handle, "Terminal Status", "status: malformed_tool_call\nreason: get_document.doc_id is not a string\n")
                    return
                runtime._record_retrieved_docids(retrieved_docids, [doc_id])
                tool_result = runtime.backend.get_document(doc_id)
                tool_call_counts["get_document"] += 1
            else:
                write_section(handle, "Terminal Status", f"status: malformed_tool_call\nreason: unsupported tool {tool_name!r}\n")
                return

            write_section(
                handle,
                f"Round {round_number} Tool Result",
                json.dumps(
                    {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "tool_call_counts": tool_call_counts,
                        "retrieved_docids": retrieved_docids,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n\n"
                + tool_result,
            )

            state.rounds.append(
                ToolRound(
                    assistant_message=Message(role="assistant", content=normalized_output),
                    tool_call=ToolCallRecord(tool_name=tool_name, arguments=arguments, raw_output=normalized_output),
                    tool_result=Message(role="tool", content=tool_result),
                )
            )

            after_tool_prompt = runtime._build_runtime_prompt(state)
            write_prompt(
                handle,
                runtime=runtime,
                generator=generator,
                title=f"Round {round_number} Context After Tool",
                prompt=after_tool_prompt,
                include_formatted_prompt=include_formatted_prompt,
            )

            compacted_state = runtime._compacted_state(state)
            compacted_tokens = context_manager.current_token_count(compacted_state)
            write_section(
                handle,
                f"Round {round_number} Threshold Check",
                json.dumps(
                    {
                        "compacted_context_tokens": compacted_tokens,
                        "context_threshold_tokens": state.context_threshold_tokens,
                        "should_summarize": context_manager.should_summarize(compacted_state),
                        "raw_tail_round_count": len(runtime._raw_tail_rounds(state)),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
            )

            if not context_manager.should_summarize(compacted_state):
                continue

            summary_state, retired_count = runtime._build_summary_state(state)
            if retired_count == 0:
                write_section(handle, f"Round {round_number} Summary Skipped", "reason: only one raw tail round is available\n")
                continue

            summary_prompt = context_manager.build_summary_context(summary_state)
            context_manager.assert_fits(summary_prompt)
            write_prompt(
                handle,
                runtime=runtime,
                generator=generator,
                title=f"Summary {state.summary_count + 1} Context",
                prompt=summary_prompt,
                include_formatted_prompt=include_formatted_prompt,
            )
            generated_summary = runtime.model.generate(summary_prompt)
            write_section(handle, f"Summary {state.summary_count + 1} Model Output", generated_summary)
            if not generated_summary.strip():
                write_section(handle, f"Summary {state.summary_count + 1} Skipped", "reason: model returned an empty summary\n")
                continue

            state.latest_summary = generated_summary
            state.summarized_round_count += retired_count
            state.summary_count += 1
            write_section(
                handle,
                f"Summary {state.summary_count} Applied",
                json.dumps(
                    {
                        "retired_rounds": retired_count,
                        "summarized_round_count": state.summarized_round_count,
                        "remaining_raw_tail_rounds": len(runtime._raw_tail_rounds(state)),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
            )
            write_prompt(
                handle,
                runtime=runtime,
                generator=generator,
                title=f"Context After Summary {state.summary_count}",
                prompt=runtime._build_runtime_prompt(state),
                include_formatted_prompt=include_formatted_prompt,
            )


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, merge_overrides(args))
    seed = config.experiment.seed if args.seed is None else args.seed
    examples = load_query_examples(
        config.experiment.bc_plus_root,
        config.dataset,
        require_answers=True,
        seed=config.experiment.seed,
    )
    example, sample_index = choose_example(examples, sample_index=args.sample_index, seed=seed)
    generator = build_generator(config.model)
    backend = build_backend(config.experiment.bc_plus_root, config.retrieval)
    runtime = build_runtime(generator, backend, config.runtime)
    output_path = Path(args.output) if args.output else default_output_path(config, example.query_id)

    trace_collection(
        runtime=runtime,
        generator=generator,
        example=example,
        sample_index=sample_index,
        output_path=output_path,
        include_formatted_prompt=args.include_formatted_prompt,
    )
    print(output_path)


if __name__ == "__main__":
    main()
