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
from self_summarization_agent.dataset import QueryExample, load_query_examples
from self_summarization_agent.generation import build_generator
from self_summarization_agent.judge import RewardJudge
from self_summarization_agent.launcher_utils import build_runtime, ensure_dir, utc_timestamp
from self_summarization_agent.models import Message
from self_summarization_agent.prompts import ConversationPrompt
from self_summarization_agent.runtime import EpisodeRuntime


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


def write_training_sequences(
    handle: TextIO,
    *,
    runtime: EpisodeRuntime,
    terminal_status: str,
    trajectory_records: list[dict[str, Any]],
) -> None:
    if not trajectory_records:
        write_section(
            handle,
            "Training Sequences",
            "No trainable intervals were produced. This rollout ended before any model-generated interval was recorded.\n",
        )
        return

    if terminal_status == "completed":
        reward_note = (
            "These are the append-only trainable intervals.\n"
            "- Every assistant message is trainable, including tool actions and the boundary completion.\n"
            "- System/user state, tool results, and appended boundary instructions are conditioning-only.\n"
            "- If the final answer is judged correct, every listed interval gets reward +1.\n"
            "- If the final answer is judged wrong, every listed interval gets reward -1.\n"
        )
    elif terminal_status == "budget_exhausted":
        reward_note = "This rollout exhausted the tool budget. Every listed interval gets reward -1.\n"
    elif terminal_status == "summary_length_exceeded":
        reward_note = "This rollout exceeded the summary length cap. Every listed interval gets reward -1.\n"
    else:
        reward_note = "This rollout ended unsuccessfully. Every listed interval gets reward -1.\n"

    write_section(
        handle,
        "Training Sequences",
        reward_note + f"trainable_interval_count: {len(trajectory_records)}\n",
    )
    for index, record in enumerate(trajectory_records, start=1):
        prompt = str(record["prompt"])
        completion = str(record["completion"])
        messages = record["messages"]
        metadata = {
            "index": index,
            "trajectory_id": record["turn_id"],
            "termination_kind": record["termination_kind"],
            "query_id": record["query_id"],
            "constituent_turn_ids": record.get("turn_ids", []),
            "assistant_completion_count": record.get("assistant_completion_count"),
            "prompt_token_count": runtime.token_counter(prompt),
            "completion_token_count": runtime.token_counter(completion),
            "prompt_character_count": len(prompt),
            "completion_character_count": len(completion),
        }
        write_section(
            handle,
            f"Training Sequence {index} Metadata",
            json.dumps(metadata, indent=2, ensure_ascii=False),
        )
        handle.write("\n--- Full interval messages ---\n")
        handle.write(json.dumps(messages, indent=2, ensure_ascii=False))
        handle.write("\n\n--- Trainable assistant completions ---\n")
        for message in messages:
            if message.get("role") == "assistant":
                handle.write(str(message.get("content", "")))
                handle.write("\n")


def _write_judge_output(
    handle: TextIO,
    judge: RewardJudge | None,
    example: QueryExample,
    status: str,
    answer: str,
) -> None:
    if judge is None:
        return
    decision = judge.evaluate(example, status, answer)
    body = {
        "outcome": decision.outcome,
        "parse_error": decision.parse_error,
    }
    write_section(handle, "Judge Evaluation", json.dumps(body, indent=2, ensure_ascii=False))
    if decision.judge_prompt:
        handle.write("\n--- Judge Prompt ---\n")
        handle.write(decision.judge_prompt)
        handle.write("\n")
    if decision.judge_response:
        handle.write("\n--- Judge Response ---\n")
        handle.write(decision.judge_response)
        handle.write("\n")


def trace_collection(
    *,
    runtime: EpisodeRuntime,
    generator: Any,
    example: QueryExample,
    sample_index: int,
    output_path: Path,
    include_formatted_prompt: bool,
    judge: RewardJudge | None = None,
) -> None:
    result = runtime.run(query_id=example.query_id, user_prompt=example.query)
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
                "generated_token_budget": runtime.generated_token_budget,
                "generator_max_model_len": getattr(generator, "max_model_len", None),
                "max_summary_tokens": runtime.max_summary_tokens,
            },
        )

        write_section(
            handle,
            "Runtime Result",
            json.dumps(
                {
                    "status": result.status,
                    "final_answer": result.final_answer,
                    "summary_turns": result.summary_turns,
                    "turn_rewards": result.turn_rewards,
                    "retrieved_docids": result.retrieved_docids,
                    "tool_call_counts": result.tool_call_counts,
                    "token_usage": result.token_usage,
                },
                indent=2,
                ensure_ascii=False,
            ),
        )
        for index, record in enumerate(result.trajectory_records, start=1):
            messages = [Message(role=message["role"], content=message["content"]) for message in record["messages"]]
            prompt = ConversationPrompt(messages)
            write_prompt(
                handle,
                runtime=runtime,
                generator=generator,
                title=f"Trajectory Interval {index}",
                prompt=prompt,
                include_formatted_prompt=include_formatted_prompt,
            )
        write_training_sequences(
            handle,
            runtime=runtime,
            terminal_status=result.status,
            trajectory_records=result.trajectory_records,
        )
        _write_judge_output(handle, judge, example, result.status, result.final_answer or "")


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
