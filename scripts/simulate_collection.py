from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Mapping
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
            "tool result, summary output, and exact model-input token sequence to a text trace."
        )
    )
    parser.add_argument("--config", default="configs/train/default.yaml", help="Path to the train YAML config.")
    parser.add_argument("--sample-index", type=int, default=None, help="Use a fixed index after dataset slicing.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random query sampling. Defaults to config seed.")
    parser.add_argument("--output", default=None, help="Trace text file path. Defaults under experiment.output_root.")
    parser.add_argument("--model-path", default=None, help="Override model.model_path.")
    parser.add_argument("--retrieval-backend", default=None, help="Override retrieval.backend.")
    parser.add_argument("--include-formatted-prompt", action="store_true", help="Also write the tokenizer chat-template prompt.")
    parser.add_argument("--training-max-seq-len", type=int, default=None, help="Override training.max_sequence_length for the fit check.")
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


def _coerce_token_ids(value: object) -> list[int]:
    """Coerce tokenizer output to a flat list of ints (mirrors trajectory._coerce_token_ids)."""
    if isinstance(value, list):
        result: list[int] = []
        for item in value:
            if isinstance(item, int):
                result.append(item)
            elif hasattr(item, "tolist"):
                result.append(int(item))
            elif hasattr(item, "__iter__"):
                for sub_item in item:
                    result.append(int(sub_item))
        return result
    if hasattr(value, "tolist"):
        return [int(x) for x in value.tolist()]
    if hasattr(value, "__iter__"):
        return [int(x) for x in value]
    return [int(value)]


def _analyze_training_lengths(
    generator: Any,
    trajectory_records: list[dict[str, Any]],
    max_sequence_length: int | None,
) -> list[dict[str, Any]]:
    """Tokenize each trajectory record the same way ``tokenize_interval_messages`` does.

    Returns one dict per record with keys ``training_total_tokens``,
    ``assistant_content_tokens``, and ``fits`` (None when no max_sequence_length).
    """
    tokenizer = getattr(generator, "tokenizer", None)
    enable_thinking = getattr(generator, "enable_thinking", None)
    chat_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None

    results: list[dict[str, Any]] = []
    for record in trajectory_records:
        collection_tokens = record.get("collection_tokens")
        if isinstance(collection_tokens, Mapping):
            full_ids = collection_tokens.get("full_token_ids")
            assistant_mask = collection_tokens.get("assistant_token_mask")
            if isinstance(full_ids, list) and isinstance(assistant_mask, list):
                training_total = len(full_ids) - 1
                results.append(
                    {
                        "training_total_tokens": training_total,
                        "assistant_content_tokens": sum(bool(value) for value in assistant_mask[1:]),
                        "fits": (
                            None
                            if max_sequence_length is None
                            else training_total <= max_sequence_length
                        ),
                        "token_source": "collection",
                    }
                )
                continue
        messages = record.get("messages", [])
        if not chat_template or not messages or not tokenizer:
            results.append(
                {
                    "training_total_tokens": None,
                    "assistant_content_tokens": None,
                    "fits": None,
                }
            )
            continue

        full_ids: list[int] | None = None
        assistant_token_mask: list[bool] | None = None
        template_kwargs: dict[str, Any] = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        if record.get("tools"):
            template_kwargs["tools"] = record["tools"]

        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
                **template_kwargs,
            )
        except (TypeError, ValueError):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    **template_kwargs,
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )

        if isinstance(rendered, Mapping):
            full_ids = _coerce_token_ids(rendered.get("input_ids"))
            raw_mask = rendered.get("assistant_masks", rendered.get("assistant_mask"))
            if raw_mask is not None:
                candidate_mask = [bool(v) for v in _coerce_token_ids(raw_mask)]
                if len(candidate_mask) == len(full_ids) and any(candidate_mask):
                    assistant_token_mask = candidate_mask
        else:
            full_ids = _coerce_token_ids(rendered)

        if full_ids is None:
            results.append(
                {
                    "training_total_tokens": None,
                    "assistant_content_tokens": None,
                    "fits": None,
                }
            )
            continue

        training_total = len(full_ids) - 1
        assistant_tokens = sum(assistant_token_mask) if assistant_token_mask else None
        fits = (
            None
            if max_sequence_length is None
            else training_total <= max_sequence_length
        )
        results.append(
            {
                "training_total_tokens": training_total,
                "assistant_content_tokens": assistant_tokens,
                "fits": fits,
            }
        )
    return results


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
                "chat_template_token_count": (
                    generator.count_prompt_tokens(prompt)
                    if callable(getattr(generator, "count_prompt_tokens", None))
                    else None
                ),
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


def format_exact_model_input_sequences(trajectory_records: list[dict[str, Any]]) -> str:
    """Render the authoritative token IDs supplied to every collected generation."""
    lines: list[str] = []
    found_per_generation_ids = False
    for trajectory_index, record in enumerate(trajectory_records, start=1):
        trajectory_id = record.get("turn_id", f"trajectory-{trajectory_index}")
        collection_tokens = record.get("collection_tokens")
        if not isinstance(collection_tokens, Mapping):
            lines.append(
                f"trajectory {trajectory_index} ({trajectory_id}): exact model-input token IDs unavailable"
            )
            continue
        generations = collection_tokens.get("generations")
        if not isinstance(generations, list) or not generations:
            final_prompt_ids = collection_tokens.get("prompt_token_ids")
            if isinstance(final_prompt_ids, list):
                lines.append(
                    f"trajectory {trajectory_index} ({trajectory_id}), final generation input token IDs:\n"
                    + json.dumps(final_prompt_ids, ensure_ascii=False)
                )
            else:
                lines.append(
                    f"trajectory {trajectory_index} ({trajectory_id}): exact model-input token IDs unavailable"
                )
            continue
        for generation in generations:
            if not isinstance(generation, Mapping):
                continue
            prompt_token_ids = generation.get("prompt_token_ids")
            if not isinstance(prompt_token_ids, list):
                continue
            found_per_generation_ids = True
            generation_index = generation.get("index", "?")
            finish_reason = generation.get("finish_reason")
            lines.append(
                f"trajectory {trajectory_index} ({trajectory_id}), generation {generation_index} "
                f"input token IDs (count={len(prompt_token_ids)}, finish_reason={finish_reason!r}):\n"
                + json.dumps(prompt_token_ids, ensure_ascii=False)
            )
    if not lines:
        return "No trainable trajectories were produced."
    if not found_per_generation_ids:
        lines.append(
            "Note: no per-generation server/engine token trace was available; displayed final prompt IDs "
            "are the best available artifact."
        )
    return "\n\n".join(lines)


def write_exact_model_input_sequences(
    handle: TextIO,
    trajectory_records: list[dict[str, Any]],
) -> None:
    write_section(
        handle,
        "Exact Model Input Token Sequences",
        format_exact_model_input_sequences(trajectory_records),
    )


def write_context_timeline(
    handle: TextIO,
    *,
    result: Any,
    generator: Any,
    context_threshold_tokens: int,
    max_context_tokens: int,
) -> None:
    """Write a chronological turn-by-turn table showing actual context sizes."""
    turn_records = getattr(result, "turn_records", None) or []
    write_section(handle, "Context Timeline")
    write_key_values(
        handle,
        {
            "model": getattr(generator, "model_path", None) or "unknown",
            "generator_max_model_len": getattr(generator, "max_model_len", None),
            "context_threshold_tokens": context_threshold_tokens,
            "max_context_tokens": max_context_tokens,
            "turn_count": len(turn_records),
        },
    )
    if not turn_records:
        handle.write("No generation turns recorded.\n")
        return

    token_usage = getattr(result, "token_usage", None) or {}
    max_prompt_seen = token_usage.get("max_prompt_tokens_seen", 0)
    prev_prompt: int | None = None
    cum_gen = 0

    header = (
        f"{'#':>3}  {'turn_id':<14} {'kind/gen':<20} {'prompt':>8} {'delta':>8} "
        f"{'ctx_util':>8} {'compl':>8} {'cum_gen':>8}  notes"
    )
    handle.write("\n")
    handle.write(header)
    handle.write("\n")
    handle.write("-" * len(header))
    handle.write("\n")

    for index, record in enumerate(turn_records, start=1):
        turn_id = record.get("turn_id", "?")
        kind = record.get("kind", "?")
        gen_kind = record.get("generation_kind", kind)
        kind_gen = f"{kind}/{gen_kind}"
        prompt_tokens = record.get("prompt_tokens")
        completion_tokens = record.get("completion_tokens", 0)

        prompt_str = str(prompt_tokens) if prompt_tokens is not None else "n/a"
        delta_str = "—"
        if prompt_tokens is not None and prev_prompt is not None:
            delta = prompt_tokens - prev_prompt
            delta_str = f"{delta:+d}"
        prev_prompt = prompt_tokens

        ctx_util_str = (
            f"{prompt_tokens / max_context_tokens * 100:.1f}%"
            if prompt_tokens is not None
            else "n/a"
        )
        compl_str = str(completion_tokens) if completion_tokens is not None else "?"
        cum_gen += completion_tokens if isinstance(completion_tokens, int) else 0
        cum_str = str(cum_gen)

        notes: list[str] = []
        if prompt_tokens is not None and prompt_tokens >= context_threshold_tokens:
            notes.append("THRESHOLD")
        if kind == "summary":
            notes.append("COMPACTED")
        if prompt_tokens == max_prompt_seen and max_prompt_seen > 0:
            notes.append("PEAK")
        if gen_kind == "forced_answer":
            notes.append("BUDGET")
        notes_str = " ".join(f"[{n}]" for n in notes) if notes else ""

        row = (
            f"{index:>3}  {turn_id:<14} {kind_gen:<20} {prompt_str:>8} {delta_str:>8} "
            f"{ctx_util_str:>8} {compl_str:>8} {cum_str:>8}  {notes_str}"
        )
        handle.write(row)
        handle.write("\n")

    # Compaction events summary
    handle.write("\n--- Compaction events ---\n")
    compaction_events = 0
    for index, record in enumerate(turn_records):
        if record.get("kind") != "summary":
            continue
        turn_id = record.get("turn_id", "?")
        pre_context = record.get("prompt_tokens")
        post_context = None
        if index + 1 < len(turn_records):
            post_context = turn_records[index + 1].get("prompt_tokens")
        delta = (
            post_context - pre_context
            if pre_context is not None and post_context is not None
            else None
        )
        handle.write(
            f"  {turn_id}: pre-context {pre_context or '?'} tokens, "
            f"post-context {post_context or '?'} tokens, "
            f"delta {delta or '?'} tokens\n"
        )
        compaction_events += 1
    if compaction_events == 0:
        handle.write("  (no compaction events)\n")


def write_context_summary(
    handle: TextIO,
    *,
    result: Any,
    runtime: EpisodeRuntime,
) -> None:
    """Write a key-value summary of context utilization and budget consumption."""
    write_section(handle, "Context Summary")
    token_usage = getattr(result, "token_usage", None) or {}
    turn_records = getattr(result, "turn_records", None) or []
    summary_turns = getattr(result, "summary_turns", None) or []

    max_prompt = token_usage.get("max_prompt_tokens_seen", 0)
    ctx_threshold = runtime.context_threshold_tokens
    max_ctx = runtime.max_context_tokens
    gen_budget = runtime.generated_token_budget
    total_gen = token_usage.get("total_generated_tokens", 0)

    peak_util = f"{max_prompt / max_ctx * 100:.1f}%" if max_prompt > 0 else "n/a"
    budget_util = f"{total_gen / gen_budget * 100:.1f}%" if gen_budget else "n/a"
    headroom = max_ctx - max_prompt if max_prompt > 0 else None
    threshold_crossed = max_prompt >= ctx_threshold if max_prompt > 0 else None

    # Count turns at or over threshold
    turns_over = sum(
        1
        for r in turn_records
        if (pt := r.get("prompt_tokens")) is not None and pt >= ctx_threshold
    )

    # Final context size (last turn record's prompt_tokens)
    final_ctx = turn_records[-1].get("prompt_tokens") if turn_records else None

    write_key_values(
        handle,
        {
            "status": getattr(result, "status", "?"),
            "turn_count": len(turn_records),
            "max_prompt_tokens_seen": max_prompt,
            "context_threshold_tokens": ctx_threshold,
            "max_context_tokens": max_ctx,
            "peak_utilization_vs_cap": peak_util,
            "peak_headroom_tokens": headroom,
            "threshold_crossed": threshold_crossed,
            "turns_at_or_over_threshold": turns_over,
            "compaction_count": len(summary_turns),
            "compaction_turn_ids": summary_turns,
            "retired_round_count": token_usage.get("retired_round_count", 0),
            "reasoning_generated_tokens": token_usage.get("reasoning_generated_tokens", 0),
            "summary_generated_tokens": token_usage.get("summary_generated_tokens", 0),
            "forced_answer_generated_tokens": token_usage.get("forced_answer_generated_tokens", 0),
            "tool_result_tokens": token_usage.get("tool_result_tokens", 0),
            "total_generated_tokens": total_gen,
            "generated_token_budget": gen_budget,
            "budget_utilization": budget_util,
            "forced_answer_reasons": token_usage.get("forced_answer_reasons", []),
            "final_context_prompt_tokens": final_ctx,
        },
    )

    # Interpretive line
    parts: list[str] = []
    parts.append(f"Context peaked at {max_prompt} tokens ({peak_util} of the {max_ctx} cap)")
    if summary_turns:
        parts.append(f"compaction triggered at {summary_turns[0]}")
    if threshold_crossed:
        parts.append(f"{turns_over} turn(s) at or over the {ctx_threshold} threshold")
    else:
        parts.append(f"context never reached the {ctx_threshold} threshold")
    if gen_budget:
        parts.append(f"generated {total_gen} of {gen_budget} token budget ({budget_util})")
    handle.write("\n")
    handle.write(" ".join(parts) + ".\n")


def write_training_sequences(
    handle: TextIO,
    *,
    runtime: EpisodeRuntime,
    terminal_status: str,
    trajectory_records: list[dict[str, Any]],
    generator: Any | None = None,
    max_sequence_length: int | None = None,
    training_lengths: list[dict[str, Any]] | None = None,
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
        tl_entry = (
            training_lengths[index - 1]
            if training_lengths and index - 1 < len(training_lengths)
            else {}
        )
        metadata = {
            "index": index,
            "trajectory_id": record["turn_id"],
            "termination_kind": record["termination_kind"],
            "query_id": record["query_id"],
            "constituent_turn_ids": record.get("turn_ids", []),
            "assistant_completion_count": record.get("assistant_completion_count"),
            "prompt_token_count": runtime.token_counter(prompt),
            "chat_template_prompt_token_count": (
                generator.count_prompt_tokens(prompt)
                if generator is not None
                and callable(getattr(generator, "count_prompt_tokens", None))
                else None
            ),
            "completion_token_count": runtime.token_counter(completion),
            "prompt_character_count": len(prompt),
            "completion_character_count": len(completion),
            "training_total_tokens": tl_entry.get("training_total_tokens"),
            "assistant_content_tokens": tl_entry.get("assistant_content_tokens"),
            "fits_in_training_max_sequence_length": tl_entry.get("fits"),
            "training_max_sequence_length": max_sequence_length,
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


def write_training_sequence_analysis(
    handle: TextIO,
    *,
    training_lengths: list[dict[str, Any]],
    trajectory_records: list[dict[str, Any]],
    max_sequence_length: int | None,
) -> None:
    """Write per-interval tokenized lengths with a FIT/EXCEEDS verdict."""
    write_section(handle, "Training Sequence Analysis")

    if not trajectory_records:
        handle.write("No trainable intervals.\n")
        return

    if max_sequence_length is None:
        handle.write(
            "  No training.max_sequence_length configured; fit check skipped. "
            "Tokenized lengths are still shown below.\n"
        )

    header = (
        f"{'#':>3}  {'trajectory_id':<16} {'termination':<16} "
        f"{'train_total':>12} {'asst_tokens':>12} {'max_seq':>10} {'headroom':>10}  verdict"
    )
    handle.write("\n")
    handle.write(header)
    handle.write("\n")
    handle.write("-" * len(header))
    handle.write("\n")

    exceeds_count = 0
    fits_count = 0
    unknown_count = 0
    worst_id = ""
    worst_total = 0

    for index, record in enumerate(trajectory_records, start=1):
        traj_id = record.get("turn_id", "?")
        term = record.get("termination_kind", "?")
        tl_entry = (
            training_lengths[index - 1]
            if index - 1 < len(training_lengths)
            else {}
        )

        train_total = tl_entry.get("training_total_tokens")
        asst_tokens = tl_entry.get("assistant_content_tokens")
        fits = tl_entry.get("fits")

        total_str = str(train_total) if train_total is not None else "n/a"
        asst_str = str(asst_tokens) if asst_tokens is not None else "n/a"
        max_seq_str = str(max_sequence_length) if max_sequence_length is not None else "—"
        headroom_str = (
            str(max_sequence_length - train_total)
            if max_sequence_length is not None and train_total is not None
            else "—"
        )

        if fits is True:
            verdict = "FITS"
            fits_count += 1
        elif fits is False:
            verdict = "EXCEEDS"
            exceeds_count += 1
            if train_total is not None and train_total > worst_total:
                worst_total = train_total
                worst_id = traj_id
        else:
            verdict = "n/a"
            unknown_count += 1

        row = (
            f"{index:>3}  {traj_id:<16} {term:<16} "
            f"{total_str:>12} {asst_str:>12} {max_seq_str:>10} {headroom_str:>10}  {verdict}"
        )
        handle.write(row)
        handle.write("\n")

    handle.write("\n--- Aggregate ---\n")
    write_key_values(
        handle,
        {
            "interval_count": len(trajectory_records),
            "fits": fits_count,
            "exceeds": exceeds_count,
            "unknown": unknown_count,
            "max_sequence_length": max_sequence_length,
        },
    )
    if exceeds_count > 0:
        handle.write(f"\nWorst: {worst_id} ({worst_total} tokens, exceeds by {worst_total - (max_sequence_length or 0)})\n")
        handle.write(
            "\nNote: Intervals marked EXCEEDS would be rejected by the training pipeline "
            "(ValueError: \"interval prefixes are never left-truncated\"). "
            "Increase training.max_sequence_length or lower context_threshold_tokens to fit.\n"
        )
    elif exceeds_count == 0 and max_sequence_length is not None:
        handle.write("\nAll intervals fit within training.max_sequence_length.\n")


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
    training_config: Any | None = None,
    training_max_sequence_length: int | None = None,
) -> Any:
    result = runtime.run(query_id=example.query_id, user_prompt=example.query)

    # Resolve the training max_sequence_length: CLI override > config > None
    max_sequence_length = training_max_sequence_length
    if max_sequence_length is None and training_config is not None:
        max_sequence_length = getattr(training_config, "max_sequence_length", None)

    # Compute tokenized training lengths once, shared by two sections
    training_lengths = _analyze_training_lengths(
        generator, result.trajectory_records, max_sequence_length
    )

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
                "training_max_sequence_length": max_sequence_length,
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

        write_context_timeline(
            handle,
            result=result,
            generator=generator,
            context_threshold_tokens=runtime.context_threshold_tokens,
            max_context_tokens=runtime.max_context_tokens,
        )

        write_context_summary(handle, result=result, runtime=runtime)

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
        write_exact_model_input_sequences(handle, result.trajectory_records)
        write_training_sequences(
            handle,
            runtime=runtime,
            terminal_status=result.status,
            trajectory_records=result.trajectory_records,
            generator=generator,
            max_sequence_length=max_sequence_length,
            training_lengths=training_lengths,
        )
        write_training_sequence_analysis(
            handle,
            training_lengths=training_lengths,
            trajectory_records=result.trajectory_records,
            max_sequence_length=max_sequence_length,
        )
        _write_judge_output(handle, judge, example, result.status, result.final_answer or "")
    return result


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

    result = trace_collection(
        runtime=runtime,
        generator=generator,
        example=example,
        sample_index=sample_index,
        output_path=output_path,
        include_formatted_prompt=args.include_formatted_prompt,
        training_config=getattr(config, "training", None),
        training_max_sequence_length=args.training_max_seq_len,
    )
    print(format_exact_model_input_sequences(result.trajectory_records))
    print(output_path)


if __name__ == "__main__":
    main()
