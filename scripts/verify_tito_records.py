"""Independent TITO verification for probe trajectory records.

Reads a probe ``*.records.jsonl`` file (one JSON trajectory record per line,
each record = one finalized compaction interval) and checks the append-only
token contract without trusting the runtime's own validation:

- collection token schema 3, contract, and renderer fingerprint present.
- Per generation: full_token_ids == prompt_token_ids + completion_token_ids.
- Append-only prefix chain: prompt of generation k+1 starts with the full
  sequence of generation k (P[k+1] = P[k] || C[k] || D[k]).
- Append spans tile [0, len(full)) exactly once, sampled flags match the
  assistant mask, and the sampled spans line up with generation boundaries.
- Every sampled completion that finished with reason "stop" ends in the
  tokenizer's <|im_end|> id (the sampled stop token belongs to the assistant).
- Successor intervals restart from the same stable prefix (system +
  original query + wrapped summary), not from the predecessor's raw tail.

Usage:
  python scripts/verify_tito_records.py FILE.records.jsonl [--tokenizer PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def check_interval(record: dict, interval_index: int, im_end_id: int | None) -> list[str]:
    problems: list[str] = []
    label = f"interval[{interval_index}]"
    tokens = record.get("collection_tokens")
    if not isinstance(tokens, dict):
        return [f"{label}: missing collection_tokens"]
    if tokens.get("version") != 3:
        problems.append(f"{label}: collection tokens version={tokens.get('version')!r}, expected 3")
    if not tokens.get("contract"):
        problems.append(f"{label}: missing TITO contract id")
    if not tokens.get("renderer_fingerprint"):
        problems.append(f"{label}: missing renderer fingerprint")

    generations = tokens.get("generations") or []
    full = tokens.get("full_token_ids") or []
    mask = tokens.get("assistant_token_mask") or []
    spans = tokens.get("spans") or []
    if not generations:
        problems.append(f"{label}: no generations recorded")
        return problems

    previous_full: list[int] | None = None
    sampled_total = 0
    completion_total = 0
    for gen_index, generation in enumerate(generations):
        gen_label = f"{label}.gen[{gen_index}]"
        prompt = generation.get("prompt_token_ids") or []
        completion = generation.get("completion_token_ids") or []
        gen_full = generation.get("full_token_ids") or []
        if gen_full != prompt + completion:
            problems.append(f"{gen_label}: full_token_ids != prompt + completion")
        if previous_full is not None and prompt[: len(previous_full)] != previous_full:
            problems.append(
                f"{gen_label}: prompt does not extend the previous generation's full sequence "
                f"(append-only violation: history rewritten or re-rendered)"
            )
        previous_full = gen_full
        if full[:len(gen_full)] != gen_full:
            problems.append(f"{gen_label}: generation is not an exact interval prefix")
        finish = generation.get("finish_reason")
        if completion:
            sampled_total += len(completion)
            completion_total += len(completion)
        if finish == "stop" and completion:
            if im_end_id is not None and completion[-1] != im_end_id:
                problems.append(
                    f"{gen_label}: sampled stop completion does not end in <|im_end|> "
                    f"({completion[-1]} != {im_end_id})"
                )

    if len(full) < len(generations[-1].get("full_token_ids") or []):
        problems.append(f"{label}: final full_token_ids shorter than last generation full")
    if len(mask) != len(full):
        problems.append(f"{label}: assistant mask length {len(mask)} != full length {len(full)}")

    # Span tiling: spans must cover [0, len(full)) exactly once in order.
    cursor = 0
    span_mask = [False] * len(full)
    for span_index, span in enumerate(spans):
        start, end = span.get("start"), span.get("end")
        sampled = span.get("sampled")
        if not isinstance(start, int) or not isinstance(end, int) or start != cursor or end > len(full):
            problems.append(f"{label}.span[{span_index}]: bad bounds ({start}, {end}), cursor={cursor}")
            break
        kind = span.get("kind")
        if kind in {"initial_state", "tool_result"} and sampled:
            problems.append(f"{label}.span[{span_index}]: conditioning span kind={kind!r} flagged sampled")
        for position in range(start, end):
            span_mask[position] = True
        cursor = end
    if cursor != len(full):
        problems.append(f"{label}: spans cover {cursor} of {len(full)} tokens")

    # Mask must equal sampled flags from spans.
    span_sampled = [False] * len(full)
    for span in spans:
        if span.get("sampled"):
            for position in range(span["start"], span["end"]):
                span_sampled[position] = True
    if mask and list(map(bool, mask)) != span_sampled:
        mismatches = sum(1 for a, b in zip(map(bool, mask), span_sampled) if a != b)
        problems.append(f"{label}: assistant mask disagrees with sampled spans at {mismatches} positions")
    if mask:
        owned = sum(1 for flag in mask if flag)
        if owned != sampled_total:
            problems.append(f"{label}: mask owns {owned} tokens but completions sum to {sampled_total}")

    # The retained final prompt must be a prefix of the interval full sequence
    # (any trailing tool/control span is conditioning-only and never re-dispatched).
    final_prompt = tokens.get("prompt_token_ids") or []
    if final_prompt and full[: len(final_prompt)] != final_prompt:
        problems.append(f"{label}: final prompt_token_ids is not a prefix of full_token_ids")

    return problems


def check_successor_prefix(records: list[dict], tokenizer) -> list[str]:
    problems: list[str] = []
    if len(records) < 2 or tokenizer is None:
        return problems
    first = records[0].get("collection_tokens") or {}
    spans = first.get("spans") or []
    if not spans or spans[0].get("kind") != "initial_state":
        return ["interval[0]: missing initial-state span"]
    # The first interval contains the stable system/query, without a summary.
    # Compare the entire committed initial prefix, not an arbitrary 64 tokens.
    stable = first["full_token_ids"][:spans[0]["end"]]
    stable_messages = records[0].get("messages", [])[:2]
    for index in range(1, len(records)):
        previous, current = records[index - 1], records[index]
        tokens = current.get("collection_tokens") or {}
        spans = tokens.get("spans") or []
        if not spans or spans[0].get("kind") != "initial_state":
            problems.append(f"interval[{index}]: missing initial-state span")
            continue
        initial = tokens["full_token_ids"][:spans[0]["end"]]
        if initial[:len(stable)] != stable:
            problems.append(f"interval[{index}]: initial state changed the stable token prefix")
        prefix = []
        for message in current.get("messages", []):
            if message["role"] in {"assistant", "tool"}:
                break
            prefix.append(message)
        # A forced-answer control may be appended immediately after a new
        # interval is initialized. It is outside the initial_state token span.
        controls = prefix[3:]
        valid_controls = not controls or (
            len(controls) == 1 and controls[0]["role"] == "system"
            and controls[0]["content"].startswith("<forced_answer_request>")
            and len(spans) > 1 and spans[1]["kind"] == "forced_answer_header")
        if len(prefix) < 3 or prefix[:2] != stable_messages or prefix[2]["role"] != "user" or not valid_controls:
            problems.append(f"interval[{index}]: successor must contain only stable system/query and wrapped summary")
            continue
        if current.get("tools") != records[0].get("tools"):
            problems.append(f"interval[{index}]: tool definitions changed")
        if previous.get("termination_kind") != "compaction":
            problems.append(f"interval[{index}]: predecessor did not successfully compact")
        generations = (previous.get("collection_tokens") or {}).get("generations") or []
        raw = tokenizer.decode(generations[-1]["completion_token_ids"], skip_special_tokens=False) if generations else ""
        body = raw.split("</think>", 1)[-1]
        summary = re.search(r"<\s*summary\s*>(.*?)<\s*/\s*summary\s*>", body, re.S | re.I)
        wrapped = "<summary>\n" + summary[1].strip() + "\n</summary>" if summary else None
        if wrapped is None or prefix[2]["content"] != wrapped:
            problems.append(f"interval[{index}]: successor does not contain the extracted predecessor summary")
        expected_suffix = "<|im_start|>user\n" + prefix[2]["content"].strip() + "<|im_end|>\n"
        suffix = tokenizer.decode(initial[len(stable):], skip_special_tokens=False)
        if suffix != expected_suffix:
            problems.append(f"interval[{index}]: successor initial state retains a tail or changes the summary")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="Path to *.records.jsonl from probe_vllm_collection.py")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path for <|im_end|> resolution and decode checks")
    args = parser.parse_args()

    records = load_records(args.records)
    if not records:
        print(f"FAIL: no records in {args.records}")
        return 1

    tokenizer = None
    im_end_id = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    all_problems: list[str] = []
    for index, record in enumerate(records):
        all_problems.extend(check_interval(record, index, im_end_id))
    all_problems.extend(check_successor_prefix(records, tokenizer))

    for record_index, record in enumerate(records):
        tokens = record.get("collection_tokens") or {}
        mask = tokens.get("assistant_token_mask") or []
        full = tokens.get("full_token_ids") or []
        owned = sum(1 for flag in mask if flag)
        coverage = owned / len(full) if full else 0.0
        print(
            f"interval[{record_index}]: kind={record.get('kind')} "
            f"termination={record.get('termination_kind')!r} generations={len(tokens.get('generations') or [])} "
            f"tokens={len(full)} assistant_owned={owned} coverage={coverage:.3f}"
        )

    if all_problems:
        print(f"\nFAIL: {len(all_problems)} problem(s)")
        for problem in all_problems:
            print(f"  - {problem}")
        return 1
    print(f"\nPASS: {len(records)} interval(s) satisfy the append-only TITO token contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
