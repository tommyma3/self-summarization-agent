# Development Contract for the Self-Summarization Agent

This repository trains a long-running research agent with reinforcement learning. The most important invariant is:

> Within an active trajectory interval, the agent only appends new events. It never rewrites, deletes, normalizes, or reconstructs earlier events. A successful summary is the only operation that retires prior context and starts a new interval.

Do not change this contract unless the user explicitly approves a redesign of the rollout and training format.

## What "append-only" means

Append-only is a strict token, message, and trajectory invariant. Production collection uses the project-owned Qwen renderer and an interval token ledger. After initialization, inference consumes the ledger's exact IDs; no backend may re-render or retokenize its historical prefix. KV prefix caching is allowed if it preserves those IDs.

The renderer encodes only new tool results, runtime controls, separators, and assistant headers. Sampled IDs are appended verbatim, with masks recorded at append time. Parsed messages are routing and diagnostic views, never a source for reconstructing the next token prompt. Only successful compaction can replace active context: finalize the immutable old ledger first, then initialize its successor. Archived intervals are never edited and every segment remains eligible for its configured GRPO or value-MC signal.

During one interval:

- Keep the original system instructions and tool definitions stable.
- Append each raw assistant completion exactly as generated, including reasoning and tool-call metadata.
- Append each tool result after its assistant tool call. Native tool results use linked `role: tool` messages and retain the `tool_call_id`.
- Append runtime-control instructions only at the boundary where they are needed. Do not replace an earlier system or user message with a control instruction.
- The summary request is an appended terminal `user` control message, not a replacement or additional leading system prompt. A forced-answer request is an appended terminal `system` control message and likewise does not replace the leading system instructions.
- Never rebuild earlier assistant reasoning from parsed or normalized tool calls.
- Never drop, left-truncate, reorder, or rewrite an interval prefix to make it fit.

Prefix caching, batching, and other inference optimizations must not change this logical sequence or the stored token evidence.

## Agent state machine

The initial interval is:

```text
stable system instructions
original user query
```

Normal research proceeds by appending:

```text
assistant reasoning and action
tool result
assistant reasoning and action
tool result
...
```

After a completed tool round, the runtime may choose one of three boundaries:

1. Continue research by appending another assistant/tool round.
2. Finish by appending the assistant's final answer. If a budget forces completion, append the forced-answer control first and then append the assistant answer.
3. Compact by appending the summary request, disabling tools for that generation, and appending the assistant's raw summary completion.

The summarization trigger is runtime-controlled, not model-controlled. Generated-token budget exhaustion takes priority over compaction.

## The only replacement operation: successful compaction

Compaction closes the current interval; it must not retroactively edit that interval.

The boundary remains part of the old interval:

```text
unchanged interval prefix
appended summary request
assistant summary reasoning and <summary>compressed state</summary>
```

Preserve the complete raw summary completion in trajectory records and token accounting. Separately extract the first complete `<summary>...</summary>` body after completed thinking and store that body as runtime compressed state.

Only after the old interval has been finalized may the runtime start the next interval as:

```text
the same stable system instructions
the same original user query
user: <summary>extracted compressed agent history</summary>
```

The system instructions and original user query remain byte-for-byte stable across every interval. The wrapper distinguishes model-generated compressed history from the original query. Do not retain a raw assistant/tool-event tail beside the summary in the new interval; the summary replaces the preceding agent history and must preserve gathered evidence, conclusions, unresolved work, and next steps without being responsible for reproducing the original query.

A malformed, empty, or over-limit summary is never installed as new state. Its raw failed output stays in the terminated interval and the rollout receives the configured penalty.

## Provider history rewrites

Under exact-token collection every mid-interval generation is verified: the server-rendered prompt of generation k+1 must start with the exact sampled tokens of generation k. The provider's chat template can normalize a malformed assistant turn (for example a think block closed with a misspelled tag such as `</thinking>` instead of `</think>`), which rewrites the interval history instead of appending to it. Training on a continuation of that prompt would condition on tokens the policy never saw. Two runtime consequences:

- An action output inside unresolved thinking, with ambiguous syntax, or without a supported sampled end-of-turn token is treated as malformed: retain the raw completion, penalize, and stop. On the TITO tagged-action path, a known `</thinking>` closing-tag typo may route an otherwise unambiguous complete action without repairing any sampled tokens. This does not permit dispatching action-looking text inside unfinished reasoning. Legacy message-based collection retains its strict unclosed-thinking guard.
- If a later backend input differs from the committed token request, fail closed. Runtime integrity checks discard an offending generation and finalize the valid prefix with status `history_rewrite_detected`; backend protocol errors abort collection. There is no fallback to reconstructed prompts.

Already-collected records that fail this verification raise `ProviderHistoryRewriteError` at extraction time and are excluded from trainable samples, so judging never trains on rewritten history.

## RL trajectory and token contract

One `trajectory_record` is one complete interval ending in compaction, final answer, forced answer, or malformed output. `turn_records` are diagnostics; training consumes `trajectory_records`.

The sparse training mask is:

- Trainable: every model-generated assistant token, including reasoning, tool actions, summary reasoning/body, and final-answer reasoning/body.
- Conditioning only: system instructions, the original query, any wrapped compressed state, tool results, summary requests, and forced-answer controls.

All intervals from one rollout share the same judged rollout reward. Do not give a rollout more independent reward-normalization weight merely because it compacted more often.

For OpenAI-compatible/vLLM collection, server-returned token IDs are authoritative:

- Preserve each generation's `prompt_token_ids`, `completion_token_ids`, and `full_token_ids`.
- Preserve the final interval `full_token_ids` and sparse assistant-token mask.
- Verify that earlier sampled completions occur in order inside later server-rendered prompts.
- Never replace missing or mismatched exact IDs with locally reconstructed tokenization when exact collection is required.
- Never train on a cleaned or normalized substitute for the raw sampled completion.

Reject an over-length interval instead of left-truncating its prefix.

## Rules for code changes

When changing runtime, prompts, chat templates, generation backends, collection, cache construction, or training:

1. Identify the interval boundary before editing.
2. Confirm that every non-compaction transition only appends messages or tokens.
3. Confirm that compaction first finalizes the complete old interval, then starts a fresh interval from the unchanged system instructions, unchanged original query, and wrapped summary.
4. Preserve raw generated output alongside any parsed runtime representation.
5. Preserve native tool-call structure and tool-call IDs.
6. Keep exact collected token IDs authoritative for training.
7. Test both no-compaction and one-or-more-compaction rollouts, including malformed summary behavior.
8. Treat any change that reconstructs history, changes preceding prompts at a boundary, or silently retokenizes training data as an RL data-contract change requiring explicit user approval.

The primary implementation paths are `src/self_summarization_agent/runtime.py`, `prompts.py`, `generation.py`, `trajectory.py`, `rollout_collection.py`, and the training-cache code. The append-only interval design in `docs/superpowers/specs/2026-07-31-append-only-compaction-interval-design.md` is the detailed reference.
