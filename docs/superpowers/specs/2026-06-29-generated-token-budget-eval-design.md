# Generated-Token Budget Evaluation Design

Date: 2026-06-29

## Goal

Compare BrowseComp-Plus evaluation behavior for two policy-runtime conditions:

1. no summarization or compaction, with a large active context window;
2. runtime-controlled summarization, with compaction triggered around an 8k-token prompt threshold.

The comparison should use the same checkpoint, eval query set, retrieval backend, judge, sampling settings, and reasoning-token budget. The target result is that compact and no-compact runs can match answer quality, while compact runs reduce active-context size and improve evaluation efficiency.

## Current Behavior

`EpisodeRuntime` already enforces a tool budget through `runtime.tool_budget`. When the tool budget is exhausted, the runtime switches to the forced-answer prompt. The runtime also triggers summarization with `runtime.context_threshold_tokens` and rejects packed prompts above `runtime.max_context_tokens`.

The current runtime does not have an episode-level generated-token budget. Existing rollout rows include `summary_turns`, `turn_records`, tool-call counts, retrieved docids, and judged rewards, but they do not summarize token usage by action, summary, forced answer, or prompt size. Existing eval metrics focus on accuracy, malformed outputs, and parse errors.

## Selected Design

Add a runtime generated-token budget that is separate from the existing tool budget.

The budget controls normal agent action generations only: search actions, document actions, malformed normal action generations, and voluntary final-answer generations. Summary completions are compression overhead and are reported separately, not counted against the agent reasoning budget.

The runtime should track these episode counters:

- `reasoning_generated_tokens`: normal action completion tokens counted against the new budget.
- `summary_generated_tokens`: summary completion tokens, reported as compression overhead.
- `forced_answer_generated_tokens`: completion tokens emitted by the forced-answer path.
- `total_generated_tokens`: reasoning plus summary plus forced-answer completion tokens.
- `prompt_tokens_by_turn`: prompt token counts for normal action, forced-answer, and summary prompts.
- `max_prompt_tokens_seen`: maximum prompt size observed for the episode.
- `summary_count`: number of successful summary turns.
- `retired_round_count`: number of history rounds retired into summaries.

The new runtime config should be optional:

```yaml
runtime:
  tool_budget: 16
  generated_token_budget: 32768
```

When `generated_token_budget` is unset, runtime behavior stays unchanged. When it is set, the normal action path checks the budget before each new normal action generation. If `reasoning_generated_tokens` has reached or exceeded the budget, the runtime uses the existing forced-answer path. If a normal action generation crosses the budget, the runtime accepts that completion and forces an answer on the next turn if the episode is still active. The runtime should not truncate model output mid-generation because truncation would create artificial parsing failures.

## Experiment Conditions

The no-compaction condition disables summarization by setting `runtime.context_threshold_tokens` above any reachable prompt size, for example `1000000000`. It should use a large active-context gate, such as `runtime.max_context_tokens=32768`, and a rollout model length high enough for prompt plus generation headroom.

The compaction condition sets `runtime.context_threshold_tokens=8000`. It should use the same checkpoint, eval split, retrieval config, model sampling config, judge config, tool budget, and generated-token budget as the no-compaction condition.

Both conditions should be collected as eval rollout artifacts. Training artifacts and policy update behavior are out of scope for this design.

## Metrics

Each rollout row should carry the token-usage counters so the artifact is self-describing. A comparison report should aggregate:

- eval accuracy, correct count, malformed count, and parse-error count;
- average reasoning generated tokens per query;
- average summary generated tokens per query;
- average forced-answer generated tokens per query;
- average total generated tokens per query;
- average and maximum prompt tokens seen;
- average search and document calls;
- average summary count;
- wall-clock seconds per query when phase timing or per-row timing is available;
- correct answers per 1k reasoning tokens;
- correct answers per 1k total generated tokens;
- correct answers per second when timing is available.

Primary quality comparison should use equal `reasoning_generated_tokens` budgets. Summary tokens should appear in efficiency and overhead reporting, not in the reasoning-budget gate.

## Failure and Resume Behavior

Budget exhaustion should not produce a new terminal status if the forced-answer path succeeds. The final status remains the normal completed, malformed, or forced-answer outcome already produced by the runtime path. Token counters should make it clear whether the forced-answer path was entered because of tool budget, generated-token budget, or both.

Resume behavior should continue to use rollout artifact completeness. The new metrics are row payload fields, so incomplete old artifacts should not be mixed with new comparison artifacts unless a migration or compatibility path is intentionally added.

## Validation

Unit tests should cover:

- generated-token budget disabled preserves current behavior;
- normal action completions increment `reasoning_generated_tokens`;
- summary completions increment `summary_generated_tokens` and do not consume reasoning budget;
- reaching the generated-token budget switches the next action to the existing forced-answer path;
- an action that crosses the budget is accepted and only forces answer on the next turn;
- serialized rollout rows include the token counters;
- comparison aggregation reports separate reasoning, summary, forced-answer, and total token metrics.

Focused Python compile checks should validate edited runtime, config, launcher utility, and reporting modules.
