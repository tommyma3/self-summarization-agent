# TITO rollout implementation plan

Status: implemented; local validation completed, remote backend/GPU validation pending. See [validation and server handoff](2026-09-05-tito-validation.md).

Implement token-in, token-out (TITO) within every compaction interval. Each interval remains a separate training sequence, and every interval retains its existing rollout reward and mode-specific RL signal. Successful compaction finalizes the old interval and constructs a new state; it never edits the archived interval.

This applies the token-buffer design from `docs/agentic-rl-token-in-token-out-done-right.pdf`, pages 10-18, using an explicit renderer for the project's own chat template. The renderer owns the formatting of new tool/control spans and their token boundaries; the ledger owns the immutable sampled history. This replaces the earlier proposal to derive production bridges by diffing dummy template renders. It deliberately replaces the paper's last-rewrite-only training fallback (page 19) with the repository's existing complete-interval training design. Earlier intervals have their own actual sampled contexts, so they can be trained separately without reconstructing them inside a later interval.

The implementation contract is strict: after an interval is initialized, every token transition only appends. No renderer, backend, parser, budget handler, or training-cache path may rewrite, remove, normalize, or retokenize its prefix. Only successful compaction may replace the active context, by closing the old ledger and creating a new one. Even compaction cannot modify an archived interval. Update `AGENTS.md` and the append-only design specification during implementation to state this stronger token-level guarantee explicitly.

## 1. Establish the contract and regression fixtures

Pre-implementation evidence:

- `generation.py`: OpenAI-compatible collection calls Chat Completions with the entire message list. Offline vLLM and SGLang also render the whole history before each generation. Returning exact IDs does not prevent the next request from changing them.
- `runtime.py`: `_history_rewrite_detected` detects changed prefixes after generation. `_collection_token_payload` derives the interval sequence and assistant mask from generation snapshots. A separate unclosed-thinking guard rejects tool actions because they may be rewritten on the next request.
- `context.py` and runtime budget checks count re-rendered messages, which can disagree with the eventual TITO input.
- `trajectory.py`: extraction verifies exact generation prefixes; the native cache maps sampled-token logprobs to the sparse assistant mask.
- `trainer.py`: `compute_group_advantages` deduplicates rewards by rollout before normalization. Both the local and VERL value paths calculate `reward - old_value` at the first-generation prompt boundary. These semantics should remain intact.

Before changing implementation, capture regression fixtures for normal tagged tool calls, native calls, whitespace/argument serialization changes, noncanonical tokenization, and the malformed-thinking example from the history-rewrite note. Use explicit sampled IDs so a tokenizer round trip cannot accidentally make a test pass.

Required invariants for interval s, generation k:

```text
P[s, k] = exact IDs submitted to inference
C[s, k] = exact IDs returned by inference
D[s, k] = newly encoded environment/control bridge

P[s, k+1] = P[s, k] || C[s, k] || D[s, k]
```

The bridge may contain tool results, runtime-control messages, separators, and the next assistant header. It must contain no re-encoded historical assistant content. Parsed messages remain useful for routing and inspection; IDs and append-time spans define inference and training.

## 2. Add an interval token ledger and typed generation requests

Primary files: new `token_stream.py`, new `token_renderer.py`, `generation.py`, `runtime.py`, `prompts.py`.

- Give each active episode its own interval ledger: token IDs, assistant provenance mask, append spans, first-generation prompt length, and generation metadata. Keep it out of shared generator state so concurrent episodes cannot mix histories.
- Render/tokenize the initial state once: stable system instructions, original query, stable tool definitions, and an assistant generation header. Save the model/tokenizer/template identity and thinking configuration used to construct it.
- Append returned completion IDs verbatim and mark every sampled token trainable, including sampled stop tokens. Preserve their raw decoded representation, logprobs, finish reason, and parsed tool metadata separately.
- Append external spans with mask zero. Build masks at append time instead of recovering assistant spans by searching text or token subsequences later.
- Expose append and finalize operations on the ledger, with immutable snapshots for requests and archived intervals. Do not expose replacement, truncation, or a writable token-list reference. Validate each append against its expected interval/version so retries cannot duplicate a bridge or mutate a finalized interval.
- Introduce a typed token request carrying `prompt_token_ids`, generation kind, and the existing sampling/constraint settings. Avoid hiding the inference source of truth in a string-only prompt interface.
- Construct the complete next bridge before dispatch. Do not append an ordinary assistant header and then remove it when the boundary turns out to require a summary. Candidate request construction must be side-effect free; commit the chosen append once.
- Assert that the backend used the exact submitted IDs. Retain inter-generation prefix checks as independent corruption checks.

## 3. Build an explicit append-only renderer for the project template

Primary files: new `token_renderer.py`, new `token_stream.py`, `chat_template.py`, `chat_templates/qwen3_5_agent.jinja`, `prompts.py`.

Implement one renderer for the repository-owned Qwen agent template, shared by all token-capable collection backends. Define its supported tokenizer, special tokens, template revision, thinking modes, and action protocols explicitly. Supporting arbitrary external templates is outside this change; a new template requires an explicit renderer implementation or verified compatibility with the existing one.

The renderer has two distinct responsibilities:

- `render_initial_state(...)`: render/tokenize stable instructions, the original query, tool definitions, and any validated wrapped summary once when an interval is created.
- `render_continuation(new_events, boundary, sampled_tail_metadata)`: return only new token spans for tool results, any terminal control, required separators, and the next assistant header. This operation never takes the full historical message list or decodes/re-encodes sampled content. It may inspect the exact sampled tail IDs and finish reason to validate a supported boundary.

The ledger accepts sampled IDs through `append_sampled(...)` and renderer-produced conditioning spans through `append_external(...)`. The runtime chooses the boundary, commits the returned spans once, and submits an immutable token snapshot to the backend. The renderer does not clear or replace ledgers. Only the runtime's successful-compaction transition can finalize one interval and initialize its successor; ordinary finalization terminates the episode without creating replacement context.

Represent new events structurally and encode their content plus template-defined wrappers without re-encoding the prior prefix. Resolve special-token IDs through the configured tokenizer rather than assuming numeric constants. The renderer owns exact separators, role markers, tool wrappers, and assistant/thinking headers. Existing helpers may supply new control/result content, but there must be one owner of each transport wrapper to prevent double wrapping.

Keep the Jinja template as the initial-state formatter and a conformance reference. Production continuation must not call `apply_chat_template` on either real history or a dummy conversation to derive a bridge. Dummy-render comparisons may remain in tests as an independent formatting oracle for canonical examples; they are not a runtime fallback.

Support the actual two tool protocols already present:

- Offline tagged actions: tool output is a user message containing the existing `<tool_response>` / `<information>` wrappers.
- Native tools: linked `role: tool` messages retain `tool_call_id`; the current single-tool-per-turn policy remains in force.

Add explicit renderer cases for terminal user summary requests and terminal system forced-answer requests. Compose a pending tool result and a chosen control in one continuation, placing the assistant header only after the final input event. Keep the original system/tool definition prefix identical when tools are disabled for the summary generation; generation restrictions do not re-render that prefix.

Validate the seam between the sampled completion and the rendered append:

- A stop token already sampled belongs to the assistant mask and appears exactly once.
- Account explicitly for template separators after the stop token. The current Qwen template writes `<|im_end|>` followed by a newline, while generation may stop at the special token; the renderer must append the required newline as conditioning without adding another sampled terminator.
- Check generation headers and thinking openers are inserted exactly once and are conditioning tokens.
- Verify the supported atomic token boundary rather than assuming ordinary-text BPE boundaries are safe.
- Never trim sampled IDs, synthesize a repaired thinking block, or deduplicate arbitrary overlapping text. If the seam cannot be established, terminate safely with the original completion retained.

Run renderer/template conformance checks for supported thinking settings, both tool formats, controls, whitespace, Unicode, and representative result contents. For canonical completed turns, compare explicit renderer formatting against the approved template, accounting for stop tokens and formatting-only suffixes. For malformed or noncanonical sampled output, test literal prefix preservation independently; a full-history template render is not the correctness oracle because it may normalize that output.

Fingerprint the template, tokenizer, and renderer protocol together. Unsupported combinations fail preflight with an actionable diagnostic; there is no full-history fallback. Treat template and renderer changes as one reviewed formatting contract and update their conformance fixtures together. This makes the unique template's append-only behavior explicit and testable instead of relying on general template prefix-preservation assumptions.

## 4. Make all training collection backends consume IDs

Primary files: `generation.py`, `config.py`, `rollout_collection.py`, generator/probe tests.

| Backend | Planned token path |
| --- | --- |
| `vllm` / `vllm_offline` | Pass tokenized prompts to `LLM.generate`; keep returned completion IDs and raw-policy token logprobs. Implement first because all checked-in training presets currently collect through offline vLLM. |
| OpenAI-compatible vLLM server | Use a token-capable Completions endpoint with integer prompts and exact output IDs, instead of sending historical messages to Chat Completions. Probe capabilities and identity before collection. |
| SGLang | Use the engine's token-input interface; record submitted IDs as the input evidence and validate exact output IDs. Do not recreate prompt IDs by encoding text after inference. |
| Transformers / trainer-owned generation | Feed explicit `input_ids` and slice generated IDs directly. Add exact metadata where absent, or reject use as a TITO training collector until supported. |

For the native-tool API path, decode/parse returned IDs locally for tool dispatch, preserving raw output and stable linked call IDs in diagnostics. Preserve the current action syntax for each existing backend; adopting TITO must not silently switch offline tagged actions to native function calling.

Token endpoints do not automatically provide Chat Completions tool parsing or `tool_choice`. Explicitly implement and test equivalent summary/forced-answer restrictions through the token generation path. If a required capability cannot be preserved, fail the configuration rather than silently weakening it.

Preserve batching, episode refill, train/eval sampling settings, and lifecycle teardown. Capability detection must use the configured collection backend, independently of the training backend. A messages-only endpoint cannot qualify as exact TITO collection. Judge-only generation need not adopt a multi-turn ledger.

API references checked while planning: [vLLM tokenized inputs](https://docs.vllm.ai/en/v0.19.0/api/vllm/inputs/), [vLLM 0.19.1 renderer token path](https://docs.vllm.ai/en/v0.19.1/api/vllm/renderers/base/), and [vLLM exact output IDs for both completion endpoints](https://vllm.ai/blog/2025-10-22-agent-lightning). Exact installed-version signatures and native-tool parity remain implementation preflight checks.

## 5. Integrate boundaries and malformed-output handling

Primary files: `runtime.py`, `context.py`, `models.py`, runtime/context tests.

Normal round: generate from the ledger, append sampled IDs, parse for routing, execute the tool, and use the renderer to append only the new result and chosen continuation. The parsed tool call is never serialized back over its original sampled token span.

Successful compaction:

1. Append the terminal user summary request to the current token interval, then generate the raw summary from that exact prefix.
2. Append the full sampled summary completion with assistant mask one; validate the extracted body for structure, non-emptiness, and length.
3. Finalize and retain the complete old interval, including summary reasoning/body and all earlier actions.
4. Initialize a new ledger from the same system instructions and original query plus the wrapped extracted summary. The new summary wrapper/body is conditioning only. No raw history tail carries over.

Failed compaction retains its raw output in a terminated interval, installs no new state, and keeps the configured penalty. Make termination metadata reflect unsuccessful compaction as well as malformed syntax.

Count actual ledger/request IDs for context thresholds and overflow checks. Include pending tool/control/header deltas and the appropriate completion reservation. Keep generated-budget priority and existing budget semantics explicit; separate raw retrieved-text counts from transport-wrapper counts. Never left-truncate a prefix to fit.

Separate two cases currently conflated by the unclosed-thinking guard:

- A complete, unambiguously parsed tool action with an unusual thinking delimiter should not be rejected solely because a chat template could rewrite it. Under TITO, allow continuation only when the actual sampled end-of-turn seam is valid. Record the format anomaly; never repair it. Replace the current blanket rewrite-prevention guard with this rule.
- Incomplete/ambiguous actions, invalid arguments, and unsupported/truncated continuation seams still terminate under the malformed-output policy. Raw sampled tokens remain eligible for the configured negative RL signal. TITO does not authorize dispatching arbitrary action-looking text inside unresolved reasoning.

Retain existing integrity rejection for genuine mismatches and invalid legacy records. Report the proposed guard change as a versioned collection/termination semantics change; do not silently mix it into a resumed experiment's old reward history.

## 6. Preserve every segment through extraction, caches, and RL

Primary files: `trajectory.py`, `cache_step.py`, `rollout_collection.py`, `trainer.py`, `verl_ray_trainer.py`, `verl_value_worker.py`, `value_model.py`.

Persist ledger IDs and spans directly. Validate prompt/completion concatenation, per-generation offsets, exact backend inputs, assistant-mask ownership, and logprob alignment independently during extraction and cache construction. For new records, a deliberately retained trailing tool/control span at termination must be entirely conditioning; update the current final-generation-equals-interval check explicitly rather than dropping that span or weakening legacy validation.

Keep one training sample per finalized interval and stable query/rollout/segment identities. Every sampled assistant token is owned once in its original segment. A summary is trained where it was generated; its wrapped copy in the following state has no loss.

| Mode | Signal preserved for segment s of rollout r |
| --- | --- |
| `group_relative` / GRPO | Normalize terminal rewards across distinct rollouts for the same query, then assign the rollout's advantage to every segment. Extra compactions do not add entries to the reward-normalization group. |
| `compaction_mc_value` | Use `A[r,s] = R[r] - V_old(S[r,s])`, where `S[r,s]` is that interval's exact first-generation prompt. Freeze the baseline during the update and fit the existing value head to the same terminal outcome, with existing rollout-normalized value weights. |

Preserve existing actor loss reduction, clipping, value objective, and distributed padding behavior. Group-relative normalization and actor loss weighting are separate concerns; this transport fix must not silently introduce a different segment-weighting objective. Retain explicit experimental loss-mask ablations as opt-in exceptions, without enabling them for the requested full-segment training runs.

Reuse authoritative rollout logprobs only where their raw-policy provenance is established. Otherwise rescore the exact stored IDs under the collection checkpoint; never retokenize messages to build the cache. Check value anchors against those same IDs and exclude all later assistant tokens from the state estimate.

Version collection/cache metadata where the ledger schema changes. Include the TITO contract, tokenizer/template/renderer fingerprint, action protocol, thinking settings, and checkpoint identity in artifact compatibility checks. Old caches must not be relabeled as TITO or silently reused in a new TITO run. Valid historical artifacts may remain readable through their existing strict legacy validator; auditing or salvaging older runs is separate work.

## 7. Acceptance tests and rollout sequence

Tests must establish behavior, not just mirror the concatenation implementation:

- A deliberately noncanonical sampled tokenization survives multiple tool rounds unchanged, even when decode/encode would change it.
- After initialization, instrument template rendering and historical-text encoding to fail if invoked by any continuation. Tool, summary-request, and forced-answer paths must still work through explicit renderer appends. Full initial-state rendering becomes available again only after successful compaction finalizes the old interval.
- Golden renderer/template fixtures establish exact wrapper, role-marker, whitespace, and header parity for canonical cases. Renderer/template fingerprint mismatches fail before collection.
- Ledger snapshots remain unchanged after later appends; finalized ledgers reject writes, and no ordinary transition can clear, replace, or truncate a prefix.
- No-compaction, one-compaction, and multiple-compaction episodes retain exact prefixes and produce all expected training segments.
- Tool/native-call serialization differences and parseable thinking anomalies do not trigger historical re-rendering. Incomplete/ambiguous calls never dispatch.
- Malformed, empty, and over-limit summaries install no state and retain all raw completion tokens.
- Summary and forced-answer controls preserve the old prefix, including when a pending tool result exists. Failed or repeated request construction cannot append controls twice.
- Sampled stop tokens are trainable exactly once; separators, tool results, headers, and control tokens have zero mask. Unsupported and truncated seams fail safely.
- Batched episodes compact at different rounds without token/state leakage. Overflow handling rejects the interval without truncation.
- Extraction and native/rescored caches agree on IDs, sparse labels, state anchors, and per-token logprobs within an explicitly recorded numerical tolerance.
- GRPO groups with unequal segment counts retain the same rollout-level mean/std and advantages. All retained segments reach the trainer, including pre-summary segments.
- Value-MC tests verify `R - V_old` per segment, frozen prefix-only values, unchanged value-loss weights, and zero-contribution distributed padding.
- Negative probes reject missing output IDs, backend-modified inputs, wrong fingerprints, masks inconsistent with sampled spans, and rewritten legacy records.

Use the existing generation, runtime, context, chat-template, trainer, cache, rollout-collection, value, and VERL suites plus focused token-ledger and renderer tests. The user authorized feasible local pytest runs for this implementation, superseding the earlier remote-only testing preference. Use isolated local dependencies without repairing the server-oriented project venv; leave inference-engine and distributed GPU validation to the remote Linux environment.

Implementation order: explicit renderer and ledger contract, with template conformance -> offline vLLM and runtime integration -> extraction/cache contract -> remaining backend parity -> trainer regression checks and experiment migration.

Before a long run, complete a real-tokenizer template preflight, a small multi-round vLLM collection with all boundary types, cache/rescore comparison, one GRPO update, one value-MC update on the configured four-GPU path, and save/resume under the same TITO contract. Measure prompt construction time, collection throughput, peak memory, mask coverage, integrity failures, and actual malformed-output reasons. Passing a short probe establishes tested coverage, not universal correctness or recovered accuracy.

Start validation experiments in a fresh output lineage. Existing model/value weights may be initialization candidates after compatibility checks, but loading them does not establish behavioral equivalence or a clean continuation of old metrics. Collect fresh TITO data and caches; record the fixed collection semantics for all compared experiments.

Completion criteria: every supported TITO continuation uses the explicit renderer only to append new spans and consumes the preceding sampled prefix verbatim; successful compaction is the sole active-context replacement operation; every valid finalized segment reaches its selected RL path; and all renderer, boundary, cache, and remote smoke checks pass. Implementation and local checks are complete; real inference-engine, distributed update, performance, and save/resume acceptance remain for the server.
