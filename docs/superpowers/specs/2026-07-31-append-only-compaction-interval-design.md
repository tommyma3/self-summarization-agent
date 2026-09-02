# Append-Only Compaction and Interval Training Design

## Goal

Preserve the model's complete reasoning and action trajectory up to each compaction or final-answer boundary. Compaction must be an ordinary continuation of the active context, not a separately reconstructed request that discards chain-of-thought or divides the trajectory into per-round training samples.

## Runtime Context Contract

The initial interval is:

```text
system instructions
user request
assistant reasoning and action
tool result
assistant reasoning and action
tool result
...
```

Assistant outputs are stored verbatim. Tool results are appended as conditioning messages. The runtime does not rebuild prior reasoning from normalized tool calls.

When the compaction threshold is reached after a completed tool round, the runtime appends the compaction instruction to the end of that unchanged interval and generates the summary as the next assistant message:

```text
system instructions
original user request
<summary>previous compressed agent history</summary> (after the first compaction)
assistant/tool events
...
compaction instruction
assistant compaction reasoning and `<summary>compressed state</summary>`
```

The compaction prompt is concise and does not state the configured summary-body limit. It requires completed thinking followed by compressed agent history inside `<summary>...</summary>`. Because the original user request remains present verbatim in every interval, the summary preserves gathered evidence, conclusions, unresolved work, and next steps without needing to restate the request. The runtime first removes everything through `</think>`, then stores only the text inside the first complete wrapper in the remainder, ignoring any other prefix or suffix. The stored body remains the value used for summary metrics and diagnostics. When it becomes the next interval's third message, the runtime wraps that body again in `<summary>...</summary>` to distinguish model-generated compressed history from the original query. Missing `</think>` or summary wrappers are malformed-tool-call failures; a present but empty wrapper is an empty-summary failure. The full raw completion, including thinking and wrappers, remains in the interval trajectory and generated-token accounting.

The compaction prompt does not insert a second copy of the previous compressed state. A valid summary retires the preceding assistant/tool history while preserving the stable task prefix. The next action begins from exactly:

```text
system instructions
original user request
<summary>new compressed agent history</summary>
```

The system instructions and original user request are byte-for-byte identical across intervals. There is no retained raw assistant/tool-event tail after compaction, and only the latest compressed state is carried forward. The complete retired interval remains immutable in its trajectory record.

## Forced-Answer Contract

When a tool-call or episode-token budget forces an answer, the runtime leaves the active interval unchanged, appends the forced-answer instruction, and generates the answer as the next assistant message. The episode-token budget counts model completions and each raw appended tool result once. It does not switch system prompts or reconstruct a special final-answer context.

## Training Sequence Contract

One RL sample represents one complete interval. An interval terminates with one of:

- successful or rejected compaction
- natural final answer
- forced final answer
- malformed model output

The sample contains the full structured message sequence. Its token mask is sparse:

- every assistant-generated token is trainable, including reasoning, tool actions, summary reasoning/body, and final-answer reasoning/body
- system instructions are conditioning-only
- the original user request and any compressed state are conditioning-only
- tool results are conditioning-only
- appended compaction and forced-answer instructions are conditioning-only

Summary and forced-answer generations therefore require no special training rows. They are the final assistant spans in otherwise ordinary interval sequences.

All intervals from one rollout receive the same judged rollout reward. Group-relative advantage is computed at rollout granularity and then broadcast to every interval from that rollout, so rollouts with more compactions do not receive extra independent reward normalization weight.

## Artifact and Cache Contract

Generation-level `turn_records` remain diagnostic metadata. Training consumes `trajectory_records`, which contain:

- a trajectory id and termination kind
- the ordered structured messages
- constituent diagnostic turn ids
- assistant-completion count
- optional sparse token cache

Trajectory schema version 3 records the stable system-plus-original-query prefix contract and must not be mixed with earlier rollout rows during resume or training. Training cache version 6 tokenizes the entire interval once, masks only assistant targets, records whether reference log probabilities came from raw vLLM rollout logits or policy rescoring, and stores those probabilities aligned with the sparse mask. It also stores the exact first-generation prompt length as the interval-start value-state anchor. Rollout-native values are accepted only when every generation has an exact token-prefix alignment with the finalized interval; otherwise collection falls back to policy rescoring. An interval that exceeds the configured training sequence length is rejected; its prefix is never left-truncated because truncation would violate the system/state/trajectory contract.

## Failure Semantics

Malformed actions, summaries over the configured body limit, and empty compressed states terminate with a penalized status. The failed boundary output remains in the interval sample, and every interval in the failed rollout receives the terminal negative reward. A rejected summary is never installed as the next compressed state.
