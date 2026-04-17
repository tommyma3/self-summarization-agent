# Self-Summarization Agent Design

Date: 2026-04-17

## Goal

Build a faithful research prototype of an AI agent with self-summarization for BrowseComp-Plus using a Qwen3.5-9B base model and RL. The prototype should isolate whether training the model to write better summaries and better final answers improves downstream benchmark performance when retrieval is fixed.

## Scope

This design covers:

- an RL-native agent runtime separate from `bc-plus/search_agent`
- fixed BrowseComp-Plus retrieval and document access
- runtime-triggered synthetic summarization
- GRPO-style RL with updates only on summary turns and final answer turns
- multi-GPU full-parameter training
- evaluation compatibility with BrowseComp-Plus run export and judge pipeline

This design does not cover:

- training the retriever
- RL on tool-call turns
- richer tool surfaces beyond benchmark search and document retrieval
- extra memory tools, planners, or note-writing actions

## Research Hypothesis

If the runtime compresses long interaction histories at safe boundaries and the model is trained with RL to produce summaries that preserve task-critical evidence, then the agent should make better use of limited context and achieve higher downstream performance on BrowseComp-Plus than the same base model without summary training.

## High-Level Approach

The system will run benchmark episodes in a custom RL environment. The model uses the benchmark's fixed `search` and `get_document` tools to gather evidence. When the runtime estimates that the next packed context would exceed a configured threshold, it waits until the current tool round is complete and then injects a synthetic summarization turn. The model writes a new summary, which replaces the older compactable history. The agent continues from a packed context containing the system prompt, user prompt, latest summary, and a recent raw tail.

RL is applied only to:

- generated summaries
- final answers

Tool-call generations are executed during rollout but excluded from the policy loss.

## Architecture

### TaskSampler

Loads a medium training slice from BrowseComp-Plus and creates:

- a train split for RL
- a held-out dev split for frequent evaluation
- metadata needed for deterministic sampling and reproducibility

### BenchmarkBackend

A thin adapter over BrowseComp-Plus assets and services. It exposes:

- `search(query)`
- `get_document(doc_id)`

It also normalizes tool inputs and outputs into an internal event schema used by the rollout runtime.

### AgentRuntime

Runs one episode at a time and owns:

- the live conversation state
- tool and token budgets
- summary count
- termination status
- transition logging

The runtime, not the model, decides when summarization is required.

### ContextPacker

Builds packed contexts for acting, summarizing, and final answering. It is responsible for:

- estimating packed token length with the system tokenizer
- preserving fixed headers such as the system prompt and user prompt
- retaining a recent unsummarized raw tail
- selecting the compactable prefix that will be summarized

### SummaryInjector

Creates a synthetic summary turn after a completed tool round when the runtime determines that compaction is needed. It produces the summary-generation input, calls the model, records the output summary, and updates the compact state.

### RewardEngine

Implements task rewards, local malformed-tool penalties, and training masks. It distinguishes between:

- global task outcomes that train summary and answer turns
- local protocol failures on tool-call turns that do not propagate backward to prior summary turns

### TrajectoryStore

Persists stepwise records required for GRPO, including:

- packed model inputs
- generated outputs
- tool calls and tool results
- summary events
- final answers
- per-token training masks
- reward metadata

### GRPOTrainer

Samples grouped rollouts, computes relative advantages, and performs full-parameter multi-GPU updates. It only includes tokens marked trainable by the runtime.

### Evaluator

Exports held-out runs in BrowseComp-Plus format and evaluates them with the benchmark's existing pipeline.

## Runtime Data Flow

### Episode Initialization

Each episode starts with:

- system prompt
- user prompt
- empty previous summary
- empty interaction history
- configured tool budget
- configured runtime context threshold

### Acting Loop

The agent acts using the base model and may emit:

- `search(query)`
- `get_document(doc_id)`
- `finish(answer)`

After each valid tool call, the runtime executes the tool and appends the tool result to history.

### Summarization Trigger

The runtime estimates token length using the system tokenizer and future packing rules. The model does not decide whether summarization is needed.

After each completed `assistant -> tool call -> tool result` round:

1. The runtime estimates the next packed context length.
2. If it remains under the configured threshold, the agent continues.
3. If it exceeds the threshold, the runtime marks summarization as required.
4. The runtime injects a synthetic summarization turn only after the current tool round is complete.

This avoids summarizing across broken action-observation pairs.

### Synthetic Summary Turn

The summary-generation sequence is:

- system prompt
- user prompt
- previous summary if present
- compactable older interaction prefix
- summary prompt
- generated new summary

After the new summary is generated, the active context becomes:

- system prompt
- user prompt
- new summary
- retained recent raw tail

### Final Answer Turn

When the model emits `finish(answer)`, the final-answer sequence is:

- system prompt
- user prompt
- latest summary if present
- retained tool history and raw tail
- generated final answer

## Prompt Contracts

### Agent System Prompt

The acting prompt should be short and stable. It should instruct the model to:

- solve the benchmark question using the provided tools
- ground claims in retrieved evidence
- avoid unsupported guesses
- provide a concise final answer once sufficient evidence has been collected

### Summary Prompt

The summary prompt should be minimal and avoid over-specifying the output. It should instruct the model to:

- write a clean summary containing only essential information for continuing the task
- preserve normalized facts, current hypotheses, unresolved questions, and useful next steps
- keep evidence-grounded facts tied to `doc_id` citations

There is no explicit summary token cap in the prompt. Overall context budget is enforced by the runtime.

## Training Objective

The prototype uses GRPO-style optimization, but only for:

- summary turns
- final answer turns

Tool-call turns are part of rollout execution but are excluded from the policy loss.

### RL-Trainable Sequence Types

#### Summary Sequence

The trainable sequence is:

- system prompt
- user prompt
- previous summary
- compactable history and tool results
- summary prompt
- new summary

Only the `new summary` tokens receive RL updates.

#### Final-Answer Sequence

The trainable sequence is:

- system prompt
- user prompt
- previous summary
- retained tool history
- final answer

Only the final-answer tokens receive RL updates.

### Training Mask Rules

- summary tokens: trainable
- final-answer tokens: trainable
- tool-call tokens: not trainable
- tool-result tokens: not trainable
- system, user, and retained context tokens: conditioning only

## Reward Semantics

### Global Task Outcomes

If the episode ends with a correct answer:

- all prior summary turns in the episode receive propagated reward `+1`
- the final-answer turn receives reward `+1`

If the episode ends with an incorrect answer:

- all prior summary turns in the episode receive propagated reward `-1`
- the final-answer turn receives reward `-1`

If the episode ends because the tool or runtime budget is exhausted:

- all prior summary turns in the episode receive propagated reward `-1`
- there is no final-answer training example

### Malformed Tool Calls

If the model emits a malformed tool call:

- terminate the rollout immediately
- assign local reward `-1` only to that offending timestep
- do not propagate this local format penalty backward to earlier summary turns
- do not create a final-answer training example

This treats protocol failure as a local acting mistake rather than a summary-quality signal. In `v1`, because tool-call turns are excluded from the RL loss, this malformed-tool penalty is recorded for termination and diagnostics rather than used for gradient updates.

## Summary Content Policy

The summary should preserve:

- normalized facts from retrieved evidence
- active answer hypotheses
- unresolved subquestions
- useful next actions
- citations to supporting `doc_id`s

The summary should avoid:

- verbatim transcript copying
- unsupported claims
- long raw excerpts unless they are necessary for future reasoning
- free-form memory that cannot be tied back to retrieved documents

## Evaluation Plan

The system should be evaluated in four stages.

### Unit Tests

Verify:

- token-threshold trigger logic
- prefix and raw-tail selection
- reward propagation masks
- malformed-tool termination semantics

### Episode Simulation Tests

Run against a deterministic fake backend to verify:

- summary injection only occurs after completed tool rounds
- summary and answer tokens are the only trainable tokens
- malformed tool calls stop the episode immediately

### Held-Out Dev Evaluation

Run a small held-out slice and export runs in BrowseComp-Plus format for compatibility checks with the benchmark evaluation pipeline.

### Research Ablations

At minimum compare:

- no summarization
- summarization enabled without RL updates
- RL-trained summarization plus RL-trained final answering

Optional ablations may vary the context threshold and retained raw-tail size.

## Implementation Notes

- Keep `bc-plus` as the benchmark backend only.
- Build all rollout, packing, reward, and training logic in a separate codepath.
- Keep prompts simple so gains come from learning rather than prompt engineering.
- Prefer deterministic logging and reproducible sampling because the main research claim depends on subtle runtime behavior.

## Open Decisions Deferred From V1

The following are intentionally deferred:

- learned summarize timing instead of runtime-fixed timing
- RL on tool-use turns
- retriever training or reranking
- richer browsing interfaces
- structured memory beyond summaries

## Success Criteria

The prototype is successful when:

- it can run end-to-end on a medium BrowseComp-Plus training slice
- it produces valid benchmark-compatible run files
- summaries are inserted at runtime-controlled safe boundaries
- only summary and final-answer tokens are trained with RL
- the system supports full-parameter multi-GPU GRPO training
- ablations can isolate the effect of RL-trained summarization
