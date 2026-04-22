# vLLM GRPO Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a process-isolated vLLM rollout and FSDP2/context-parallel training handoff that keeps one optimizer step per full training pass.

**Architecture:** Add explicit rollout/training config sections, checkpoint-tagged rollout artifact helpers, a parent iteration launcher, a rollout collection entrypoint, and a train-one-step entrypoint. The GPU-only FSDP2/context-parallel path is represented by a separate backend entrypoint and guarded with clear errors when optional distributed dependencies are unavailable.

**Tech Stack:** Python dataclasses, PyYAML config loading, existing runtime/judge/trainer modules, subprocess orchestration, pytest unit tests.

---

### Task 1: Config and Checkpoint Contract

**Files:**
- Modify: `src/self_summarization_agent/config.py`
- Create: `src/self_summarization_agent/checkpoints.py`
- Test: `tests/test_config.py`

- [x] **Step 1: Add tests for rollout/training backend config and latest checkpoint manifest**

Add assertions that train config reads `rollout.backend=vllm_offline`, `rollout.tensor_parallel_size=4`, and `training.backend=fsdp2_context_parallel`.

- [x] **Step 2: Implement config dataclasses**

Add `RolloutConfig` and extend `TrainingConfig` with backend/GPU/context-parallel fields while preserving existing defaults.

- [x] **Step 3: Implement checkpoint helpers**

Create helpers to resolve `latest`, write a complete marker, validate a vLLM-loadable checkpoint directory, and atomically advance latest.

### Task 2: Rollout Artifact Collector

**Files:**
- Create: `src/self_summarization_agent/rollout_collection.py`
- Test: `tests/test_rollout_collection.py`

- [x] **Step 1: Test checkpoint-tagged rollout artifacts**

Use fake backend/generator/judge to collect rollouts and assert each JSONL row records `policy_checkpoint_id`, `policy_checkpoint_path`, query metadata, judge payload, and runtime result.

- [x] **Step 2: Implement reusable collection function**

Build a function that receives config, examples, backend, generator, judge, checkpoint id/path, and output path. It should mirror current rollout generation and reward application without doing policy updates.

### Task 3: Train-One-Step Artifact Consumer

**Files:**
- Create: `src/self_summarization_agent/train_step.py`
- Modify: `src/self_summarization_agent/trainer.py`
- Test: `tests/test_train_step.py`

- [x] **Step 1: Test checkpoint mismatch and one-step update**

Load checkpoint-tagged rollout JSONL and assert mismatch fails before trainer mutation. Assert matched artifacts call trainer once and save/export the next checkpoint.

- [x] **Step 2: Implement artifact reader and step runner**

Read rollout JSONL, validate checkpoint id, extract trainable samples, group by query, call `trainer.step`, save next checkpoint, write complete marker.

- [x] **Step 3: Stream trainer gradients**

Change `TransformersPolicyTrainer.step` to call backward per sample/microbatch instead of accumulating live loss tensors before one backward call.

### Task 4: Parent Iteration Launcher

**Files:**
- Create: `src/self_summarization_agent/iteration_launcher.py`
- Test: `tests/test_iteration_launcher.py`

- [x] **Step 1: Test subprocess ordering and latest update**

Use injectable command runner to assert rollout runs first, train runs second, and latest advances only after training writes a complete checkpoint.

- [x] **Step 2: Implement launcher**

Resolve latest checkpoint, run rollout command, run train-step command, validate next checkpoint, atomically advance latest.

### Task 5: Entrypoints and Docs

**Files:**
- Modify: `src/self_summarization_agent/cli.py` or package modules as needed
- Modify: `README.md`
- Test: focused pytest subset

- [x] **Step 1: Add command-line entrypoints**

Expose module `main()` functions for rollout collection, train one step, and iteration launcher.

- [x] **Step 2: Document workflow**

Add README notes for the new process-isolated flow and the FSDP2/context-parallel backend status.

- [x] **Step 3: Run tests**

Run config, rollout collection, train step, iteration launcher, and launcher tests.
