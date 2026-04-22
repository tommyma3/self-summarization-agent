# vLLM GRPO Training Design

Date: 2026-04-22

## Purpose

The training loop should use vLLM for fast rollout generation while preserving the on-policy requirement of GRPO. The current Transformers rollout path is too slow, taking around 10 minutes per query. The machine has four A100 80GB GPUs, which is enough to run rollout collection and training as separate full-GPU phases.

## Decision

Use offline vLLM for rollout collection and keep Transformers as the trainable policy/logprob engine. Run them in separate subprocesses so each phase can use all four GPUs and release memory cleanly before the next phase starts.

The training loop should keep the current high-level cadence: process the full configured training set, then apply one gradient step.

1. Launch the rollout subprocess with offline vLLM loaded from the current policy checkpoint.
2. Iterate over all configured training queries for the current epoch or step.
3. For each query, generate `group_size` full agent trajectories through vLLM.
4. Judge rewards for each trajectory.
5. Extract trainable summary and final-answer turns.
6. Store rollout records as text, rewards, and metadata.
7. Compute group-relative advantages within each query group.
8. Flatten trainable turns across all query groups in the epoch.
9. Recompute current-policy logprobs in the Transformers trainer with microbatching and gradient accumulation.
10. Apply exactly one optimizer step after all configured training queries have been processed.
11. Discard the rollouts.

This remains on-policy because all rollouts for the update are generated from the current checkpoint before the policy changes, and the trainer applies only one optimizer step from that rollout set. It would become stale-policy training only if the same rollout set were reused for additional optimizer steps after the policy update.

## GPU Layout

The default hardware layout should be phase-based:

- Rollout phase: GPUs 0-3 run offline vLLM inference.
- Training phase: GPUs 0-3 run the Transformers trainer and optimizer.

The two phases should not coexist in one long-lived process. A parent launcher should run rollout collection in one subprocess, wait for it to exit, then run the training step in a second subprocess. This avoids contention between vLLM KV cache allocation and training memory, and it avoids relying on partial CUDA cleanup inside one Python interpreter.

## Gradient Step Signal

Each gradient step should use all configured training queries, not a single query or a small sampled batch.

For example:

```yaml
dataset:
  train_limit: 200

training:
  group_size: 8
```

This produces:

```text
200 queries x 8 rollouts/query = 1600 trajectories per optimizer step
```

Advantages are normalized within each query group. The loss is then averaged across all trainable summary and final-answer turns from all groups.

Single-query or small-batch steps are allowed for debugging, but the default design should match the current configuration: one optimizer step after the configured training split is processed. This produces a large, stable update signal, at the cost of slower policy refreshes.

## Architecture

### Rollout Generator

Introduce a rollout collector separate from the trainable policy object.

The existing runtime depends on a small interface:

```python
generate(prompt: str) -> str
count_tokens(text: str) -> int
```

The first implementation can use the existing in-process `VLLMGenerator` inside a short-lived rollout subprocess. That subprocess should load vLLM from the current checkpoint, collect one full training pass, write rollout artifacts, and exit.

The simple `generate(prompt)` interface is acceptable for the first version. A later collector should batch across active episodes at each generation point so offline vLLM can exploit higher throughput:

```text
active episodes need next action -> batch prompts into vLLM
execute tools
episodes needing summaries -> batch summary prompts into vLLM
repeat until all episodes finish
```

The batched collector is an optimization, not required to validate the process-isolated design.

### Trainer

Keep `TransformersPolicyTrainer` as the owner of trainable weights, optimizer state, logprob recomputation, gradient clipping, and checkpoint saving.

vLLM should not be used for gradient computation. It only generates candidate trajectories. The training subprocess should load the checkpoint used by rollout collection, consume the rollout JSONL, compute the single update, and save the next checkpoint.

### Parent Launcher

Add an iteration launcher that owns the phase boundary:

```text
for iteration:
    run collect_rollouts_vllm subprocess on GPUs 0-3
    verify rollout artifacts were written for the current checkpoint
    run train_one_step subprocess on GPUs 0-3
    verify the next checkpoint was written
```

The checkpoint path is the synchronization contract between phases. The rollout artifact should record the checkpoint id or path used to generate it.

### Config

Decouple rollout backend from training backend. The current `model.backend` field conflates both roles.

A future config shape should distinguish:

```yaml
model:
  model_path: /124090467/Qwen/Qwen3.5-9B
  dtype: bfloat16
  trust_remote_code: false
  enable_thinking: true

rollout:
  backend: vllm_offline
  gpu_ids: [0, 1, 2, 3]
  tensor_parallel_size: 4
  max_new_tokens: 8192
  temperature: 0.7
  top_p: 0.95
  do_sample: true

training:
  backend: transformers
  gpu_ids: [0, 1, 2, 3]
  group_size: 8
  gradient_accumulation_microbatch_size: 1
```

The existing `model` settings can remain as shared defaults, but rollout-specific and training-specific behavior should not depend on a single backend switch.

### Logprob Microbatching

The optimizer step should still happen once per full training pass, but the backward computation must not retain every long-sequence graph at once.

For 24k-26k token prompts, the trainer should process trainable turns in tiny microbatches, usually one trainable turn at a time:

```text
optimizer.zero_grad()
for sample in full_epoch_samples:
    loss = per_sample_loss(sample) / contributing_sample_count
    loss.backward()
clip gradients
optimizer.step()
```

This preserves the current "one gradient step after all training sequences" behavior while keeping GPU memory bounded. Rollout artifacts can be stored as JSONL text; only the active microbatch needs a live computation graph.

## Weight Synchronization

Because GRPO is on-policy, rollout weights must be current at the start of each full training pass that feeds one gradient step.

### Phase 1: Process-Isolated Checkpoint Boundary

The first implementation should sync through process isolation and checkpoint files:

1. The parent launcher selects the current checkpoint.
2. The rollout subprocess loads that checkpoint into offline vLLM.
3. The rollout subprocess writes JSONL artifacts tagged with that checkpoint id and exits.
4. The training subprocess loads the same checkpoint with Transformers.
5. The training subprocess consumes the rollout JSONL, applies one optimizer step, saves the next checkpoint, and exits.

This is simple and correct enough to validate the training loop. It pays model reload cost twice per iteration, but that cost is likely acceptable while rollout generation is the dominant bottleneck.

### Phase 2: Batched Offline Collector

After the sequential offline collector is correct, replace per-episode generation with an active-episode batched collector. This keeps the same checkpoint boundary but improves vLLM utilization during agentic rollouts.

### Phase 3: Persistent Server or Online Weight Transfer

If model reload time becomes the bottleneck, consider a persistent vLLM server or online weight transfer. This should be treated as a later optimization because it adds distributed synchronization complexity.

The persistent path is not required for correctness. The process-isolated checkpoint path already preserves the GRPO on-policy contract as long as each rollout set feeds exactly one optimizer step.

## Training Loop Shape

The current loop is close to the desired cadence:

```text
for epoch:
    generate rollouts for all train queries
    apply one optimizer update after the epoch
```

The target loop should be:

```text
for iteration:
    current_checkpoint = latest checkpoint

    collect_rollouts_vllm subprocess:
        load current_checkpoint with offline vLLM on GPUs 0-3
        for query in all configured train queries:
            generate group_size trajectories
            judge rewards
            extract trainable samples
        write rollout JSONL tagged with current_checkpoint
        exit and release GPUs

    train_one_step subprocess:
        load current_checkpoint with Transformers on GPUs 0-3
        read rollout JSONL for current_checkpoint
        group samples by query
        compute group-relative advantages
        stream trainable samples through the trainer in microbatches
        accumulate gradients across the full rollout set
        update Transformers policy once
        save next checkpoint
        exit and release GPUs
```

Evaluation can run less frequently than training. It does not need to be part of every gradient step, especially when a gradient step already covers the full configured training split.

## Error Handling

- If vLLM returns an empty response, treat the rollout as malformed through the existing runtime path.
- If the rollout subprocess fails, do not launch the training subprocess for that iteration.
- If the training subprocess fails, do not mark the next checkpoint as current.
- If the rollout artifact checkpoint id does not match the checkpoint loaded by the trainer, fail before mutating trainer state.
- If all rollouts in a full training pass are malformed or produce no trainable samples, skip the optimizer update and log the skipped step.
- If a query group has zero reward variance, its advantages should remain centered at zero and contribute no gradient signal.
- If the number of contributing samples is zero after filtering zero-advantage samples, skip backward and log a zero-loss update record.

## Testing

Add tests around the orchestration boundaries rather than requiring vLLM in unit tests:

- config loading supports separate rollout and training backend fields
- training step processes all configured training queries and `group_size` rollouts per query
- advantages are normalized per query group, not globally
- exactly one optimizer update is applied per full training pass
- trainer backpropagates through microbatches while applying one optimizer step at the end
- rollout samples are discarded after the update
- offline vLLM rollout subprocess writes checkpoint-tagged JSONL artifacts
- parent launcher does not start training if rollout collection fails
- trainer refuses rollout artifacts from a different checkpoint id

Integration testing with real vLLM should be optional and gated by environment variables or a separate launcher.

## Open Implementation Notes

- The first code change should be the process-isolated iteration launcher and rollout artifact contract, not vLLM weight transfer.
- The existing `VLLMGenerator` can be used inside the rollout subprocess for the first version.
- `eval_interval` should become effective so full train/eval accuracy is not recomputed after every optimizer step unless explicitly requested.
- Rollout artifacts should keep the policy checkpoint or step id used to generate them, so stale rollout bugs are visible in logs.
- The current trainer accumulates `losses` as live tensors before one backward call; that must be replaced with streaming gradient accumulation before using 24k-26k token sequences.
