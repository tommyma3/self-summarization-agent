# vLLM GRPO Training Design

Date: 2026-04-22

## Purpose

The training loop should use vLLM for fast rollout generation while preserving the on-policy requirement of GRPO. The current Transformers rollout path is too slow, taking around 10 minutes per query. The machine has four A100 80GB GPUs, which is enough to separate rollout serving from policy training.

## Decision

Use vLLM as a dedicated rollout service and keep Transformers as the trainable policy/logprob engine.

The training loop should keep the current high-level cadence: process the full configured training set, then apply one gradient step.

1. Ensure the vLLM rollout service is serving the current policy weights.
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

This remains on-policy because all rollouts for the update are generated before the policy changes, and the trainer applies only one optimizer step from that rollout set. It would become stale-policy training only if the same rollout set were reused for additional optimizer steps after the policy update.

## GPU Layout

The default hardware layout should be:

- GPU 0: vLLM rollout server.
- GPUs 1-3: Transformers policy trainer and optimizer state.

This avoids contention between vLLM KV cache allocation and training memory. It also keeps the first implementation simpler than colocating vLLM with the trainer.

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

Introduce a rollout generator separate from the trainable policy object.

The runtime already depends on a small interface:

```python
generate(prompt: str) -> str
count_tokens(text: str) -> int
```

The vLLM integration should satisfy this interface through either:

- an OpenAI-compatible HTTP client pointed at a vLLM server, or
- the existing in-process `VLLMGenerator` for offline experiments.

The first production path should be an HTTP/server generator, because it maps cleanly onto the four-GPU layout and lets the trainer and rollout engine have independent processes.

### Trainer

Keep `TransformersPolicyTrainer` as the owner of trainable weights, optimizer state, logprob recomputation, gradient clipping, and checkpoint saving.

vLLM should not be used for gradient computation. It only generates candidate trajectories.

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
  backend: vllm_server
  endpoint: http://127.0.0.1:8000/v1
  gpu_ids: [0]
  tensor_parallel_size: 1
  max_new_tokens: 8192
  temperature: 0.7
  top_p: 0.95
  do_sample: true

training:
  backend: transformers
  gpu_ids: [1, 2, 3]
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

### Phase 1: Step-Boundary Checkpoint Reload

The first implementation can sync vLLM by checkpoint reload:

1. Trainer saves a checkpoint after each optimizer step.
2. vLLM reloads or restarts from that checkpoint before the next full training pass.
3. The next set of training rollouts is generated from the current policy.

This is simple and correct enough to validate the training loop. It may be slower than online weight transfer, but it should still be much faster than Transformers generation if rollout currently dominates runtime.

### Phase 2: Online Weight Transfer

After the server path is working, replace checkpoint reload with vLLM online weight transfer. This keeps vLLM alive and pushes updated weights into the inference engine after each optimizer step.

This is the right long-term path for fast on-policy GRPO, but it adds distributed synchronization complexity and should not block the first integration.

## Training Loop Shape

The current loop is close to the desired cadence:

```text
for epoch:
    generate rollouts for all train queries
    apply one optimizer update after the epoch
```

The target loop should be:

```text
for epoch_or_step:
    sync rollout engine to current policy
    for query in all configured train queries:
        generate group_size trajectories with vLLM
        judge rewards
        extract trainable samples
    group samples by query
    compute group-relative advantages
    stream trainable samples through the trainer in microbatches
    accumulate gradients across the full rollout set
    update Transformers policy once
    save/sync weights for next epoch_or_step
```

Evaluation can run less frequently than training. It does not need to be part of every gradient step, especially when a gradient step already covers the full configured training split.

## Error Handling

- If vLLM returns an empty response, treat the rollout as malformed through the existing runtime path.
- If the vLLM server is unavailable, fail the training step clearly before mutating trainer state.
- If weight sync fails, do not generate the next full training pass.
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
- vLLM server generator maps prompts, sampling settings, and returned text into the existing `TextGenerator` interface
- sync failure prevents the next full training pass

Integration testing with real vLLM should be optional and gated by environment variables or a separate launcher.

## Open Implementation Notes

- The first code change should be config/interface separation, not vLLM weight transfer.
- The existing `VLLMGenerator` can remain useful for run-only experiments, but training should prefer the server generator.
- `eval_interval` should become effective so full train/eval accuracy is not recomputed after every optimizer step unless explicitly requested.
- Rollout artifacts should keep the policy checkpoint or step id used to generate them, so stale rollout bugs are visible in logs.
- The current trainer accumulates `losses` as live tensors before one backward call; that must be replaced with streaming gradient accumulation before using 24k-26k token sequences.
