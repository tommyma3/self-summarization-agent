# vLLM GRPO Training Design

Date: 2026-04-22

## Purpose

The training loop should use vLLM for fast rollout generation while preserving the on-policy requirement of GRPO. The current Transformers rollout path is too slow, taking around 10 minutes per query. The machine has four A100 80GB GPUs, which is enough to separate rollout serving from policy training.

## Decision

Use vLLM as a dedicated rollout service and keep Transformers as the trainable policy/logprob engine.

The training loop should run one on-policy rollout batch per gradient step:

1. Ensure the vLLM rollout service is serving the current policy weights.
2. Sample a batch of multiple training queries.
3. For each query, generate `group_size` full agent trajectories through vLLM.
4. Judge rewards for each trajectory.
5. Extract trainable summary and final-answer turns.
6. Compute group-relative advantages within each query group.
7. Flatten trainable turns across all query groups in the batch.
8. Recompute current-policy logprobs in the Transformers trainer.
9. Apply exactly one optimizer step.
10. Discard the rollouts.

This keeps the rollout policy and update policy aligned at the gradient-step boundary. It avoids the stale-policy behavior that would result from generating many epochs of rollouts from one old vLLM checkpoint.

## GPU Layout

The default hardware layout should be:

- GPU 0: vLLM rollout server.
- GPUs 1-3: Transformers policy trainer and optimizer state.

This avoids contention between vLLM KV cache allocation and training memory. It also keeps the first implementation simpler than colocating vLLM with the trainer.

## Gradient Step Signal

Each gradient step should use multiple queries, not a single query.

For example:

```yaml
training:
  batch_size: 4
  group_size: 8
```

This produces:

```text
4 queries x 8 rollouts/query = 32 trajectories per optimizer step
```

Advantages are normalized within each query group. The loss is then averaged across all trainable summary and final-answer turns from all groups.

Single-query steps are allowed for debugging, but the default design should prefer multiple queries because BrowseComp queries vary heavily in difficulty, rollout length, tool count, and reward variance.

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
  batch_size: 4
  group_size: 8
```

The existing `model` settings can remain as shared defaults, but rollout-specific and training-specific behavior should not depend on a single backend switch.

## Weight Synchronization

Because GRPO is on-policy, rollout weights must be current at the start of each gradient step.

### Phase 1: Step-Boundary Checkpoint Reload

The first implementation can sync vLLM by checkpoint reload:

1. Trainer saves a checkpoint after each optimizer step.
2. vLLM reloads or restarts from that checkpoint before the next rollout batch.
3. The next rollout batch is generated from the current policy.

This is simple and correct enough to validate the training loop. It may be slower than online weight transfer, but it should still be much faster than Transformers generation if rollout currently dominates runtime.

### Phase 2: Online Weight Transfer

After the server path is working, replace checkpoint reload with vLLM online weight transfer. This keeps vLLM alive and pushes updated weights into the inference engine after each optimizer step.

This is the right long-term path for fast on-policy GRPO, but it adds distributed synchronization complexity and should not block the first integration.

## Training Loop Shape

The current loop is close to:

```text
for epoch:
    generate rollouts for all train queries
    apply one optimizer update after the epoch
```

The target loop should be:

```text
for step:
    sync rollout engine to current policy
    sample batch_size queries
    for query in query_batch:
        generate group_size trajectories with vLLM
        judge rewards
        extract trainable samples
    group samples by query
    compute group-relative advantages
    update Transformers policy once
    save/sync weights for next step
```

Evaluation can run less frequently than training. It does not need to be part of every gradient step.

## Error Handling

- If vLLM returns an empty response, treat the rollout as malformed through the existing runtime path.
- If the vLLM server is unavailable, fail the training step clearly before mutating trainer state.
- If weight sync fails, do not generate the next rollout batch.
- If all rollouts in a step are malformed or produce no trainable samples, skip the optimizer update and log the skipped step.
- If a query group has zero reward variance, its advantages should remain centered at zero and contribute no gradient signal.

## Testing

Add tests around the orchestration boundaries rather than requiring vLLM in unit tests:

- config loading supports separate rollout and training backend fields
- training step samples `batch_size` queries and `group_size` rollouts per query
- advantages are normalized per query group, not globally
- exactly one optimizer update is applied per generated rollout batch
- rollout samples are discarded after the step
- vLLM server generator maps prompts, sampling settings, and returned text into the existing `TextGenerator` interface
- sync failure prevents the next rollout batch

Integration testing with real vLLM should be optional and gated by environment variables or a separate launcher.

## Open Implementation Notes

- The first code change should be config/interface separation, not vLLM weight transfer.
- The existing `VLLMGenerator` can remain useful for run-only experiments, but training should prefer the server generator.
- `eval_interval` should become effective so full train/eval accuracy is not recomputed after every optimizer step unless explicitly requested.
- Rollout artifacts should keep the policy checkpoint or step id used to generate them, so stale rollout bugs are visible in logs.
