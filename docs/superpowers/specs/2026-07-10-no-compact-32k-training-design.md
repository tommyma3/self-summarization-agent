# No-compaction 32k training preset

## Goal

Provide a dedicated training configuration for the no-compaction baseline used
by the compact-vs-no-compact comparison. The baseline must preserve the full
long-context training sequence rather than generating a long rollout and then
silently truncating it before the verl update.

## Configuration design

Create `configs/train/no_compact_32k.yaml` as a self-contained copy of
`configs/train/default.yaml`. Normal training-config loading does not implement
`base_config` inheritance, so a standalone YAML is required. It keeps the
existing model, dataset, optimizer, four-GPU verl/FSDP backend, judge, and
artifact contracts, differing only in the baseline-specific values:

- Disable runtime compaction with `runtime.context_threshold_tokens: 1000000000`.
- Match the comparison runtime envelope with
  `runtime.max_context_tokens: 40960`,
  `runtime.generated_token_budget: 32768`, and
  `rollout.max_model_len: 49152`.
- Set `rollout.max_concurrent_episodes: 8` so long-context vLLM KV-cache usage
  is conservative on the two rollout GPUs.
- Set `training.max_sequence_length: 49152` so cached no-compaction sequences
  are not left-truncated by the training adapter.
- Set both verl per-GPU token limits to `49152`:
  `ppo_max_token_len_per_gpu` and `log_prob_max_token_len_per_gpu`.
- Set `training.verl.fsdp.ulysses_sequence_parallel_size: 4` to split
  activation-heavy sequence work across the four 80 GB A100 training GPUs.

`training.backend: verl_ray` and
`training.verl.worker_backend: verl_fsdp` remain inherited from the default.
The preset therefore continues to run the policy update through official verl,
not through the legacy Transformers backend.

## Memory and operational behavior

The launcher runs collection, judging, caching, and policy update in separate
processes. The FSDP policy update can use all four configured GPUs after the
vLLM rollout processes exit. The long sequence configuration is designed for
microbatch size one, dynamic batching, activation checkpointing, and FSDP
sharding, all inherited from the default.

Four 80 GB A100s make this a reasonable full-context starting point, but the
exact peak allocation depends on the installed verl, FlashAttention, and model
build. Before a full run, launch one remote iteration with one training query
and inspect the per-rank peak memory. If that passes, restore the normal
collection count for the experiment.

## Validation

Validate that the new YAML resolves with the normal training-config loader and
that it preserves the verl backend fields. Add a focused configuration test
covering the no-compaction runtime, rollout, and full-context training limits.
No local GPU execution is required; the memory preflight is intentionally a
remote-server command.
