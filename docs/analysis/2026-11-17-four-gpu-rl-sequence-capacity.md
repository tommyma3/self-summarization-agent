# Four-GPU RL sequence capacity

## Finding

The available hardware is four NVIDIA A100-SXM4-80GB GPUs. **A subsequent synthetic
65,536-token test passed two forward/backward/optimizer steps on all four GPUs.**
See the measured results below. For production rollouts with the current
Qwen3.5-9B actor, completed runs demonstrate 45,416-token intervals; the existing
long-context configuration admits 49,152 tokens. This is a configuration ceiling,
not a measured out-of-memory boundary. No measured maximum above this length is
available from this analysis.

64K (65,536 tokens) is now demonstrated for a single synthetic interval. 96K and 128K are
further candidates, not validated capacities. The analytical tensor sizes below
do not suffice to certify that any of these lengths fits. In particular, the
four GPUs do not provide a single 320 GB allocation pool.

## Evidence and scope

- `configs/train/no_compact_32k.yaml` sets a 49,152-token training limit,
  four-way Ulysses, activation checkpointing, parameter/optimizer offloading,
  and 12,288 tokens per GPU for actor updates and log-probability evaluation.
- Line 20 of `artifacts/train/qwen-bcplus-no-compact-32k-train/step_metrics.jsonl`
  reports 45,416 tokens and four completed optimizer steps. This run has 30
  completed update records. Its longest interval is 45,416 tokens.
- The currently running experiment uses `configs/train/no_compaction_loss.yaml`,
  whose per-interval cap is 27,000 tokens. A snapshot during its actor update
  showed approximately 64.3 GiB of device memory occupied per GPU. This is total
  device occupancy, including allocator reservations, not measured live tensor
  memory at a known sequence length; extrapolating it linearly would be invalid.
- Existing rank-zero worker logs for the November 17 15:42 UTC Ray session show
  4.38 GiB allocated and 12.89 GiB reserved after FSDP initialization. These are
  initialization figures, not backward-pass peaks.
- Local checkpoint headers contain 9,653,104,368 tensor elements, equivalent to
  17.98 GiB at two bytes per element including vision weights. A quarter is
  4.50 GiB. The loaded text actor log reports approximately 9.41B parameters.
- The model configuration declares 262,144 positions. This is an architectural
  limit, not evidence that training at that length fits these GPUs.

The initial analysis was read-only apart from this report. No production records,
configuration, model weights, or runtime behavior were changed. No new GPU
benchmark was run during that initial analysis: although GPUs were temporarily empty, `/proc` showed the
ongoing collection process preparing the next iteration. Competing for those
devices would interfere with the active experiment.

## Why 49,152 is currently the configured ceiling

The installed verl engine's `workers/engine/utils.py` computes its dynamic batch
budget as `max_token_len_per_gpu * sp_size`. The current long-context budget is
therefore `12,288 * 4 = 49,152`. The batcher requires every complete sequence to
fit this budget. Increasing only `training.max_sequence_length` is insufficient.
Actor batch construction can add one token to the shifted training-cache length;
boundary tests should check the actual full input length against the batch budget.

## Memory that grows with sequence length

The actor uses BF16, hidden size 4,096, 32 decoder layers, and vocabulary size
248,320. The normal verl forward materializes vocabulary logits; fused linear
log-probability kernels are disabled in the observed run.

For one long interval per microbatch, with sequence parallel size four:

| Full input tokens | Minimum per-GPU token budget | BF16 logits per GPU | One BF16 checkpoint input per decoder layer, per GPU |
| ---: | ---: | ---: | ---: |
| 49,152 | 12,288 | 5.68 GiB | 3.00 GiB |
| 65,536 | 16,384 | 7.58 GiB | 4.00 GiB |
| 98,304 | 24,576 | 11.37 GiB | 6.00 GiB |
| 131,072 | 32,768 | 15.16 GiB | 8.00 GiB |
| 262,144 | 65,536 | 30.31 GiB | 16.00 GiB |

These are component sizes, not total peak-memory predictions, and they need not
all reach their maxima simultaneously. Logits use `S/4 * 248320 * 2` bytes;
the checkpoint-input estimate uses `32 * S/4 * 4096 * 2` bytes. Extra checkpoint
metadata, other saved tensors, layer recomputation, temporary projections,
gradients, FSDP all-gathers/prefetch, communication buffers, CUDA context, and
allocator fragmentation are excluded.

There is a particularly important non-sharded cost in this repository:
`src/self_summarization_agent/qwen_interval_attention.py` gathers hidden states
across sequence-parallel ranks, then runs each overlapping interval's original
GatedDeltaNet mixer on that entire interval before slicing its output. For one
long interval, all four ranks evaluate that full interval in recurrent layers.
One full-interval BF16 hidden tensor alone is 0.5 GiB at 64K and 1 GiB at 128K.
The mixer also creates QKV projections, convolution outputs, expanded query/key
heads, gates, recurrent-kernel intermediates, and their backward tensors.
Consequently these workspaces cannot be divided by four in a memory estimate.

Sparse assistant loss masks do not remove the full conditioning sequence or
avoid computing the current implementation's logits at conditioning positions.
Inference KV-cache capacity also does not establish backward-pass capacity.

## Settings needed for a capacity test

Use a separate benchmark configuration/output directory. Preserve the current
four-way Ulysses, BF16, checkpointing, and CPU offloading to make the comparison
meaningful. Keep the effective microbatch to one long interval; dynamic batching
can otherwise pack multiple shorter intervals into the same token budget.

For a full input of 65,536 tokens:

```yaml
training:
  max_sequence_length: 65536
  verl:
    fsdp:
      ppo_max_token_len_per_gpu: 16384
      log_prob_max_token_len_per_gpu: 16384
      ulysses_sequence_parallel_size: 4
```

For 98,304 tokens use 24,576 in both per-GPU budgets; for 131,072 use 32,768.
Also align `rollout.max_model_len` and the runtime context limit with the desired
complete interval, reserving space for appended controls and terminal output.
These are proposed benchmark settings; none were installed.

An empirical ceiling requires an isolated run through log-probability scoring,
forward, backward, gradient clipping, and a real optimizer step after optimizer
state initialization. Record peak allocated and reserved memory on every rank,
finite losses/gradients, and elapsed time. Warm up and repeat passing lengths.
Start at 48K and 64K, then test 96K and 128K as permitted by measured headroom;
refine between the largest passing and first failing length. Require several GiB
of headroom for an operational limit rather than choosing the last allocation
that barely fits. Test the critic independently for PPO variants: the separate
critic's data parallelism does not split a single interval across GPUs.

Training length always means the complete immutable interval, including system,
query, compressed state when applicable, tool results, controls, and assistant
tokens. A longer episode can comprise several such intervals after successful
compaction. Capacity testing must never truncate or reconstruct production
intervals to obtain a passing result.

## Possibility of substantially longer sequences

The following are engineering assessments, not additional measurements:

| Full input length | Assessment on four 80 GB GPUs |
| --- | --- |
| 128K | Worth benchmarking with the current implementation; not certified to fit. |
| 192K | An aggressive candidate; fused output-head computation and recurrent context parallelism would improve feasibility. |
| 256K | A credible optimization target, not a supported capacity claim. Close attention to temporary workspace and throughput is necessary. |
| 512K or 1M | Requires more than raising YAML limits: output-head/activation memory changes, positional-support work, and long-context quality validation. No evidence of practical feasibility for this complete RL pipeline. |

At 192K, ordinary BF16 logits occupy 22.73 GiB per GPU and the checkpoint-input
estimate is 12 GiB. At 256K those components are 30.31 and 16 GiB. Thus 256K
already accounts for approximately 46.3 GiB in just these two components, before
the weight shard and the substantial omitted terms. Their combined growth is a
warning against extrapolating the current four-way parallelism to arbitrarily
long intervals.

At 512K, those component sizes become 60.63 and 32 GiB. Under the current
checkpointed, materialized-logits implementation, their expected overlapping
storage already exceeds a GPU's capacity. At 1M tokens, the ordinary BF16 logits
tensor alone is 121.25 GiB per GPU. These calculations assume lengths of
524,288 and 1,048,576 respectively; they are not benchmark results.

### Optimization priorities

1. **Avoid materializing all vocabulary logits.** The installed verl
   `models/transformers/qwen3_5.py` includes fused torch and Triton output paths;
   the latter calls `linear_cross_entropy`. The normal path allocates
   `self.lm_head(hidden_states)` for the whole local shard. A fused/chunked
   implementation can remove the need for the complete 30.31 GiB logits tensor
   at 256K, although it still needs tile buffers and backward storage. The
   repository's `build_verl_fsdp_worker_config` does not currently expose the
   relevant model options. Integration must verify shifted-label alignment
   across Ulysses shards, temperature handling, log probabilities, sparse masks,
   and gradients. Library availability is not proof of a safe toggle.

2. **Distribute recurrent computation without gathering whole intervals.** The
   installed FLA `ops/gated_delta_rule/chunk.py` accepts `cp_context` and contains
   explicit forward and backward context-parallel processing. The repository's
   interval wrapper instead gathers full hidden states and computes complete
   overlapping intervals. Replacing this with correct context parallelism could
   reduce that duplicated work and memory. It requires convolution boundary
   halos, recurrent-state propagation, backward state gradients, and exact
   interval resets. Resetting state or detaching it at arbitrary shard boundaries
   would change the model's conditioning or training gradient. This is a
   substantive integration project, not a batch-size adjustment.

3. **Offload saved activations if still necessary.** The installed verl
   `HFModelConfig` has `enable_activation_offload`, and its FSDP engine invokes
   activation offloading when enabled. This option is not exposed by the current
   repository worker-config builder. Approximately 2 TiB host RAM was present
   during inspection, so host capacity is less constraining than GPU capacity;
   transfer overhead, pinning, and compatibility still need measurement.
   Offloading checkpoint inputs does not eliminate recomputation's transient
   workspace. The installed model configuration also offers tiled MLPs, another
   possible transient-memory reduction requiring validation.

4. **Preserve the sparse objective.** A custom output head could compute policy
   logits only at assistant-token loss positions while keeping every conditioning
   token in the transformer and recurrent computation. This can reduce head
   memory substantially for sparse trajectories. It requires careful alignment
   and equivalent losses/gradients; it does not justify deleting conditioning
   tokens or using a different trajectory representation.

### Context support and throughput

The local model's declared context is 262,144 positions. RoPE-based code may
execute beyond a declared maximum, but that does not establish trained context
support or good retrieval, tool use, and policy behavior there. Extending to 512K
or 1M requires an appropriate positional configuration and quality evaluation;
it cannot be represented as a supported configuration-only capacity increase.

Eight of the 32 decoder layers use full attention. With exact full attention,
their attention arithmetic grows quadratically even when FlashAttention avoids
storing the quadratic attention matrix. Relative to 64K, the attention component
at 128K, 256K, 512K, and 1M has approximately 4x, 16x, 64x, and 256x the work.
These are ratios for attention arithmetic, not predictions of total wall time:
linear layers, recurrent layers, communication, and offloading have different
scaling. Long-context rollout collection and any separate critic must also be
validated, independently of actor backward-pass capacity.

The strongest near-term path is to benchmark the unchanged implementation beyond 64K up to
128K, then evaluate fused output computation and correct recurrent context
parallelism for 192K/256K. No precise maximum beyond the previously demonstrated
45,416-token interval can be claimed without actual forward/backward/optimizer
measurements. The subsequent 64K probe below establishes a new synthetic capacity
point, not the maximum. No optimization or production configuration was changed here.

## Measured 64K forward/backward test

The user subsequently requested a 64K backpropagation test. The isolated
`scripts/benchmark_actor_sequence.py` probe loaded the base Qwen3.5-9B actor and
used the production interval-preserving worker, ordinary materialized logits,
BF16, four-way Ulysses, gradient checkpointing, parameter/optimizer offloading,
and 16,384 tokens per GPU. The actor path ran a real clipped policy loss,
backward pass, gradient clipping, and Adam optimizer step twice. No model
checkpoint was saved and no production process was stopped.

The input was one synthetic, seeded random-ID interval containing exactly
65,536 tokens (65,535 shifted next-token targets), with every target marked
trainable. The constant synthetic reference log probability and positive
advantage are capacity-test inputs, not collected RL evidence or quality scores.
This tests a microbatch of one long interval; it does not establish rollout,
critic, or arbitrary larger-microbatch capacity.

| Step | Elapsed time | Peak allocated per GPU | Maximum reserved across GPUs | Gradient norm before clipping |
| --- | ---: | ---: | ---: | ---: |
| First | 44.51 s | 43.58 GiB | 62.69 GiB | 1,720 |
| Second | 16.01 s | 52.35 GiB | 71.59 GiB | 11 |

Both steps passed finite-loss and finite, nonzero-gradient checks. Peak allocated
memory was identical across the four ranks; second-step reserved memory ranged
from 71.50 to 71.59 GiB. The second-step memory includes initialized optimizer
state and is the more relevant sizing result. Its extra optimizer-state memory
must not be mistaken for sequence-length-dependent activation memory.

Initialization took 61.55 s; total probe execution before Ray shutdown was
122.23 s. Timing includes worker RPC, CPU offloading, synchronization, and memory
inspection, rather than a GPU-kernel-only measurement. The first step can also
include initialization and kernel warmup. The complete result is preserved in
`docs/analysis/2026-11-17-actor-64k-backprop.json` and the original output in
`artifacts/benchmarks/actor-64k-2026-11-17-test.json`. GPU occupancy returned to
4 MiB on every device after the isolated Ray job exited.

This demonstrates that **64K actor training fits with the tested settings**.
It does not establish the maximum length or validate long-context task quality.
