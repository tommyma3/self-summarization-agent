# Token-MC value training

Each value-network input remains a complete, exact collected interval (shifted by
one token for next-token prediction). Tool results and controls remain conditioning
context. Sparse sampled positions select the hidden states used by the two-bin
value head. Intervals are never concatenated: the Qwen hybrid recurrent layers must
not carry state from one interval into another. There is no truncation,
retokenization, or change to compaction boundaries.

The trainer prepares token and sampled-position tensors once per fit, length-buckets
intervals, and right-pads within each microbatch. It evaluates the sparse head and
cross entropy in batches. Loss accumulation stays on the GPU until the epoch ends.
Diagnostic weighting counts sampled tokens once per interval instead of repeatedly
constructing position lists.

The objective remains the mean over rollouts, then over nonempty intervals in each
rollout, then over sampled tokens in each interval. Microbatches accumulate into
one optimizer step per rollout minibatch. Frozen pre-fit actor advantages, optimizer
state, clipping, and checkpoint serialization retain their existing semantics.

## Four GPUs

Use `configs/train/compact_separate_token_mc_ppo_4gpu.yaml` for a fresh run. It sets:

```yaml
training:
  value:
    gpu_ids: [0, 1, 2, 3]
    prefix_microbatch_size: 4
    eval_prefix_microbatch_size: 4
    max_prefix_tokens_per_batch: 108000
```

The training and pretraining launchers set GPU visibility before loading the
critic. `gpu_ids: []` preserves the existing single `gpu_id` behavior. The critic
phase finishes before policy training allocates these devices. Direct Python
callers must set `CUDA_VISIBLE_DEVICES` to the selected devices before CUDA
initialization and use local `cuda:0`.

This is single-process data parallelism: each GPU gets independent intervals, runs
the backbone and sparse head locally, and returns only two-bin logits. PyTorch
reduces gradients to the primary replica; one optimizer owns and saves state.
Replicas are created per forward, so very short sequences or small batches can be
slower than one GPU. Optimizer memory stays on the primary GPU. A single interval
is not split across GPUs. At least four intervals and a sufficient aggregate token
budget are needed to engage all four devices. The 108000 budget permits four
27000-token inputs; it does not change the 27000-token per-interval limit.

Use a new run/output directory when changing the value configuration: frozen
artifact fingerprints intentionally reject configuration mismatches on retries.
Do not change the configuration of an active staged update.

## Validation and measurement

`test_parallel_token_update_matches_single_gpu` compares sparse predictions and
parameter updates against a single-GPU reference, including unequal interval
counts, sparse masks, and a partial microbatch. It passed on four A100 GPUs.
The token tests also cover no compaction, repeated compaction, malformed summaries,
checkpoint retries, and preservation of raw token evidence.

A reproducible capacity/throughput probe (synthetic data, not an RL quality score):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 .venv/bin/python \
  scripts/benchmark_token_value.py --length 27000 --samples 12 \
  --devices 4 --microbatch 4 --output /tmp/token-value-4gpu.json
```

Use `--devices 1 --microbatch 1` with one visible GPU for the comparison. The probe
uses distinct exact ID sequences and alternating signed rewards, and performs a
real optimizer update and final evaluation. Timing fields separate sequence
preparation, updates, and final evaluation. Fit time also includes diagnostics.
Results depend on interval lengths, padding, minibatch size, and competing GPU work.

On three idle A100 GPUs, a 12-interval, 27000-token Qwen3.5-0.8B probe
measured 23.12 seconds on one GPU versus 10.09 seconds on three (2.29×).
Update time was 18.93 versus 8.51 seconds; final evaluation was 4.07 versus
1.47 seconds. Sequence preparation took 0.12 seconds in both cases. Both
runs had finite outputs and a nonzero gradient norm of 18.875. These are
single synthetic measurements, not a four-GPU throughput guarantee. Raw
measurement fields and conditions are in `value-training-benchmark.json`.
