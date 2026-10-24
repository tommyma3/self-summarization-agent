# Separate value-MC performance plan

Status: phase timings, batching, critic-specific checkpointing, evaluation cadence,
and reduced diagnostics implemented. Multi-GPU work is deferred. The user installed
FLA and causal-conv1d; Qwen fast-path imports and GPU forward/backward execution
have now been verified. No further dependency changes were made during validation.

Goal: reduce time from collected trajectories to saved fitted advantages, preserving the existing three-epoch critic objective and the append-only/exact-token training contract.

## Evidence from the current implementation and recorded run

Inspected `artifacts/train/qwen-bcplus-compact-separate-mc/` and the separate-critic preset. These are recorded measurements, not a new controlled benchmark.

| Completed iteration | Critic fit, including internal evaluations | Actor update | Whole train-update phase |
| --- | ---: | ---: | ---: |
| 1 | 27.5 min | 162.1 min | 192.5 min |
| 2 | 27.1 min | 161.5 min | 193.5 min |
| 3 | 35.0 min | 157.1 min | 198.7 min |

Sources: `step_metrics.jsonl`, `phase_timings.jsonl`, and `train-iterations-20261021T164723Z.log`. `value/fit_seconds` excludes model loading, checkpoint saving, and evaluation-collection scoring. The actor timer wraps `trainer.step` after the critic process exits. The remaining phase time is not attributable to held-out evaluation alone.

- `train_step.py` exposes only `training.value.gpu_id` to a single critic subprocess. Actor training already uses multiple GPUs.
- `separate_value_trainer.py` processes one prefix per microbatch, for both training and evaluation, under the current configuration.
- Three epochs produce four complete training-set inference passes: before fitting and after each epoch. The final epoch predictions are already reused for advantages; `eval_after` repeats diagnostic calculations, not model inference.
- `critic_step.py` saves the critic, then unconditionally attempts same-policy evaluation-collection scoring, before writing fitted advantages. This blocks actor startup when evaluation states exist.
- Completed runs used 200 training rollouts and 436–445 states, with 21 critic optimizer updates. Evaluation collections in iterations 2–3 contain 256 states.
- Critic peak allocated memory is about 7.0 GiB. Logged training-state prefix p95 is 583–617 tokens. This suggests batching headroom, but longest-prefix memory and reserved memory still need measurement.
- The critic startup logs explicitly warn that Qwen3.5 fast-path dependencies are unavailable. The local Transformers implementation checks causal-conv1d and FLA gated-delta-rule kernels. FlashAttention for ordinary attention is a separate concern.
- The lockfile specifies PyTorch 2.10.0+cu128 and Transformers 5.5.3; kernel compatibility must be tested against the actual training environment.

## Contract and phase boundary

The optimization boundary is after collection/cache validation and before actor construction. Collection and compaction behavior remain unchanged: successful compaction finalizes the complete old interval before installing the next interval; malformed summaries create no successor.

Critic inputs remain the exact collected IDs through `first_generation_prompt_end`. Stored full intervals, raw completions, masks, rewards, tool IDs, and sample identity remain immutable. Batch sorting changes execution order only, with results restored by sample key. Never truncate a prefix, retokenize, change the anchor, drop states, or increase a rollout's statistical weight because it contains more intervals.

Keep fitting epochs, global rollout minibatches, learning rates, and optimizer-step count unchanged in the first optimization series. Preserve all eligible critic-only states. Final values must come from the final fitted critic, and advantages remain `reward - fitted_value`, frozen for the entire subsequent PPO update.

## 1. Establish phase-level timings and repair kernel fallback

Files: `separate_value_trainer.py`, `critic_step.py`, dependency/environment setup, benchmark script.

- Measure sample loading/validation, model loading, each fit epoch, optional diagnostics, mandatory final predictions, checkpoint save, eval-collection preparation/inference, advantage export, and total critic subprocess time.
- Record useful prefix tokens/s, states/s, padding ratio, optimizer steps, and per-rank allocated/reserved memory. Synchronize only at coarse measurement boundaries; use CUDA events or a short profiler trace for detailed GPU timing.
- Verify which fast-path symbols are actually available in the critic process, including backward execution. Install and pin compatible FLA and causal-conv1d versions in a benchmark environment, then compare outputs, gradients, and throughput with the fallback implementation.
- Confirm the dense-attention backend separately. Do not assume that installing FlashAttention enables Qwen's recurrent/linear-attention kernels.
- Include startup/kernel compilation in end-to-end measurements and report warm throughput separately. Do not upgrade unrelated dependencies or modify an active training environment as part of the benchmark.

Acceptance: complete timing breakdown, confirmed kernel dispatch, finite gradients, numerical parity within measured dtype-appropriate tolerances, and repeatable speed improvement before promotion.

## 2. Increase batching and avoid repeated host/device work

Files: `config.py`, `separate_value_trainer.py`, separate critic preset.

- Separate train and inference microbatch settings. Sweep train sizes 4, 8, 16 and evaluation sizes 16, 32, 64, subject to measured memory. These are benchmark candidates, not guaranteed defaults.
- Add a padded-token budget and length bucketing. Within each existing global rollout minibatch, group similar prefix lengths while preserving each state's original rollout weight. Evaluation may bucket across the entire dataset and restore original order afterward.
- Validate prefixes once per prepared dataset. Prebuild CPU token tensors, lengths, targets, and identity/weight metadata. Avoid rescanning every token in Python during every forward.
- Compute maximum padded length on the CPU; the current `int(lengths.max())` reads a CUDA scalar. Collate on CPU and transfer batches together, using pinned memory/nonblocking copies where useful.
- Accumulate evaluation values/losses on-device in bounded buffers and copy them in batches, avoiding two CPU transfers per single-state forward. Benchmark `inference_mode` for the read-only evaluation path.
- Add critic-specific activation checkpointing, independent of the 9B actor setting. Benchmark disabling it for the 0.8B critic; retain a configurable enabled mode for long-prefix or memory-constrained runs.
- Keep bf16 and the current optimizer semantics initially. Fused optimizer experiments are secondary, after kernel dispatch and batching.

Acceptance: mixed-length batched predictions match unbatched predictions; single-device optimizer updates match the rollout-weighted reference within tolerance; longest valid prefixes fit without truncation.

## 3. Remove optional evaluation from the critical path

Files: `config.py`, `separate_value_trainer.py`, `critic_step.py`, `value_diagnostics.py`, metric consumers/tests.

Proposed fast preset:

| Work | Policy |
| --- | --- |
| Pre-fit full training-set evaluation | Disabled by default; opt-in for debugging |
| Intermediate epoch full evaluations | Disabled; report loss accumulated during training, clearly labeled as online training loss |
| Final fitted predictions for all training states | Always run exactly once; required for PPO advantages |
| Final basic statistics | Compute from the same mandatory predictions |
| Eval-collection critic scoring | Every 5 training iterations initially; explicit 0 disables it; independent of policy rollout evaluation |
| Calibration bins and state-stratified diagnostic payloads | Detailed mode on scheduled diagnostic iterations |
| Progress output | Rank zero, phase boundaries plus a time-based heartbeat |

With three epochs this removes three of four training-set evaluation passes: a 75% reduction in that inference work, not a claimed 75% reduction in fitting time. It also removes eval-collection scoring from four out of five iterations under the proposed cadence.

Keep final rollout-weighted CE, basic value/advantage summaries, counts, gradient norm, step count, timing, and memory. Preserve nonfinite checks and artifact validation. Compute the final diagnostic set once; avoid duplicating identical final-epoch/eval-after payloads. Metric consumers must tolerate explicit skipped evaluations and must not carry old measurements forward as current ones.

Scheduled eval collections must still match the behavior policy checkpoint. If subsampling becomes necessary, select whole rollouts deterministically and report coverage. Do not sample or skip the mandatory final training-state predictions.

Only rank zero writes progress and metric files. Do not expect suppressing a few print statements to solve the slowdown: unnecessary forward passes and synchronization are higher-priority work. Optional diagnostics must not perturb training RNG or optimizer state.

## 4. Add multi-GPU critic fitting and prediction

Files: `train_step.py`, `critic_step.py`, `separate_value_trainer.py`, `config.py`, `evaluate_value.py`, new distributed tests.

- Add `training.value.gpu_ids` and a single-device/DDP backend selection, retaining legacy `gpu_id` configurations. Reject ambiguous settings. Launch one process per selected GPU with torchrun and map physical device IDs through local rank.
- Wrap backbone and value head together in one DDP module. Set the local CUDA device before model construction, including any fused kernel initialization. Use NCCL on production GPUs.
- Benchmark 1, 2, and 4 GPUs. The model fits on one GPU, making replicated data parallelism the first candidate. FSDP is a later option only if critic size or memory requirements justify sharding.
- Keep the global minibatch at 32 rollouts across all ranks, rather than 32 per GPU. Generate the same deterministic global rollout schedule, then partition work with prefix-token cost balancing. All eligible states appear exactly once per epoch with their global weights.
- For B real rollouts and n_r states in rollout r, retain the objective `L = sum_r sum_i CE(r,i) / (B*n_r)`. Because ordinary DDP averages gradients, rank-local weighted sums must be multiplied by world size before reduction. Normalize by real global counts, including short final minibatches.
- Accumulate microbatches under `no_sync()` and synchronize once per optimizer step; include forward and backward inside that context. Coordinate microstep schedules across ranks. Handle empty/uneven ranks with zero-weight execution-only padding so every rank participates without dropping or double-counting real data. Clip gradients after synchronization.
- Shard mandatory final predictions and scheduled diagnostic predictions too. Gather compact `(sample_key, value, loss)` results, verify exact coverage/no duplicates, and restore canonical order. Avoid collectives in uneven inference loops; gather after rank-local inference finishes.
- Rank zero saves the unwrapped model, optimizer state, manifests, and advantages. All ranks restore the same committed state and update counter. Preserve current checkpoint completeness and critic-identity checks; test world-size changes on resume.
- Preserve strict critic-process exit before actor GPU allocation. Confirm collection/judge/retrieval resources are released. Do not overlap GPU diagnostics with the actor's four-GPU training in the initial implementation.
- Propagate worker failure, terminate peers, and reject partial output before starting the actor. Finalize advantage files atomically only after coverage and provenance validation.

DDP design follows [PyTorch's documented gradient averaging and accumulation behavior](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html). Verify the used APIs against pinned PyTorch 2.10 before implementation; newer optional features are not required.

## 5. Verification and rollout

Use one saved checkpoint and immutable cached collection in isolated output directories. Do not launch benchmark jobs against occupied production GPUs.

Benchmark variants in order: baseline; kernel repair; batching/checkpointing; reduced diagnostics; optimized DDP on two GPUs; optimized DDP on four GPUs. Measure cold end-to-end critic time and warmed phase timings with repeated trials. Compare total time to valid advantages, not just training tokens/s. Retain the fastest configuration; four GPUs can lose to two when synchronization/startup dominates a small dataset.

Extend existing `test_separate_value.py`, `test_critic_step_evalset.py`, and config tests, and add distributed critic tests covering:

- Rollout-weighted update parity across one/two/four ranks, unequal compaction counts, uneven final batches, empty ranks, and variable lengths.
- Stable optimizer update count, learning rates, resumed moments, critic identity, and exact advantage coverage.
- Batched and distributed evaluation parity, correct last-real-token pooling, and independence from completion tails.
- Diagnostic cadence and modes that leave final values, sample coverage, and required artifact validation unchanged.
- No-compaction, multiple-compaction, and malformed-summary trajectories; immutable exact IDs/masks and rejection of history rewrites/over-length intervals.
- Distributed failure cleanup, rank-zero-only writes, and no actor launch after a critic failure.
- Representative Qwen3.5 GPU smoke tests for padding, fused-kernel backward, and checkpointing toggles; small CPU models alone do not validate the production kernels.

Keep full diagnostics for controlled validation comparisons. Then use the fast preset for a short continuation, checking critic quality, advantage distributions, and actor outcomes against baseline variability. Reduced fitting epochs, smaller models, and early stopping are later algorithm experiments, not part of this semantics-preserving optimization.

## Expected benefit and remaining bottleneck

The initial target was at least 2x lower total critic-phase time; the sample measurements below now establish gains for fitting, but not for a full production iteration. At the original timings, a 3x improvement in the recorded fit portion alone would save about 18–23 minutes per update, roughly 9–12% of total train-update wall time. Actor updates remain the dominant cost.

After critic work, profile the actor's 157–162 minutes separately. Its logs also contain fast-path warnings, so kernel repair may help both components, but this needs independent validation with the actor's FSDP/sequence-parallel setup.

The host/device and checkpointing experiments are consistent with [PyTorch's performance tuning guidance](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html). Priorities and proposed settings above are inferred from this repository's code and recorded run, not benchmark guarantees.

## Implementation and validation results

Implemented the code-only work on 2026-10-22. The separate preset now uses training
microbatches of 8, evaluation microbatches of 32, an 8192 padded-token budget, and
critic activation checkpointing disabled. Three fitting epochs and the global
32-rollout minibatch remain unchanged. Final predictions run once; optional
eval-collection scoring and detailed training diagnostics run every five output
iterations. The README describes all overrides, timing metrics, and the benchmark.

At the initial benchmark, the environment had flash-attn 2.8.3.post1 but neither
FLA nor causal-conv1d. The Qwen implementation reported its linear-attention fast
path unavailable. The code-only optimization did not modify dependency manifests,
lockfiles, or installed packages. The user subsequently installed both missing
packages; their validation is recorded below.

GPU benchmark: Qwen3.5-0.8B, bf16, one A100 80 GiB, checkpoint iteration-00003,
reservoir sample of 16 whole rollouts from iteration-00004: 37 states, including
21 post-compaction states, maximum prefix length 611. Both variants fit three
epochs with three optimizer updates, starting from the same saved weights and
optimizer state. The baseline uses the previous batching/checkpointing/evaluation
settings on the new implementation; it is not an execution of the old source tree.

| Trial | Baseline fit | Optimized fit | Baseline final CE | Optimized final CE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 132.85 s | 13.43 s | 0.42866 | 0.42695 |
| 2 | 136.01 s | 13.09 s | 0.42945 | 0.42888 |

This sample shows 9.9–10.4x lower fit time, with peak allocated memory increasing
from 7.1 to 36.0 GiB. Model-load-plus-fit time was 143–146 s versus 21–22 s; loading
can benefit from filesystem/kernel warmup, so fit timing is the cleaner comparison.
Maximum same-weight batched/unbatched value differences were 0.00490 and 0.00630;
maximum post-update differences between variants were 0.01353 and 0.00957. These
are observed bf16 differences, not bitwise parity or a long-run quality guarantee.
Checkpoint saving, eval-collection scoring, longer prefixes, full-size updates,
and the actor were outside this GPU comparison. All benchmark GPU memory was
released, and saved model/rollout inputs were not modified.

Full measurements are in the local artifact
`artifacts/benchmarks/separate-value-performance-mixed.json`. Reproduce with:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/benchmark_separate_value.py \
  --config configs/train/compact_separate_mc_ppo.yaml \
  --checkpoint artifacts/train/qwen-bcplus-compact-separate-mc/checkpoints/iteration-00003 \
  --rollouts artifacts/train/qwen-bcplus-compact-separate-mc/rollouts/iteration-00004.jsonl \
  --output /tmp/new-separate-value-benchmark.json --rollout-limit 16 --repeats 2
```

Focused CPU regression result: 95 passed, 3 deselected. Coverage includes weighted
gradient parity, mixed-length pooling/order, mandatory prediction coverage,
diagnostic scheduling, checkpoint/resume, and immutable no-compaction,
multiple-compaction, and malformed-summary trajectories. The deselected tests
already disagree with unchanged committed presets: shared-value head LR expects
1e-4 versus YAML 1e-5; shared-value minibatch expects 4 versus YAML 128; the
no-compaction-loss preset differs from default in more than identity and masking.
Those unrelated configuration files and tests were left unchanged. Whitespace
validation (`git diff --check`) passed.

## Fast-kernel validation after user installation

Verified flash-linear-attention 0.5.2 and causal-conv1d 1.7.0 with the unchanged
PyTorch 2.10.0+cu128 and Transformers 5.5.3. Qwen reports all fast-path components
available: causal convolution, chunked/recurrent gated-delta-rule functions, and
fused gated normalization import successfully. Dense attention remains SDPA.

Repeated the same 16-rollout / 37-state sample and three-epoch schedule on GPU 0.
Both settings completed forward/backward execution and three optimizer updates.

| Kernel-enabled trial | Previous settings fit | Optimized settings fit | Previous settings final CE | Optimized settings final CE |
| --- | ---: | ---: | ---: | ---: |
| First pass, including startup overhead | 106.23 s | 9.24 s | 0.42621 | 0.42768 |
| Repeated pass, warmed kernels | 32.59 s | 2.73 s | 0.42665 | 0.42754 |

The warmed optimized fit is about 4.8–4.9x faster than the earlier optimized
PyTorch-fallback fit (13.09–13.43 s). Peak allocated memory fell from 36.0 GiB
to 15.2 GiB. Model loading plus optimized fitting took 19.04 s initially and
11.12 s on the repeat; these figures exclude sample loading, checkpoint saving,
eval-collection scoring, and actor training.

The first kernel-enabled pass showed substantial startup overhead, consistent
with first-use kernel compilation: pre-fit evaluation took 30.54 s initially
versus 1.48 s on repeat; the first training epoch took 54.01 s initially versus
8.52 s on repeat under the previous settings. Compilation time was not separately
profiled, so warmed timing must not be presented as cold-start latency.

Maximum same-weight batched/unbatched value differences were 0.01179 and 0.00627;
maximum post-update differences between the two settings were 0.00966 and 0.01055.
Final losses and gradient norms were finite. These bf16 sample checks do not
establish bitwise equality, full-iteration speed, or long-run policy quality.

Results: `artifacts/benchmarks/separate-value-performance-fast-kernels.json`.
All benchmark GPU memory was released. No saved checkpoint, rollout, dependency,
or training-source changes were needed for this validation.
