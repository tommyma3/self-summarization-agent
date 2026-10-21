# Critic loss normalization and gradient audit

This applies to `verl_fsdp` compaction Monte Carlo value training. The change is
observability only: optimizer objectives, coefficients, clipping, rollout IDs,
interval token evidence, and collection weights are unchanged.

## Which curve to read

- `value/eval_loss_before`: frozen critic cross-entropy on the current collection,
  evaluated before any optimizer step, in the same pass used to obtain advantages.
- `value/eval_loss_after`: the same states and weights, evaluated after all update
  epochs, with no optimizer steps between probe microbatches.
- Both equal `sum(w_i * CE_i) / sum(w_i)`, where a rollout with `n_r` intervals
  receives weight `1/R` split equally across its intervals. Only real states are
  gathered by sample ID; alignment duplicates are excluded. CE comes directly
  from logits, so saturated expected rewards cannot truncate the loss.
- These are training-batch evaluations, not held-out validation losses. The batch
  changes between updates; within an update the before/after comparison is paired.
- `value/loss` retains its historical meaning for compatibility, also exposed as
  `value/train_loss_contribution_mean`. It averages optimizer-minibatch
  contributions from successively updated models. Multiplying it by minibatches
  per epoch removes its mass scale but does not produce a frozen-model loss.
- `value/minibatches_per_epoch`, `value/train_weight_mass_mean`,
  `value/eval_weight_sum`, and `value/expected_coefficient_per_minibatch` expose the
  normalization. The observed mean training mass should equal `1/M`, including
  multiple epochs. `value/eval_seconds` measures the additional post-update pass.

The training-curve script now uses `value/eval_loss_after` and explicitly reports
it unavailable for old logs; it never substitutes or rescales the legacy curve.
The existing `value/classification_accuracy` remains the unweighted pre-update
state accuracy and must not be mistaken for post-update rollout-weighted accuracy.

## Gradient audit before coefficient tuning

Audited the installed verl FSDP engine and PPO loss path, pinned in `uv.lock` to
`d15a9d3f751c4d67a38221bb69561c64e0c9ddab`:

1. `FSDPEngine.forward_backward_batch` computes the masked token count `T_m`
   across DP ranks for the current optimizer minibatch, then splits microbatches.
   It calls `backward()` on each microbatch loss without further averaging.
2. With the configured `token-mean`, verl `agg_loss` supplies each rank's actor
   contribution as `D/T_m * sum(token PPO losses)`. After DP averaging and
   microbatch accumulation, the actor objective is a token mean for this minibatch.
3. Our critic contributes `D * sum(w_i * CE_i)` per rank. After averaging and
   accumulation it is `sum(i in m, w_i * CE_i)`. The `D` factor correctly cancels
   DP gradient averaging; adding another microbatch divisor would be incorrect.
4. Write `W_m = sum(i in m, w_i)`. The combined objective is therefore
   `actor_token_mean_m + c * W_m * critic_weighted_mean_m`. With `M` minibatches,
   `sum_m W_m = 1` each epoch, so the average critic coefficient relative to its
   within-minibatch mean is `c/M`. Individual masses differ with rollout lengths
   and minibatch composition. Changing four minibatches to five changes this
   average from `c/4` to `c/5`; it affects gradients as well as the displayed loss.
5. The replicated value head averages gradients over the full worker world.
   Sequence-parallel ranks replicate the state loss, so their replication cancels
   in that average. The backbone gather uses verl's default SP gradient scaling.
   CPU tests check DP/microbatch algebra; live multi-GPU/SP execution is not part
   of this verification, to avoid interfering with the active run.

These coefficients are normalization factors, **not measured gradient norm
ratios**. Actor and critic derivatives differ in direction and magnitude. A
zero-initialized head initially sends zero critic gradient to the shared backbone;
head and backbone clipping also act separately in this worker. Before tuning `c`,
choose whether collection-normalized critic contributions or minibatch-normalized
critic means are intended, then measure component gradients on a fixed checkpoint
and batch in a separate experiment. This patch does not silently change that
optimization choice.

The standalone Transformers trainer has a separate concern: it sums independently
normalized actor microbatch means while retaining collection-normalized critic
weights. Its coefficient also depends on microbatch partitioning. The new frozen
metrics and the FSDP audit above apply specifically to the distributed path;
coefficients should not be transferred between backends without checking that
additional difference.

## Same-policy eval-set diagnostics (`value/evalset/`)

A value network is only defined relative to the policy it was fit against. After
each separate-critic fit, `critic_step` therefore also scores the freshly fitted
critic on the states of that policy's own checkpoint-evaluation rollouts
(`iteration-<behavior>.eval.jsonl`, whose rows carry the matching
`policy_checkpoint_id`) and records the frozen diagnostics under `value/evalset/`:
classification accuracy, BCE, Brier score, explained variance, calibration, and
the initial/post-compaction strata, plus `value/evalset/policy_id` and
`value/evalset/data_policy_id` tags. These metrics measure same-policy
generalization to unseen evaluation queries; they are computed in memory only
(native caches attached to deep-copied rows, never persisted) and never affect
fitting or advantages.

Scope distinction: `value/eval_before`/`value/eval_after` are the training
collection; `value/evalset/` is the same policy's evaluation collection; and
`value/held_out/` (in `evaluate_value.py`) is the fixed query-disjoint probe
records. Note that eval-set states come from greedy decoding of the same policy
weights, while training collections use the configured sampling temperature.

## Interval contract

Evaluation reads only the exact cached state prefix at the interval-start anchor.
It does not alter trajectory records, sampled IDs, masks, compaction boundaries,
or the immutable interval prefix. The old interval still finalizes before any
summary successor is created. No new tokenization or history reconstruction is
introduced. Runtime regressions cover no compaction, repeated compaction, and
malformed summaries alongside the critic-specific CPU tests.
