# Joint training with a separate value-head learning rate

Date: 2026-10-15

Status: implementation complete; validated with focused regressions and tiny-model GPU tests. Full agent training, held-out dataset preparation, and the LR sweep have not been run.

Implemented:
- Optional head LR in one joint optimizer on Transformers, Ray Transformers, and verl/FSDP, including the existing LambdaLR schedule.
- Equal per-rollout value weights preserved, with no new minibatch loss scaling.
- Frozen rollout-weighted diagnostics and metadata-derived state strata.
- A read-only held-out evaluator with query-disjointness and exact-token checks.
- Matched experiment preset and README usage documentation.

Validation:
- 266 focused regression tests passed; 4 environment-dependent tests skipped.
- Two config tests were deselected after reproducing their failures before implementation: `test_load_compaction_mc_value_training_preset` expects minibatch size 4 although the checked-in preset uses 128; `test_no_compaction_loss_preset_only_changes_ablation_identity_and_mask` assumes presets with other existing differences are identical.
- The tiny two-GPU FSDP worker test passed, including separate LRs, replicated gradient reductions, checkpoint save/load, and the actual Ulysses packed-prefix gather/unpadding path.
- Two tiny local update/save/reload cycles preserved cached IDs/masks and the existing optimizer-reset behavior. CLI help, Python compilation, and whitespace checks passed.
- No full training, policy evaluation, or pilot LR sweep was launched. Tests validate mechanics, not critic usefulness or an optimal LR.

Source: [Give the critic an independent training schedule](../../../artifacts/train/compact_value_mc_training_curves/value_mc_analysis_and_next_steps.md#1-give-the-critic-an-independent-training-schedule).

## Objective and scope

The first experiment keeps the joint objective and one optimizer, but gives the value head a separate parameter group and configurable learning rate:

```text
L_total = L_policy + loss_coefficient * L_value
backbone group: training.learning_rate
value-head group: training.value.head_learning_rate
```

The backbone receives gradients from both losses. The head receives gradients from the weighted value loss. Both groups step together on the existing schedule. Preserve frozen actor advantages throughout the update.

Test whether the zero-initialized linear head is learning too slowly at the actor's learning rate before separating critic training. Keep equal total value-loss weight per rollout as an explicit requirement. The weighting helper already implements this; preserve it and strengthen coverage rather than claiming it is newly introduced.

Implement this for Transformers, its Ray wrapper, and verl/FSDP. Keep value-disabled GRPO unchanged and retain the existing unsupported-backend guard for value training on `fsdp2_context_parallel`. Preserve the linear head, its dtype, loss coefficient, clipping behavior within each backend, batch order, update epochs, and checkpoint/optimizer reset behavior for the initial comparison.

Independent critic passes, detached feature caches, new validation partitions in the training loop, optimizer-persistence redesign, new loss scaling, and MLP heads are follow-up experiments. The source filename is retained so existing links to this plan remain valid.

## Implementation baseline (before this change)

- `CompactionValueConfig` in `src/self_summarization_agent/config.py` exposes `loss_coefficient` and initialization, but no head-specific LR.
- `TransformersPolicyTrainer` in `trainer.py` adds head parameters to the actor AdamW optimizer and backpropagates the joint loss.
- `CompactionValueTrainingWorker.configure_compaction_value()` in `verl_value_worker.py` adds the head to the actor's existing optimizer group after the scheduler has been created. The replicated head has its own gradient synchronization and clipping hooks.
- Both value-capable training paths use `rollout_value_weights()` from `value_model.py`, assigning equal total weight to each rollout.
- `_step_compaction_value_fsdp()` freezes values before updating. Its existing prefix-only probes and normalized frozen-model `value/eval_loss_before` and `value/eval_loss_after` metrics are useful foundations.
- Head weights and a manifest are saved under `compaction_value/`. The normal collection-iteration lifecycle does not gain persistent critic optimizer moments from this change; keep that lifecycle matched across experiment arms.

## Training boundary and ordering

All training acts on finalized immutable intervals. A successful summary remains in the old interval, and its successor still starts from unchanged system instructions, the original query, and wrapped compressed state. No runtime transition changes.

For each collection iteration:

1. Load actor and head using the existing checkpoint lifecycle.
2. Validate exact cached IDs, state anchors, binary rewards, and consistent rewards within each rollout.
3. Evaluate old interval-start values before optimization. Freeze `A = R - V_old` and existing behavior-policy log probabilities.
4. Compute rollout value weights once over the collection's real eligible states.
5. Run the existing joint update epochs and minibatches. Backpropagate `L_policy + c * L_value`, then step the single optimizer with separate backbone/head LRs.
6. Evaluate frozen-model diagnostics and save the actor/head through the existing checkpoint path.

Do not detach training features, add critic-only passes, or recompute actor advantages after head updates. A higher head LR changes subsequent head predictions and can change value gradients entering the backbone; it does not isolate the backbone from the critic.

## Task 1: Configure the head learning rate

Files: `src/self_summarization_agent/config.py`, `src/self_summarization_agent/verl_ray_trainer.py`, `tests/test_config.py`, `tests/test_verl_ray_trainer.py`.

- [x] Add optional `training.value.head_learning_rate: float | None = None`. Omission preserves the historical optimizer construction and inherits `training.learning_rate`; an explicit value selects the separate head group.
- [x] Reject booleans, non-numeric values, non-finite values, and zero/negative rates. Propagate the optional field through worker configuration without dropping or replacing it.
- [x] Keep `training.value.loss_coefficient` active in the joint loss for both groups. Do not reinterpret the coefficient as an LR multiplier or remove it from head gradients.
- [x] Add `configs/train/compact_value_mc_head_lr.yaml` based on the current value-MC config, with a distinct output directory and an explicit head LR. Preserve the baseline YAML.
- [x] Document the resolved backbone/head rates, how to reproduce the original configuration, and how to override the experimental rate. No new optimization-mode selector is needed for this milestone.

## Task 2: Give each rollout equal total value-loss weight

Files: `src/self_summarization_agent/value_model.py`, existing weighting call sites in `trainer.py` and `verl_ray_trainer.py`, `verl_value_worker.py`; tests in `tests/test_value_model.py`, `tests/test_trainer.py`, `tests/test_verl_value_worker.py`, and `tests/test_verl_ray_trainer.py`.

For R real rollouts, let rollout r have m_r eligible interval states. Define:

```text
w_(r,j) = 1 / (R * m_r)
target_r = (reward_r + 1) / 2

L_value(collection) =
    sum over rollouts r and intervals j:
        w_(r,j) * BCE(predicted_success_(r,j), target_r)
```

The implementation can retain two-class cross-entropy from logits, which expresses the same binary classification objective stably. Every interval in a rollout shares that rollout's judged reward but retains its own prefix/state prediction.

Example: rollout A has one interval, and rollout B has three:

| State | Value-loss weight |
|---|---:|
| A1 | 1/2 |
| B1 | 1/6 |
| B2 | 1/6 |
| B3 | 1/6 |

A and B each contribute total weight 1/2. B does not receive three times the total weight because it compacted more often. This equalizes objective weight, not gradient norms or final parameter displacement.

- [x] Reuse `rollout_value_weights()` as the authoritative helper. Compute weights over all real eligible collection states before batching or alignment padding, not separately per minibatch, microbatch, or rank.
- [x] Require rollout identity and consistent binary targets. Include existing critic-only eligible states even if they have no policy-loss contribution; do not silently filter away valid states when constructing optimizer batches.
- [x] Preserve weights when states are split across batches/devices. Padding duplicates have zero weight and are excluded from rollout/state counts. Verify equal per-rollout mass and total collection mass one.
- [x] Preserve current joint-training scale for the first LR experiment: each minibatch contributes its collection-weighted sum, and accumulation sums those contributions. Do not add the earlier plan's `N / n` multiplier, divide by a minibatch's observed weight mass, or change policy-loss reduction.
- [x] Preserve the current distributed compensation for gradient averaging, applied exactly once. Test value gradients against an explicit reference with uneven rollout lengths, several accumulation partitions, and zero-weight padding. Sequence-parallel replication must not multiply a rollout's weight.
- [x] For frozen evaluation, aggregate `sum(w_i * loss_i) / sum(w_i)` over the entire evaluated set. Recompute weights within a separate evaluation set and label its scope.
- [x] Keep historical loss-contribution metrics distinct from normalized frozen-model BCE. Report collection weight mass, minibatch count, and the coefficient so the current scale remains interpretable.

Equal rollout weighting and minibatch-scale normalization are separate choices. The existing minibatch contribution scale depends on how the collection is divided; hold batching fixed across the first LR comparison. Any later change to that scale must be a separate controlled experiment because it changes critic pressure on the backbone as well as on the head.

## Task 3: Local optimizer parameter groups

Files: `src/self_summarization_agent/trainer.py`, `tests/test_trainer.py`.

- [x] When head LR is explicit, build one AdamW optimizer with disjoint backbone and value-head groups. Every trainable parameter appears exactly once. Inherit existing optimizer options such as betas, epsilon, and weight decay; only the head LR differs.
- [x] Preserve the existing optimizer construction when the field is omitted. An explicit head LR equal to actor LR should match the old update numerically within the backend's normal tolerance.
- [x] Retain the joint forward/backward path, including value gradients into the backbone. Preserve the existing combined clipping behavior in Transformers; changing clipping would be a separate experiment.
- [x] Keep one optimizer step per existing minibatch, unchanged update epochs, and frozen advantages. Do not introduce a head-only step or detached-feature cache.
- [x] Exercise the same behavior through the Ray Transformers wrapper and verify initialization from saved head weights.

## Task 4: verl/FSDP optimizer and scheduler integration

Files: `src/self_summarization_agent/verl_value_worker.py`, `src/self_summarization_agent/verl_ray_trainer.py`, `tests/test_verl_value_worker.py`, `tests/test_verl_ray_trainer.py`.

- [x] Replace the explicit-LR path's extension of the actor parameter list with a distinct head parameter group. Copy the intended actor-group optimizer options and override only the head LR. Keep the head out of all actor groups.
- [x] Preserve replicated-head initialization, gradient averaging, finite checks, and current clipping hooks. Retain the joint loss and value-to-backbone gradient flow.
- [x] Handle the already-created scheduler explicitly. Inspect the pinned verl engine/scheduler construction and register both groups before scheduler construction where practical; otherwise provide a narrowly scoped, tested extension of the actual scheduler's group bookkeeping.
- [x] For the configured constant schedule, assert backbone/head rates remain at their configured values before and after scheduler steps. Preserve existing actor scheduler step counts. Reject unsupported scheduler integration clearly rather than silently freezing or overwriting a group's LR.
- [x] Test group membership, actual per-group LRs, and effective head updates, not only serialized config. Do not assume `add_param_group()` alone updates scheduler base rates or per-group functions.
- [x] Retain existing frozen prefix evaluation and joint actor/value RPCs; no separate critic-training RPC is needed.

## Task 5: Diagnostics and checkpoint compatibility

Files: `src/self_summarization_agent/trainer.py`, `verl_ray_trainer.py`, `train_step.py`, `value_model.py`, plotting/docs as needed; relevant trainer, value-model, train-step, and launcher tests.

- [x] Log actual backbone/head group LRs, their ratio, active loss coefficient, shared optimizer step count, value weight mass, state/rollout counts, and existing timing/token metrics.
- [x] Retain normalized frozen-model BCE before/after the joint update; add equivalent local-backend reporting if absent. Preserve legacy contribution metrics without presenting them as normalized validation loss.
- [x] Add rollout-weighted Brier score and explained variance, predicted-value summaries, and calibration diagnostics. Use consistent rollout weights for reward and residual variance; report EV as unavailable for zero target variance. Keep classification accuracy secondary.
- [x] Separate initial and post-compaction diagnostics using verified interval metadata. Preserve frozen pre-update advantage statistics.
- [ ] Prepare and use a fixed query-disjoint critic diagnostic set, excluded from training in every arm, when selecting LR. The evaluator and leakage checks are implemented; dataset preparation and selection are deferred until experiments are authorized. Reuse exact recorded prefixes and targets; do not reconstruct history. Label fixed-record results with their generating policy because targets can become stale as the policy changes. Keep final policy evaluation on its own untouched holdout.
- [x] Preserve head checkpoint format and the current iteration-level optimizer reset/persistence behavior. Confirm the configured head LR is applied on each fresh trainer construction and old head-only checkpoints still load.
- [x] Do not add an optimizer sidecar or promise uninterrupted Adam-state equivalence across collection iterations in this milestone. Document this limitation and keep it identical across comparison arms.

## Task 6: Acceptance tests and controlled pilot

- [x] **Default compatibility:** omitted head LR preserves existing joint training and value-disabled behavior. Explicit head LR equal to actor LR reproduces the old parameter update within tolerance under matched inputs.
- [x] **LR isolation:** from identical parameters/optimizer state and one fixed batch, changing only head LR changes the head update while preserving the backbone's first update. Use matched optimizer options/clipping. Later backbone updates may diverge because the learned head changes value gradients.
- [x] **Gradient flow:** prove value loss can update both backbone and head, using a nonzero head for the test; a zero-initialized linear head can initially give zero backbone gradient. Verify policy-only loss does not directly generate a head gradient.
- [x] **Schedule:** both groups step together, actual rates survive scheduler steps, and original actor advantages/reference log probabilities remain fixed through all epochs.
- [x] **Weighting:** unequal compaction counts, critic-only states, accumulation, distributed averaging, and padding preserve the reference value objective. Frozen evaluation is invariant to evaluation batching.
- [x] **Data contract:** cover no compaction, one/multiple compactions, and malformed summaries. Preserve all raw completions, exact IDs, masks, anchors, immutable intervals, shared rollout rewards, history-rewrite rejection, and over-length rejection.
- [x] **Checkpoint lifecycle:** complete two tiny-model update/save/reload cycles, verify requested LRs on restart, and retain staged checkpoint publication regression coverage. Full collection/training iterations remain unrun.
- [x] **Focused suites:** run config/value-model/trainer/verl-worker/verl-trainer/train-step/checkpoint/launcher tests plus runtime, TITO, rollout-collection, and cache contract regressions.
- [x] **GPU smoke:** use the real value worker with a tiny two-GPU FSDP model and explicit head LR. Check replicated-head synchronization against a single-process reference, DP/SP-style gradient reductions, actual Ulysses gather/unpadding, scheduler behavior, and checkpoint save/load. Full Qwen training remains unrun.

Pilot arms (not run), all with backbone LR `1e-6`:

1. Original joint baseline: inherited head LR `1e-6`.
2. Joint training with explicit head LR `1e-5`.
3. Joint training with explicit head LR `1e-4`.

Verify the explicit `1e-6` configuration as a compatibility check before the sweep. Hold loss coefficient, rollout weighting, minibatch rules, epochs, splits, seed set, collection budgets, and optimizer lifecycle matched. Use normalized frozen losses to avoid interpreting variable minibatch contribution counts as learning progress.

Select settings using critic diagnostics on held-out queries, including comparison with a training-derived constant predictor. Check policy accuracy, malformed outcomes, generated tokens, and wall time on a separate fixed evaluation holdout. Do not call lower training BCE alone a success or assume higher head LR is always better.

## Follow-up decision

If a faster head improves held-out prediction without harming the policy, retain it before adding complexity. If head predictions improve but policy behavior deteriorates, test whether critic/backbone interference is responsible using detached-head training. If prediction remains weak, investigate supervision, optimization scale, and extra critic passes before inferring insufficient model capacity.

A subsequent independent-schedule plan can add post-actor exact-prefix feature extraction, detached vectors, 2–5 critic-only passes, query-level validation, and persistent critic optimizer state. It must retain equal per-rollout value weights and freeze actor baselines before fitting the current collection. Evaluate loss-rescaling, extra passes, detachment, and MLP capacity as separate changes where practical.
