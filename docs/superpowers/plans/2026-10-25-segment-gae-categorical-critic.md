# Segment GAE with a categorical critic and pre-fit advantages

Date: 2026-10-25 (Asia/Shanghai)
Status: implemented and validated. A bounded production smoke was run; no full research training experiment was launched.

## Implementation and validation record

- Added the separate-critic GAE preset, configurable categorical head, two-hot CE fitting, pre-fit frozen advantages/targets, and versioned provenance/checkpoint validation. Historical post-fit MC, pre-fit MC, and two-bin GAE are supported.
- Final combined regression: **325 passed, 4 skipped**, covering value/config, cache, collection, runtime/TITO, launchers, checkpoint publication, and actor dispatch. Compilation and `git diff --check` passed.
- Added real tiny-model two-iteration tests for 51-bin GAE, two-bin GAE, and pre-fit MC. Fault injection covers target-file save, critic fit/save, and actor update/save. Retries retain byte-identical frozen targets and restart fitting from the committed input critic. Completed critic-phase reuse is deliberately not implemented; the existing restart policy is preserved.
- Exact-token tests cover no compaction, multiple successful compactions, a malformed first summary, and a malformed summary after successful compaction. Original collection records remain unchanged. Terminal/forced/malformed/overflow boundaries use zero bootstrap; missing links are rejected.
- Production smoke: one Qwen3.5-0.8B critic optimizer step and one Qwen3.5-9B PPO optimizer step on four A100s, using three complete saved base-policy rollouts (four intervals). The batch includes a successful compaction and a malformed completion. Actor loss was -0.7384662, measured behavior-policy KL 0.00064546, and clip fraction 0.00140647.
- Frozen actor advantages were [-1.0, 0.95, 1.0, 1.0]. The updated critic predicted approximately 0.0024–0.0026 after reload, confirming that fitted predictions did not replace the actor's pre-fit signal. Actor/critic pairing, critic optimizer reload, and repository export-loadability checks passed; original input JSONL remained byte-identical. Ray shut down and released the GPUs.
- Production smoke artifacts: `/tmp/segment-gae-production-jeo_89rp/` (config, immutable selected rollouts, smoke.log, metrics.jsonl, validation.json, and iteration-00001 checkpoint). Final test log: `/tmp/segment-gae-final-regression.log`.
- The smoke used existing exact on-policy cached records from the base actor and changed no source rollouts. It did not perform new live retrieval/judging, launch vLLM inference from the exported actor, or measure policy-quality improvements. Fresh stochastic validation and the research ablations below remain experiments to run separately.

## Objective and agreed algorithm

Extend the separate-critic PPO path with segment-level GAE and a configurable categorical value head. Compute actor advantages and critic targets from the carried critic **before** fitting the current batch. Freeze both for the entire iteration. Use 51 evenly spaced support values in [-1, 1], two-hot target encoding, gamma = 1, and lambda = 0.95 for the first experimental preset. These are starting settings, not established optima.

Keep existing MC presets and their historical behavior available. Add explicit pre-fit MC and two-bin GAE controls. The new method initially supports separate critics only; shared actor/critic GAE is outside this implementation.

For a completed rollout with M existing intervals, state s_j is the exact prefix ending at first_generation_prompt_end. Evaluate v_j = V_old(s_j) in eval/no-grad mode. Define transition rewards r_j = 0 except r_(M-1) = R. Calculate backward:

```text
next_value = 0
next_advantage = 0
for j in reversed(range(M)):
    delta[j] = r[j] + next_value - old_value[j]
    advantage[j] = delta[j] + lambda * next_advantage
    target[j] = old_value[j] + advantage[j]
    next_value = old_value[j]
    next_advantage = advantage[j]
```

Equivalently, target[M-1] = R and target[j] = (1-lambda) * old_value[j+1] + lambda * target[j+1]. Use this equivalence as an independent numerical test. Lambda = 1 recovers pre-fit MC exactly within floating-point tolerance. A one-interval rollout has the MC target for any lambda.

For support z_k and probabilities p_k = softmax(logits)_k, V = sum_k p_k*z_k. Encode a target y between adjacent supports z_k and z_(k+1) with weights (z_(k+1)-y)/(z_(k+1)-z_k) and (y-z_k)/(z_(k+1)-z_k). At support endpoints or an exact support value, use a single mass. Fit soft cross-entropy to this fixed target distribution. Two-bin encoding is soft binary supervision with p_target(success) = (y+1)/2.

Use raw frozen advantages for PPO; do not whiten them or normalize within query groups. Keep existing actor aggregation, clipping, behavior log probabilities, and critic rollout weighting. No reward KL, new step costs, or changes to the judged reward.

## Interval and token contract

- A segment remains one complete trajectory_record. Research and its raw summary completion belong to the old interval. Successful compaction finalizes it before initializing the successor.
- Successful compaction is a nonterminal transition for GAE. Bootstrap from the successor's exact wrapped-summary prefix. Actual terminal answers, forced answers, and malformed termination use zero bootstrap according to existing rollout semantics. A failed summary has no successor.
- Preserve the same judged rollout reward R on every stored interval. The transient transition reward array counts it only at the end; never overwrite sample.reward or trajectory reward fields with GAE targets.
- Derive ordering and termination from authoritative rollout/trajectory metadata, not lexicographic turn IDs, minibatch order, or the last sample left after filtering. Validate original record order/segment indices and their mapping to extracted sample keys.
- Reject missing, duplicate, inconsistent, or incomplete successor chains before an update. Never connect across an excluded interval, fabricate a terminal transition, or bootstrap across rollout IDs. Preserve existing ProviderHistoryRewriteError exclusions and backend-integrity behavior.
- Preserve exact collected IDs, masks, raw completions, anchors, native tool-call IDs, and all eligible actor/critic states. No re-rendering, retokenization, prefix truncation, new runtime controls, or changes to the compaction format.
- Extra compactions do not create additional independent reward-normalization weight. Critic loss remains a mean within each rollout followed by a mean over rollouts.

## Repository findings

- critic_step.run currently calls critic.fit, saves the fitted critic, and exports R - V_fitted.
- separate_value.load_advantages requires version 1, A = R - V, and identity matching the newly fitted critic. That contract must not be silently reused for pre-fit GAE.
- value_model and SeparateValueTrainer._prepare assume two logits and hard binary reward labels. Predictions, training targets, and outcome diagnostics need separate interfaces.
- separate_checkpoint_complete assumes class_order = [-1, 1]. Head and paired-checkpoint validation must recognize a versioned support specification.
- samples_from_rollout_rows flattens extracted samples and can skip records. GAE needs a validated ordered rollout view before that flattening can lose chain information.
- The actor already accepts fixed keyed interval advantages. Reuse that path rather than introducing a different PPO objective.

## 1. Configuration and compatibility

Files: config.py, value configuration tests, new configs/train/compact_separate_gae_ppo.yaml, README.md.

- [x] Add training.advantage_estimator = compaction_gae_value to estimator validation and dispatch. Preserve compaction_mc_value and group_relative semantics.
- [x] Add proposed value settings num_bins (default 2), baseline_timing (historical separate-MC default post_fit), and gae_lambda (default 0.95). The new preset explicitly selects pre_fit and 51 bins. Fix gamma = 1 and support = [-1, 1] for this first implementation.
- [x] Require a separate enabled critic and pre_fit timing for GAE. Validate finite lambda in [0, 1] and integer num_bins >= 2; reject unsupported shared-mode bin/estimator combinations before allocating models.
- [x] Allow pre-fit MC for the control experiment. Keep diagnostic eval_before_fit separate from the mandatory baseline prediction pass: disabling optional diagnostics cannot disable GAE predictions.
- [x] Preserve existing presets and output directories. Match collection settings, actor updates, critic fitting settings, and seeds when producing ablation configs/overrides.
- [x] Define legacy config fingerprint normalization explicitly. Adding default fields must not silently break version-1 artifact verification or permit changed settings to pass as historical defaults.

## 2. Categorical head and fixed-target fitting

Files: value_model.py, separate_value_trainer.py, tests/test_value_model.py, tests/test_separate_value.py, tests/test_separate_value_performance.py.

- [x] Generalize the separate critic's projection to K logits, add expectation-over-support and two-hot encoding helpers, and retain compatible two-class helpers for existing shared paths.
- [x] Preserve zero initialization. Symmetric uniform logits initially yield V = 0. Do not interpret their bin entropy as calibrated uncertainty.
- [x] Separate prefix preparation, scalar prediction, target encoding, fitting, and evaluation against observed outcomes. Accept detached, keyed scalar targets in fit rather than replacing sample.reward.
- [x] Compute softmax, target interpolation, expectations, and soft CE in float32 where needed. Reject nonfinite/materially out-of-range targets; permit only documented machine-precision endpoint correction.
- [x] Preserve exact-prefix evaluation, padding/position handling, length bucketing, microbatch accumulation, rollout-normalized loss, gradient clipping, and persistent critic optimizer state.
- [x] Verify prediction/target key alignment after sorting and batching. Freeze targets across all fitting epochs and ensure they have no gradient path to old predictions.

## 3. Ordered rollout view and GAE builder

Files: new segment_returns.py, train_step.py extraction helpers, separate_value.py; inspect trajectory.py metadata without changing collection records.

- [x] Build a training-only ordered rollout view containing sample key, segment index, termination kind, successor key, and terminal rollout reward. Validate it against original records and exact extracted samples before computing targets.
- [x] Implement pure backward GAE/return construction from keyed pre-fit values. Return old values, transition rewards, terminal flags, successors, advantages, and scalar targets.
- [x] Include critic-only states when valid under existing extraction; they participate in the chain but supply no actor tokens. Reject a missing required chain state rather than silently shortening the rollout.
- [x] Validate targets in [-1, 1], advantages in [-2, 2], terminal zero bootstrap, and A = target - old_value. Preserve R as separate outcome metadata.
- [x] Compute this once for the complete collection, before actor/critic shuffling. Never calculate independent GAE traces per minibatch.

## 4. Pre-fit orchestration and immutable artifacts

Files: critic_step.py, separate_value.py, train_step.py, tests/test_critic_step_evalset.py, tests/test_separate_value.py.

- [x] Load the carried critic from the committed input checkpoint; on the first iteration use the configured initialization and zero head. Use the same configured lambda from iteration one, with no hidden warm-up schedule.
- [x] Evaluate all state prefixes before any optimizer step, build returns/advantages, then atomically save a version-2 frozen batch-target artifact before fitting.
- [x] Record behavior checkpoint identity, source critic identity (or initialization fingerprint), input/token-cache fingerprint, ordered chain/termination fingerprint, estimator settings, support, scalar old values/targets/advantages, and artifact digest.
- [x] Fit on the artifact's targets, save the updated critic, and publish a completion manifest binding the frozen artifact to the output critic identity. Distinguish the critic that generated targets from the critic trained on them.
- [x] Load frozen advantages for actor training; neither post-fit evaluation nor repeated PPO epochs may regenerate them. Keep critic GPU release before actor allocation.
- [x] Preserve version-1 loading with its original MC checks. Version 2 verifies chain provenance, recurrence, target/value/advantage relationships, exact key coverage, and source/output identities as appropriate. Do not weaken validation globally to admit GAE.
- [x] Keep retries anchored to the last committed input pair. An interrupted fit restarts from that source, using a verified frozen artifact if available; never warm-start from a partially fitted critic. A completed critic phase may be reused only when its artifact and completion manifest validate. A target file alone does not mean the phase is complete.
- [x] Handle empty batches and critic-only batches explicitly, preserving existing rules for actor optimizer steps and publication.

## 5. Checkpoint persistence and actor integration

Files: value_model.py sidecar loading/saving, separate_value.py checkpoint helpers, train_step.py, checkpoints.py, evaluate_value.py; audit trainer.py and verl_ray_trainer.py dispatch.

- [x] Version categorical head metadata and separate/pair manifests. Persist support values, bin count, encoding, estimator/timing settings, and optimizer compatibility information.
- [x] Load historical two-class checkpoints as their original format. Reject changing the bin count/support or incompatible method settings during ordinary resume; do not silently reset a head or its optimizer moments. Cross-method initialization/migration is a separate explicit operation, not part of this milestone.
- [x] Exercise actor-only supplied-advantage updates for local Transformers and supported Ray/verl paths. Broadcast each interval's frozen advantage to its existing assistant-token mask, including summary reasoning/body. Preserve partial minibatches and critic-only state handling.
- [x] Keep behavior token log probabilities fixed. No new reference-policy KL term or group-relative advantage preparation.
- [x] Ensure final pair publication requires the completed critic, valid frozen artifact, and actor checkpoint; reject mismatched or incomplete pairs on resume.

## 6. Diagnostics

Files: value_diagnostics.py, separate_value_trainer.py, critic_step.py, evaluate_value.py, relevant diagnostics tests.

- [x] Log target-fitting CE under a distinct name from outcome evaluation. Raw CE across different encodings/bin counts is not a fair comparison metric.
- [x] Report pre-fit and post-fit value MSE/explained variance against observed R, plus Brier/log loss using the implied success probability (V+1)/2. Apply explicit numerical handling at probability endpoints. These scalar metrics are comparable across head types.
- [x] Log target/advantage quantiles and near-zero fractions by initial/post-compaction state, outcome, and segment depth; log target deviation from R and bin occupancy as descriptive checks.
- [x] Preserve equal rollout weighting. Keep diagnostics against greedy evaluation labeled as a different distribution; use query-disjoint fresh stochastic validation to assess critic generalization under training decoding settings.
- [x] Avoid treating bin entropy as epistemic uncertainty or as a validated distribution of final returns: these are encoded scalar GAE targets.

## 7. Validation and completion criteria

- [x] Pure numerical tests: one interval; multiple compactions; lambda = 0 and 1; hand-calculated intermediate lambda; failed/malformed summary; final and forced answer; distinct rollouts; shuffled input; missing/duplicate states; excluded history; inconsistent rewards and terminal metadata.
- [x] Two-hot tests: endpoints, exact/intermediate supports, mass conservation, expected value equals target, two-bin equivalence, finite gradients, out-of-range rejection, and categorical save/reload compatibility.
- [x] Ordering regression: use a critic whose predictions change markedly during fitting and assert all actor advantages and targets still equal the pre-fit calculations across every epoch and reload. Verify fit never reads post-fit values to build actor advantages.
- [x] Fault-injection tests: interruption before/after frozen-artifact save, during fitting, after critic save, and during actor update/publication. Verify reproducible source selection and rejection of altered provenance.
- [x] Contract integration: both no-compaction and multiple-compaction rollouts, including malformed summary output; assert original queries/system text, raw completions, exact IDs, masks, and records remain unchanged. Confirm successful compaction retains the full old interval before its successor and malformed summaries install no state.
- [x] Run focused value/config/extraction/checkpoint and separate-actor suites, then relevant runtime/TITO/trajectory regressions. Compare any failures against unchanged HEAD rather than changing unrelated presets.
- [x] Run a tiny-model two-iteration integration that trains, saves, reloads, and proves iteration 2 predictions come from iteration 1's committed critic. Test at least one compacted and one uncompacted rollout.
- [x] Before a research run, execute a bounded production-backend smoke with exact on-policy cached records; verify finite losses/gradients, target provenance, actor token coverage, and export/loadability. Do not infer quality gains from smoke success.

## Experiment sequence after implementation

Use a fresh run for changed head formats. Hold actor/critic update budgets, initial backbone, task schedule, collection settings, and seeds matched:

1. Historical two-class post-fit MC as the existing reference.
2. Two-class pre-fit MC to isolate baseline timing.
3. Two-class pre-fit GAE, lambda = 0.95, with soft labels to isolate bootstrapping.
4. 51-bin pre-fit GAE with identical lambda and two-hot labels to isolate head parameterization.

First verify lambda = 1 recovers MC targets, then compare 0.95 and 0.9 if the initial experiment supports further testing. Select on stochastic validation; report terminal task accuracy, malformed rate, value calibration, actor signal by segment, and token/compute cost. Lower training CE alone is not evidence of improvement. Full research runs are separate from implementing and testing this plan.
