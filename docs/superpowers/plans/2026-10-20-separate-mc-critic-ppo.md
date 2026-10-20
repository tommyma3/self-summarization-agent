# Separate MC critic with critic-first PPO updates

Date: 2026-10-20
Status: implemented and validated. No full research training experiment was launched.

## Implementation and validation record

- Added the separate critic, process-isolated MC fitting, post-fit fixed advantages, actor-only PPO for local Transformers/Ray Transformers/verl FSDP, paired checkpoints, evaluation support, and the new preset.
- Final broad regression run: **276 passed, 4 skipped, 3 failed**. All three failures were reproduced against unchanged HEAD in `/tmp/separate-mc-baseline-check`: the existing compact-value preset minibatch assertion, the no-compaction ablation preset equality assertion, and the head-LR preset assertion (`1e-4` expected versus the checked-in `1e-5`). Existing presets were not modified to hide these failures.
- Final dedicated separate-critic suite: **19 passed**, including real tiny actor/critic fitting and reload over two iterations, post-fit advantage checks, accumulation weighting, interrupted update/save retries, paired publication, tokenizer/provenance rejection, and runtime interval immutability.
- Real-model smoke: Qwen3.5-0.8B completed two fitting epochs/optimizer steps; Qwen3.5-9B completed four actor optimizer steps over two PPO epochs on four A100s with DP=2/SP=2, including masked padding in the final minibatch. Actor/critic export and the repository's vLLM-loadability checks passed; Ray shut down successfully.
- The real-model smoke used three short synthetic cached intervals representing initial and post-compaction states. Its fixed illustrative token log probabilities were for exercising the update mechanics; it was not an on-policy research experiment, a long-context benchmark, or evidence of improved task accuracy. vLLM inference on the exported checkpoint was not launched.
- Runtime collection/compaction, judging labels, and cache invariants are covered by fixture-based tests. The two-cycle trainer integration supplies validated cached examples and runs real tiny networks. Live retrieval/judging and a full collection-to-training experiment remain unrun.
- Real-model smoke log: `/tmp/separate-production-smoke2.log`; paired export: `/tmp/separate-production-smoke-_rwfj1gr/iteration-00001`. Final regression log: `/tmp/separate-final-regression.log`; final separate suite: `/tmp/separate-final-integration.log`.
- Python compilation and `git diff --check` passed. The existing actor optimizer reset behavior is preserved; critic optimizer state persists.

## Agreed method

Keep Qwen3.5-9B as the actor. Initialize an independent trainable critic from `/124090467/Qwen/Qwen3.5-0.8B`, with the existing zero-initialized two-class linear head adapted to hidden size 1024. Fit the critic backbone and head for multiple epochs using MC cross-entropy over all eligible intervals of the current trajectories. Evaluate the fitted critic on those same trajectories, freeze `A = R - V_fitted`, and update the actor for multiple PPO epochs.

No cross-fitting, fitting holdout, bootstrap targets, GAE, or replay buffer. Initialize the critic once and continue from its saved weights across iterations. Optional diagnostic evaluation does not remove fitting trajectories or determine advantages.

For N rollouts with K_i eligible intervals each:

```text
p_phi(s_ij) = softmax(value_head(critic_hidden(s_ij)))[success]
V_phi(s_ij) = 2 * p_phi(s_ij) - 1
y_i = int(R_i == +1)
L_value = (1/N) * sum_i [(1/K_i) * sum_j CE(logits_phi(s_ij), y_i)]
A_ij = R_i - V_phi_fitted(s_ij)
ratio_ijt = exp(log_pi_theta(token_t | exact_prefix_t) - cached_old_logprob_ijt)
L_actor = -token_mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))
```

Use raw advantages without group-relative normalization or whitening. Retain eligible intervals from identical-reward groups. Separate optimizers replace the joint actor/value objective in this mode.

## Boundary and data contract

The segment remains the existing complete `trajectory_record`. Successful compaction finalizes the old interval, including the raw summary completion, before starting its successor from unchanged system instructions, unchanged original query, and wrapped summary. Final answers, forced answers, and malformed outputs follow existing termination rules.

Do not change collection, runtime transitions, prompts, renderers, tool-call IDs, rewards, or interval boundaries. Critic inputs are validated exact-ID prefixes ending at `first_generation_prompt_end`, never re-rendered messages or retokenized strings. Prefix-only critic evaluation does not mutate or truncate stored intervals or actor inputs. No completion tail or reward label enters the critic input.

Preserve raw assistant-token eligibility, including summary reasoning/body and malformed completions where existing extraction permits them. Preserve history-rewrite rejection and over-length rejection. Failed summaries create no successor interval. Extra compactions do not create extra independent reward-normalization weight.

## Repository findings

- `value_model.py` already provides the head, binary reward conversion, expected value, and rollout weighting.
- `trainer.py` and `verl_value_worker.py` currently attach the head to the actor, freeze values before fitting, and optimize joint losses. Separate mode needs an independent path.
- `verl_ray_trainer.py` provides exact-prefix probes and a padded value actor-batch builder. Its ordinary GRPO path computes group-relative advantages and can trim real rows for alignment; do not reuse that preparation path unchanged.
- `train_step.py` constructs the actor before updating. Insert critic training before actor construction to avoid GPU overlap.
- `checkpoints.py`, `train_step.py`, and `iteration_launcher.py` assume value-enabled checkpoints contain a shared-head sidecar. Extend checks to recognize separate critics.
- `evaluate_value.py` loads the actor for shared-head evaluation. Separate mode should load only the critic.
- Local 0.8B and 9B `tokenizer.json` hashes match. Validate vocabulary/ID compatibility, including special and added tokens, at runtime rather than relying on model names.

## 1. Configuration and compatibility

Files: `config.py`, new `configs/train/compact_separate_mc_ppo.yaml`, configuration tests, README.

- [x] Keep `advantage_estimator: compaction_mc_value`; add `training.value.architecture: shared | separate`, defaulting to `shared`. Architecture selects the new schedule while the MC estimator remains unchanged.
- [x] Add separate-critic initialization path, backbone LR, fitting epochs, rollout minibatch size, prefix microbatch size, max gradient norm, and GPU selection. Reuse/configure the head LR. Validate positive finite rates, positive integer counts, the supported anchor, and a required model path.
- [x] Define critic minibatches in rollout units: shuffle rollout IDs, include all eligible intervals of each selected rollout, average losses within each rollout, then average over minibatch rollouts. This preserves equal rollout weight and supports prefix microbatching.
- [x] Keep actor LR, `update_epochs`, minibatch size, clip range, and token limits in existing fields. Document `loss_coefficient` as shared-mode-only; separate critic gradients are not scaled by a joint-loss coefficient. Reject misleading explicit overrides where detectable.
- [x] Add a distinct preset/output directory; preserve old GRPO/shared-value presets. Suggested pilot settings: critic 3 epochs, backbone LR 1e-5, head LR 1e-4, rollout minibatch 32, prefix microbatch 1; actor 2 epochs, LR 1e-6, clip 0.2. Keep collection budgets and token limits matched. These are starting settings, not measured optima.
- [x] Support local Transformers and production `verl_ray/verl_fsdp` actors. Extend the Ray Transformers wrapper through the same explicit-advantage API if it remains advertised. Reject unsupported backend combinations before allocation; do not silently fall back.

## 2. Independent critic implementation

Files: reuse `value_model.py`; add `separate_value_trainer.py` and a critic-phase entry point.

- [x] Load the 0.8B text backbone with its own value head. Use the supported Qwen3.5 text forward path without full vocabulary logits or unused visual computation. Both backbone and head are trainable.
- [x] Build examples from `sample.input_ids[:sample.state_prefix_length]`, keyed by `(rollout_id, turn_id)`. Validate unique keys, consistent rollout rewards, exact anchors, cache provenance, and tokenizer mapping.
- [x] Use correct attention masks/positions and gather the last real prefix token in padded batches. Changing a later interval tail must not change its critic prediction.
- [x] Reject missing exact data, incompatible IDs, and oversized inputs. Retain complete-interval length checks before either phase. Never reconstruct or truncate inputs to fit.
- [x] Implement fit/evaluate/save/load with independent AdamW groups for backbone and head. Persist critic optimizer state and update counter across collection iterations.
- [x] Accumulate the rollout-normalized minibatch objective correctly across prefix microbatches. Do not independently normalize microbatches or divide CE by prefix length. Handle short final minibatches and zero-weight padding.
- [x] Use every eligible current trajectory in every configured critic epoch, with reproducible rollout shuffling. No class rebalancing, failure oversampling, or fitting folds.
- [x] Preserve zero head initialization. Test backbone learning after the head starts learning; the first step can legitimately have zero backbone gradient.

## 3. Critic-first orchestration

Files: `train_step.py`, critic-phase module, reusable subprocess helpers if needed.

- [x] Run the critic phase in a dedicated subprocess on one configured training GPU before constructing the 9B actor/Ray workers. Exit it before actor initialization to release model, optimizer, and CUDA allocations. Validate actual Qwen prefix memory in the smoke test.
- [x] First iteration loads the configured 0.8B initialization; later iterations require the input checkpoint's saved critic. Fit exactly the configured epoch count on the current collection.
- [x] After all fitting epochs, evaluate once in eval/no-grad mode and compute `A = R - V_fitted`. Before-fit values may be diagnostics only, never actor baselines.
- [x] Write the staged critic and an atomic advantage artifact outside trajectory records. Record sample keys, fitted values, advantages, behavior checkpoint ID, critic identity, input/cache fingerprint, fitting-config fingerprint, and completion status.
- [x] Require finite values, exact key coverage, and matching provenance before the actor update. Reject missing/duplicate/extra rows. Keep advantages and old log probabilities fixed throughout every actor epoch.
- [x] Restart interrupted updates from the last committed actor/critic pair by default. Partial artifacts are diagnostic and must not silently warm-start a retry. Mid-epoch resumability is outside this milestone.
- [x] Handle empty collections and critic-only states explicitly. Critic-only states may fit/evaluate the critic but contribute no actor tokens or optimizer steps.

## 4. Actor-only PPO with supplied advantages

Files: `trainer.py`, `verl_ray_trainer.py`, Ray wrapper and batch helpers.

- [x] Add an explicit update API accepting fixed keyed interval advantages. Reuse clipping calculations without invoking group-relative preparation or filtering same-reward groups.
- [x] Construct ordinary actor workers in separate mode, with no shared value head or CE term. Keep external configuration validation intact; avoid pretending the method is GRPO simply to disable shared-worker construction.
- [x] Refactor/reuse padded interval batching from `build_verl_value_actor_dataproto` into an actor-only path. Preserve exact IDs, sparse masks, old log probabilities, and every real interval. Alignment padding has zero loss weight; never trim real intervals.
- [x] Broadcast each advantage to its interval's assistant tokens. All system/query/summary-state/tool-result/control tokens remain conditioning-only.
- [x] Preserve production token-mean loss and make accumulation/distributed reduction match that objective. Count actual steps and handle short batches correctly.
- [x] Use unchanged behavior-policy denominators for all PPO epochs. Preserve native-versus-rescored logprob provenance and the existing likelihood convention; no implicit temperature correction is introduced here.
- [x] Make `target_kl` operational in the new path: check approximate behavior-policy KL after each complete optimizer minibatch and stop subsequent updates when exceeded. Audit pinned verl support; if necessary drive minibatches externally while preserving the fixed old-logprob tensors. This is a stopping threshold, not a reference-model penalty.
- [x] Verify disjoint actor/critic gradients. Do not introduce an entropy bonus or reference-model KL objective.

## 5. Paired checkpoints and restart

Files: `checkpoints.py`, `train_step.py`, `iteration_launcher.py`, lifecycle tests.

- [x] Keep actor HF files at the checkpoint root for vLLM. Store the critic backbone, head/config, compatibility metadata, optimizer state, and update counter under `critic/`, outside the root weight glob.
- [x] Add a versioned training manifest with architecture, logical iteration, actor/critic identities, critic initialization provenance, head class order, and optimizer lifecycle.
- [x] Separate policy-loadability from training-resume completeness. Separate-mode resume requires a complete matching critic bundle; shared mode retains sidecar checks.
- [x] Mark complete and promote only after both saves succeed. Advance `latest` only after paired validation. Actor export/promotion must preserve the staged critic directory.
- [x] Reject missing, partial, incompatible, or wrong-iteration critics. Never silently reinitialize on resume or load a 9B shared-head sidecar into the 0.8B critic.
- [x] Use a fresh output directory. Explicitly initializing a new experiment from an existing actor checkpoint may create a fresh 0.8B critic, with provenance; it is not a resumed separate-critic run.
- [x] Preserve the current actor optimizer lifecycle, which resets at iteration construction, and document it. Persist critic optimizer state independently. Do not promise uninterrupted actor optimizer equivalence across iterations.

## 6. Diagnostics and documentation

Files: `value_diagnostics.py`, `evaluate_value.py`, progress/metrics writers, README.

- [x] Report per-epoch rollout-weighted critic CE, frozen before/after CE, calibration/Brier score, explained variance, value distribution, gradients, LRs, actual steps, memory, and timing. Clearly label same-collection measurements.
- [x] Report frozen post-fit advantage statistics and near-zero fraction, actor loss/clip fraction/behavior KL, actual epochs and steps, and KL-stop reason. Keep actor and critic losses separate.
- [x] Preserve initial/post-compaction strata and correct rollout weighting. Handle zero reward variance without invalid explained-variance values.
- [x] Add progress phases for critic load/fit/evaluate/save, advantage readiness, actor load/update, and paired publication, with counters and timeout visibility.
- [x] Extend optional read-only evaluation to load only the separate critic. Preserve query-disjoint diagnostic checks; this evaluation never withholds fitting trajectories or generates actor advantages.

## 7. Validation and acceptance

Run focused checks as components land, then integrated validation. A long production training experiment is not part of implementing the feature.

- [x] Config compatibility and early rejection of unsupported combinations.
- [x] Critic head/class order, hard-label CE, causal anchor, tokenizer mismatch rejection, multiple epochs, gradients, and two-cycle weight/optimizer reload.
- [x] Equal rollout weighting with unequal compaction counts, prefix lengths, microbatch splits, partial batches, and padding; compare losses/gradients with a direct reference.
- [x] Schedule test distinguishing before-fit and after-fit values, proving advantages use the latter and all eligible trajectories participate in every fitting epoch.
- [x] PPO fixed advantages/logprobs across multiple steps; identical-reward groups; zero advantages; sparse masks; padding; accumulation/reduction; KL stopping; disjoint gradients.
- [x] Contract regression: no compaction, one/multiple compactions, malformed/empty/over-limit summaries, forced answers, malformed thinking, history-rewrite exclusion, and over-length rejection. Compare raw records, exact IDs/masks, stable prompts, and native tool IDs before/after training.
- [x] Failure/retry tests for interrupted critic, actor, and save phases; missing optimizer files; partial checkpoints; mismatched manifests; atomic latest publication; no double updates on retry.
- [x] Tiny local two-iteration fit/update/save/reload integration with real actor/critic networks and both reward signs, plus fixture-based runtime/cache tests covering compaction. Live research collection/judging is outside the completed validation scope.
- [x] Distributed GPU validation uses the actual 9B production actor path with uneven real row counts, masked alignment padding, and multiple PPO epochs. A small loss reference separately verifies token-mean reduction semantics; a separate tiny distributed model run was not needed.
- [x] Bounded real-model smoke with 0.8B critic and 9B actor on short synthetic exact-ID cached intervals representing initial and post-compaction states. Verify model loading, memory release, multiple critic/actor steps, finite gradients, preserved token evidence, and vLLM-loadable paired export. Report unavailable hardware checks as unverified.

Extend/run relevant existing suites: `test_value_model.py`, `test_value_config.py`, `test_trainer.py`, `test_verl_ray_trainer.py`, `test_train_step.py`, `test_checkpoints.py`, `test_iteration_launcher.py`, `test_evaluate_value.py`, `test_tito.py`, `test_value_training_integration.py`, and runtime/collection/cache regressions. Add dedicated separate-critic tests without repurposing shared-value baseline assertions.

## Delivery order

1. Configuration, independent critic, exact-prefix validation, and unit tests.
2. Critic phase, fitted-advantage artifact, and local actor schedule.
3. Production verl/FSDP actor-only path with fixed advantages and full interval retention.
4. Paired checkpoint/resume lifecycle and separate-critic evaluator.
5. Diagnostics, documentation, regressions, two-cycle integration, and GPU smokes.

Completion means the new preset executes the agreed critic-first schedule on the production backend, preserves the token/interval contract, continues a second iteration from its saved critic, and keeps existing methods compatible. Training efficacy is a separate experimental result.
