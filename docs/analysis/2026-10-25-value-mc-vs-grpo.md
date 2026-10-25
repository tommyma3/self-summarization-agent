# Separate value-MC versus GRPO on BrowseComp-Plus

Analysis date: 2026-10-25 (Asia/Shanghai). Runs: `qwen-bcplus-compact-separate-mc` and `qwen-bcplus-train`.

## Findings

**Value-MC shows an early learning advantage per collection iteration, but this run does not beat GRPO in final accuracy, stability, token efficiency, or elapsed training time.** At the last shared evaluated checkpoint, 28, value-MC scores **82/100**, versus **91/100** for GRPO. Their best recorded scores are **89%** and **95%**, respectively. Value-MC improves from its base policy but regresses late in training.

**The critic successfully fits each iteration’s training data, but does not demonstrate a useful generalizing value function.** Final training explained variance rises to approximately 0.65–0.81 in later iterations, whereas held-out explained variance is negative at four of six measured updates and only 0.011 at update 30. At that final diagnostic, predicted success is 97.7%, while observed success is 84%. Its 84% classification accuracy is exactly what an always-success classifier achieves.

**The most promising intervention is to stop measuring baseline quality on the data just fitted, and test out-of-fold or pre-fit baselines.** Post-compaction training predictions become exceptionally accurate, but this also nearly removes the corresponding actor signal. At update 20, every post-compaction interval has an advantage magnitude below 0.1. Better held-out calibration, broader critic training data, and matched actor update budgets should precede a larger critic or more fitting epochs.

These are findings about the recorded runs, not proof that MC value baselines are intrinsically inferior. The runs use different actor training paths and update counts, have one training seed each, and show different base-policy evaluation outcomes despite matching evaluation settings. The critic’s held-out evaluation also uses greedy rollouts, while its training targets come from stochastic rollouts.

![Policy results and critic generalization](/124090467/self-summarization-agent/docs/analysis/value-mc-vs-grpo-assets/comparison.png)

## Evidence and comparison boundaries

The primary evidence is each run’s `eval_metrics.jsonl`, `step_metrics.jsonl`, `phase_timings.jsonl`, launcher log, and selected saved rollout files. Critic manifests and fitted advantages were inspected directly. Current source was used to interpret metric semantics; current YAML presets are **not** treated as an immutable record of every historical actor setting.

- Value-MC has 30 completed updates, producing checkpoints 1–30, but evaluation only for checkpoints 0–29.
- GRPO has 29 completed updates, producing checkpoints 1–29, but evaluation only for checkpoints 0–28. The iteration-30 collection phase exits with code 1; its log ends with a train-policy collection failure. This is not evidence of an algorithmic accuracy failure, and there is no valid final evaluation to invent for checkpoint 29.
- Therefore the main matched comparison ends at **evaluated checkpoint 28**. Update `k` fits the critic on behavior checkpoint `k−1`; its scheduled held-out diagnostics also score checkpoint `k−1` rollouts. For example, critic update 30 is evaluated on policy checkpoint 29, not checkpoint 30.
- Both runs start from Qwen3.5-9B. Recorded collection contract and sampling-profile IDs match. Training uses four stochastic rollouts per task, temperature 0.7, top-p 0.95. Evaluation uses one greedy rollout per task, temperature 0, top-p 1.
- Raw rollout checks at training updates 1, 15, and 29 confirm identical sets of 50 questions and identical question text between methods, with 200 rollouts each. These three batches span 140 unique questions; this is not repeated fitting on a fixed 50-question set. Their union has no overlap with the checked evaluation set.
- Raw evaluation files at checkpoints 0, 3, 24, and 28 contain the same 100 question IDs and texts. Both runs use the same collection-profile identifier, `86ab1533…651e`, and evaluation-profile identifier, `2387221b…7b8b`. Membership was audited at these checkpoints, not independently re-parsed for every intervening file.
- The current presets specify a 12,000-token compaction threshold, a 36,000-token episode budget including tool results, and a 27,000-token maximum training sequence. The recorded collection-profile match supports common collection settings; it does not establish identical hardware scheduling or a bitwise identical execution environment.

The saved critic manifest establishes a separate Qwen3.5-0.8B backbone, two-class reward head, backbone/head learning rates both `1e-5`, three fit epochs, rollout minibatches of 32, and a persistent critic optimizer. Each update records 21 critic optimizer steps, accumulating to 630. Critic inputs end at `first_generation_prompt_end`: the initial query prefix or the successor interval’s wrapped summary prefix. This is an **interval-state value model**, not a value prediction for every action or token.

## Does value-MC have an advantage?

### Policy quality and learning speed

| Measurement | Separate value-MC | GRPO |
| --- | ---: | ---: |
| Base policy accuracy, checkpoint 0 | 47% | 43% |
| Checkpoint 3 accuracy | 66% | 55% |
| Mean accuracy, checkpoints 1–5 | 62.4% | 56.4% |
| Mean accuracy, checkpoints 1–10 | 71.4% | 67.5% |
| First checkpoint reaching 80% | 7 | 8 |
| First checkpoint reaching 85% | 15 | 10 |
| First checkpoint reaching 90% | Never observed | 16 |
| Best recorded accuracy | 89%, first at 16 | 95%, first at 24 |
| Mean accuracy, checkpoints 19–28 | 84.9% | 92.5% |
| Checkpoint 28 accuracy | 82% | 91% |
| Improvement from own base to checkpoint 28 | +35 percentage points | +48 percentage points |
| Value-MC-only additional evaluation, checkpoint 29 | 84% | Not available |

The early result is a real descriptive advantage: value-MC leads by 11 percentage points at checkpoint 3 and by 7 points at checkpoint 7. However, it already starts four points ahead at checkpoint 0. Subtracting each run’s own baseline makes the mean checkpoint-1–5 improvement 15.4 points versus 13.4 points: only a two-point difference. Baseline subtraction is descriptive, not a causal adjustment.

The late regression is substantial: value-MC averages 87.7% over checkpoints 15–24, then 81.25% over 25–28. GRPO averages 91.5% and 92.5% over those same windows. Averaging repeated checkpoints measures the curve’s shape; it does not create independent experimental replications.

Paired question-level checks reinforce the uncertainty around individual checkpoint differences:

| Checkpoint | Both correct | MC only correct | GRPO only correct | Neither correct | Exact two-sided McNemar p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 34 | 13 | 9 | 44 | 0.523 |
| 3 | 42 | 24 | 13 | 21 | 0.099 |
| 24 | 84 | 5 | 11 | 0 | 0.210 |
| 28 | 76 | 6 | 15 | 3 | 0.078 |

These are exploratory, unadjusted tests on 100 questions, with no seed replication. Neither the early lead nor the final nine-point gap individually establishes a conventional 5% significance claim. The sustained late advantage favors GRPO descriptively; the six value-MC-only successes at checkpoint 28 show complementary behavior, not overall superiority.

### Formatting reliability and inference cost

At checkpoint 28, value-MC has **13 malformed rollouts**, versus **3** for GRPO. Among completed rollouts it gets 82/87 correct (94.3%); GRPO gets 91/97 (93.8%). Thus, at this checkpoint the overall gap is primarily associated with completing valid trajectories, rather than lower conditional correctness after successful completion. This conditional comparison is descriptive because completion selects different questions for each method. Judge parse errors also occur: four versus two; they are not interchangeable with malformed model trajectories.

| Checkpoint-28 evaluation metric | Value-MC | GRPO |
| --- | ---: | ---: |
| Mean model-generated tokens / question | 3,780 | 3,637 |
| Mean total budget tokens / question, including tools | 31,070 | 31,675 |
| Correct answers / 1,000 model-generated tokens | 0.217 | 0.250 |
| Correct answers / 1,000 total budget tokens | 0.0264 | 0.0287 |
| Mean compactions / question | 1.59 | 1.67 |

Value-MC’s slightly lower total budget consumption is not an efficiency win once correctness is included. GRPO yields approximately 15.4% more correct answers per model-generated token and 8.9% more per total budget token. Failed trajectories can also consume fewer tokens, so lower average consumption alone is misleading.

### Training cost and useful signal coverage

| Through completed update 28 | Value-MC | GRPO |
| --- | ---: | ---: |
| Actor optimizer steps | 266 | 70 |
| Mean actor update duration | 24.24 min | 7.77 min |
| Mean complete train-update phase | 28.11 min | 10.09 min |
| Mean collection phase / iteration | 54.51 min | 41.75 min |
| Sum of successful collection + update phase durations | 38.56 h | 24.19 h |

Phase totals include startup and collection/judging overhead represented by the logged phases, not just GPU compute. The duplicate near-zero collection record at value-MC iteration 1 is summed into that iteration, rather than counted as another independent observation. These are phase-duration totals, not normalized GPU-hours or a controlled hardware benchmark.

The **1.59× total elapsed-phase cost** and **3.80× actor step count** rule out an observed compute advantage. Even reaching 80% takes about 9.41 hours to produce value-MC checkpoint 7, versus 7.67 hours to produce GRPO checkpoint 8; these totals end at checkpoint creation, excluding the next collection that evaluates it.

The critic itself is no longer the dominant cost: through update 28 it averages 39.4 seconds for fitting/final training predictions and 86.9 seconds for the complete critic subprocess, about 5.2% of the train-update phase. Earlier performance notes describing 27–35 minute critic fits concern an older execution regime and do not describe these current records.

Value-MC has a structural advantage in **signal coverage**. GRPO assigns zero advantage to all-success or all-failure groups. At training update 29, 46/50 GRPO groups are homogeneous; only 39 intervals have nonzero advantages, and 36 are actually passed to the actor. Value-MC retains a residual for each training state, including homogeneous groups, and its `abs(advantage) < 1e-6` fraction is zero throughout. But numerical nonzero coverage does not imply useful gradient strength: many residuals are extremely small after fitting.

There is an additional implementation confound. GRPO rounds eligible sample counts down for batching, beyond removing zero-advantage groups; at update 3 it trains on 128 of 255 eligible intervals. The separate actor path keeps partial final minibatches with execution padding. Therefore this is a comparison of the recorded systems, not an isolated change of advantage estimator. The current GRPO alignment trimming also deserves a separate audit against this repository’s requirement that every interval remain eligible; this analysis makes no training-code changes.

## Does the value network fit properly in every iteration?

### Optimization works; generalization is weak

There are two different answers:

1. **Does optimization reduce the current training objective? Yes.** All 30 iterations have lower epoch-3 online cross-entropy than epoch 1, followed by a lower final frozen-model training CE. Update 2 briefly worsens from epoch 1 to 2 before improving. Optimizer step counts advance continuously, and the recorded metrics show no nonfinite failure or missing update.
2. **Does the fitted model estimate unseen state values well? Not convincingly.** Held-out diagnostics exist only at updates 5, 10, 15, 20, 25, and 30. Four have negative explained variance; the last is barely positive and badly overconfident. Generalization cannot be asserted for the other 24 updates because it was not measured there.

Online epoch CE aggregates losses while parameters change, so it is not the loss of a single frozen checkpoint. Pre-fit evaluation is disabled and intermediate frozen-model evaluation is disabled. Therefore the records do not establish exact before/after improvement on every iteration, nor justify declaring epoch 3 optimal.

The diagnostic weighting gives each rollout equal total weight, divided among its interval states. For success probability `p = (V + 1)/2`, the value head minimizes binary cross-entropy against terminal rollout reward `R ∈ {−1,+1}`. Explained variance is:

`EV = 1 − Var_rollout-weighted(R − V) / Var_rollout-weighted(R)`.

Positive EV means reduced scalar reward-residual variance relative to a constant baseline; it does not by itself prove reduced policy-gradient variance. EV also ignores constant calibration bias, so Brier score and cross-entropy must be checked alongside it.

| Critic update | Behavior checkpoint | Train CE | Train EV | Held-out CE | Held-out EV | Held-out Brier | Constant-prevalence Brier | Predicted / observed success |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 4 | 0.411 | 0.392 | 0.595 | 0.063 | 0.2033 | 0.2139 | 74.4% / 69% |
| 10 | 9 | 0.149 | 0.695 | 0.547 | −0.027 | 0.1671 | 0.1600 | 85.3% / 80% |
| 15 | 14 | 0.114 | 0.720 | 0.540 | −0.028 | 0.1539 | 0.1411 | 92.4% / 83% |
| 20 | 19 | 0.101 | 0.774 | 0.532 | −0.152 | 0.1485 | 0.1275 | 89.1% / 85% |
| 25 | 24 | 0.110 | 0.705 | 0.409 | −0.051 | 0.1063 | 0.0979 | 94.9% / 89% |
| 30 | 29 | 0.100 | 0.695 | 0.622 | 0.011 | 0.1516 | 0.1344 | 97.7% / 84% |

The constant-prevalence reference uses the held-out set’s own success rate and is an oracle diagnostic, not a deployable baseline. The conclusion also holds against the **training rollout success-rate constant**, which requires no held-out labels: at update 30 its held-out CE is 0.460 and Brier is 0.1386, both better than the critic’s 0.622 and 0.1516. The fitted critic loses to that training-prior constant in CE and Brier at all five scheduled checks after update 5.

At update 30, 97.4% of held-out rollout-weighted state mass lies in the 0.9–1.0 predicted-success bin. That bin predicts 97.9% success but achieves 84.3%. All held-out values are positive, so every state is classified as successful. This is poor failure discrimination, not a well-calibrated critic with merely noisy labels.

**Distribution caveat:** these held-out trajectories use the same checkpoint but a different sampling distribution: greedy evaluation versus stochastic training. These results directly show poor transfer to the evaluation distribution. They strongly motivate an overfitting investigation, but cannot by themselves distinguish training-set memorization from decoding-distribution shift or prove the critic is useless on fresh stochastic rollouts. A separate stochastic validation collection is the missing decisive measurement.

### Post-compaction fitting suppresses the signal the method needs

The strongest evidence concerns states after compaction:

| Critic update | Initial-state train EV | Post-compaction train EV | Post-compaction held-out EV |
| ---: | ---: | ---: | ---: |
| 10 | 0.343 | 0.903 | −0.091 |
| 15 | 0.284 | 0.970 | −0.025 |
| 20 | 0.335 | 0.998 | −0.168 |
| 25 | 0.353 | 0.917 | −0.048 |
| 30 | 0.262 | 0.912 | 0.009 |

At update 20, saved advantages show mean post-compaction `A = R − V` of **+0.0367** for 269 successful states and **−0.0615** for 60 failed states; all 329 have `|A| < 0.1`. At update 30, all 258 successful post-compaction states have `|A| < 0.1`, averaging +0.0265; 92.1% of the 38 failed post-compaction states are also below that magnitude. These are interval counts and unweighted conditional means, not the rollout-weighted diagnostics above.

This suggests a failure mechanism: fit a high-capacity critic to the just-collected outcomes, subtract those fitted values from the **same** outcomes, and nearly erase learning on distinctive summary states. The critic can fit individual post-compaction prefixes far more easily than initial prefixes shared by four rollouts with potentially conflicting labels. The train/held-out gap is consistent with this mechanism. Proving its causal effect on policy quality requires a baseline timing or cross-fitting ablation.

The state prefix itself is causally valid: it excludes future generated actions in that interval. The concern is statistical dependence introduced by fitting on the same trajectory’s terminal label, not evidence of future tokens being included in the input. An action-independent baseline has the usual policy-gradient cancellation property when fixed independently of the sampled continuation; in-sample fitted baselines need not retain that independence in a finite batch.

The reported last gradient norm exceeds the configured clip threshold of 1 at every update, from 3.61 to 194. The source records the norm returned **before clipping**, so this does not mean clipping failed or that gradients diverged. It does justify measuring clipping frequency across all critic steps; one final-step norm per iteration cannot establish that frequency.

GRPO has no learned value network to fit. Its comparison baseline is the reward mean/std within each four-rollout question group, computed once per rollout and then assigned to that rollout’s intervals.

## Changes most likely to improve value-MC

These are proposed experiments, not guaranteed improvements or implemented changes. The first three have the strongest direct support from these artifacts.

### 1. Separate baseline prediction from fitting on the same outcomes

Start with two inexpensive ablations: use the critic carried from the previous iteration to produce fixed actor advantages before fitting the new critic; and use query-grouped cross-fitting, where each query’s actor advantages are predicted by a critic that did not train on that query’s current outcomes. All intervals of a rollout must stay together, and grouping all sibling rollouts by question prevents closely related prefixes crossing folds.

Cross-fitting costs additional critic fits, but the present critic subprocess is only about 5% of training-update time. A pre-fit baseline is cheaper but may be stale as the policy changes. Compare both on fresh stochastic trajectories from the behavior checkpoint. Retain critic learning for subsequent updates and preserve the final rollout reward on every interval.

**Success criterion:** positive held-out stochastic EV, better Brier/CE than a training-prior constant, and stronger post-compaction residuals without increasing malformed rollouts. Log advantage magnitude quantiles by success/failure and state kind; the existing `1e-6` near-zero threshold misses the observed attenuation.

### 2. Select critic fitting strength using an independent stochastic validation set

Reserve query-disjoint validation questions within the training partition. Collect them with the training temperature and top-p. Evaluate the carried critic and each candidate fitted epoch on those same trajectories, with rollout-clustered uncertainty. Keep the existing 100 greedy evaluation questions for policy comparison, rather than tuning critic hyperparameters against them repeatedly.

Sweep one versus three fitting epochs and backbone learning rates such as `1e-6`, `3e-6`, and `1e-5`, with separate head learning rates. Consider a frozen-backbone/head-only or low-rank-adaptation control. These are experiment settings, not established optimal values. More epochs at the current settings is a poor first response to near-perfect post-compaction training fit and weak validation performance.

Temperature-calibrate the reward logits or shrink values toward a training-prior constant using validation data. A constant offset changes calibration but not EV; temperature or shrinkage also changes prediction spread. Track CE, Brier, EV, failure recall, and calibration separately for initial and post-compaction states. Increase model size only if these checks establish underfitting on fresh data after regularization and coverage have been addressed.

### 3. Test a conservative hybrid with the existing group baseline

Use a validation-selected blend such as `b = (1−α)b_group + αV_crossfit`, and `A = R−b`, comparing `α=0`, an intermediate value, and `α=1`. In the first ablation use raw group-mean and raw value residuals on a consistent scale; current GRPO also divides by group reward standard deviation, so mixing its normalized advantages directly with raw MC advantages changes more than the baseline.

The learned component should earn its weight through held-out variance reduction. Evaluate homogeneous groups separately: they are where GRPO has zero signal and where a generalizing critic could supply an advantage. Do not assume every nonzero residual is useful. Keep query/rollout weighting explicit so more compactions do not create additional independent normalization weight.

The motivation is the tradeoff established by [DeepSeekMath’s GRPO formulation](https://arxiv.org/abs/2402.03300): a learned critic is an extra estimation problem, while group-relative rewards provide a cheap baseline. The blend proposed here is an experimental inference from this run’s diagnostics, not a result established by that paper.

### 4. Match actor optimization and address malformed trajectories

Value-MC currently receives 8–10 actor optimizer steps per update versus GRPO’s 2–4, with late clip fractions averaging 1.24% versus 0.36% over updates 19–28. These fractions do not establish instability by themselves, but the exposure difference makes actor optimization a necessary ablation. Compare one versus two actor epochs and lower learning rates, and report results both per collected rollout and per elapsed time. Audit loss reduction, advantage scale, interval ordering, and partial minibatch handling before attributing the remaining gap to value estimation.

Log a nonnegative approximate KL estimator and its tail statistics. The currently reported mean `old_logprob − new_logprob` can be negative, so it is a weak sole early-stop signal. Establish KL thresholds empirically; do not interpret low average clip fraction as proof that rare trajectories remain stable.

Because the checkpoint-28 gap closely tracks malformed outputs, preserve their configured penalties and inspect their advantage magnitudes, state kinds, and format anomalies. If targeted format training or balanced critic batches are tested, weight samples to preserve the original objective and compare them as explicit ablations. Never repair malformed sampled tokens, reconstruct their history, or silently remove them from evidence.

### 5. Test the potential rollout-efficiency benefit

A learned baseline need not require four samples of every question. Once validation demonstrates a useful critic, compare 50 questions × 4 rollouts with 100 × 2 and 200 × 1 at a fixed 200-rollout budget, keeping training/evaluation partitions disjoint. This may broaden question coverage and reduce reliance on repeated sibling outcomes. It is a prospective benefit: the present value-MC run uses the same four-rollout grouping as GRPO and does not demonstrate it.

A short, policy-tagged critic replay window may help rare failures and summary-state coverage, but stale outcomes reflect older policies. Measure performance on current-policy validation, limit age, and keep replay restricted to critic fitting unless off-policy actor correction is explicitly designed.

### 6. Consider richer credit assignment only after validating the baseline

The present value model supplies one terminal-reward residual per interval. A validated model could support bootstrapped or GAE-style interval credit assignment to distinguish progress before and after compaction. [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) explains the bias–variance tradeoff of such estimators. Bootstrapping from the currently weak held-out critic could amplify error, so this is a later experiment rather than the first fix.

Changing state anchors, interval targets, or how signals cross compaction boundaries requires an explicit design review under this repository’s rollout/training contract. No proposed experiment should rewrite a ledger, retokenize historical context, omit archived intervals, or replace the terminal reward with an unapproved signal.

## Experiment order and a credible definition of “beat GRPO”

1. **Diagnostics first:** add fresh stochastic critic validation, pre-fit/epoch-wise evaluation, group/state-stratified advantage quantiles, and failure discrimination. Record source revision, resolved config, task IDs, and environment fingerprints for each new run.
2. **Small controlled ablations:** compare in-sample, pre-fit, and cross-fitted baselines with the same actor update implementation; then tune fitting strength and calibration. Use a constant baseline and raw group-mean baseline to separate the benefit of state prediction from advantage scaling.
3. **End-to-end test:** promote the best validated baseline or hybrid to replicated policy training. Use at least three seeds as a first replication stage, common task schedules, a common initial evaluation, and equal rollout and compute budgets reported separately. Choose checkpoints by a declared validation rule rather than the maximum on the test curve.
4. **Claim superiority only with sustained paired gains:** compare a predeclared final checkpoint or late-window metric, malformed rate, accuracy per generated token, and accuracy per wall-clock budget. Use question-clustered paired uncertainty and seed-to-seed variability. A single 95% peak on these same 100 repeatedly inspected questions is not a sufficient new superiority claim.

The current evidence supports pursuing value-MC as a potentially broader-coverage baseline, but **the route to beating GRPO is better out-of-sample value estimation and better use of the resulting actor signal, not simply a lower training value loss**.

## Sources and reproducibility

- [Value-MC evaluation metrics](/124090467/self-summarization-agent/artifacts/train/qwen-bcplus-compact-separate-mc/eval_metrics.jsonl), [update metrics](/124090467/self-summarization-agent/artifacts/train/qwen-bcplus-compact-separate-mc/step_metrics.jsonl), and [phase timings](/124090467/self-summarization-agent/artifacts/train/qwen-bcplus-compact-separate-mc/phase_timings.jsonl).
- [GRPO evaluation metrics](/124090467/self-summarization-agent/artifacts/train/qwen-bcplus-train/eval_metrics.jsonl), [update metrics](/124090467/self-summarization-agent/artifacts/train/qwen-bcplus-train/step_metrics.jsonl), and [phase timings](/124090467/self-summarization-agent/artifacts/train/qwen-bcplus-train/phase_timings.jsonl).
- [Saved critic configuration](/124090467/self-summarization-agent/artifacts/train/qwen-bcplus-compact-separate-mc/checkpoints/iteration-00030/critic/critic_manifest.json), [update-20 fitted advantages](/124090467/self-summarization-agent/artifacts/train/qwen-bcplus-compact-separate-mc/checkpoints/iteration-00020/fitted_advantages.json), and [update-30 fitted advantages](/124090467/self-summarization-agent/artifacts/train/qwen-bcplus-compact-separate-mc/checkpoints/iteration-00030/fitted_advantages.json).
- Implementation semantics: [critic fitting](/124090467/self-summarization-agent/src/self_summarization_agent/separate_value_trainer.py), [held-out scoring and advantage export](/124090467/self-summarization-agent/src/self_summarization_agent/critic_step.py), [value diagnostics](/124090467/self-summarization-agent/src/self_summarization_agent/value_diagnostics.py), [GRPO advantages](/124090467/self-summarization-agent/src/self_summarization_agent/trainer.py), [actor paths and batching](/124090467/self-summarization-agent/src/self_summarization_agent/verl_ray_trainer.py), and [separate actor loss](/124090467/self-summarization-agent/src/self_summarization_agent/verl_separate_actor_worker.py).
- Derived compact artifacts: [evaluation CSV](/124090467/self-summarization-agent/docs/analysis/value-mc-vs-grpo-assets/evaluation.csv), [critic-fit CSV](/124090467/self-summarization-agent/docs/analysis/value-mc-vs-grpo-assets/critic-fit.csv), and [paired contingency counts](/124090467/self-summarization-agent/docs/analysis/value-mc-vs-grpo-assets/paired-comparison.json).

All statistics were computed from saved records; no models were retrained or re-evaluated. Raw trajectories, checkpoints, and training code were left unchanged. The raw-rollout audit covers training files `iteration-{00001,00015,00029}.jsonl` and evaluation files `iteration-{00000,00003,00024,00028}.eval.jsonl` in both run directories. Paired p-values use an exact two-sided binomial test on discordant question outcomes. Cross-iteration means are simple descriptive means; value metrics preserve the implementation’s equal-total-weight-per-rollout convention.

## Appendix: every critic fitting iteration

“Update” denotes the output checkpoint. Online CE is the moving-model training loss; final train CE and train EV use the final frozen critic. A dash means held-out evaluation was not scheduled, not zero or a carried-forward value. No value network is trained by the GRPO run.

| Update → checkpoint | Online CE epoch 1 | Epoch 2 | Epoch 3 | Final train CE | Train EV | Held-out EV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.680 | 0.656 | 0.633 | 0.586 | 0.143 | — |
| 2 | 0.684 | 0.688 | 0.624 | 0.595 | 0.215 | — |
| 3 | 0.667 | 0.556 | 0.473 | 0.412 | 0.476 | — |
| 4 | 0.677 | 0.559 | 0.465 | 0.402 | 0.491 | — |
| 5 | 0.681 | 0.517 | 0.461 | 0.411 | 0.392 | 0.063 |
| 6 | 0.595 | 0.486 | 0.379 | 0.283 | 0.621 | — |
| 7 | 0.574 | 0.349 | 0.274 | 0.217 | 0.706 | — |
| 8 | 0.353 | 0.275 | 0.216 | 0.179 | 0.605 | — |
| 9 | 0.391 | 0.248 | 0.155 | 0.111 | 0.772 | — |
| 10 | 0.507 | 0.259 | 0.186 | 0.149 | 0.695 | -0.027 |
| 11 | 0.501 | 0.289 | 0.192 | 0.150 | 0.696 | — |
| 12 | 0.471 | 0.227 | 0.158 | 0.124 | 0.750 | — |
| 13 | 0.464 | 0.245 | 0.172 | 0.132 | 0.683 | — |
| 14 | 0.487 | 0.277 | 0.160 | 0.135 | 0.716 | — |
| 15 | 0.448 | 0.263 | 0.159 | 0.113 | 0.720 | -0.028 |
| 16 | 0.432 | 0.264 | 0.208 | 0.140 | 0.679 | — |
| 17 | 0.387 | 0.202 | 0.149 | 0.118 | 0.706 | — |
| 18 | 0.532 | 0.350 | 0.211 | 0.153 | 0.671 | — |
| 19 | 0.356 | 0.189 | 0.120 | 0.089 | 0.775 | — |
| 20 | 0.476 | 0.206 | 0.128 | 0.101 | 0.774 | -0.152 |
| 21 | 0.503 | 0.245 | 0.136 | 0.100 | 0.802 | — |
| 22 | 0.402 | 0.173 | 0.121 | 0.092 | 0.744 | — |
| 23 | 0.466 | 0.155 | 0.117 | 0.093 | 0.807 | — |
| 24 | 0.439 | 0.262 | 0.166 | 0.114 | 0.717 | — |
| 25 | 0.331 | 0.207 | 0.144 | 0.110 | 0.705 | -0.051 |
| 26 | 0.469 | 0.261 | 0.153 | 0.107 | 0.750 | — |
| 27 | 0.429 | 0.181 | 0.125 | 0.103 | 0.728 | — |
| 28 | 0.532 | 0.226 | 0.137 | 0.109 | 0.758 | — |
| 29 | 0.315 | 0.197 | 0.127 | 0.096 | 0.645 | — |
| 30 | 0.373 | 0.170 | 0.130 | 0.100 | 0.695 | 0.011 |
