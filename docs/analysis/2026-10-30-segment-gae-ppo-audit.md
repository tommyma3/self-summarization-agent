# Segment-value GAE/PPO audit

Audit date: 2026-10-30, Asia/Shanghai. The comparison snapshot ends at evaluated checkpoint 16 and completed update 16. The GAE run was still active; this report does not use an incomplete checkpoint or compare its checkpoint 16 with the baseline's checkpoint 29.

Runs: `qwen-bcplus-compact-separate-gae-ppo` and `qwen-bcplus-train`.

## Assessment

The critic eventually provides useful information about post-compaction states, but its initial-state predictions remain weak and its calibration is poor. A long cold start makes the first eight policy updates effectively use a nearly constant critic. Value warmup is worth testing, with an independent validation gate; more fitting on the same small batch is not supported by the generalization results.

The core segment-return and training-data wiring passed the audit. However, the actor objectives differ: the baseline uses verl's default **dual-clip PPO**, whereas the separate-critic worker implements **standard clipped PPO**. Their clip-fraction metrics also have different definitions. Aligning these is a prerequisite for attributing an experimental difference to the critic alone.

The saved judge decisions parse consistently, use the correct checked gold answers, and assign rewards consistently. There is no observed judge parsing failure explaining this comparison.

## Current policy results

The baseline uses query-group normalized advantages (GRPO) with a PPO-family loss. It is not a conventional learned token-value PPO baseline.

| Measurement | Segment GAE/PPO | GRPO baseline |
|---|---:|---:|
| Checkpoint 0 accuracy | 15% | 15% |
| Mean accuracy, checkpoints 1–8 | 26.375% | 31.875% |
| Mean accuracy, checkpoints 9–16 | 31.25% | 32.50% |
| Mean accuracy, checkpoints 1–16 | 28.8125% | 32.1875% |
| Best accuracy through checkpoint 16 | 37%, checkpoint 12 | 44%, checkpoint 6 |
| Checkpoint 16 accuracy | 26% | 31% |
| Actor optimizer steps, updates 1–16 | 124 | 142 |
| Malformed evaluation rollouts, checkpoint 8 | 31/100 | 20/100 |
| Malformed evaluation rollouts, checkpoint 16 | 39/100 | 41/100 |

The early deficit is larger than the later deficit. These are descriptive results from one seed and 100 repeated evaluation questions; checkpoint averages are not independent experimental replications. Different interval counts also produce different optimizer-step budgets, despite a shared minibatch loop.

At checkpoint 16, conditional correctness among completed rollouts is 26/61 versus 31/59. Thus the final gap is not explained by a higher malformed rate in GAE at that checkpoint. Earlier, formatting reliability does favor the baseline. Both methods still have substantial malformed-output rates, which limits the attainable reward.

![Matched policy results and critic diagnostics](/124090467/self-summarization-agent/artifacts/train/segment_value_audit/comparison.png)

## Are the values informative when PPO actually uses them?

The important predictions are in `frozen_value_targets.json`, generated **before** the current critic fit. Reported `value/eval_after/*` instead evaluates the critic after fitting on that batch. The latter measures fitting ability and substantially overstates the signal available to that update's actor.

The following metrics were independently recomputed from the frozen predictions and recorded rollout chains. Each rollout receives equal total weight, divided among its states. Initial and post-compaction strata recompute these weights independently.

| Update | Pre-fit EV, all states | Initial EV | Post-compaction EV | After-fit training EV |
|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| 5 | 0.0017 | 0.00005 | 0.0020 | 0.0013 |
| 8 | 0.0030 | 0.0007 | 0.0053 | 0.0668 |
| 10 | 0.1335 | 0.0509 | 0.2280 | 0.5346 |
| 15 | 0.0913 | 0.0209 | 0.2823 | 0.6118 |
| 16 | 0.1662 | 0.0227 | 0.3424 | 0.6463 |

Through update 8, the all-state pre-fit value standard deviation is at most 0.0066; the values mostly learn a global prior. Later post-compaction predictions have useful signal. At update 16 their reward-residual variance is about 34% lower than a constant's, while initial-state reduction is only about 2.3%. This is scalar residual variance, not a measured reduction in policy-gradient variance.

Calibration remains important because EV ignores constant bias. At update 16, mean pre-fit value is −0.6637, implying **16.8% success**, while observed success is **32.5%**. Initial-state predictions imply **13.7% success**, versus the same 32.5% observed success. Initial-state value MSE is 0.9993, worse than the same batch's optimal constant MSE of 0.8775. The optimal constant uses batch outcomes and is an oracle diagnostic, not a deployable validation baseline.

Held-out greedy diagnostics tell a similar story:

| Critic update | After-fit training EV | Held-out EV | Held-out initial EV | Held-out post-compaction EV | Critic Brier | Training-prior constant Brier |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.0013 | 0.0009 | −0.00004 | 0.0020 | 0.20069 | 0.19600 |
| 10 | 0.5346 | 0.0624 | −0.0027 | 0.1858 | 0.20218 | 0.19733 |
| 15 | 0.6118 | 0.1145 | 0.0042 | 0.2438 | 0.21377 | 0.21550 |

The constant here uses the current training batch's rollout success rate, without held-out labels. At update 15 the critic only slightly beats it; its Brier score is also barely below the held-out oracle constant's 0.21390. Positive held-out EV therefore does not imply good calibration. Held-out predicted success is 15.4%, versus 31% observed.

These held-out rollouts use greedy decoding, whereas critic fitting uses temperature 0.7/top-p 0.95. They establish weak transfer to the evaluation distribution, but do not isolate overfitting from decoding-distribution shift. The pre-fit stochastic-batch results are the more direct evidence about values actually used by PPO.

## Training-code audit

### Correct behavior verified

- `segment_returns.py` applies gamma 1, terminal reward once, zero intermediate rewards, and the usual backward GAE recurrence. Every frozen row through update 16 agrees with an independent recurrence calculation, including terminal/malformed chains.
- `critic_step.py` freezes old values, lambda returns, and advantages before fitting; PPO receives those frozen advantages. The fitted critic only affects subsequent updates. The post-fit leakage problem discussed in the older MC experiment is not present here.
- Critic inputs use the exact recorded first-generation prompt prefix. They do not include the current segment's future answer. Initial states share a query prefix across its four rollouts; differing outcomes are legitimate noisy value labels.
- The 51-bin head uses two-hot scalar return targets and its categorical mean as the value. This is a consistent scalar-return fitting method. Diagnostic `bce` is the binary outcome log loss implied by that mean; `target_ce` is the actual categorical training objective. They measure different quantities.
- Critic fitting gives each rollout equal total weight regardless of its number of compactions. Its optimizer state persists across iterations; the actor optimizer resets each iteration, as recorded in the training manifest.
- Both actor paths retain eligible intervals and use zero-masked alignment padding. Exact IDs, shifted labels, sparse assistant masks, and old per-token log probabilities remain aligned. No history reconstruction or left truncation is introduced by the audited critic views.

On training updates 1, 8, and 16, **1,200 saved rollouts / 2,525 interval samples** passed extraction, exact-cache checks, and ordered-chain checks across the two runs. Their 50-question sets and question text match at each checked update. Evaluation sets match at checkpoints 0, 1, 8, and 16. The checked training batches cover 133 unique questions with no overlap with the 100-question evaluation set.

### Actor mismatch requiring an experimental correction

Implementation follow-up: the separate-value worker now uses the same dual-clip bound (`clip_ratio_c=3`), log-ratio stability clamp, and surrogate clipping metrics as the baseline. The common worker configuration pins the bound explicitly. The findings below describe the implementation used by the audited runs before this change; historical artifacts remain unchanged. Regression tests compare losses and gradients directly with installed verl.

`verl_separate_actor_worker.py:12` installs a custom standard clipped PPO loss. The baseline inherits installed verl's `compute_policy_loss_vanilla`, whose default includes dual clipping with `clip_ratio_c=3`, plus clamping of the log ratio for stability.

For advantage −1 and probability ratio 4, a CPU check against installed verl gives baseline loss **3**, while the custom worker's formula gives **4**. This can change gradients on negative-advantage tokens. No saved post-update per-token ratios establish how often this occurred in the actual run, so it is a demonstrated comparison confound, not a proven cause of the accuracy deficit.

The custom worker reports `abs(ratio−1)>epsilon` as clip fraction. Verl reports whether the clipped surrogate actually determines the loss. With advantage −1 and ratio 1.3, both losses are 1.3, but the custom clip fraction is 1 and verl's is 0. Cross-run clip-fraction numbers are therefore not comparable. The custom worker also lacks verl's log-ratio stability clamp; finite-metric checks passed, and no observed overflow is established.

Both actor paths optimize raw temperature-1 probabilities, matching stored vLLM `raw_logprobs`, while generation samples at temperature 0.7 with top-p 0.95. Numerator and denominator are internally on the same raw-probability convention. The sampling distribution nevertheless differs from that raw policy; this is an additional shared approximation, not exact on-policy sampling. A controlled temperature-1/top-p-1 ablation would clarify its influence. The sampled mean log ratio logged as `mean_policy_kl` can be negative and should not be interpreted as a guaranteed nonnegative divergence.

### Judge audit

For matched checkpoints 0–16 and training updates 1–16:

| Check | GAE | Baseline |
|---|---:|---:|
| Judged rollout records inspected | 4,900 | 4,900 |
| Actual judge generations reparsed | 2,377 | 3,018 |
| Recorded interval rewards checked | 10,931 | 12,074 |
| Stored parse errors | 0 | 0 |
| Reparse / independent verdict / reward mismatches | 0 | 0 |

The independent verdict check reads the final line-oriented verdict after completed thinking. Noncompleted malformed trajectories receive the runtime penalty without invoking the answer judge; they are not judge parse failures. On selected training/evaluation files, **973 judge prompts** also matched the original dataset's gold answer exactly, with no question-text mismatches. Representative positive judgments, differing-answer rejections, and abstention rejections agree with the intended answer-matching criteria. Judges sometimes add unnecessary background assertions; this audit does not certify every semantic judgment or every sentence of their reasoning.

## Recommended next experiment

1. **Align actor loss, stability handling, and metric definitions first.** Use one explicit PPO variant for both estimators and record it in run metadata. Match optimizer-step or trained-token budgets in addition to collection counts. Existing runs should retain their recorded evidence and configuration.
2. **Test critic warmup before the first actor update.** Freeze the base actor, collect broader stochastic training rollouts with the intended collection settings, and train only the critic. Split validation by question/rollout, never by interval: intervals and repeated initial prefixes are correlated. Use train-only data and keep evaluation questions out of optimization.
3. **Warm up against Monte Carlo returns**, or test lambda 1 initially. This avoids relying on an untrained critic's bootstrap values. The present lambda 0.95 differs from terminal MC targets by only about 0.01–0.03 reward units on average, so lambda bias alone is unlikely to explain the whole deficit. Preserve pre-fit advantages when actor training starts; fitting on a batch and subtracting those same fitted outcomes would reintroduce leakage.
4. **Require useful validation predictions before spending actor updates.** Compare with a training-prevalence constant using Brier/log loss and EV, separately for initial and post-compaction states. Use a stochastic validation set matching collection; keep greedy validation as a separate transfer measure. Initial-state metrics are the weak point that all-state averages hide.
5. **Tune head learning speed separately.** The saved initialization is a zero head with both learning rates at `1e-5`, three fitting epochs, 21 critic steps/update. Zero initialization gives no backbone gradient on the first backward pass. A head-LR sweep such as `1e-4` versus `1e-5`, and a head-only warmup followed by backbone fitting, are reasonable ablations. A larger rate is not yet demonstrated to improve held-out quality.
6. Compare aligned-loss **cold critic**, **warmed critic**, and a **constant baseline** with matched actor budgets; estimate that actor baseline from independent warmup data or a preceding batch. Repeat promising settings across seeds. Consider more critic capacity or replay only after distinguishing calibration, optimization speed, and generalization. Uncorrected old-policy replay labels do not directly estimate the new policy's value.

Warmup is justified as an experiment by the observed cold start. It is not established as a sufficient fix: initial-state generalization remains close to zero after hundreds of critic steps. More epochs on the same 200 rollouts can improve training EV without improving the baseline that the next actor update needs.

## Validation and reproduction

**290 targeted tests passed** across segment GAE, separate critic/value models, value diagnostics/integration, judge/parser/workers, verl batching, append-only runtime, rollout extraction, and train-step orchestration. These CPU tests and saved-artifact checks do not replace a GPU end-to-end numerical comparison.

The only repository additions are this report and a read-only audit script. Runtime/training behavior, active runs, and collected trajectories were not modified.

```bash
.venv/bin/python scripts/audit_segment_value_runs.py --judges --max-iteration 16 --output artifacts/train/segment_value_audit/audit.json
.venv/bin/python scripts/audit_segment_value_runs.py --evidence-only --output artifacts/train/segment_value_audit/evidence.json
```

Machine-readable evidence: [audit.json](/124090467/self-summarization-agent/artifacts/train/segment_value_audit/audit.json), [evidence.json](/124090467/self-summarization-agent/artifacts/train/segment_value_audit/evidence.json). The plot is available as [PNG](/124090467/self-summarization-agent/artifacts/train/segment_value_audit/comparison.png) and [SVG](/124090467/self-summarization-agent/artifacts/train/segment_value_audit/comparison.svg).

Method references: [GAE](https://arxiv.org/abs/1506.02438), [PPO](https://arxiv.org/abs/1707.06347), [dual-clip PPO](https://arxiv.org/abs/1912.09729). Conclusions about these runs come from repository artifacts and installed source, rather than extrapolation from those papers. The October 25 MC report concerns older contents of these mutable run directories; its 80–95% results are not the current October 27–30 runs.
