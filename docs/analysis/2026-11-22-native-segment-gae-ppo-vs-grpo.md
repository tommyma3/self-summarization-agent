# Segment-value PPO versus GRPO on BrowseComp-Plus with native tools

**Segment-value PPO leads GRPO by 2.2 percentage points on average across matched post-initial checkpoints and by 7 points at the last matched checkpoint.** The clearest behavioral difference is that segment PPO makes fewer malformed tool calls but more often exhausts its compaction budget. These results warrant a single-seed finding in a paper. A claim that segment-level credit assignment improves accuracy would require replication and an experiment that isolates the learned baseline from GAE.

## Matched policy performance

This November 22, 2026 snapshot compares `qwen-bcplus-native-compact-separate-gae-ppo` and `qwen-bcplus-native-compaction-train`. Both start from Qwen3.5-9B, use the same 50 training questions at each matched update, and evaluate on the same 100 held-out questions with one stochastic rollout per question. Checkpoint 14 is the last complete shared evaluation. GRPO has 96 judged outcomes at checkpoint 15. Segment PPO reaches 33% accuracy at checkpoint 18. Failed trajectories count in the accuracy denominator. [Segment PPO evaluation log](../../artifacts/train/qwen-bcplus-native-compact-separate-gae-ppo/eval_metrics.jsonl) · [GRPO evaluation log](../../artifacts/train/qwen-bcplus-native-compaction-train/eval_metrics.jsonl)

| Held-out measurement | Segment PPO | GRPO | Difference |
|---|---:|---:|---:|
| Initial accuracy | 20% | 24% | −4 points |
| Mean accuracy, checkpoints 1–14 | 28.5% | 26.3% | +2.2 points |
| Mean accuracy, checkpoints 10–14 | 31.4% | 27.4% | +4.0 points |
| Checkpoint 14 accuracy | 34% | 27% | +7 points |
| Checkpoint 14 malformed tool calls | 10% | 35% | −25 points |
| Checkpoint 14 compaction-budget exhaustion | 53% | 30% | +23 points |

At checkpoint 14, both methods answered 20 questions correctly, segment PPO alone answered 14 correctly, GRPO alone answered 7 correctly, and both missed 59. The paired 95% bootstrap interval for the accuracy difference is **−2 to +16 points** (exact two-sided McNemar *p* = 0.19). For the +2.2-point average over checkpoints 1–14, the paired interval is **+0.3 to +4.2 points**. The bootstrap resamples questions, keeping each question’s checkpoint history together. Both intervals condition on these two trained policies. The late window is retrospective. [Segment PPO checkpoint 14 outcomes](../../artifacts/train/qwen-bcplus-native-compact-separate-gae-ppo/rollouts/iteration-00014.eval.jsonl) · [GRPO checkpoint 14 outcomes](../../artifacts/train/qwen-bcplus-native-compaction-train/rollouts/iteration-00014.eval.jsonl)

Compute also differed. Through update 14, segment PPO used 162 actor optimizer steps and 34.6 hours of successful logged phases, including critic pretraining. GRPO used 152 steps and 32.4 hours. These are elapsed phase times. A reproducible 2–4-point gain could matter for a paper, especially if it persists at matched compute. This comparison leaves training-seed variation unresolved. [Segment PPO timings](../../artifacts/train/qwen-bcplus-native-compact-separate-gae-ppo/phase_timings.jsonl) · [GRPO timings](../../artifacts/train/qwen-bcplus-native-compaction-train/phase_timings.jsonl)

## Where the seven extra correct answers came from

Segment PPO produced 36 final answers and got 34 right. GRPO produced 29 and got 27 right. Both submitted two incorrect answers, so the entire net accuracy difference at checkpoint 14 comes from completing seven more answers. The completed question sets differ, which limits conclusions about reasoning quality.

**On 25 of the same 100 questions, GRPO made a malformed call and segment PPO ran out of compactions.** The runtime records an `overlong` failure when another context-threshold crossing would require a fourth compaction. A direct question for the next experiment is whether the agent already had enough evidence to answer before that crossing. [Runtime compaction boundary](../../src/self_summarization_agent/runtime.py)

## What the critic learned

The separate Qwen3.5-0.8B critic predicts success at the start of each interval. Its predictions are frozen before fitting on the current batch. The actor uses those pre-fit values to calculate its advantages. The held-out diagnostics show that the critic becomes informative once it can see a compacted history. [Critic diagnostics](../../artifacts/train/qwen-bcplus-native-compact-separate-gae-ppo/step_metrics.jsonl) · [Saved estimator configuration](../../artifacts/train/qwen-bcplus-native-compact-separate-gae-ppo/checkpoints/iteration-00015/training_manifest.json)

| Held-out critic explained variance | Update 5 | Update 10 | Update 15 |
|---|---:|---:|---:|
| Initial state | −0.06 | −0.03 | −0.02 |
| Post-compaction state | 0.43 | 0.43 | 0.59 |

At update 15, overall training-batch explained variance rose from **0.51 before fitting to 0.83 after fitting**. Held-out explained variance was **0.38**. The critic predicted **27.1%** success on average against **34.0%** observed. There is room to improve generalization and calibration. The training fit gives little reason to make additional epochs the first intervention. The more relevant capacity test is whether the critic can rank two summaries of the *same history* by their downstream utility.

The frozen advantages also suggest a cheaper explanation for the policy gain. Across all 9,140 intervals in updates 1–14, the correlation between the actual GAE advantage at λ = 0.95 and the segment Monte Carlo advantage `R − V(s)` was **0.9989**. Only **32 of 2,800 rollouts** gave positive credit to one interval and negative credit to another. In GRPO, **436 of 700 four-rollout question groups** had identical rewards and therefore zero group-relative advantage. A learned baseline can still supply an update in such groups and vary its magnitude across states. That mechanism could account for some of the lead even if GAE adds little over segment Monte Carlo. [Frozen segment advantages](../../artifacts/train/qwen-bcplus-native-compact-separate-gae-ppo/checkpoints) · [GRPO training rollouts](../../artifacts/train/qwen-bcplus-native-compaction-train/rollouts) · [Group advantage calculation](../../src/self_summarization_agent/trainer.py)

## Experiments in priority order

1. **Test whether the agent should answer sooner.** For each frozen policy, sample late-interval states on separate validation questions. From each exact state, run one branch that requests an answer immediately and independent branches that allow more research under the original remaining budget. Judge every branch and compare correctness within state. If answering immediately recovers many budget failures, target the stopping decision. If both choices usually fail, examine which evidence was missing after compaction.

2. **Separate the learned baseline from temporal credit assignment.** Compare GRPO, segment Monte Carlo PPO (λ = 1), and current segment GAE PPO (λ = 0.95) across matched seeds and actor-update and rollout-token budgets. Include a cheap PPO control that estimates success from prior batches using compaction depth alone. This supplies credit on unanimous groups and tests how much the critic gains from reading the state. If the critic ranks alternative summaries reliably, add a lower-λ arm to test whether putting more weight on successor values improves the policy. [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

3. **Compare critic sizes offline.** Generate alternative summaries from the same frozen history. Sample several independent continuations from each exact successor state to estimate its success probability. Train the 0.8B critic and a larger critic on identical fitting data, then compare their within-question, same-depth ranking accuracy and calibration on held-out questions. A separate arm giving the 0.8B critic more independently labeled states tests whether supervision is the limiting factor. A larger online critic is worth the cost if it materially improves these decision-relevant predictions and the improvement transfers to policy training. [VinePPO’s Monte Carlo credit-assignment experiments](https://proceedings.mlr.press/v267/kazemnejad25a.html)

For a paper, prioritize the matched-seed estimator comparison and the answer-now experiment. Choose the primary checkpoint or learning-curve metric in advance, repeat stochastic evaluation, and report accuracy, malformed calls, budget exhaustion, and compute together. That design tests both whether the gain replicates and which decisions the critic improves.
