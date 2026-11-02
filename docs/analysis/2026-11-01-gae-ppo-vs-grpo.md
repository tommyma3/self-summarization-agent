# Latest segment-value GAE PPO versus GRPO

Snapshot: 2026-11-01 20:04 CST. Runs: `qwen-bcplus-compact-separate-gae-ppo` and `qwen-bcplus-train`.

**The latest segment-value run shows better late-stage evaluation accuracy and substantially better format reliability. It has not established general superiority over GRPO: early learning is slower, whole-run average accuracy is nearly tied, the final checkpoint difference is uncertain, and recorded training cost is higher.**

Both runs completed 30 updates, but their latest saved policy evaluation is **checkpoint 29**, not 30. The GAE evaluation was recorded on November 1 at 11:43 CST; the baseline evaluation was recorded on October 29 at 06:58 CST. Each checkpoint evaluates the same 100 questions with greedy decoding and one rollout per question.

The current GAE artifacts record **critic pretraining**, using the Qwen3.5-9B actor and a separate Qwen3.5-0.8B critic. A 51-bin two-hot value head supplies pre-fit segment GAE advantages with gamma 1 and lambda 0.95. The critic is fitted afterward for subsequent updates. Its initialization manifest selects pretraining epoch 2, from 200 additional rollouts over 50 training-partition questions (40 fitting questions and 10 validation questions). Those questions do not overlap the policy evaluation set. These current artifacts differ from the older cold-critic snapshot discussed in `2026-10-30-segment-gae-ppo-audit.md`; its numerical results should not be mixed into this comparison.

| Policy measurement | Segment GAE PPO | GRPO |
|---|---:|---:|
| Initial accuracy, checkpoint 0 | 14% | 15% |
| Early mean accuracy, checkpoints 1–10 | 26.5% | 32.5% |
| Mean accuracy, checkpoints 1–29 | 32.97% | 32.48% |
| Late mean accuracy, checkpoints 20–29 | **38.2%** | 33.0% |
| Latest accuracy, checkpoint 29 | **45%** | 39% |
| Highest observed accuracy | 45%, checkpoint 29 | 44%, checkpoint 6 |
| Late mean malformed rate, checkpoints 20–29 | **15.8%** | 40.2% |
| Latest malformed rate | **8%** | 31% |
| Latest correct / completed | 45/92 = 48.9% | 39/69 = 56.5% |
| Late correct / completed, pooled | 382/842 = 45.4% | 330/598 = 55.2% |

![Performance and paired uncertainty](/124090467/self-summarization-agent/docs/analysis/gae-ppo-vs-grpo-assets/performance.png)

GAE's late advantage is associated with completing more valid trajectories. At checkpoint 29, completion rates are 92% versus 69%, while correctness among completed trajectories favors GRPO. Completion selects different subsets of questions, so conditional correctness does not establish which method reasons better on the same successfully completed tasks. All failed trajectories remain in the main accuracy denominator. Saved evaluation judge parse-error counts are zero at every checkpoint in both runs; malformed model outputs are distinct from judge parsing failures.

The paired final-checkpoint outcomes are 32 questions correct for both, 13 correct only for GAE, 7 correct only for GRPO, and 48 correct for neither. The exact two-sided McNemar test gives **p = 0.263**. A paired question bootstrap gives a 95% interval of **−3 to +15 percentage points** for the final +6-point difference.

For the late window, averaging each question's paired correctness difference across checkpoints 20–29 and then bootstrapping the 100 questions gives **+5.2 points, 95% interval +1.6 to +9.0 points**. All 20,000 bootstrap replicates preserve a question's entire checkpoint history together. This avoids treating the 1,000 late evaluations as independent questions. The late window is a retrospective descriptive comparison, and its interval conditions on these two trained runs; it does not measure training-seed variability or correct for inspecting many possible windows. Across all post-initial checkpoints the gap is only +0.48 points, with interval −1.97 to +2.97 points. The early-window gap is −6 points.

| Cost / efficiency measurement | Segment GAE PPO | GRPO |
|---|---:|---:|
| Successful recorded pipeline phases, all 30 updates | 44.15 h | 36.28 h |
| Critic pretraining, included above | 0.69 h | 0 |
| Collection + evaluation + judging phases | 29.90 h | 23.50 h |
| Training-update phases | 13.56 h | 12.78 h |
| Actor optimizer steps through update 30 | 272 | 252 |
| Actor steps through evaluated checkpoint 29 | 264 | 246 |
| Latest generated tokens / question | 3,212 | 3,838 |
| Latest total budget tokens / question, including tools | 33,550 | 27,923 |
| Latest correct / 1,000 generated tokens | **0.1401** | 0.1016 |
| Latest correct / 1,000 total budget tokens | 0.01341 | **0.01397** |
| Latest searches / question | 11.09 | 8.83 |
| Latest compactions / question | 1.85 | 1.52 |

![Efficiency and recorded cost](/124090467/self-summarization-agent/docs/analysis/gae-ppo-vs-grpo-assets/efficiency.png)

GAE takes **21.7% more successful phase time** and **7.9% more actor optimizer steps** over 30 updates. Most of the additional pipeline time is in collection/evaluation/judging, rather than critic fitting. The online critic subprocess averages 94.8 seconds per update, about 5.8% of the complete GAE training-update phase. Mean actor update duration is approximately 23.3 minutes for each method; different numbers of interval samples and optimizer steps make those averages imperfect compute controls.

At checkpoint 29, GAE yields **37.9% more correct answers per generated token**, but **4.0% fewer per total budget token** after tool results are included. It uses more search/tool context. There is therefore an observed generated-token advantage, but no demonstrated overall compute-efficiency advantage. Pipeline totals are sums of logged durations, not normalized GPU-hours or a controlled hardware benchmark. Adding the baseline's one failed collection attempt changes its total to 36.49 h. Restart gaps are excluded. The time curve assigns checkpoint k the pretraining plus successful collection/update phases through update k, excluding the later collection that evaluates it.

The learned critic provides useful state prediction, especially after compaction. The important pre-fit predictions are made before fitting the current batch and determine that batch's actor advantages. The much higher after-fit training explained variance is a fitting diagnostic, not the quality of the baseline used for that update.

| Critic update | Held-out EV | Initial-state EV | Post-compaction EV | Held-out Brier | Training-prior constant Brier |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.176 | -0.046 | 0.498 | 0.1540 | 0.1825 |
| 10 | 0.309 | 0.023 | 0.495 | 0.1513 | 0.2182 |
| 15 | 0.238 | 0.099 | 0.345 | 0.1832 | 0.2356 |
| 20 | 0.291 | 0.068 | 0.411 | 0.1616 | 0.2287 |
| 25 | 0.253 | 0.077 | 0.374 | 0.1777 | 0.2244 |
| 30 | 0.168 | -0.052 | 0.307 | 0.2420 | 0.2475 |

![Critic quality and stochastic training outcomes](/124090467/self-summarization-agent/docs/analysis/gae-ppo-vs-grpo-assets/critic.png)

The critic beats the training-prevalence constant in held-out Brier score at all six scheduled checks. By update 30, however, its advantage is small: Brier 0.2420 versus 0.2475. It predicts 26.0% success while observed success is 45.0%, and initial-state EV falls to −0.052. Post-compaction EV remains positive at 0.307. At that same update, pre-fit training EV is 0.702 and after-fit training EV is 0.889; held-out EV is only 0.168. Thus the baseline does carry useful information, with substantial generalization and calibration limitations.

Held-out critic rollouts use greedy decoding; training rollouts use stochastic decoding. This difference prevents attributing the entire train/held-out gap to memorization. Held-out metrics at critic update k evaluate trajectories from behavior checkpoint k−1; the critic's update-30 diagnostic is **not** a policy evaluation of checkpoint 30. Late stochastic training rollout accuracy is also higher for GAE (37.55% versus 26.05% over updates 21–30), but these are training questions and support a different claim from held-out policy accuracy.

A credible next comparison would repeat both methods across seeds, match actor optimization and collected/processed token budgets, and choose the final evaluation rule before inspecting its curve. The saved run evidence already supports continuing the segment-value approach for reliability and late accuracy. It does not isolate the effects of critic pretraining, the learned baseline, and segment GAE; that requires ablations. The current source has a dual-clip actor implementation, but the saved manifests do not provide a complete immutable source/environment fingerprint for every phase, so this report does not certify exact implementation equivalence across the two historical runs.

Validation covered every saved evaluation checkpoint (6,000 outcomes) and all 30 training batches per method (12,000 rollouts). Per-question evaluation counts reproduce the aggregate metrics, question IDs and text match across runs, training question sets match at every update, and no evaluated question appears in training or critic pretraining. Retained completed GAE checkpoints 14–18 and 27–30 pass an independent frozen-target recurrence check. Pruned checkpoints cannot be rechecked from unavailable frozen-target files. These checks do not constitute a fresh audit of every token ledger or every semantic judge decision.

Reproduction: run `.venv/bin/python scripts/analyze_gae_ppo_vs_grpo.py` from the repository root. Use `--figures-only` to regenerate figures from extracted evidence. The script reads saved artifacts and writes only derived outputs. No models were retrained or re-evaluated, and no trajectories or training code were changed.

Evidence: [summary and paired statistics](/124090467/self-summarization-agent/docs/analysis/gae-ppo-vs-grpo-assets/summary.json), [evaluation metrics](/124090467/self-summarization-agent/docs/analysis/gae-ppo-vs-grpo-assets/evaluation.csv), [per-question outcomes](/124090467/self-summarization-agent/docs/analysis/gae-ppo-vs-grpo-assets/per-question-evaluation.csv), [training outcomes](/124090467/self-summarization-agent/docs/analysis/gae-ppo-vs-grpo-assets/training.csv), [critic metrics](/124090467/self-summarization-agent/docs/analysis/gae-ppo-vs-grpo-assets/critic.csv), [source inventory and metric hashes](/124090467/self-summarization-agent/docs/analysis/gae-ppo-vs-grpo-assets/sources.json), [retained frozen-target audit](/124090467/self-summarization-agent/docs/analysis/gae-ppo-vs-grpo-assets/frozen-target-audit.json), and [analysis script](/124090467/self-summarization-agent/scripts/analyze_gae_ppo_vs_grpo.py). Each figure is available as PNG, SVG, and PDF in the assets directory. Metric JSONL snapshots are included so later run-directory updates do not change this report's evidence.
