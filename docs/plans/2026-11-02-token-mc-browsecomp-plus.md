# Token-level two-bin Monte Carlo critic for BrowseComp-Plus

Status: implementation plan, 2026-11-02. Repository paths and integration points were inspected; implementation and training validation remain to be done. Existing unrelated working-tree changes must be preserved.

## Objective and scope

Add `token_mc_value`: a separate causal critic predicting the final signed rollout reward before every sampled assistant token. Use exactly two output classes, ordered `[-1, +1]`, and frozen pre-fit Monte Carlo advantages for PPO. Support the existing BrowseComp-Plus collection, retrieval, judge, critic warmup, actor training, evaluation, and checkpoint/resume workflow.

The first release supports the Transformers reference actor path and the production `verl_ray`/`verl_fsdp` actor path. Reuse the independent critic process and its compatible smaller backbone. Shared-backbone token critics, token GAE, reward shaping, other benchmark presets, and post-fit token baselines are outside this implementation. Unsupported configurations must fail at validation rather than silently use interval behavior.

No collection-format redesign is required or authorized. The append-only contract in `AGENTS.md` remains authoritative.

## Mathematical contract

For rollout r, interval i, and sampled token at full-ledger position j, the state is precisely the active prefix before j. If cached shifted arrays are `input_ids = full_ids[:-1]` and `labels = full_ids[1:]`, prediction index k uses hidden state k to baseline label k, which is full-ledger token k+1. The critic cannot see that label or any future observation.

The head produces two logits. With `p = softmax(logits)`, compute `V = p[1] - p[0]`. Every eligible position has the same observed target `R_r in {-1, +1}` and class label `int(R_r == +1)`. Critic loss is stable cross-entropy directly from logits. Freeze `A[k] = R_r - V_old[k]` before fitting the critic on the current collection; advantages remain fixed for every actor epoch. Thus V is in [-1, 1] and A is in [-2, 2]. There is no reward discount, token penalty, reward whitening, or GAE recurrence.

The critic learns expected eventual reward, not individual token correctness. Reasoning, tool-call syntax and arguments, summary reasoning/body/tags, final answers, forced answers, and sampled stop tokens all receive supervision. Original prompts, tool observations, runtime controls, renderer-inserted headers/separators, and the wrapped summary copy in a successor interval do not.

Preserve the current critic's rollout/interval weighting. For B contributing rollouts, K_r nonempty eligible intervals per rollout, and N_ri supervised tokens in an interval:

`L_V = (1/B) sum_r (1/K_r) sum_i (1/N_ri) sum_k CE(logits_rik, outcome_r)`.

Zero-assistant-token intervals have no token target: retain their evidence, count them diagnostically, and exclude them from these denominators. A rollout with no eligible tokens produces no training contribution. Never synthesize an interval-start target to fill this case. An interval with tokens and zero advantages still participates in critic fitting and actor batch accounting.

Optimizer minibatches should contain complete rollout groups. Compute weights using the actual rollout count in that minibatch, including the final short minibatch; accumulate microbatch weighted sums without renormalizing each microbatch. Use whole-collection denominators for frozen evaluation metrics. Preserve the existing actor reduction, clipping, behavior log probabilities, KL controls, and interval inclusion rules. Confirm this reduction agrees with the existing separate-critic implementation and use the identical reduction in the matched interval control.

## Existing integration points and gaps

All source paths below are under `src/self_summarization_agent/`.

| Component | Current behavior and required extension |
| --- | --- |
| `config.py` | Recognizes group-relative, interval MC, and interval GAE estimators. Add explicit token-MC validation and method identity. |
| `trajectory.py`, `cache_step.py` | Exact shifted IDs, sparse masks, and behavior log probabilities already exist (token cache v7). Consume them without retokenizing or changing stored evidence. |
| `value_model.py` | Existing `CompactionValueHead` supports two logits, signed expectation, stable classification loss, and head serialization. Reuse these primitives. |
| `separate_value_trainer.py` | Reads only interval-start prefixes and selects one hidden state. Add full-interval causal evaluation/fitting and token masks. |
| `separate_value.py` | Advantages and frozen-target rows are scalar per interval. Add a separately versioned token artifact and strict validators. |
| `critic_step.py`, `train_step.py` | Dispatch critic evaluation/fitting and load frozen advantages. Route token mode and bind the actor update to its source critic. |
| `trainer.py` | Scalar advantage types and broadcasting assume one advantage per interval. Add explicit token-array handling. |
| `verl_ray_trainer.py` | Batch construction fills response positions with one scalar advantage. Populate aligned token advantages instead, preserving masks and padding. |
| `value_pretraining.py`, `value_pretraining_step.py` | Warmup uses interval targets and hard-codes `compaction_mc_value` when constructing its trainer. Add token-mode warmup and selection metrics. |
| `value_diagnostics.py`, `evaluate_value.py` | Metrics assume one prediction per state/interval. Add weighted token metrics without flattening tokens into independent rollouts. |
| `iteration_launcher.py`, `scripts/train_iterations.sh` | Existing separate-critic startup automatically enables warmup. Ensure method identity and completion validation cover token mode at every phase. |

## Phase 1 — Configuration and training-view contract

1. Register `training.advantage_estimator: token_mc_value`. Require enabled separate critic, `num_bins: 2`, `baseline_timing: pre_fit`, signed rewards, all compaction tokens enabled, exact collection IDs, and a supported training backend. Initially require `benchmark.name: browsecomp` for this estimator.
2. Derive token granularity from the estimator rather than adding a second independent granularity switch. Give the token method manifest the explicit anchor `before_each_sampled_token`; retain the legacy `state_anchor` configuration solely for existing interval estimators. Document that distinction and reject ambiguous overrides.
3. Add a small training-only module, proposed `token_value.py`, owning token position extraction, value/advantage validation, and token loss weights. Reuse `RLSample`; do not create one trajectory/sample per token.
4. Validate shifted-ID continuity, exact mask lengths, vocabulary limits, complete ordered interval membership, finite rewards/log probabilities, and raw sampled-token coverage. Preserve the upstream history-rewrite rejection and exclusion policy.
5. Test causal index alignment with known full-ledger IDs, disjoint assistant spans, and a tool observation between generations. Changing a target token or later suffix must not alter its pre-token prediction. Run the causality check with a real supported tiny causal model as well as deterministic fixtures.

Acceptance: every supervised token maps to exactly one authoritative mask position; non-assistant tokens have no value loss; invalid alignment fails before fitting.

## Phase 2 — Token critic forward pass and loss

1. Extend `SeparateValueTrainer` with an explicit token-mode prepared representation holding full shifted interval inputs, lengths, sparse target positions, and rollout/interval weights. Keep legacy prefix paths unchanged.
2. Use `AutoModel` hidden states and the two-logit head in one causal pass per interval. Gather eligible hidden states before applying the head; do not materialize vocabulary logits or independently forward every token prefix.
3. Preserve actor/critic tokenizer vocabulary and special-token signature checks. Reuse actor token IDs directly even when the critic backbone is smaller. Enforce the critic's own positional/context capacity for full intervals, not just initial prompts.
4. Batch by full interval lengths, with explicit right-padding attention masks and correct positions. A valid interval larger than the soft token-batching budget runs alone. Exceeding the hard model/training limit fails; never truncate or use independently conditioned sliding windows.
5. Default token-mode fit/eval microbatch size to one and enable activation checkpointing for the initial long-context production preset. Measure peak memory before raising either batch size. Keep model mode and RNG stable for diagnostic passes.
6. Reduce token CE according to the stated rollout/interval/token weights. Preserve backbone/head optimizer state and separately report effective weight mass, number of predictions, padded tokens, throughput, and memory.

Acceptance: hand-computed CE and gradients agree on a mixed-length, multi-rollout fixture; microbatch partitioning preserves the update; the first token's prediction agrees with the existing interval-start prediction under identical weights and inputs.

## Phase 3 — Frozen token artifacts and resumable orchestration

1. Add a token-specific frozen artifact schema, leaving existing scalar artifacts backward-compatible. Store compact per-interval arrays of shifted positions, frozen values, and advantages, plus one rollout reward; do not repeat a full prefix or reward per token. Prefer tensor shards with a small JSON manifest to millions of JSON objects.
2. Bind manifests to behavior checkpoint, source critic identity, tokenizer signature, estimator/granularity, class order, pre-fit timing, weighting policy, ordered sample keys, exact IDs/masks, and behavior-logprob provenance. Store shard digests and explicit shape/dtype metadata.
3. Validate exact coverage of each eligible position with no duplicates/extras; verify finite values, ranges, and `A = R - V_old` within numeric tolerance. Store float32 values/advantages. Dense arrays reconstructed for a minibatch have zero values in masked slots.
4. Iteration order: collect and judge; validate/build exact caches; evaluate carried critic; atomically freeze token values/advantages; fit critic to terminal classes; run actor PPO using frozen arrays; publish complete actor/critic pair.
5. Include token artifact identity in checkpoint completion checks and restart fingerprints. Retries start from the committed input pair and reuse only validated frozen artifacts; fitted critic values never overwrite the frozen baseline. Ordinary resume rejects interval/token method changes even though both heads have two logits.
6. Avoid changing historical configuration fingerprints accidentally when adding fields. Preserve valid resumes for all existing methods. No token-cache version bump is needed unless inspection finds genuinely missing evidence; new derived artifacts have their own version.

Acceptance: uninterrupted and interrupted/retried fixture updates use identical frozen arrays; altered masks, reordered arrays, corrupt shards, mismatched critics, and wrong method resumes fail closed.

## Phase 4 — PPO token advantages in both actor paths

1. Add an explicitly validated token-advantage representation rather than relying on permissive scalar/list coercion. Update `train_step.py` dispatch and Transformers/Ray fixed-advantage interfaces.
2. Transformers: align each advantage with cached labels and completion masks, pad with zeros, and apply the existing clipped surrogate elementwise. Retain scalar broadcasting for legacy estimators only.
3. verl/FSDP: fill `DataProto` response advantages at the exact shifted positions used by response masks and old log probabilities. Preserve full interval inputs, remove-padding reconstruction, sequence parallelism, and minibatch packing/order mappings.
4. Distributed alignment duplicates have zero response masks, zero advantages, and no metric weight. Do not drop intervals for divisibility or because an interval's average advantage happens to be zero.
5. Compute token-mode diagnostics from masked token arrays; remove scalar-only `float(advantage)`, scalar zero tests, and scalar averaging assumptions on this path. Do not introduce group normalization over tokens or intervals.

Acceptance: a fixture with distinct token advantages matches a hand-computed PPO loss and gradient; constant token arrays reproduce the existing scalar objective; padding, DP/SP layouts, and token reordering preserve alignment. Verify on the pinned real verl worker, not only mocks.

## Phase 5 — BrowseComp-Plus warmup, configuration, and evaluation

1. Add `configs/train/compact_separate_token_mc_ppo.yaml` based on the current BrowseComp-Plus separate-MC preset. Explicitly set `benchmark.name: browsecomp`, signed rewards, `token_mc_value`, separate critic, two bins, pre-fit timing, and all assistant tokens enabled. Give it a fresh experiment/output root.
2. Preserve the fixed split seed and 730 training / 100 evaluation questions; existing local decrypted dataset, corpus/index, retrieval configuration, actor and compatible critic tokenizer, judge, sampling policy, and context/generated-token budgets. Keep the default 50 training tasks with four attempts and the existing production actor FSDP topology. Do not invent a new benchmark name for BrowseComp-Plus.
3. Configure critic batches for full intervals instead of copying the interval-prefix preset's batch sizes 8/32 and 8192-token budget. Begin with fit/eval microbatch one, activation checkpointing, and a batching budget consistent with the 27000-token training ceiling. Record measured feasibility on the configured hardware.
4. Extend automatic value pretraining to the token estimator. Preserve its 50-question/four-attempt collection and query-level 40/10 fit/validation split. Train all eligible tokens, select by rollout/interval/token-weighted validation Brier score, keep the actor unchanged, and bind warmup artifacts to token granularity. Retain the existing MC warmup behavior for interval GAE; do not globally replace its estimator with GAE.
5. Preserve explicit `training.value_pretraining.enabled=false` for a cold-critic control. Warmup must complete before checkpoint-zero online collection, or fail clearly; never silently substitute an interval-pretrained critic.
6. Extend same-policy evaluation and `evaluate_value.py` to token outputs. Report weighted CE/Brier/calibration and explained variance (with undefined cases handled), plus token counts and strata for research/tool calls, summaries, and final answers when authoritative span metadata is available. Do not parse and re-render text to create missing span metadata.
7. Keep held-out splits at query level and distinguish greedy benchmark evaluation from sampling-policy calibration probes. Report rollout counts separately from correlated token counts. Keep benchmark evaluation data out of warmup selection and fitting.
8. Add a matched pre-fit, two-bin interval-MC control with the same critic backbone, warmup budget, optimizer reduction, actor objective, and data split. Compare task success at matched generated-token budgets; report extra critic compute and memory separately.

## Phase 6 — Validation and release gates

Extend existing tests in `test_value_config.py`, `test_value_model.py`, `test_separate_value.py`, `test_trainer.py`, `test_verl_ray_trainer.py`, `test_value_pretraining*.py`, `test_value_diagnostics.py`, `test_train_step.py`, and `test_train_iterations.py`; add focused token-value tests where clearer.

Required checks:

- Exact token alignment, causality, CE/expected-value math, weighted losses, vector PPO advantages, and reduction invariance.
- No compaction, one compaction, repeated compaction, forced answers, malformed tool output, and malformed/empty/over-limit summaries. All old-interval sampled tokens remain eligible with the final rollout reward; copied summaries remain masked; failed summaries have no successor.
- Successful compaction changes only the active interval; it does not create a new reward horizon. Hash original interval evidence before/after training extraction and compare.
- History rewrite, missing exact IDs/log probabilities, invalid tokenizer mapping, overlength intervals, and malformed derived artifacts fail closed with no reconstructed substitute.
- Empty sampled spans, all-success/all-failure task groups, zero advantages, unequal interval/token counts, and zero-weight distributed padding.
- Warmup best-checkpoint selection, interrupted warmup, cold start, two online updates with resume, matching behavior/source/fitted critic identities, and legacy estimator regression coverage.

After CPU tests, run a small real BrowseComp-Plus training-subset collection through retrieval, judge, cache, critic fit, PPO, checkpoint save, and resume. Use a clearly labeled smoke config with small task/attempt counts and thresholds that exercise compaction; never report it as the full benchmark result. Run a bounded production-length memory/throughput probe, then one production-config iteration and explicit final-checkpoint evaluation. Check active GPU allocations before launching and keep smoke artifacts separate from existing runs.

Release requires finite real token-critic and PPO updates, a complete reloadable actor/critic pair, matching frozen-advantage digests across retry, all required interval cases, and documented peak GPU memory/runtime. CPU tests alone do not establish production support.

## Documentation and launch deliverables

Update README with the method, two-bin semantics, warmup behavior, memory differences, resume incompatibilities, and the matched control. Document this command only once the preset and integration exist (paths are absolute here for clarity):

```bash
/124090467/self-summarization-agent/scripts/train_iterations.sh \
  --iterations 1 \
  --latest-root /124090467/self-summarization-agent/experiments/token-mc-bcplus \
  --config /124090467/self-summarization-agent/configs/train/compact_separate_token_mc_ppo.yaml
```

Configure `experiment.output_root` consistently with the new run and use the existing explicit final-checkpoint evaluation flow. Save the effective config, split/provenance identities, validation results, and performance measurements alongside the run.

Implementation order is phases 1–4, then warmup/evaluation/configuration in phase 5, followed by the complete phase-6 acceptance run. Work is complete when the standard BrowseComp-Plus launcher can warm up, train, evaluate, and resume the new method without changing append-only collection or existing methods.
