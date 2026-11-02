# SWE-Dev environment integrated with self-summarization training

Status: revised implementation plan, 2026-11-02. Reviewed against the working tree at base commit `a5f9263`, including existing uncommitted training changes. This document changes the plan only; it does not claim SWE-Dev tasks or training runs have been implemented or validated.

## 1. Objective and architectural decision

Add reproducible SWE-Dev coding tasks to this repository's existing self-summarization training pipeline. Completion means collecting fresh on-policy terminal rollouts, verifying the resulting code, and performing training updates with GRPO and the existing segment-value methods.

Reuse Harbor for environments and verification, the project's `terminus-kira-tito` scaffold for terminal actions, and `EpisodeRuntime` for all policy generations and compaction. Keep the existing interval records, token ledger, cache, and trainers authoritative. Do not introduce a second Terminus-2 agent loop, a second token renderer, or a parallel generation-based training format.

SWE-Dev uses `0/1` training rewards: `1` for eligible success and `0` for task failure or penalized malformed output. Infrastructure failures remain excluded. This user-selected reward scale requires explicit value-model and validation support; it does not change the append-only token or interval contract.

The implementation includes environment construction and integration with existing optimization methods; it does not implement a new RL algorithm. Runtime-controlled, trainable self-summarization is part of the required path, alongside a no-compaction comparison.

The governing contract is [AGENTS.md](../../AGENTS.md) and the [append-only interval design](../superpowers/specs/2026-07-31-append-only-compaction-interval-design.md). No rollout-format redesign is authorized.

## 2. Review of the supplied plan

| Supplied proposal | Revision and reason |
| --- | --- |
| Build a standalone environment and connect a trainer later | Integrate collection, verifier rewards, cache construction, critic scheduling, actor update, and checkpoint resume in this repository. |
| Reuse Terminus-2's execution loop and stopping conditions | Reuse the existing Terminus-KIRA adapter, whose policy loop already runs through `EpisodeRuntime`. Its completion and forced-completion behavior are part of the experiment identity. |
| Build a token-native model interface and incremental renderer | Reuse `IntervalTokenLedger`, `TokenRequest`, `QwenAgentTokenRenderer`, and token-input generation backends. Add tests for the SWE environment, not another implementation of TITO. |
| Make OpenEnv capture a prerequisite | Remove it from the production dependency chain. Capture can be a separate diagnostic experiment; recording per-call tokens does not enforce this project's immutable cross-call prefix or compaction boundaries. |
| Initially support only non-compacted traces | Require no-compaction, one-compaction, multiple-compaction, and failed-summary coverage before training integration is accepted. |
| Use standalone `GenerationRecord` and `EpisodeRecord` training schemas | Extend existing rollout metadata. Training continues to consume `trajectory_records`, one record per interval; `turn_records` remain diagnostic. |
| Use raw binary reward directly | Use `0/1` for SWE-Dev training as requested. Preserve runtime failure precedence with reward `0`, and adapt critic support/validation to `[0,1]`. |
| Require collection logprobs for every accepted generation | Exact IDs are mandatory. Reuse native caches only for validated raw policy logprobs; otherwise rescore those exact IDs under the frozen collection checkpoint. |
| Treat downloadable conversations as executable tasks | Add a source-recoverability gate before promising a task count. Conversations are neither executable task specifications nor on-policy exact-token RL data. |
| Finish when trajectories can be exported | Finish after real verifier-backed traces drive updates and resumes for the supported training modes. |

## 3. What already exists, and what still needs work

All source paths below are relative to `src/self_summarization_agent/`.

| Existing component | Reuse and integration work |
| --- | --- |
| `benchmarks/base.py:AgentScaffold` | Stable prompt/tool/parse/execute/finish interface. SWE-Dev should use the existing terminal scaffold. |
| `benchmarks/terminal_bench/{harbor_agent,scaffold,tools}.py` | Harbor external agent, persistent tmux session, linked tool observations, completion confirmation, cancellation, raw observation artifacts, and runtime-driven compaction. |
| `benchmarks/terminal_bench/{dataset,collection,verification}.py` | Local task paths, grouped attempts, fresh environments, verifier reward mapping, collection/resume, and merged collection. Generalize dataset identity and dispatch instead of copying the collector. |
| `runtime.py`, `token_stream.py`, `token_renderer.py`, `generation.py` | Own immutable intervals and all trainable generations, including summary and forced-completion output. |
| `trajectory.py`, `cache_step.py` | Schema v3 interval extraction, current token cache v7, exact token validation, native raw-logprob caches, and exact-ID policy rescoring. Do not copy older README references to cache v6. |
| `trainer.py:compute_group_advantages`, `verl_ray_trainer.py` | Normalize GRPO rewards once per distinct rollout within each task, then broadcast the advantage to its intervals. |
| `value_model.py`, `critic_step.py`, `separate_value*.py`, `segment_returns.py` | Shared value-MC, separate MC/PPO, and separate segment GAE/PPO, with interval-start anchors and critic provenance. |
| `iteration_launcher.py`, `merged_collect_step.py`, `judge_step.py`, `train_step.py` | Existing staged collection, reward import, cache, critic/actor phases, and checkpoint handling. Extend all benchmark branches consistently. |
| `collection_contract.py` | Resume identity already includes terminal task content, scaffold, runtime, and tokenizer information. Add explicit SWE manifest, image, and verifier provenance. |

Concrete gaps found in the current code:

- `BenchmarkConfig.name` and several dispatch sites recognize only `browsecomp` and `terminal-bench`; the terminal collector hard-codes the emitted benchmark label.
- Local Harbor task directories already work. They are a useful plumbing smoke test, but the current loader does not provide a dedicated SWE-Dev manifest or enforce train/eval disjointness.
- Launcher expected-count helpers derive membership from generic count/BrowseComp split settings. An explicit terminal evaluation task list alone does not establish the evaluation count.
- `run_launcher.py` still loads BrowseComp examples and retrieval. Use `rollout_collection` for SWE evaluation initially; do not document `run_launcher` as compatible until it dispatches correctly.
- Existing terminal tests cover verifier/cache flow and CPU GRPO/shared value-MC updates. They do not establish complete SWE-Dev or separate GAE integration.
- `LockedPolicy` serializes inference calls. Concurrent Harbor trials overlap environment work, but this is not cross-episode model batching.
- Final repository patch/snapshot export and SWE-specific verification isolation need implementation.
- Current reward conversion and value code assume `-1/+1` outcomes and `[-1,1]` value support. SWE-Dev needs a benchmark-scoped `0/1` reward contract threaded through shared and separate critics, GAE, diagnostics, and checkpoint validation; changing the verifier alone is insufficient.
- Value pretraining is explicitly restricted to BrowseComp by `load_train_config`. Leave it disabled for initial SWE configurations; ordinary online separate-critic fitting is a distinct, supported path.
- Existing documentation reports terminal transport validation, while Docker/oracle execution was unverified on that host. Recheck current infrastructure rather than assuming the environment works.

Dependency baseline: Python 3.12, the existing locked environment, and optional `terminal-bench` extra with Harbor `0.1.44`. Keep these pins unless a demonstrated incompatibility requires a separately tested update.

## 4. Target data flow and ownership

```text
Pinned source data + recovered executable task specifications
    -> validated SWE-Dev task bundles + disjoint manifests
    -> shared Harbor collector: K fresh attempts per training task
         Harbor: environment allocation / teardown / verifier lifecycle
         Terminus-KIRA: terminal operations / observations / completion checks
         EpisodeRuntime: policy generation / TITO ledger / compaction
    -> raw rollout rows + immutable interval evidence + code artifacts
    -> verifier reward mapping + training eligibility
    -> exact-token cache under the collection policy checkpoint
    -> group by task / retain rollout identity and ordered interval chain
         GRPO
         shared interval value-MC
         separate MC critic -> fixed advantages -> PPO
         pre-fit segment GAE -> frozen targets -> critic fit -> PPO
    -> existing checkpoint, evaluation, and resume pipeline
```

Each rollout owns one environment throughout all its intervals. Compaction changes model context only: repository edits, shell state, processes, and completion-confirmation state persist. Neither compaction nor critic evaluation resets or clones the environment.

The value state is the exact model-visible interval-start prompt. It includes the original task and, after compaction, the wrapped summary. It is a partial observation of the live coding environment, not a privileged repository snapshot. Do not inject hidden tests, the final patch, future observations, or verifier results into critic input.

## 5. Phase A — Establish recoverable, executable tasks

### A1. Audit source availability before building images

Pin the SWE-Dev repository (https://github.com/THUDM/SWE-Dev) commit and Hugging Face dataset (https://huggingface.co/datasets/zai-org/SWE-Dev-train) revision. Inspect each dataset split and record its schema, row origin, available IDs, and recoverable source links.

The public dataset preview exposes conversation `input`, origin `from`, outcome/count fields, and `instance_id`; preview rows include multiple origins and blank instance IDs. This does not establish availability of a complete repository/base-commit/test specification for every conversation. The upstream repository provides a test-generation pipeline; it is not proof that every released conversation has a recoverable executable fixture. See the [dataset](https://huggingface.co/datasets/zai-org/SWE-Dev-train) and [upstream repository](https://github.com/THUDM/SWE-Dev).

Produce a coverage report: total rows, unique identifiable tasks, duplicate trajectories, recoverable task specifications, validated tasks, and exclusions by reason. Do not infer a base commit or trustworthy hidden test solely from an assistant transcript. If source assets cannot be recovered, exclude that task; newly synthesized tests form a separately versioned dataset variant.

### A2. Version a manifest independent of policy transcripts

Each validated task entry should contain:

- Namespaced stable `task_id`, original source instance ID/origin, source revision, repository URL, exact base commit, and original task instruction digest.
- Task bundle path and digest, image reference pinned by digest, Dockerfile/build-context digest, and dependency/setup provenance.
- Trusted verifier version/digest, fail-to-pass and pass-to-pass test specifications, timeout/resource profile, and oracle-validation report reference.
- Reference patch availability and digest, stored outside agent-visible assets.
- Split membership, split-construction version/seed, and duplicate/leakage-group identity.

Retain the provenance of mixed upstream origins; do not silently call every downloaded row an independently sourced SWE-Dev task. Assign splits at the task/issue identity level before choosing trajectories or attempts. Check repository/issue/patch overlap with held-out SWE-Dev and any intended SWE-bench evaluation set. Record remaining overlap limitations.

Training consumes fresh rollouts from these tasks. Released SFT/RFT conversations and their historical rewards must never be converted into current on-policy cache entries.

### A3. Construct and validate Harbor bundles

Keep the proposed Harbor layout (`task.toml`, `instruction.md`, `environment/Dockerfile`, trusted `tests/`, optional `solution/`). Use `/app/repo` consistently in images and task instructions. Package only agent-visible assets into the rollout image; keep oracle solutions and hidden verification material outside its build context and accessible mounts.

Prebuild images and restore the exact base state for every attempt. Eligibility requires successful setup, the specified baseline failures, passing baseline regression tests, and successful reference-patch verification when a reference is available. For the initial pilot, select tasks with reproducible oracle validation; tasks lacking an oracle require a separately documented validation basis.

Repeat baseline and oracle verification to reject flaky tasks. Save complete commands, test selection, environment identity, and logs. An empty/misdiscovered test suite cannot pass.

**Exit:** a source coverage report and 10–20 reproducibly executable tasks, or an explicit smaller validated set with documented source blockers. Task recoverability is the first feasibility gate.

## 6. Phase B — Extend the existing Harbor path for SWE-Dev

### B1. Add dataset identity without another runtime

Introduce `benchmark.name: swe-dev` and proposed `benchmark.train_manifest_path` / `benchmark.eval_manifest_path` fields. These fields do not exist today and require loader and validation changes.

Implement SWE-Dev manifest resolution under `benchmarks/swe_dev/`. Extract reusable task collection/verification helpers into `benchmarks/harbor_common/` only where needed, preserving existing Terminal-Bench entry points and behavior. Keep the existing terminal agent and scaffold importable; moving all terminal code is unnecessary.

Use a small shared dispatch helper for Harbor-backed benchmarks in configuration, collection, merged collection, judging, cache lineage, and launcher count resolution. Remove hard-coded output labels from the reusable collector. Reject missing or overlapping SWE manifests rather than falling back to official Terminal-Bench tasks.

An early smoke test can use existing `benchmark.name: terminal-bench` with explicit local task paths. Label this as a local Harbor plumbing test, not final SWE-Dev support.

### B2. Preserve lifecycle and bound observations

A fresh environment is allocated per task/attempt; image layers may be shared, writable workspaces may not. Apply CPU, memory, storage, network, command, agent, verifier, and collection limits at their actual enforcement layers. Record the resolved values.

Retain existing command fidelity, linked `role: tool` observations, and `task_complete` confirmation behavior. Never replay a state-changing command after an ambiguous timeout. Inference retry may reuse an identical committed request only if no completion was accepted.

Bound a new observation before its first append, reserving token space for a legal boundary. Existing `max_output_chars` is a useful coarse bound, not a token-capacity guarantee. Preserve raw output separately and include an explicit truncation notice in the newly constructed observation. Never shorten historical observations.

Start with the stable `text-only` tool profile. A future local vision profile must remain separately labeled and use the existing frozen external-helper contract.

### B3. Capture final state and isolate verification

Capture the candidate changes before cleanup and before a verifier can modify them. Save the base commit, tracked changes, deletions, additions/untracked deliverable files, file modes, and binary changes as appropriate, plus a content digest. A plain `git diff` alone misses untracked files. Preserve a reproducible candidate snapshot/export as well as a readable patch.

For this SWE profile, verify in a fresh trusted environment built from the pinned base plus the candidate changes. This requires an explicit Harbor-compatible verification hook/adapter; stock verification of the live agent workspace does not establish that isolation.

Keep hidden tests, oracle patches, verifier executables, and reward output outside agent control. Freeze or stop agent execution before verification; no background agent process may inspect or alter verification. The trusted verifier controls test discovery and expected counts and writes the reward from trusted logs. Agent changes to tests/configuration must not disable required checks or forge a pass.

Report patch-application or build failures caused by candidate changes as task failures. Distinguish them from image-fetch failures, host/resource failures, unavailable dependencies in a supposedly pinned image, and verifier infrastructure errors. Version this classification. Quiesce/snapshot, verify, export evidence, then clean up on all paths.

**Exit:** baseline/oracle/incorrect-patch trials, concurrent independent attempts, and a successful compaction trial all preserve the correct filesystem and verifier isolation.

## 7. Phase C — Preserve the existing interval and token contract

No new message reconstruction layer is needed. For every generation within an interval, validate the actual token request and returned evidence:

```text
P[k+1] = P[k] || C[k] || E[k]
```

`C[k]` is the exact sampled completion; `E[k]` contains only newly encoded observations, runtime controls, boundaries, and the next assistant header. Keep sampled stop tokens, raw completion text, reasoning, tool metadata, exact prompt/completion/full IDs, and append-time masks.

A successful compaction must:

1. Append the terminal user summary request to the old interval and disable tool use for that generation.
2. Append the complete raw summary completion, including thinking and summary tags; validate it and extract the first complete summary body after completed thinking.
3. Finalize the old immutable interval.
4. Initialize the successor with byte-identical system instructions and original task instruction, followed by a conditioning-only user message containing the wrapped extracted summary.

Keep no raw assistant/tool tail in the successor. Preserve the live environment and scaffold state. Empty, malformed, over-limit, wrongly closed thinking, or unsupported-termination output must not install a successor state. Keep its raw sampled output and the configured penalty.

Generated-token budget exhaustion takes precedence over compaction. Forced completion appends the existing terminal system control and retains the final generated output; it must not execute further shell commands. Context overflow rejects/terminates rather than truncating the prefix.

Every sampled assistant token is trainable, including reasoning, commands, summaries, and final/forced completions. Initial prompts, copied summaries, tool results, and controls are conditioning-only. Set `training.train_compaction_tokens: true` in the SWE methods.

Backend token mismatches fail closed; protocol errors abort collection. Retain valid-prefix diagnostics on runtime history-rewrite termination, and never repair sampled output. Extraction still verifies saved evidence and rejects rewritten histories. In particular, no summary-success exception may conceal a rewrite inside the old interval.

**Exit:** existing token-contract tests plus SWE fixtures pass for zero, one, and multiple compactions, failed summaries, and malformed completion of every generation kind.

## 8. Phase D — Rewards, records, cache, and selection

### D1. Use binary 0/1 SWE-Dev training rewards

| Event | Stored raw verifier result | Training treatment |
| --- | --- | --- |
| Correct patch, required tests pass | `1` | `1` if runtime status is eligible |
| Valid attempt, correctness/regression fails | `0` | `0` |
| Penalized runtime output with no verifier exception | Preserve any result | SWE runtime failure reward `0`; a passing patch does not override it |
| Infrastructure or verifier exception | Preserve evidence; no invented correctness result | Exclude from training; no reward imputation |
| Missing/invalid verifier reward on an otherwise normal attempt | Missing/invalid | Infrastructure error; exclude |
| History rewrite/corrupt evidence/incomplete interval chain | Preserve diagnostics | Reject for SWE training selection; never silently salvage a partial GAE chain |

Generalize `apply_verifier_reward` with an explicit reward profile: SWE-Dev uses `0/1`, while existing BrowseComp and Terminal-Bench defaults retain `-1/+1`. Preserve and test the current precedence: a penalized runtime outcome remains a failure when no verifier exception exists, even if a valid raw reward is absent. For SWE-Dev that failure receives `0`, including malformed actions and failed summaries; it is still a trainable negative outcome, not an infrastructure exclusion. Record runtime status, verifier status, benchmark pass, and training eligibility separately. Apply the profile consistently to runtime reward helpers and final verifier conversion so provisional `-1` labels cannot enter SWE training artifacts.

Broadcast one terminal training reward to every eligible interval in that rollout using the existing `turn_rewards` field. Its historical name does not mean per-tool rewards. Do not add summary bonuses, intermediate test rewards, token penalties, or per-command targets in this integration.

### D2. Extend metadata, preserve training evidence

Retain existing `query_id`, `rollout_index`, `policy_checkpoint_id`, `sampling_profile_id`, `collection_profile_id`, `trajectory_records`, `turn_records`, and `turn_rewards`.

Map `query_id` to the unique namespaced task ID. Within the collection lineage, preserve `rollout_id = query_id + ":" + rollout_index` and each interval's `turn_id`, `segment_index`, `termination_kind`, `turn_ids`, and `prefix_summary_turn_id`. Include lineage/checkpoint identity when combining data across iterations.

Add versioned SWE metadata for task/source/image/verifier digests, environment limits, candidate export/patch digests and paths, trial identity, timings, and failure classification. Keep privileged oracle/test contents out of model observations.

Current evidence uses trajectory schema `3`, TITO collection version `3`, contract `qwen-agent-tito-v2`, and training cache `7`. Add environment metadata outside the token evidence when possible. Import version constants rather than duplicating them; version any necessary schema change explicitly without rewriting archived intervals.

### D3. Use the existing exact-token cache

For final interval IDs `F` and assistant mask `M`, the cache must retain:

```text
input_ids       = F[:-1]
labels          = F[1:]
completion_mask = M[1:]
state_prefix_length = len(first_generation.prompt_token_ids)
```

Native cache construction is valid only when all sampled-token probabilities align with their exact interval positions and have `logprobs_mode: raw_logprobs`. Its provenance is `vllm_raw_rollout`. Sampling-temperature/top-p/constraint-adjusted logprobs must not be mislabeled as raw policy probabilities.

If those raw probabilities are unavailable, use the existing `policy_rescore` path with the unchanged behavior checkpoint and exact collected IDs. This is a probability computation, not retokenization. Do not fabricate missing stop-token logprobs or rescore under an updated actor. Reject over-length intervals; validate the configured training sequence ceiling against the runtime/inference ceiling before launch.

### D4. Make rollout eligibility and group completeness explicit

Preserve all raw attempts and failures. Create an auditable training-selection view without modifying interval evidence. For the initial controlled comparison, require all configured attempts of a selected task to be eligible; omit incomplete task groups from the update and report the omission. Apply the same group-selection policy to the compared methods. Valid wrong answers and malformed-output penalties remain eligible; they are not infrastructure exclusions.

This complete-group rule is proposed SWE selection behavior, not an existing collector guarantee. Keep it versioned and localized to SWE selection; do not silently change BrowseComp training. Do not retry based on correctness. Infrastructure retries, if later enabled, need bounded, recorded retry semantics and fresh sandboxes.

Reject a rollout as a unit when integrity validation or missing intervals would break its ordered chain. GAE explicitly rejects `history_rewrite_detected`; generic GRPO/MC code may otherwise see a retained valid prefix, so selection must reconcile this before dispatch.

Resume counts must distinguish completed raw attempts, eligible attempts, selected groups, and optimizer samples. Empty selected batches must produce an explicit no-update/error result, never a misleading successful training checkpoint.

**Exit:** verifier -> selection -> exact cache -> sample extraction passes for successes, failures, runtime penalties, infrastructure failures, and rejected chains.

## 9. Phase E — Wire all existing training methods

### E0. Carry the reward scale through value training

Add a proposed `benchmark.reward_profile: binary_zero_one` configuration for SWE-Dev and resolve it once into the training reward contract. Its outcome classes are `[0,1]`, malformed-output reward is `0`, and critic support is `[0,1]`. Keep the existing signed profile as the default for other benchmarks. These are new capabilities, not currently accepted configuration fields.

Required changes span `rewards.py`, verifier conversion, `value_model.py`, `segment_returns.py`, `separate_value.py`, `value_diagnostics.py`, and all shared/separate trainer and worker call sites:

- Validate terminal labels against the configured classes; retain the same label on every interval in a rollout. A zero reward must not be mistaken for a missing reward or an ineligible sample.
- For binary heads, use `V(s) = P(training reward = 1 | s)` on SWE-Dev, rather than `P(success) - P(failure)`. Map labels `0/1` to the failure/success classes for cross-entropy. A zero-initialized binary head now predicts `0.5`.
- For categorical heads, use support `k / (num_bins - 1)`, with targets and predictions in `[0,1]`; adapt two-hot encoding, expectation, range checks, and outcome log loss. MC and segment-GAE advantages lie in `[-1,1]`; do not clip away negative advantages merely because rewards are nonnegative.
- Update diagnostics to use `value` directly as the predicted success probability, reward directly as the binary target, and `0.5` as the classification threshold. Preserve rollout-normalized weighting.
- Record the reward profile/support in collection lineage, training selection, value-head manifests, method specifications, frozen targets, and actor/critic checkpoint pairing. Reject incompatible resume artifacts. Legacy signed critics and targets must not be silently reinterpreted as probability-valued critics or rewritten in place; a fresh SWE initialization may reuse compatible backbone weights with a newly initialized head.

The GRPO normalization formula already accepts `0/1` rewards; its normalized advantages are invariant to this affine change apart from the existing epsilon. MC/PPO advantage magnitudes do change, so log them and evaluate update/learning-rate settings on the new scale. Compute GAE anew from `0/1` outcomes and the corresponding value predictions; do not rescale an old frozen-target artifact, particularly with lambda < 1.

### E1. GRPO

Use `training.advantage_estimator: group_relative` with value disabled and `training.group_size = K` independent fresh attempts per task; begin with `K = 4`.

For each task, compute mean and population standard deviation over the distinct rollout rewards. The existing implementation uses:

```text
A[i] = (R[i] - mean(R)) / (std(R) + 1e-6), when std > 0
A[i] = 0, when all rewards are equal
```

Broadcast `A[i]` to all trainable assistant tokens in all intervals of rollout `i`. A rollout with five intervals contributes one reward to normalization, just like a rollout with one interval. Preserve the trainer's existing masked actor-loss reduction; do not claim this also makes actor loss equal per rollout.

Log zero-variance groups and effective updates. Sparse binary coding rewards can produce many all-failure groups; report this instead of changing rewards or selectively retaining successes.

### E2. Shared value-MC

Use `compaction_mc_value`, `value.enabled: true`, `value.architecture: shared`, and `state_anchor: first_generation_prompt_end`.

The value prediction is anchored at the last token of the exact first-generation prompt of each interval, before that interval's new assistant output. For SWE-Dev, `R` is `0/1` and `V_old` is in `[0,1]`. Freeze old values for actor advantages `A[i,j] = R[i] - V_old(s[i,j])`; train the binary value head toward the shared terminal outcome using the reward-profile-aware path.

Preserve rollout-normalized critic weighting: each interval of rollout `i` has weight `1 / (N * m[i])`, where `N` is the number of rollouts and `m[i]` its number of intervals. More summaries must not create more independent critic outcome weight.

### E3. Separate value-MC/PPO

Use `compaction_mc_value` with `value.architecture: separate` and an explicit critic model path. Reuse `critic_step`, `SeparateValueTrainer`, `fitted_advantages.json`, and the existing actor update.

The existing separate MC preset uses `baseline_timing: post_fit`: fit the critic to SWE-Dev terminal `0/1` outcomes, then freeze `R - V_fitted(s)` for PPO with values on `[0,1]`. The implemented `pre_fit` option is a distinct experiment and must be labeled as such. Do not describe all MC paths as using old pre-fit values.

The critic uses stored prefix IDs, not a newly rendered prompt. Verify actor/critic tokenizer ID and special-token compatibility, context capacity, rollout-normalized critic loss, and matching actor/critic checkpoint provenance.

### E4. Separate segment GAE/PPO

Use `compaction_gae_value`, separate critic, `baseline_timing: pre_fit`, and 51 bins with lambda 0.95 as in the current preset, but with SWE-Dev categorical support `[0,1]`. Extend `ordered_rollout_chains`, `segment_gae`, and frozen-target provenance for this reward profile.

Across the ordered intervals of one rollout, with gamma = 1:

```text
r[j]     = 0 for nonterminal intervals; terminal rollout reward for the last
V[next]  = 0 after the terminal interval
delta[j] = r[j] + V_old(s[j+1]) - V_old(s[j])
A[j]     = delta[j] + lambda * A[j+1]
target[j]= V_old(s[j]) + A[j]
```

Freeze old values, lambda-return targets, and actor advantages before fitting the critic. For SWE-Dev, terminal reward is `0` or `1`, values and targets are in `[0,1]`, and the terminal bootstrap remains zero. Fit categorical targets with two-hot encoding on the configured support, then run the actor with those frozen advantages. Never recompute them from the fitted critic.

The repeated rollout reward in stored `turn_rewards` is an outcome label. It is counted only once as a terminal transition reward in GAE. Validate consecutive segment indices, predecessor-summary links, and terminal kinds; a missing interval cannot be bridged.

A single-interval rollout reduces to `R - V_old(s)`. With lambda = 1, complete chains reduce to the MC return. Test these properties and a multi-interval lambda < 1 example.

### E5. Orchestration, evaluation, and resume

- Use the existing iteration sequence: collect under a fixed checkpoint -> import verifier outcomes -> selection/cache -> optional critic phase -> actor update -> checkpoint.
- Dispatch SWE judging to recorded verifier results before any BrowseComp answer loading. No answer-judge model or retrieval worker should start.
- Derive expected task/attempt keys from the resolved SWE manifests and selected counts, and use identical selection logic in collection and resume. Keep evaluation membership/order fixed across iterations; the current terminal collector can randomly sample evaluation tasks when a count is set, so define stable selection explicitly.
- Maintain separate training and evaluation sampling profiles; do not assume `dataset.train_limit/eval_limit` partitions Harbor tasks.
- Retain both actor and critic optimizer/checkpoint state and the existing pair manifest for separate methods. Reject stale caches, mismatched behavior/critic identities, incompatible targets, or changed task/image/verifier/scaffold/runtime profiles.
- Ensure same-policy critic evaluation sees SWE verifier-backed eval artifacts at the filenames the current critic diagnostics expect. Report skipped/missing diagnostics explicitly.
- Leave `training.value_pretraining.enabled: false`. A future warm-start phase must generalize the BrowseComp-only selection/collection code and tests; removing the config guard alone is insufficient.

**Exit:** one finite update and a valid resume for GRPO, shared value-MC, separate MC/PPO, and separate GAE/PPO, including a multi-interval rollout in each mode.

## 10. Proposed files and configuration

New files are proposals, not existing interfaces:

```text
docs/plans/2026-11-02-swe-dev-training-environment.md
docs/swe-dev.md
src/self_summarization_agent/benchmarks/swe_dev/
    dataset.py        # versioned manifest loading, splits, task identity
    task_builder.py   # recover/convert/build/validate task specifications
    verification.py   # trusted candidate export and isolated verification hooks
    selection.py      # explicit SWE training eligibility and complete groups
src/self_summarization_agent/benchmarks/harbor_common/
    collection.py     # shared orchestration extracted only as needed
configs/train/
    swe_dev_grpo.yaml
    swe_dev_value_mc.yaml
    swe_dev_separate_mc_ppo.yaml
    swe_dev_separate_gae_ppo.yaml
configs/eval/
    swe_dev.yaml
    swe_dev_no_compaction.yaml
tests/test_swe_dev_dataset.py
tests/test_swe_dev_verification.py
tests/test_swe_dev_training.py
tests/fixtures/swe_dev/<small-local-task>/
```

Reuse `benchmarks/terminal_bench/{harbor_agent,scaffold,tools}.py`. Targeted edits also include `config.py`, `collection_contract.py`, `rollout_collection.py`, `merged_collect_step.py`, `judge_step.py`, `cache_step.py`, `iteration_launcher.py`, and launch-script validation where necessary. Keep runtime/token changes limited to the reward-profile plumbing and demonstrated missing extension hooks; adapt shared and separate value trainers/workers as specified in E0 without changing the interval representation.

Keep generated task bundles, images, datasets, trial logs, and large patches in configurable artifact storage rather than source control. Commit small deterministic fixtures and manifest/schema examples.

Proposed common configuration excerpt, valid only after the new fields/dispatch are implemented:

```yaml
benchmark:
  name: swe-dev
  reward_profile: binary_zero_one  # proposed; outcomes and critic support are 0/1
  dataset: swe-dev-executable-v1
  dataset_revision: <pinned-source-revision>
  train_manifest_path: <validated-train-manifest>
  eval_manifest_path: <validated-eval-manifest>
  environment: docker
  image_profile: text-only
dataset: {}
retrieval: {}
collection:
  train_task_count: 8
  eval_task_count: 4
rollout:
  backend: vllm_offline
  require_exact_token_ids: true
  max_concurrent_episodes: 4
  overlap_judge: false
judge:
  enabled: false  # verifier conversion still runs through benchmark dispatch
training:
  group_size: 4
  train_compaction_tokens: true
  advantage_estimator: group_relative
  value:
    enabled: false
  value_pretraining:
    enabled: false
```

Complete configs must also specify model/template/checkpoint, GPU placement, runtime token budgets, trainer/backend settings, and compatible sequence ceilings. The pilot counts above assume sufficiently large disjoint manifests. Start with the existing locked vLLM token-input path; other engines are separate capability-tested profiles.

Hold task selection, policy initialization, sampling, tool schema, environment/verifier limits, generated-token budget, and context capacity fixed across method comparisons. A no-compaction run must still terminate legally at context capacity; compare and report the resulting budget differences rather than truncating its history.

## 11. Validation and delivery milestones

| Milestone | Acceptance evidence |
| --- | --- |
| M0 — source feasibility | Pinned source/schema audit, provenance coverage, missing assets, deduplication and split rules; no guessed executable metadata. |
| M1 — executable pilot | 10–20 validated tasks if recoverable; repeated baseline/oracle checks, pinned images, hidden-test isolation, candidate export. |
| M2 — shared collection | Dedicated SWE manifest/identity dispatch; independent grouped attempts; correct terminal lifecycle and verifier outcomes; no retrieval/judge workers. |
| M3 — interval integrity | Exact-token no-compaction, single/multiple compaction, malformed summaries, forced completion, and immutable archived intervals. |
| M4 — training integration | Verifier -> selection -> cache -> GRPO/shared MC/separate MC/separate GAE updates with correct rewards, anchors, weights, links, and checkpoint resume. |
| M5 — real pilot | Real Docker/oracle and token-capable vLLM runs; all accepted traces pass saved-record TITO verification; small GPU update/resume for each method. |
| M6 — measured scale | Resource-bounded concurrency, reproducible held-out evaluation, measured inference/environment bottlenecks, and operating documentation. |

Required test matrix:

- Task identity: duplicate/blank source IDs, disjoint splits, content/image/verifier changes, deterministic evaluation selection, and resume mismatch.
- Environment: clean reset, independent attempts, files/cwd/process/confirmation preservation across compaction, candidate additions/deletions, timeout classification, cancellation, cleanup, and verifier tampering.
- Tokens: multiline/Unicode observations, command bytes, linked call IDs, sampled end tokens, exact thinking closure, oversized observations, missing IDs, rewritten backend inputs, and over-length rejection.
- Boundaries: no compaction, one and multiple successful summaries, empty/malformed/oversized summaries, forced completion, simultaneous budget/summary trigger, and no tool execution during summaries.
- Training: mixed success/failure groups with unequal interval counts, all-equal GRPO rewards, complete-group exclusion after infrastructure failure, no reward imputation, summary-token masks, and unchanged interval data after caching.
- Critic: exact interval-start anchors, no future/verifier input, compatible tokenizer IDs, rollout-normalized weights, MC baseline timing, hand-computed GAE, lambda=1/single-interval reductions, and missing-chain rejection.
- Probability/checkpoint provenance: native raw-logprob versus exact-ID rescoring paths, shifted mask alignment, frozen targets, stale checkpoint rejection, and actor/critic resume pairing.
- Reward profiles: successful SWE attempts receive `1`; task failures and malformed summaries receive `0`; infrastructure errors remain excluded. Cover zero-valued sample retention, binary/categorical `[0,1]` support, probability diagnostics, negative advantages, and reward-profile resume rejection.
- Regression: existing BrowseComp and Terminal-Bench tests retain their signed reward/value defaults.

Use `uv run --locked --extra terminal-bench pytest` with focused existing modules (`test_terminal_bench.py`, `test_tito.py`, `test_rollout_collection.py`, `test_cache_step.py`, `test_iteration_launcher.py`, `test_segment_gae.py`, `test_separate_value.py`, and `test_value_training_integration.py`) plus the new SWE tests. Follow with real Docker/oracle checks and `scripts/verify_tito_records.py` on collected intervals. Synthetic/CPU tests are necessary but are not evidence of a working Docker/GPU pipeline.

Report task coverage and raw verifier pass rate separately from training reward/eligibility; include infrastructure errors, malformed outputs, history rewrites, dropped groups, zero-variance groups, actual actor updates, tokens, summaries, wall time, sandbox resource use, and critic diagnostics. Record both valid-attempt and all-attempt denominators so exclusions cannot inflate an unexplained headline rate.

Keep SWE-bench Verified and official Terminal-Bench 2.0 evaluation-only by default. Their adapters and provenance checks are later evaluation work, not prerequisites for the first SWE training update.

## 12. Definition of done and external references

The environment is integrated when reproducible SWE tasks produce verifier-backed, exact-token, compacted and non-compacted rollouts through the existing runtime, and those unchanged intervals successfully drive all four existing training profiles and resume their checkpoints. Exporting conversations alone is insufficient.

External references checked during this review:

- [SWE-Dev repository and construction pipeline](https://github.com/THUDM/SWE-Dev).
- [SWE-Dev training dataset and visible schema](https://huggingface.co/datasets/zai-org/SWE-Dev-train).
- [OpenEnv Harbor integration](https://huggingface.co/docs/openenv/environments/harbor): describes per-call token/logprob capture and a proxy. That capability is useful for auditing; it does not replace this repository's ledger or compaction contract. Its linked documentation is a moving main-version surface, not a dependency pin.

Local source code takes precedence over older design notes and README descriptions. Remaining uncertainties are recoverable SWE task assets and real environment/GPU execution; both have explicit acceptance gates above.
