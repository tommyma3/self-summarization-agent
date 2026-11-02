# Value pretraining before the first policy update

Date: 2026-10-30 (Asia/Shanghai)
Status: implemented and verified with bounded CPU tests; no production run launched.

## Objective and agreed settings

Add an optional, one-time pretraining stage for the separate critic. Keep the starting actor fixed, fit the critic to Monte Carlo outcomes, select its best validation checkpoint, then start the existing segment-GAE / dual-clip PPO loop on fresh rollouts.

| Setting | Initial implementation |
|---|---|
| Collection budget | 50 training questions × 4 rollouts = 200 rollout traces |
| Fitting / validation split | 40 questions / 10 questions: 160 / 40 rollouts |
| Critic training | Full backbone and head jointly from the first minibatch |
| Maximum epochs | 20 |
| Early stopping | Five consecutive epochs without validation improvement |
| Validation frequency | Every completed epoch, plus an epoch-zero reference |
| Selection metric | Rollout-weighted validation Brier score; lower is better |
| Backbone / head learning rates | `1e-5` / `1e-4` during pretraining |
| Rollout minibatch size | 32, using the existing prefix batching configuration |
| Initial fitting budget | Five optimizer steps per epoch; at most 100 steps |
| State / head representation | Existing exact prefix anchor and 51-bin head |
| Targets | Observed terminal reward, −1 or +1, for each eligible interval |
| Online training afterward | Existing pre-fit GAE, lambda 0.95, dual-clip PPO |

These are experiment settings, not established optima. Do not introduce a head-only phase, automatically increase collection, or balance the dataset by discarding failures.

The existing 730-question training / 100-question evaluation partition stays intact. The 10-question holdout is for selecting the initial critic during pretraining; it is not a new permanent benchmark split. Stop using its fixed-policy labels as validation once the actor changes. A persistent, refreshed stochastic critic-validation service is outside this change.

## State and trajectory contract

The new stage operates outside the actor's interval state machine. It must not add a runtime instruction, alter a summary boundary, or rewrite a collected interval.

- Read initial and post-compaction states from the authoritative exact-token prefix ending at `first_generation_prompt_end`. Never include future assistant output in a value input.
- Preserve system instructions, original queries, raw completions, linked native tool results, sampled IDs, assistant masks, and failed summary evidence.
- A successful compaction still finalizes the entire old interval before creating its successor. Pretraining does not change this transition.
- Retain existing penalty semantics for malformed outputs and failed summaries. Preserve history-rewrite exclusions and missing/mismatched exact-ID failures.
- Use a separate keyed MC-target view; do not change stored interval rewards or substitute targets into trajectory records.
- Give each rollout equal total critic weight, divided among its eligible interval states. Additional compactions must not increase its independent weight.
- Reused artifacts remain read-only. New cached artifacts may be derived through existing exact-token cache helpers; no historical prefix may be re-rendered or retokenized as a fallback.

## Repository integration decisions

1. Add settings under **`training.value_pretraining`**, outside `training.value`. Existing frozen targets fingerprint `asdict(config.training.value)`; adding default pretraining fields inside that object would unnecessarily invalidate historical artifacts.
2. Reuse `merged_collect_step` in train-only mode. Its `has_eval` condition already requires an evaluation output path, so omit all evaluation output arguments rather than modifying `dataset.eval_limit` or creating another dataset partition.
3. Reuse `SeparateValueTrainer` for fitting and prediction. A pretraining-local config with `fitting_epochs=1` allows an external epoch/validation/early-stopping loop without changing the normal online fitting algorithm.
4. Treat pretraining as MC fitting with explicit pretraining provenance. The first online GAE critic is initialized from those weights; this is initialization, not resuming an MC actor/critic pair as a GAE pair. Do not weaken existing paired-checkpoint method checks.
5. Start online critic optimization with fresh Adam state and the existing online learning rates. Preserve optimizer moments across pretraining epochs and pretraining retries, then intentionally reset them at the initialization handoff. Subsequent online iterations retain their existing persistent critic optimizer.
6. Add a new experimental preset and run name. Do not enable this automatically for current runs or change the existing GAE preset's defaults.

## 1. Configuration and backward compatibility

Files: `src/self_summarization_agent/config.py`, `tests/test_value_config.py`, `tests/test_config.py`.

- [x] Introduce a `ValuePretrainingConfig` dataclass, nested in `TrainingConfig` as `value_pretraining`.
- [x] Extend the explicit nested parsing in `_load_training_config` and configuration serialization/CLI override tests.
- [x] Use the following proposed configuration:

```yaml
training:
  value_pretraining:
    enabled: false
    task_count: 50
    samples_per_task: 4
    validation_task_count: 10
    seed: 0
    max_epochs: 20
    patience: 5
    min_delta: 0.0
    learning_rate: 1.0e-5
    head_learning_rate: 1.0e-4
    reuse_rollouts_path: null
```

- [x] Require an enabled separate critic when pretraining is enabled. Initial collection support is BrowseComp; reject unsupported benchmark combinations before allocating models.
- [x] Validate positive integer counts, `0 < validation_task_count < task_count`, positive finite learning rates, nonnegative finite `min_delta`, and boolean flags. Reject booleans used as numeric values.
- [x] Keep the 51-bin support, state anchor, gradient clipping, prefix batching, and tokenizer mapping inherited from the existing critic configuration.
- [x] Confirm disabled pretraining preserves existing value-config fingerprints, checkpoint compatibility, collection profiles, commands, and online behavior.

## 2. Prepare exactly one fixed-policy batch

Files: new `src/self_summarization_agent/value_pretraining_step.py`; existing `merged_collect_step.py`, `collection_contract.py`, and cache/extraction helpers.

- [x] Add a process-isolated pretraining entry point accepting the config, starting actor checkpoint, output directory, and resume option. It must not construct an actor optimizer or update actor weights.
- [x] If no reusable batch is supplied, invoke existing merged collection with only training raw/judged/cached output paths. Apply pretraining-local task/sample counts and a recorded sampling seed without changing the online configuration.
- [x] Generate exactly 200 attempted rollout traces under the current training sampling profile, including normal failures. Do not generate the normal 100 evaluation rollouts, retry until success, or replace failures to increase class balance.
- [x] Reuse the existing collection → teardown → judging/cache lifecycle. Release policy/retrieval/judge processes before allocating the critic; keep the supervising launcher CUDA-free.
- [x] Support an explicitly supplied, already judged and cached batch from the same starting actor. Validate collection/sampling lineage, exact token evidence, rewards, 50 unique training questions, and four distinct rollout indices per question. Do not silently take the first 200 rows from an incompatible file.
- [x] When reusing a batch, use its actual question membership and record its source provenance. The split seed must not force a different collection question set and thereby prevent reuse of an otherwise compatible batch.
- [x] Create a deterministic split by question: all rollouts and intervals from 40 questions fit the critic; the remaining 10 questions only select the checkpoint. Never split by interval or select questions using observed outcomes.
- [x] Record source file digests, actor identity, collection/sampling profile IDs, selected query IDs, rollout keys, split seed, and requested/eligible counts in an immutable dataset manifest.
- [x] Preserve existing exclusions. If invalid records reduce eligible counts, report the counts without silently collecting extra traces. Require nonempty fitting and validation sets; an all-failure validation set produces undefined EV, not fabricated EV or extra collection.

## 3. MC fitting, validation, and early stopping

Files: new `src/self_summarization_agent/value_pretraining.py`; reuse `separate_value_trainer.py`, `value_model.py`, `value_diagnostics.py`.

- [x] Load the separate critic from the configured initialization. Validate actor/critic tokenizer compatibility and exact prefix capacity with existing helpers.
- [x] Keep all backbone and head parameters trainable from the first minibatch. No actor parameter is part of this optimizer.
- [x] Construct detached keyed targets `target[sample_key] = sample.reward`. Preserve raw sample rewards and equal total weight per rollout.
- [x] Use the existing categorical/two-hot CE objective. Do not compute GAE bootstrap targets or export actor advantages during pretraining.
- [x] Keep one trainer/optimizer instance alive across epochs; use a pretraining-local one-epoch fitting config so repeated `fit` calls do not reset Adam moments. Track global epoch numbers separately from the existing per-call epoch metrics.
- [x] Evaluate the initial critic and the training-prevalence constant on validation data. Compute prevalence from unique fitting rollouts, not interval counts or validation labels.
- [x] After each full epoch, evaluate validation prefixes in eval/no-grad mode using existing RNG/mode-preserving evaluation. Never use validation outcomes for gradient updates, class balancing, or target construction.
- [x] Record training CE and validation Brier/log loss, EV, predicted/observed success, state counts, and initial/post-compaction strata. Record elapsed time and optimizer steps as well.
- [x] Save a new best checkpoint only when validation Brier improves by more than `min_delta`. Ties retain the earlier checkpoint and count toward patience. Use the epoch-zero critic as a finite fallback if every fitting epoch worsens validation.
- [x] Stop after five consecutive non-improving epochs or epoch 20. Export the best candidate, not the last fitted epoch.
- [x] Log whether the selected model beats the training-prevalence constant. With only 40 validation rollouts, avoid an arbitrary required EV threshold or a hyperparameter search against this same small holdout. Nonfinite metrics/weights and integrity failures still fail closed.

## 4. Artifacts, completion, and interruption handling

Files: new pretraining modules; reuse atomic JSON/checkpoint helpers from `separate_value.py` without changing normal paired-checkpoint validation.

Proposed directory:

```text
<run>/value_pretraining/
  dataset_manifest.json
  rollouts.raw.jsonl       # only when newly collected
  rollouts.judged.jsonl
  rollouts.cached.jsonl
  metrics.jsonl
  progress.json
  last/                   # private resumable MC critic state
  best/                   # selected backbone, head, and tokenizer
  pretraining_manifest.json
  .complete
```

For reused data, the dataset manifest may reference the immutable source files instead of copying them. Bind those references to digests.

- [x] Keep only the latest resumable state and selected best state, using staged writes and completion markers; do not retain 20 full model checkpoints.
- [x] Save completed epoch, best epoch/score/identity, patience counter, optimizer step count/state, and CPU/CUDA RNG state needed for deterministic epoch-boundary resume.
- [x] Bind resumable state to actor identity, source critic identity, dataset/split digests, support/tokenizer metadata, and pretraining configuration. Reject mismatches instead of silently resetting or mixing datasets.
- [x] Resume from the last completely committed pretraining epoch. An interrupted epoch restarts from that checkpoint; incomplete writes are never selected as best.
- [x] Publish the selected best export and manifest before the top-level `.complete` marker. Distinguish MC-pretraining provenance from online GAE frozen-target provenance.
- [x] Record the actual selected epoch, stop reason, validation score, total collection/fitting cost, and that online optimizer initialization will be fresh.
- [x] Never create a fake `iteration-00000` actor/critic training pair or synthesize an online frozen-advantage file to make the initializer loadable.

## 5. Launcher and first-update handoff

Files: `iteration_launcher.py`, `train_step.py` only if needed for a narrowly scoped initialization argument, and launcher tests.

- [x] Add a `value_pretraining` phase before the first normal iteration's collection/update, after resolving the initial actor and checking the already-advanced resume case.
- [x] Run the new subprocess only when enabled and starting from the initial actor. Reuse a completed, provenance-matching pretraining artifact on retry.
- [x] Resolve the selected `best/` directory as the initial critic source for the first train step. Prefer a persisted, first-step-only effective override of the existing `training.value.model_path` field, passed consistently to the train step and its critic subprocess.
- [x] Keep pretraining's original source config separate from this effective initialization override so retries do not mistakenly pretrain the already pretrained model again.
- [x] Preserve normal online hyperparameters: only the initialization path changes. The pretraining head LR `1e-4` and 20-epoch budget must not replace the online head LR or three fitting epochs.
- [x] Use fresh actor-training collection artifacts and a separate sampling run after pretraining. Never promote the pretraining batch, its caches, or any warmup-derived residuals into the first actor update.
- [x] The normal critic step then evaluates the selected initial weights, freezes GAE targets/advantages, fits the critic, and supplies the unchanged frozen advantages to dual-clip PPO.
- [x] Once a complete online actor/critic pair exists, resume from that pair and its persistent critic optimizer. Never rerun pretraining or fall back to its best weights for an advanced checkpoint. This feature does not retroactively pretrain an existing run.
- [x] Record the pretraining identity/source in run metadata and include phase costs in timing output without changing normal rollout reward, iteration numbering, or checkpoint publication rules.

## 6. Experimental preset and user documentation

Files: new `configs/train/compact_separate_gae_ppo_pretrained.yaml`, `README.md`.

- [x] Add a separate preset based on the current GAE configuration, with a distinct run name such as `qwen-bcplus-compact-separate-gae-ppo-pretrained` and pretraining enabled.
- [x] Match actor initialization, collection limits, GAE settings, dual-clip PPO settings, and online optimizer budgets to the cold-critic control.
- [x] Document fresh collection versus reuse of a compatible base-policy batch; identify proposed CLI commands as new functionality until implemented.
- [x] Explain that the 200-trace budget includes the validation split and all rollout failures, and that no extra evaluation collection is part of pretraining.
- [x] Explain optimizer reset at handoff, checkpoint selection/resume, the temporary warmup validation split, and the limitation of its small positive-outcome count.
- [x] Report warmup cost separately and include it in total experiment cost. Do not launch a production run as part of implementation validation.

## 7. Tests and acceptance criteria

New tests: `tests/test_value_pretraining.py`, `tests/test_value_pretraining_step.py`. Extend relevant config, launcher, and separate-critic tests.

- [x] Configuration defaults/round trips and invalid values; disabled-path behavior and legacy value fingerprints remain unchanged.
- [x] Exactly 50 × 4 collection requests and 40/10 question split; no eval requests, no query leakage within warmup, no success-only sampling, deterministic replay of the split.
- [x] Valid reused batch avoids generation/judging; wrong actor/profile, duplicate keys, changed source files, and missing exact token evidence are rejected.
- [x] MC targets agree across all intervals of a rollout, including malformed summary termination; raw inputs remain byte-identical and weighting is independent of compaction count.
- [x] Tiny real-model fitting updates backbone and head, retains optimizer state across epochs, and never changes actor files or constructs an actor optimizer.
- [x] Controlled validation sequences verify patience five, ties/min-delta handling, max-epoch cap, epoch-zero fallback, and best-versus-last export. Check all-failure validation EV handling and nonfinite failures.
- [x] Fault injection at collection, epoch checkpoint, best export, and completion publication; resume matches uninterrupted epoch-boundary state and never accepts incomplete artifacts.
- [x] Reloaded best weights reproduce validation predictions; first online critic evaluation uses those weights with fresh optimizer state and online learning rates.
- [x] First actor batch is independently collected. Its frozen GAE values/advantages remain fixed through critic fitting and PPO epochs; subsequent paired-checkpoint resume does not revisit pretraining.
- [x] Exercise no-compaction, multiple-compaction, and malformed-summary rollouts under the exact-token contract. Missing/rewritten histories must retain existing fail-closed behavior.

Regression command (includes the additional selection/failure tests):

```bash
.venv/bin/python -m pytest -q \
  tests/test_value_pretraining.py tests/test_value_pretraining_selection.py \
  tests/test_value_pretraining_step.py \
  tests/test_value_config.py tests/test_config.py tests/test_iteration_launcher.py \
  tests/test_separate_value.py tests/test_segment_gae.py \
  tests/test_value_training_integration.py tests/test_critic_step_evalset.py \
  tests/test_dual_clip_ppo.py tests/test_train_step.py \
  tests/test_runtime.py tests/test_rollout_collection.py tests/test_checkpoints.py
```

Implementation is complete when the disabled path remains compatible; a bounded CPU integration test demonstrates collect/reuse → full-critic MC fitting → best-checkpoint selection/resume → fresh-batch GAE/PPO handoff; and all required interval, provenance, and regression checks pass. Policy-quality improvement remains a subsequent controlled experiment.


## Implementation verification

The opt-in preset and CLI are implemented. Automatic launcher startup uses
`vllm_offline`; a distinct engine seed is recorded for the first online training
batch. The engine is seeded once, so identical prompts within a four-attempt
group are not pinned to the same per-request seed. Disabled presets preserve
their previous sampling profiles and value-config fingerprints.

The CPU integration tests use a tiny real transformer and scripted exact-token
rollouts, including no compaction, multiple compactions, and malformed summaries.
They cover whole-critic MC fitting, checkpoint selection, epoch-boundary RNG and
optimizer recovery, collection/reuse orchestration, fresh optimizer handoff, and
pretrained predictions frozen before online GAE fitting. Separate regression
tests exercise PPO and paired checkpoints. Across the feature and regression
suites, 368 distinct tests passed and four were skipped; `git diff --check`
passed. Production GPU training and policy-quality improvement remain subsequent
experimental validation, not claims of this implementation.


Follow-up: `scripts/train_iterations.sh` now automatically enables warmup for
separate value configs after applying CLI overrides, and runs a resumable
iteration-zero startup phase before entering the loop. Explicit
`--set training.value_pretraining.enabled=false` keeps the cold-critic control.
Direct launcher calls keep their configured default unless
`--auto-value-pretraining` is requested. Completed initializers are reused;
advanced paired checkpoints skip startup fitting and retain their optimizer.
