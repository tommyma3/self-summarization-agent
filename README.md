# Self-Summarization Agent

This repo trains a self-summarization agent on BrowseComp-Plus and supports Terminal-Bench 2.0 with a Terminus-KIRA scaffold that preserves token-in/token-out intervals.

For the pinned Harbor setup, evaluation configs, local vision profile, and verifier-backed collection, see [Terminal-Bench usage](docs/terminal-bench.md). Use `uv sync --locked --extra terminal-bench` to install the optional benchmark dependencies.

Current scope:
- runtime loop with `search`, `get_document`, and `finish`
- runtime-controlled summarization after completed tool rounds
- interval-level RL trajectories that preserve reasoning, tool actions, tool results, and boundary completions
- BrowseComp-style run export
- smoke and unit-test entrypoints

Partially implemented / still minimal:
- the legacy training loop is a small custom group-normalized policy-gradient update, structured to swap to `trl` later
- retrieval and judge dependencies still come from the local `bc-plus` environment
- process-isolated rollout/training orchestration supports OpenAI-compatible vLLM serving, plus legacy offline vLLM or SGLang collection
- the FSDP2/context-parallel full-training backend is represented in config and launcher contracts, but must be run in the GPU training environment

## Setup

Python requirement:
- `Python 3.12+`

Install dependencies with `uv`:

```powershell
uv sync --group dev
```

The default Qwen3.5 rollout path expects an OpenAI-compatible vLLM server. The server must support native chat tool calls and vLLM's `return_token_ids` Chat Completions extension. The GPU training environment also needs an Accelerate release with FSDP2/context-parallel support available to the Python environment used for training subprocesses.

For the optional official verl/Ray training backend, install the extra in the remote GPU environment:

```powershell
uv sync --extra verl --group dev
```

If you prefer the project virtualenv directly:

```powershell
.venv\Scripts\Activate.ps1
```

## Repo Layout

- [main.py](/D:/M/CS/self-summarization-agent/main.py): CLI entrypoint for the smoke run
- [src/self_summarization_agent/runtime.py](/D:/M/CS/self-summarization-agent/src/self_summarization_agent/runtime.py): episode runtime and summarization logic
- [src/self_summarization_agent/trajectory.py](/D:/M/CS/self-summarization-agent/src/self_summarization_agent/trajectory.py): interval extraction, assistant-token masks, and training-cache validation
- [src/self_summarization_agent/train_grpo.py](/D:/M/CS/self-summarization-agent/src/self_summarization_agent/train_grpo.py): query grouping helper for RL samples
- [src/self_summarization_agent/export.py](/D:/M/CS/self-summarization-agent/src/self_summarization_agent/export.py): BrowseComp-style run export
- [tests](/D:/M/CS/self-summarization-agent/tests): unit and integration-style tests
- [bc-plus](/D:/M/CS/self-summarization-agent/bc-plus): benchmark checkout used as the eventual backend source

## What You Can Run Today

### 0. Real experiment launchers

These are the new primary entrypoints:

```powershell
python -m self_summarization_agent.run_launcher --config configs/run/default.yaml
python -m self_summarization_agent.iteration_launcher --config configs/train/default.yaml --iteration 1 --latest-root /path/to/train-artifacts
```

Small CLI override layer:

```powershell
python -m self_summarization_agent.run_launcher --config configs/run/default.yaml --limit 25 --retrieval-backend bm25
python -m self_summarization_agent.iteration_launcher --config configs/train/default.yaml --iteration 1 --set runtime.tool_budget=12
```

Sample configs live at:

- [configs/run/default.yaml](/D:/M/CS/self-summarization-agent/configs/run/default.yaml)
- [configs/train/default.yaml](/D:/M/CS/self-summarization-agent/configs/train/default.yaml)

### Reproducible BrowseComp train/eval splits

Set the same `dataset.split_seed` (default: `0`) and dataset sizes for every
method being compared:

```yaml
dataset:
  split_seed: 0
  train_limit: 730
  eval_limit: 100
```

When `train_limit` is set, the loader sorts unique query IDs, shuffles with a
local RNG seeded by `dataset.split_seed`, applies `offset`/`limit`, and takes
the train and eval ranges. The same questions, split seed, and size settings
produce the same ordered splits even if source-file order, experiment name,
method, `experiment.seed`, or `dataset.shuffle` differ. Duplicate query IDs are
rejected to prevent train/eval overlap. `dataset.shuffle` only controls ordinary
loading without a train/eval partition; it cannot disable seeded partitioning.

Use `--set dataset.split_seed=42` to select another partition. Experiment and
per-iteration sampling seeds still control training-task sampling within the
fixed training pool. Evaluation uses a fixed prefix of the eval pool, so also
keep `collection.eval_task_count` equal across comparisons.

This replaces the historical shuffle-dependent splits. Start new experiment
output directories for comparisons; do not mix the new partition with old
training/evaluation artifacts. Production collection lineage includes the split
seed and size settings and rejects resuming artifacts with an incompatible split.
Existing artifacts are not rewritten.

### 1. Smoke run

This is the quickest end-to-end check of the current runtime.

```powershell
uv run python main.py
```

What it does:
- uses a `FakeBackend`
- runs a scripted episode with `search -> get_document -> finish`
- exports the result in BrowseComp-style JSON

Expected output shape:

```json
{
  "query_id": "smoke-q1",
  "retrieved_docids": ["smoke-doc"],
  "result": [
    {
      "output": "smoke answer",
      "type": "output_text"
    }
  ],
  "status": "completed",
  "tool_call_counts": {
    "get_document": 1,
    "search": 1
  }
}
```

### 2. Run the test suite

For remote validation, run:

```powershell
uv run --group dev pytest -q
```

If you want to focus on the runtime path first:

```powershell
uv run --group dev pytest tests/test_runtime.py -q
```

Other useful subsets:

```powershell
uv run --group dev pytest tests/test_context.py -q
uv run --group dev pytest tests/test_rewards.py -q
uv run --group dev pytest tests/test_export.py -q
```

## Current Experiment Workflow

There are now two primary experiment launchers:

- `run_launcher` for benchmark execution and artifact export
- `iteration_launcher` for process-isolated API rollout collection, judging, and clipped GRPO training updates

The legacy `train_launcher` remains available only for `training.backend: transformers`.

The runtime modules are still reusable directly if you want a custom server-side driver.

The practical workflow is:

1. Prepare local `bc-plus` assets:
   - decrypted dataset JSONL
   - retrieval indexes for `faiss` or `bm25`
2. Edit a config under `configs/run/` or `configs/train/`.
3. Launch a benchmark run with `python -m self_summarization_agent.run_launcher --config ...`.
4. Inspect:
   - per-query BrowseComp run files
   - `trajectories.jsonl`
   - `manifest.json`
5. Launch one process-isolated training iteration with `python -m self_summarization_agent.iteration_launcher --config ...`.
6. Inspect:
   - rollout JSONL under `artifacts/train/.../rollouts/`
   - `metrics.jsonl`
   - `eval_metrics.jsonl`
   - `accuracy_history.jsonl`
   - checkpoints

## Minimal Runtime Integration Pattern

The runtime is designed to be imported by a separate experiment or training driver.

```python
from self_summarization_agent.runtime import EpisodeRuntime
from self_summarization_agent.trajectory import extract_trainable_samples
from self_summarization_agent.train_grpo import group_samples_by_query

runtime = EpisodeRuntime(
    model=my_model,
    backend=my_backend,
    context_threshold_tokens=24000,
    max_context_tokens=32768,
)

result = runtime.run(query_id="q1", user_prompt="your benchmark question")
samples = extract_trainable_samples(result.trajectory_records, result.turn_rewards)
grouped = group_samples_by_query(samples)
```

What the runtime returns:
- `turn_records` as generation-level diagnostics for tool-call, summary, and final-answer steps
- `trajectory_records` as the actual RL samples, one per compaction/final-answer interval
- `turn_rewards` aligned with trajectory ids and shared across every interval in a rollout
- `summary_turns`, `retrieved_docids`, and `tool_call_counts`

Each training interval begins with the same system instructions and the same original user request. After the first compaction, the latest compressed agent history is appended as a separate conditioning-only user message wrapped in `<summary>...</summary>`. On the OpenAI-compatible path, `search`, `get_document`, and `finish` are native function tools and tool results are linked `role: tool` messages. Summary boundaries are terminal runtime-control `user` messages, while forced-answer boundaries remain terminal runtime-control `system` messages; both are appended without changing the earlier interval prefix. API `tool_choice` disables tools for a summary or requires `finish` for a forced answer. The generated summary remains a normal assistant message and must contain `<summary>...</summary>`. The runtime stores its extracted body for metrics and diagnostics, then rewraps that body when constructing the next interval after the unchanged task prefix. The runtime preserves the structured assistant reasoning, tool calls, tool-call IDs, and full raw sampled completion.

This context contract is trajectory schema version 3. Older rollout rows must be recollected under a fresh artifact root rather than resumed or mixed with schema-v3 data.

Every collection request asks vLLM to return its exact server-side prompt and completion token IDs. The runtime stores every request under `collection_tokens.generations`; at an interval boundary it also stores the final inference sequence as `collection_tokens.full_token_ids` plus an assistant-token mask. Earlier sampled completions must be exact subsequences of the final server-rendered prompt. Missing IDs or a mismatch terminates collection with an error; the cache step never substitutes locally retokenized text for these API trajectories. Cache v6 and training then use the stored IDs directly, including the exact first-generation prompt length used as the optional critic state anchor.

## How To Run Real Experiments

### Run launcher

Example:

```powershell
python -m self_summarization_agent.run_launcher --config configs/run/default.yaml
```

Outputs:

- `runs/<experiment_name>/<query_id>.json`
- `runs/<experiment_name>/trajectories.jsonl`
- `runs/<experiment_name>/manifest.json`

Typical overrides:

```powershell
python -m self_summarization_agent.run_launcher --config configs/run/default.yaml --limit 50
python -m self_summarization_agent.run_launcher --config configs/run/default.yaml --retrieval-backend bm25
python -m self_summarization_agent.run_launcher --config configs/run/default.yaml --model-path /models/qwen
```

### Legacy train launcher

Example:

```powershell
python -m self_summarization_agent.train_launcher --config configs/train/default.yaml --set training.backend=transformers --set rollout.backend=transformers
```

Outputs:

- `artifacts/train/<experiment_name>/metrics.jsonl`
- `artifacts/train/<experiment_name>/accuracy_history.jsonl`
- `artifacts/train/<experiment_name>/rollouts/step-xxxxx.jsonl`
- `artifacts/train/<experiment_name>/checkpoints/...`
- `artifacts/train/<experiment_name>/manifest.json`

Typical overrides:

```powershell
python -m self_summarization_agent.train_launcher --config configs/train/default.yaml --set training.backend=transformers --set rollout.backend=transformers --set training.epochs=20
python -m self_summarization_agent.train_launcher --config configs/train/default.yaml --set training.backend=transformers --set rollout.backend=transformers --set training.group_size=4
python scripts/plot_accuracy.py artifacts/train/qwen-bcplus-train/accuracy_history.jsonl
```

Training notes:

- the default training config uses queries 1-780 for training and 781-830 for evaluation
- reward verification is done in-process with the same local base model family as judge
- `bm25` and `faiss` retrieval are both supported from config
- legacy `train_launcher` expects `training.backend: transformers`
- the default `rollout.backend: openai_compatible` uses Chat Completions against `rollout.api_base_url`; `openai`, `openai-compatible`, and `openai_compatible` are accepted aliases
- `rollout.api_model` must be the model name exposed by the server; when it is unset the resolved checkpoint path is sent as the model name
- `rollout.enable_prefix_caching` defaults to `true`; `vllm_offline` passes it directly to the owned vLLM engine
- the external server must already be serving the exact checkpoint selected by the launcher, with Qwen3.5 reasoning, native tool parsing, and `--enable-prefix-caching`; prefix caching is an engine-startup option and cannot be enabled by a Chat Completions request, while restarting or hot-swapping that server at an iteration boundary remains an external orchestration responsibility
- legacy `rollout.backend: vllm_offline` and `rollout.backend: sglang` remain available; SGLang is an optional runtime dependency and must be installed separately
- the training backend remains independent of the rollout backend; for example, `training.backend: fsdp2_context_parallel` and `training.backend: verl_ray` consume the same cached trajectory contract
- each iteration evaluates the selected checkpoint, collects its training trajectories, caches the exact collected token sequences plus reference logprobs, runs clipped GRPO updates, writes the next vLLM-loadable checkpoint, then advances the `latest` checkpoint pointer

Before collection, verify the server with the included compatibility probe:

```powershell
uv run python scripts/probe_openai_compatible_runtime.py --base-url http://127.0.0.1:8000/v1 --model /path/to/checkpoint --tokenizer /path/to/checkpoint
```

The probe makes a forced native `finish` call and fails unless the response contains one structured tool call plus exact `prompt_token_ids` and completion `token_ids`. The client automatically sends `extra_body.return_token_ids: true` and `chat_template_kwargs.enable_thinking`.

### Process-isolated rollout/training loop

The new orchestration path uses checkpoint files as the weight-sync boundary:

```powershell
python -m self_summarization_agent.rollout_collection --config configs/train/default.yaml --checkpoint /path/to/checkpoint --output /path/to/rollouts.raw.jsonl --judged-output /path/to/rollouts.judged.jsonl
python -m self_summarization_agent.rollout_collection --config configs/train/default.yaml --checkpoint /path/to/checkpoint --output /path/to/rollouts.raw.jsonl --judged-output /path/to/rollouts.judged.jsonl --resume
python -m self_summarization_agent.judge_step --config configs/train/default.yaml --checkpoint /path/to/checkpoint --rollouts /path/to/raw-rollouts.jsonl --output /path/to/judged-rollouts.jsonl
python -m self_summarization_agent.cache_step --config configs/train/default.yaml --checkpoint /path/to/checkpoint --rollouts /path/to/judged-rollouts.jsonl --output /path/to/cached-rollouts.jsonl --resume
python -m self_summarization_agent.train_step --config configs/train/default.yaml --checkpoint /path/to/checkpoint --rollouts /path/to/cached-rollouts.jsonl --output-checkpoint /path/to/next-checkpoint
python -m self_summarization_agent.iteration_launcher --config configs/train/default.yaml --iteration 1 --latest-root /path/to/train-artifacts --resume
```

For the intended GPU run:

- each iteration's eval-then-train collection uses one collection-scoped FAISS worker; it is shut down and joined together with the policy collection boundary before judging begins
- eval and training policy engines run in isolated subprocesses with their corresponding sampling profiles; process exit is the authoritative vLLM teardown boundary
- rollout collection keeps up to `rollout.max_concurrent_episodes` active episodes, emits completed rollouts after each runtime round, and immediately refills freed slots instead of waiting for the slowest episode in a fixed batch
- exact `vllm_offline` training generations retain raw sampled-token logprobs and assemble v6 training caches directly from authoritative collection IDs and assistant masks before the policy process exits
- after both raw artifacts are complete and policy/retrieval teardown is confirmed, one judge vLLM loads on GPUs 0-3 with tensor parallel size 4 and judges all pending eval rows followed by all pending training rows
- `rollout.overlap_judge: true` instead overlaps merged collection with judging: the collection-scoped retrieval worker starts first, the judge engine loads before the policy children and sizes its GPU budget around the resident retrieval footprint, streamed raw rows are judged as they complete, and the post-collection judge phase only catches rows the stream missed; this requires the judge and policy engines to fit in GPU memory simultaneously (disjoint `judge.gpu_ids`/`rollout.gpu_ids` or reduced `judge.gpu_memory_utilization`) and falls back to sequential judging if the judge worker fails to start
- `evaluation` owns checkpoint-eval sampling independently from the GRPO rollout policy; its generator is constructed from the resolved evaluation profile instead of mutating the training generator
- raw and judged rollout rows record the resolved sampling profile and its SHA-256 ID; eval metrics copy that identity, and resume rejects eval artifacts produced by a different profile
- `judge_step` remains the resume/fallback path when only raw rollout artifacts exist; it can use a different judge model from `judge.model_path` and writes judged rollouts with `turn_rewards`
- `cache_step` preserves complete rollout-native v6 caches without loading a model; when fallback rescoring is required, a GPU-0 policy scorer starts only after the judge process exits. Resume preserves completed v6 rows and regenerates older cache versions as v6
- interrupted iterations can be resumed with `--resume`; the launcher skips completed collection, judge, cache, training, and eval phases based on artifact validation, and `--resume-rollouts` remains a deprecated alias
- training checkpoints are written through `.iteration-NNNNN.incomplete`; a complete staged checkpoint is promoted on resume, while an incomplete one is retained as `.stale-NNN` before the update is retried
- training loads the same checkpoint on GPUs 0-3 through the distributed long-context backend
- training consumes cached rollout JSONL and applies `training.update_epochs` clipped GRPO passes over every assistant-token span in each interval using token-level reference logprobs
- iteration `N` evaluates checkpoint `N-1` before collecting the training batch for update `N`; eval artifacts and `eval_metrics.jsonl` retain the evaluated checkpoint's `N-1` label
- after the last requested update, run `iteration_launcher --iteration N --eval-only --resume` to evaluate final checkpoint `N`, because there is no following training iteration to evaluate it
- the launcher advances `latest` only after the next checkpoint is complete and vLLM-loadable

### Separate MC critic with PPO updates

[`configs/train/compact_separate_mc_ppo.yaml`](configs/train/compact_separate_mc_ppo.yaml)
keeps the Qwen3.5-9B actor and initializes a separate Qwen3.5-0.8B critic from
`/124090467/Qwen/Qwen3.5-0.8B`. The critic trains its text backbone and a
zero-initialized two-class linear head at exact interval-start prefixes.

Each iteration fits the critic on all current trajectories for
`training.value.fitting_epochs`, using the final binary rollout outcome as the
MC cross-entropy label. Loss is averaged over intervals within each rollout,
then over rollouts. After fitting, the same trajectories receive fixed
`R - V_fitted` advantages, broadcast over every assistant token. The actor runs
`training.update_epochs` clipped PPO passes with the original cached behavior
log probabilities. There is no cross-fitting, GAE, bootstrapping, group-relative
normalization, or shared actor/value gradient. The critic subprocess exits
before actor workers allocate GPUs.

Use a fresh artifact root with its `latest` pointer initialized to the chosen
base actor, then use the existing iteration launcher:

```bash
scripts/train_iterations.sh --iterations 10 \
  --latest-root /path/to/separate-mc-run \
  --config configs/train/compact_separate_mc_ppo.yaml
```

The preset starts with 3 critic epochs (backbone LR `1e-5`, head LR `1e-5`)
and 2 actor epochs (LR `1e-6`, clip range `0.2`). Critic minibatches are measured
in rollouts (`rollout_minibatch_size`); `prefix_microbatch_size` controls prefix
memory use. `training.value.gpu_id` selects the critic subprocess's GPU.
`training.target_kl` stops further actor minibatches after the measured
behavior-policy KL exceeds the threshold. `training.value.loss_coefficient`
is for the older shared-head mode and is not used by the separate critic.
These settings are pilot defaults, not evidence of better task performance.

The separate critic performance preset uses 8 training prefixes and up to 32
evaluation prefixes per microbatch, length bucketing, and an 8192 padded-token
budget (`max_prefix_tokens_per_batch`). A valid prefix above this batching budget
runs alone; it is never truncated. `eval_prefix_microbatch_size: null` inherits
the training microbatch size. Batching preserves the global rollout minibatch
and each interval's rollout-normalized weight. These batch sizes should be tuned
against available memory and the actual prefix-length distribution.

`training.value.activation_checkpointing` controls only the critic (the preset
disables it); `null` inherits the actor's setting. The default fit path skips
pre-fit and intermediate full evaluations, then computes all final fitted values
exactly once for PPO advantages. Set `eval_before_fit: true` and
`eval_every_n_epochs: 1` to restore intermediate diagnostics. The epoch `online_bce`
metric averages losses observed during training; it is not frozen-model validation.

`evalset_interval: 5` scores the matching behavior-policy evaluation collection
every fifth output iteration. `diagnostics_interval: 5` enables calibration bins
and state-stratified training diagnostics on that cadence. Each accepts `0` to
disable optional work and `1` for every update; custom non-iteration checkpoint
names are skipped unless the interval is `1`. Neither setting skips final training
predictions. Unscheduled measurements have explicit skip markers, not stale values.
Progress updates are throttled by `progress_interval_seconds`, with phase boundaries
always reported. Basic final value/advantage metrics and integrity checks remain enabled.

Metrics include model/sample loading, epoch training, final prediction throughput
and padding, diagnostic evaluation, saving, and advantage preparation. The
`critic_phase_timings.json` sidecar records actual advantage export and total phase
time without rewriting the advantage artifact; the parent adds subprocess wall time.
Kernel diagnostics report which Qwen linear-attention kernels are available.
`flash-attn` alone does not provide the separate FLA and causal-conv1d kernels.

For an isolated comparison on saved data, run the following only on an idle GPU.
It reads the checkpoint, fits in memory, and writes a new benchmark JSON without
modifying the checkpoint or input trajectories:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark_separate_value.py \
  --config configs/train/compact_separate_mc_ppo.yaml \
  --checkpoint /path/to/checkpoints/iteration-00003 \
  --rollouts /path/to/rollouts/iteration-00004.jsonl \
  --output /path/to/new-benchmark.json --rollout-limit 8 --repeats 2
```

The benchmark compares the previous evaluation/batching/checkpointing settings
against the configured settings, using the same global rollout batches and epochs.
It reports model-load/fit time and final-value differences; checkpoint saving and
evaluation-collection scoring are outside its scope. `--fitting-epochs 1` is an
optional benchmark-only override for a short smoke run.

Actor files stay at the checkpoint root for vLLM. The `critic/` directory stores
the critic model, head, tokenizer metadata, AdamW state, and update count.
`training_manifest.json` identifies the paired checkpoint, and
`fitted_advantages.json` records immutable update inputs and provenance.
Publication requires both models. Resume requires the saved critic; it never
silently reinitializes it. An incomplete update restarts from the prior
committed pair. Critic optimizer state persists across iterations; the actor
retains the existing per-iteration optimizer reset behavior.

The local Transformers actor and `verl_ray` Transformers/FSDP actor backends
support this mode. Training consumes exact token caches and never truncates or
retokenizes interval histories. The existing read-only `evaluate_value` command
also supports paired separate-critic checkpoints and loads only the small
critic. Same-collection critic metrics remain labeled as training metrics.

### Value for compaction experiment

Compare the compact value MC run with the default GRPO run (requires `matplotlib`):

```bash
python scripts/plot_compact_value_mc_comparison.py
```

This reads `artifacts/train/{qwen-bcplus-train,qwen-bcplus-compact-value-mc}/eval_metrics.jsonl`
and writes a PNG/SVG dashboard, standalone accuracy SVG, matched-checkpoint CSV,
and text summary under `artifacts/train/compact_value_mc_comparison/`. The dashboard
compares accuracy, accuracy differences, malformed-output rate, budget and generated
tokens, correct answers per 1,000 budget tokens, summary frequency/size, and search
usage. The summary reports latest, best, and last-five-checkpoint accuracy; these
are descriptive comparisons of the two runs, not independent statistical trials.
Missing optional metrics remain gaps. Use `--artifacts-dir`, `--default-run`,
`--mc-run`, `--output-dir`, or `--last-n` to customize inputs and output.

[`configs/train/compact_value_mc.yaml`](configs/train/compact_value_mc.yaml) enables the Monte Carlo actor-critic experiment. At the start of each interval, the shared language-model backbone feeds a zero-initialized two-class value head from the hidden state at the exact end of the first collected generation prompt. Before any update, the trainer freezes `V_old` for the whole batch and uses `R - V_old` as every generated token's interval advantage. Critic cross-entropy is averaged within each rollout before averaging across rollouts, so extra compactions do not give a rollout extra critic weight.

Both value backends exploit causal prefix equivalence for the frozen `V_old` pass and evaluate only through the exact state anchor. The Transformers compatibility backend additionally uses Qwen's `logits_to_keep` path to materialize vocabulary logits only for the state anchor and trainable assistant-token positions. The distributed backend uses verl's packed remove-padding/Ulysses log-probability path. The value-specific changes are gated by `training.value.enabled`; ordinary group-relative GRPO keeps its existing loss path.

The policy remains a normal Hugging Face checkpoint for vLLM. The value head is saved in a `compaction_value/` sidecar subdirectory as `compaction_value/compaction_value_head.safetensors` plus `compaction_value/compaction_value_config.json`; value-enabled resumes after iteration zero require both policy and sidecar state. The preset uses `training.backend: verl_ray` with `training.verl.worker_backend: verl_fsdp`. A repository-owned verl actor adds the shared-backbone value loss while the four FSDP/Ulysses ranks keep the exported policy namespace unchanged. Frozen values are evaluated from exact causal prefixes, and alignment padding is explicitly zero-weighted instead of dropping intervals. `runtime.train_update_timeout_seconds` can give this update a separate wall-clock limit without weakening collection and judge timeouts; progress is written atomically to `iteration-NNNNN.train-progress.json`.

#### Joint loss with a separate value-head learning rate

[`configs/train/compact_value_mc_head_lr.yaml`](configs/train/compact_value_mc_head_lr.yaml) changes only the run identity/output directory and sets `training.value.head_learning_rate: 1.0e-4`. The backbone keeps `training.learning_rate: 1.0e-6`. The objective remains `L_policy + loss_coefficient * L_value`: critic gradients still reach the backbone, both optimizer groups step together, and old actor advantages remain frozen. Omitting the head LR (or setting it to null) keeps the historical shared parameter group. An explicit head LR equal to the actor LR is the compatibility control. Compare `1e-5` and `1e-4` while keeping the coefficient, batching, epochs, splits, and seed set matched; use the existing `--set training.value.head_learning_rate=1.0e-5` override when launching an experiment.

For R rollouts with m_r eligible states each, the value loss uses `w_(r,j) = 1 / (R * m_r)`. One single-interval rollout and one three-interval rollout therefore receive weights `[1/2, 1/6, 1/6, 1/6]`. Weights are computed before batching/padding, and padding contributes zero. Each minibatch retains its collection-weighted contribution; this experiment does not rescale value gradients by minibatch count or normalize each minibatch independently.

The verl integration extends its existing LambdaLR to include the new group, preserving scheduler progress and the head/backbone LR ratio. Unsupported scheduler types fail explicitly. Head weights retain the existing checkpoint sidecar format. **This change does not add optimizer-state persistence across collection iterations**; Adam moments follow the existing reset lifecycle in every comparison arm.

Metrics include actual `value/backbone_learning_rate`, `value/head_learning_rate`, and their ratio. `value/eval_loss_before` and `value/eval_loss_after` are frozen, rollout-weighted BCE on the training collection. `value/eval_before/*` and `value/eval_after/*` also report Brier score, explained variance (null for constant targets), value distribution summaries, and weighted calibration bins/error. Initial/post-compaction breakdowns use explicit interval metadata; legacy records without it are labeled unknown. Each breakdown recomputes equal rollout weights within its own subset. Historical `value/loss` and contribution metrics remain available but are not held-out validation losses.

For LR selection, reserve a fixed critic diagnostic set by query before any training, separate from both all training queries and the final policy holdout. Prepare exact-token, judged, cached records with the existing collection/cache workflow. Evaluate these same records without updating weights:

```bash
python -m self_summarization_agent.evaluate_value \
  --config configs/train/compact_value_mc_head_lr.yaml \
  --checkpoint /path/to/checkpoint \
  --rollouts /path/to/fixed-critic-records.cached.jsonl \
  --generating-checkpoint-id iteration-00000 \
  --training-query-ids /path/to/all-training-query-ids.json \
  --policy-eval-query-ids /path/to/final-policy-eval-query-ids.json \
  --output /path/to/critic-diagnostics.json
```

Both query manifests are JSON arrays of string IDs and must cover the full intended splits, not just one collection batch. The evaluator rejects overlap and missing exact token evidence, records the generating policy separately from the evaluated checkpoint, and never calls an optimizer update or saves weights. Fixed outcome labels belong to the generating policy, so account for their staleness when evaluating later checkpoints. Compare against a constant predictor estimated from training outcomes; keep final policy evaluation separate.

To use the legacy SGLang policy rollout path, install SGLang in the runtime environment and set the backend explicitly (the `sglang_offline` alias is also accepted):

```powershell
python -m self_summarization_agent.iteration_launcher --config configs/train/default.yaml --iteration 1 --latest-root /path/to/train-artifacts --set rollout.backend=sglang --set rollout.attention_backend=flashinfer
```

### Qwen3.5 recurrent interval isolation

All repository-owned verl FSDP actors (GRPO, shared value, and separate value) install
`qwen_interval_attention.py` before FSDP wrapping. Qwen3.5's recurrent layers
require interval boundaries in addition to the position resets already handled
by full attention. The adapter gathers recurrent inputs across Ulysses shards,
runs the native convolution and recurrence separately for each complete interval,
and returns the local output slice. Its autograd gather sums cross-rank input
gradients; FSDP retains responsibility for parameter-gradient averaging. This
also applies during activation-checkpoint recomputation.

Keep the current packing and Ulysses configuration: the adapter fixes both
cross-interval state leakage and missing state at shard boundaries. It leaves
collected token IDs, masks, cached probabilities, model parameters, and checkpoint
namespaces unchanged. Full attention and feed-forward layers remain sequence
parallel; recurrent computation may be repeated on ranks that share an interval,
so its memory and runtime costs are higher than the incorrect shard-local path.
This adapter supports the text-only, cache-free training path and rejects
ambiguous packed position metadata.

### Optional verl/Ray training backend

The `training.backend: verl_ray` path is an optional infrastructure path for running the policy update through Ray while preserving the existing rollout, judge, cache, checkpoint, and `latest` pointer contracts. It converts cached rollout samples into a `verl.DataProto` batch, sends that batch to a Ray worker, saves a normal vLLM-loadable checkpoint, and then returns metrics to `step_metrics.jsonl`.

Example override:

```powershell
python -m self_summarization_agent.iteration_launcher --config configs/train/default.yaml --iteration 1 --latest-root /path/to/train-artifacts --set training.backend=verl_ray --set training.verl.worker_backend=verl_fsdp --set training.verl.num_gpus_per_worker=4
```

Useful `training.verl` knobs:

- `address`: connect to an existing Ray cluster; leave unset for local Ray initialization inside the remote training job.
- `num_gpus_per_worker`: GPU resources requested by the training actor; defaults to `len(training.gpu_ids)` or `training.data_parallel_size`.
- `worker_backend`: `transformers` runs the compatibility Ray actor around the existing trainer; `verl_fsdp` uses official verl's Ray WorkerGroup plus `ActorRolloutRefWorker` for the policy update.
- `fsdp`: native verl/FSDP knobs such as `strategy`, per-GPU microbatch/token limits, remove-padding, torch compile, Ulysses sequence parallelism, and CPU offload.
- `shutdown_ray`: shuts down Ray after checkpoint save when the train step owns Ray initialization.

Rollback is config-only: set `training.backend` back to `fsdp2_context_parallel` or `transformers`.

## Notes

- The summarization trigger is runtime-controlled, not model-controlled.
- Summarization happens only after a completed tool round.
- `runtime.generated_token_budget` limits `budget_consumed_tokens`: all model completion tokens, including thinking, actions, summaries, and forced answers, plus each raw appended tool result counted once. `total_generated_tokens` remains model-output-only for reporting. Budget exhaustion takes priority over compaction.
- Compaction and forced-answer instructions are appended to the current interval; they never replace or reconstruct its preceding context.
- `model.chat_template_path` selects the agent chat template. The bundled Qwen3.5 template permits a tagged summary user control or forced-answer system control at the end of an interval and preserves all reasoning since the interval's real user query.
- A successful compaction starts a fresh interval with the unchanged system instructions, unchanged original user request, and the latest `<summary>...</summary>`-wrapped compressed agent history; no raw assistant/tool tail is retained.
- RL training data is produced as one sparse-masked sample per interval: action, tool-call, summary, and final-answer tokens are trainable, while tool results and runtime-control prompts are context-only.
- Malformed tool calls terminate the rollout immediately and assign negative reward to every interval in that failed rollout.
