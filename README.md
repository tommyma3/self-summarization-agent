# Self-Summarization Agent

This repo contains the first runtime slice for a self-summarization agent on `BrowseComp-Plus`.

Current scope:
- runtime loop with `search`, `get_document`, and `finish`
- runtime-controlled summarization after completed tool rounds
- interval-level RL trajectories that preserve reasoning, tool actions, tool results, and boundary completions
- BrowseComp-style run export
- smoke and unit-test entrypoints

Partially implemented / still minimal:
- the legacy training loop is a small custom group-normalized policy-gradient update, structured to swap to `trl` later
- retrieval and judge dependencies still come from the local `bc-plus` environment
- process-isolated rollout/training orchestration defaults to owned offline SGLang collection, with optional offline vLLM and OpenAI-compatible serving paths
- the FSDP2/context-parallel full-training backend is represented in config and launcher contracts, but must be run in the GPU training environment

## Setup

Python requirement:
- `Python 3.12+`

Install dependencies with `uv`:

```powershell
uv sync --group dev
```

The default Qwen3.5 rollout and judge paths use an owned offline SGLang engine. SGLang receives pre-tokenized prompt IDs and returns sampled-token logprobs during decoding, allowing the collection worker to persist authoritative training tokens and materialize native caches without a second policy-model load. The GPU training environment also needs an Accelerate release with FSDP2/context-parallel support available to the Python environment used for training subprocesses.

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

Every exact-token collection request preserves the exact prompt and completion token IDs seen by the inference engine. SGLang receives the tokenizer-produced prompt IDs directly; the offline vLLM and OpenAI-compatible paths use their engine-returned token evidence. The runtime stores every request under `collection_tokens.generations`; at an interval boundary it also stores the final inference sequence as `collection_tokens.full_token_ids` plus an assistant-token mask. Earlier sampled completions must be exact subsequences of the final prompt. Missing IDs or a mismatch terminates collection with an error; the cache step never substitutes retokenized text for an exact-token trajectory. Cache v5 and training use the stored IDs directly.

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
- the default training presets use the owned offline SGLang engine for both policy rollout and judging; `sglang_offline` is accepted as an alias
- `rollout.api_model` must be the model name exposed by the server; when it is unset the resolved checkpoint path is sent as the model name
- `rollout.enable_prefix_caching` defaults to `true`; `vllm_offline` passes it directly to the owned vLLM engine
- the external server must already be serving the exact checkpoint selected by the launcher, with Qwen3.5 reasoning, native tool parsing, and `--enable-prefix-caching`; prefix caching is an engine-startup option and cannot be enabled by a Chat Completions request, while restarting or hot-swapping that server at an iteration boundary remains an external orchestration responsibility
- `rollout.backend: vllm_offline` and the OpenAI-compatible aliases remain available when their corresponding runtime dependencies or external server are provided
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
- eval and training policy engines run in isolated subprocesses with their corresponding sampling profiles; process exit is the authoritative policy-engine teardown boundary
- rollout collection keeps up to `rollout.max_concurrent_episodes` active episodes, emits completed rollouts after each runtime round, and immediately refills freed slots instead of waiting for the slowest episode in a fixed batch
- exact SGLang and `vllm_offline` training generations retain raw sampled-token logprobs; SGLang submits pre-tokenized prompt IDs directly to its engine so those IDs remain authoritative. Both paths assemble v5 training caches from the collected IDs and assistant masks without policy rescoring
- after both raw artifacts are complete and policy/retrieval teardown is confirmed, one judge engine loads on GPUs 0-3 with tensor parallel size 4 and judges all pending eval rows followed by all pending training rows
- `rollout.overlap_judge` and `rollout.overlap_queue_max_batches` remain parseable for compatibility but do not enable policy/judge overlap in `merged_collect`
- `evaluation` owns checkpoint-eval sampling independently from the GRPO rollout policy; its generator is constructed from the resolved evaluation profile instead of mutating the training generator
- raw and judged rollout rows record the resolved sampling profile and its SHA-256 ID; eval metrics copy that identity, and resume rejects eval artifacts produced by a different profile
- `judge_step` remains the resume/fallback path when only raw rollout artifacts exist; it can use a different judge model from `judge.model_path` and writes judged rollouts with `turn_rewards`
- `cache_step` preserves complete rollout-native v5 caches without loading a model; when fallback rescoring is required, a GPU-0 policy scorer starts only after the judge process exits. Resume preserves completed v5 rows and regenerates older cache versions as v5
- interrupted iterations can be resumed with `--resume`; the launcher skips completed collection, judge, cache, training, and eval phases based on artifact validation, and `--resume-rollouts` remains a deprecated alias
- training loads the same checkpoint on GPUs 0-3 through the distributed long-context backend
- training consumes cached rollout JSONL and applies `training.update_epochs` clipped GRPO passes over every assistant-token span in each interval using token-level reference logprobs
- iteration `N` evaluates checkpoint `N-1` before collecting the training batch for update `N`; eval artifacts and `eval_metrics.jsonl` retain the evaluated checkpoint's `N-1` label
- after the last requested update, run `iteration_launcher --iteration N --eval-only --resume` to evaluate final checkpoint `N`, because there is no following training iteration to evaluate it
- the launcher advances `latest` only after the next checkpoint is complete and vLLM-loadable

To override another preset onto the SGLang policy rollout path (`sglang_offline` is also accepted):

```powershell
python -m self_summarization_agent.iteration_launcher --config configs/train/default.yaml --iteration 1 --latest-root /path/to/train-artifacts --set rollout.backend=sglang --set rollout.attention_backend=flashinfer
```

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
