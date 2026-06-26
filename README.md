# Self-Summarization Agent

This repo contains the first runtime slice for a self-summarization agent on `BrowseComp-Plus`.

Current scope:
- runtime loop with `search`, `get_document`, and `finish`
- runtime-controlled summarization after completed tool rounds
- trajectory extraction for RL on tool-call, `summary`, and `final_answer` turns
- BrowseComp-style run export
- smoke and unit-test entrypoints

Partially implemented / still minimal:
- the legacy training loop is a small custom group-normalized policy-gradient update, structured to swap to `trl` later
- retrieval and judge dependencies still come from the local `bc-plus` environment
- process-isolated rollout/training orchestration is wired for offline `vllm` rollout collection and checkpoint handoff
- the FSDP2/context-parallel full-training backend is represented in config and launcher contracts, but must be run in the GPU training environment

## Setup

Python requirement:
- `Python 3.12+`

Install dependencies with `uv`:

```powershell
uv sync --group dev
```

The GPU training environment also needs vLLM and an Accelerate release with FSDP2/context-parallel support available to the Python environment used for rollout and training subprocesses.

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
- [src/self_summarization_agent/trajectory.py](/D:/M/CS/self-summarization-agent/src/self_summarization_agent/trajectory.py): trainable turn extraction
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
- `iteration_launcher` for process-isolated offline vLLM rollout collection, judging, and clipped GRPO training updates

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
samples = extract_trainable_samples(result.turn_records, result.turn_rewards)
grouped = group_samples_by_query(samples)
```

What the runtime returns:
- `turn_records` for tool-call, `summary`, and `final_answer` turns
- malformed tool-call records as negative trainable examples
- `turn_rewards` aligned with those turn ids
- `summary_turns`, `retrieved_docids`, and `tool_call_counts`

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
- the new process-isolated path uses `rollout.backend: vllm_offline` and `training.backend: fsdp2_context_parallel`
- in the new path, each iteration samples 100 training questions, collects raw rollout JSONL from the latest checkpoint, judges those rollouts, caches tokenized trainable sequences plus reference logprobs, runs clipped GRPO updates over the cached batch, evaluates the new checkpoint on the 50 held-out questions, writes a vLLM-loadable checkpoint, then advances the `latest` checkpoint pointer

### Process-isolated vLLM rollout/training loop

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

- each train or eval rollout uses a phase-scoped FAISS worker; the embedding model is unloaded before judging, caching, and policy weight updates
- rollout collection starts the overlap judge worker on GPU 1, then restricts offline vLLM to GPUs 2-3 with tensor parallel size 2
- rollout collection keeps up to `rollout.max_concurrent_episodes` active episodes and batches their next model prompts through vLLM
- rollout collection writes raw trajectories and, by default, overlaps judging into the paired judged rollout artifact; `--judge-inline` is only a compatibility path
- `judge_step` remains the resume/fallback path when only raw rollout artifacts exist; it can use a different judge model from `judge.model_path` and writes judged rollouts with `turn_rewards`
- `cache_step` loads the rollout checkpoint, writes tokenized trainable turns and frozen reference logprobs, and supports `--resume` for partially cached artifacts
- interrupted iterations can be resumed with `--resume`; the launcher skips completed collection, judge, cache, training, and eval phases based on artifact validation, and `--resume-rollouts` remains a deprecated alias
- training loads the same checkpoint on GPUs 0-3 through the distributed long-context backend
- training consumes cached rollout JSONL and applies `training.update_epochs` clipped GRPO passes over the collected batch
- evaluation collects one rollout for each held-out eval question from the new checkpoint, judges those rows, and writes one accuracy row per iteration/checkpoint to `eval_metrics.jsonl`
- the launcher advances `latest` only after the next checkpoint is complete and vLLM-loadable

### Optional verl/Ray training backend

The `training.backend: verl_ray` path is an optional infrastructure pilot for running the policy update inside a Ray worker while preserving the existing rollout, judge, cache, checkpoint, and `latest` pointer contracts. It converts cached rollout samples into a `verl.DataProto` batch, sends that batch to a Ray actor, runs the current clipped GRPO update in the actor, saves a normal vLLM-loadable checkpoint, and then returns metrics to `step_metrics.jsonl`.

Example override:

```powershell
python -m self_summarization_agent.iteration_launcher --config configs/train/default.yaml --iteration 1 --latest-root /path/to/train-artifacts --set training.backend=verl_ray --set training.verl.num_gpus_per_worker=4
```

Useful `training.verl` knobs:

- `address`: connect to an existing Ray cluster; leave unset for local Ray initialization inside the remote training job.
- `num_gpus_per_worker`: GPU resources requested by the training actor; defaults to `len(training.gpu_ids)` or `training.data_parallel_size`.
- `worker_backend`: currently `transformers`; this keeps the first verl/Ray path runnable without changing rollout artifacts.
- `shutdown_ray`: shuts down Ray after checkpoint save when the train step owns Ray initialization.

Rollback is config-only: set `training.backend` back to `fsdp2_context_parallel` or `transformers`.

## Notes

- The summarization trigger is runtime-controlled, not model-controlled.
- Summarization happens only after a completed tool round.
- RL training data is produced from tool-call, `summary`, and `final_answer` turns.
- Malformed tool calls terminate the rollout immediately and assign negative trainable rewards to the recorded generated steps in that failed trajectory.
