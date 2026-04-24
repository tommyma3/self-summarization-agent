# Self-Summarization Agent

This repo contains the first runtime slice for a self-summarization agent on `BrowseComp-Plus`.

Current scope:
- runtime loop with `search`, `get_document`, and `finish`
- runtime-controlled summarization after completed tool rounds
- trajectory extraction for RL on `summary` and `final_answer` turns
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
- `iteration_launcher` for process-isolated offline vLLM rollout collection plus one training update

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
- `turn_records` for `summary` and `final_answer` turns
- local malformed-tool records for non-trainable `tool` turns
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

- the default training config uses queries 1-200 for training and 201-210 for evaluation
- reward verification is done in-process with the same local base model family as judge
- `bm25` and `faiss` retrieval are both supported from config
- legacy `train_launcher` expects `training.backend: transformers`
- the new process-isolated path uses `rollout.backend: vllm_offline` and `training.backend: fsdp2_context_parallel`
- in the new path, each iteration collects rollout JSONL from the latest checkpoint, runs one training update, writes a vLLM-loadable checkpoint, then advances the `latest` checkpoint pointer

### Process-isolated vLLM rollout/training loop

The new orchestration path uses checkpoint files as the weight-sync boundary:

```powershell
python -m self_summarization_agent.rollout_collection --config configs/train/default.yaml --checkpoint /path/to/checkpoint --output /path/to/rollouts.jsonl
python -m self_summarization_agent.train_step --config configs/train/default.yaml --checkpoint /path/to/checkpoint --rollouts /path/to/rollouts.jsonl --output-checkpoint /path/to/next-checkpoint
python -m self_summarization_agent.iteration_launcher --config configs/train/default.yaml --iteration 1 --latest-root /path/to/train-artifacts
```

For the intended GPU run:

- rollout collection builds the FAISS searcher before vLLM, then restricts offline vLLM to GPUs 2-3 with tensor parallel size 2
- rollout collection keeps up to `rollout.max_concurrent_episodes` active episodes and batches their next model prompts through vLLM
- training loads the same checkpoint on GPUs 0-3 through the distributed long-context backend
- training applies one optimizer step after processing the full configured training split
- the launcher advances `latest` only after the next checkpoint is complete and vLLM-loadable

## Notes

- The summarization trigger is runtime-controlled, not model-controlled.
- Summarization happens only after a completed tool round.
- RL training data is produced only from `summary` and `final_answer` turns.
- Malformed tool calls terminate the rollout immediately and only penalize the offending tool turn.
