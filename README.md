# Self-Summarization Agent

This repo contains the first runtime slice for a self-summarization agent on `BrowseComp-Plus`.

Current scope:
- runtime loop with `search`, `get_document`, and `finish`
- runtime-controlled summarization after completed tool rounds
- trajectory extraction for RL on `summary` and `final_answer` turns
- BrowseComp-style run export
- smoke and unit-test entrypoints

Partially implemented / still minimal:
- the training loop is a small custom group-normalized policy-gradient update, structured to swap to `trl` later
- retrieval and judge dependencies still come from the local `bc-plus` environment
- rollout backend `vllm` is wired for generation, but training currently requires `transformers`

## Setup

Python requirement:
- `Python 3.12+`

Install dependencies with `uv`:

```powershell
uv sync --group dev
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
python -m self_summarization_agent.train_launcher --config configs/train/default.yaml
```

Small CLI override layer:

```powershell
python -m self_summarization_agent.run_launcher --config configs/run/default.yaml --limit 25 --retrieval-backend bm25
python -m self_summarization_agent.train_launcher --config configs/train/default.yaml --set training.steps=100 --set runtime.tool_budget=12
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

There are now two real launchers:

- `run_launcher` for benchmark execution and artifact export
- `train_launcher` for rollout generation plus minimal in-repo policy updates

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
5. Launch training with `python -m self_summarization_agent.train_launcher --config ...`.
6. Inspect:
   - rollout JSONL under `artifacts/train/.../rollouts/`
   - `metrics.jsonl`
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

### Train launcher

Example:

```powershell
python -m self_summarization_agent.train_launcher --config configs/train/default.yaml
```

Outputs:

- `artifacts/train/<experiment_name>/metrics.jsonl`
- `artifacts/train/<experiment_name>/rollouts/step-xxxxx.jsonl`
- `artifacts/train/<experiment_name>/checkpoints/...`
- `artifacts/train/<experiment_name>/manifest.json`

Typical overrides:

```powershell
python -m self_summarization_agent.train_launcher --config configs/train/default.yaml --set training.steps=20
python -m self_summarization_agent.train_launcher --config configs/train/default.yaml --set training.batch_size=4 --set training.group_size=4
```

Training notes:

- reward verification is done in-process with the same local base model family as judge
- `bm25` and `faiss` retrieval are both supported from config
- training currently expects `model.backend: transformers`
- `model.backend: vllm` is intended for later rollout-serving swap, not the first training path

## Notes

- The summarization trigger is runtime-controlled, not model-controlled.
- Summarization happens only after a completed tool round.
- RL training data is produced only from `summary` and `final_answer` turns.
- Malformed tool calls terminate the rollout immediately and only penalize the offending tool turn.
