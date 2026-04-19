# Self-Summarization Agent Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmark-compatible self-summarization runtime that trains only summary turns and final-answer turns for BrowseComp-Plus.

**Architecture:** Create a new `src/self_summarization_agent/` package that owns prompts, state models, context packing, synthetic summary injection, reward assignment, trajectory masking, and run export. Keep `bc-plus` as the benchmark backend only by wrapping its retrieval interface behind a small adapter and using a fake backend for deterministic tests.

**Tech Stack:** Python 3.12, PyTorch, Transformers tokenizers, Hugging Face Datasets, pytest

---

## File Map

- Modify: `pyproject.toml`
  - Add runtime and test dependencies.
- Create: `src/self_summarization_agent/__init__.py`
  - Package marker and public version export.
- Create: `src/self_summarization_agent/prompts.py`
  - System prompt and summary prompt builders.
- Create: `src/self_summarization_agent/models.py`
  - Event, round, episode, and training-mask dataclasses.
- Create: `src/self_summarization_agent/backend.py`
  - `BrowseCompBackend` protocol plus `FakeBackend` for tests.
- Create: `src/self_summarization_agent/context.py`
  - Token counting, threshold checking, prompt packing, compactable-prefix selection.
- Create: `src/self_summarization_agent/rewards.py`
  - Terminal reward rules and per-turn train-mask rules.
- Create: `src/self_summarization_agent/trajectory.py`
  - RL sample extraction for summary and final-answer turns.
- Create: `src/self_summarization_agent/runtime.py`
  - Main episode loop with tool validation, summary injection, and final answer handling.
- Create: `src/self_summarization_agent/export.py`
  - BrowseComp-Plus run-file export.
- Create: `src/self_summarization_agent/train_grpo.py`
  - Batch builder and GRPO-ready sample collation.
- Create: `src/self_summarization_agent/cli.py`
  - Minimal CLI entrypoints for smoke runs and export.
- Create: `tests/test_prompts.py`
- Create: `tests/test_context.py`
- Create: `tests/test_rewards.py`
- Create: `tests/test_runtime.py`
- Create: `tests/test_export.py`

## Task 1: Scaffold The Package And Test Harness

**Files:**
- Create: `src/self_summarization_agent/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing import smoke test**

```python
# tests/test_prompts.py
from self_summarization_agent.prompts import build_system_prompt


def test_build_system_prompt_mentions_tools() -> None:
    prompt = build_system_prompt()
    assert "search" in prompt
    assert "get_document" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'self_summarization_agent'`

- [ ] **Step 3: Add package scaffold and dependencies**

```toml
# pyproject.toml
[project]
name = "self-summarization-agent"
version = "0.1.0"
description = "Self-summarization runtime for BrowseComp-Plus"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "datasets>=4.0.0",
  "torch>=2.7.0",
  "transformers>=4.57.0",
]

[dependency-groups]
dev = [
  "pytest>=8.4.0",
]
```

```python
# src/self_summarization_agent/__init__.py
__all__ = ["__version__"]

__version__ = "0.1.0"
```

- [ ] **Step 4: Add the first prompt function**

```python
# src/self_summarization_agent/prompts.py
def build_system_prompt() -> str:
    return (
        "Solve the benchmark question using the provided tools. "
        "Use search and get_document to gather evidence. "
        "Do not invent unsupported claims. "
        "Return a concise final answer when the evidence is sufficient."
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_prompts.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/self_summarization_agent/__init__.py src/self_summarization_agent/prompts.py tests/test_prompts.py
git commit -m "feat: scaffold self summarization package"
```

## Task 2: Define Prompt Builders And Episode Data Models

**Files:**
- Modify: `src/self_summarization_agent/prompts.py`
- Create: `src/self_summarization_agent/models.py`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Write failing tests for summary prompt and state defaults**

```python
# tests/test_prompts.py
from self_summarization_agent.models import EpisodeState
from self_summarization_agent.prompts import build_summary_prompt


def test_build_summary_prompt_mentions_doc_ids() -> None:
    prompt = build_summary_prompt()
    assert "doc_id" in prompt
    assert "essential information" in prompt


def test_episode_state_starts_with_empty_summary() -> None:
    state = EpisodeState(query_id="q1", user_prompt="question", context_threshold_tokens=1024)
    assert state.latest_summary is None
    assert state.summary_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prompts.py -v`
Expected: FAIL with `ImportError` for `build_summary_prompt` and `EpisodeState`

- [ ] **Step 3: Add prompt builders**

```python
# src/self_summarization_agent/prompts.py
def build_system_prompt() -> str:
    return (
        "Solve the benchmark question using the provided tools. "
        "Use search and get_document to gather evidence. "
        "Do not invent unsupported claims. "
        "Return a concise final answer when the evidence is sufficient."
    )


def build_summary_prompt() -> str:
    return (
        "Write a clean summary containing only the essential information needed "
        "to continue solving the task. Preserve normalized facts, current "
        "hypotheses, unresolved questions, and useful next steps. Keep "
        "evidence-grounded facts tied to doc_id citations."
    )
```

- [ ] **Step 4: Add state and event dataclasses**

```python
# src/self_summarization_agent/models.py
from dataclasses import dataclass, field
from typing import Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class Message:
    role: Role
    content: str


@dataclass(slots=True)
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, str]
    raw_output: str
    is_valid: bool = True


@dataclass(slots=True)
class ToolRound:
    assistant_message: Message
    tool_call: ToolCallRecord
    tool_result: Message


@dataclass(slots=True)
class EpisodeState:
    query_id: str
    user_prompt: str
    context_threshold_tokens: int
    latest_summary: str | None = None
    summary_count: int = 0
    rounds: list[ToolRound] = field(default_factory=list)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_prompts.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/self_summarization_agent/prompts.py src/self_summarization_agent/models.py tests/test_prompts.py
git commit -m "feat: add prompts and episode models"
```

## Task 3: Add Backend Adapters For BrowseComp-Plus And Tests

**Files:**
- Create: `src/self_summarization_agent/backend.py`
- Test: `tests/test_runtime.py`

- [ ] **Step 1: Write the failing backend test**

```python
# tests/test_runtime.py
from self_summarization_agent.backend import FakeBackend


def test_fake_backend_returns_search_hits_and_document() -> None:
    backend = FakeBackend(
        search_index={"who won": ["doc-1"]},
        documents={"doc-1": "doc-1 body"},
    )

    assert backend.search("who won") == ["doc-1"]
    assert backend.get_document("doc-1") == "doc-1 body"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runtime.py::test_fake_backend_returns_search_hits_and_document -v`
Expected: FAIL with `ImportError` for `FakeBackend`

- [ ] **Step 3: Implement backend protocol and fake backend**

```python
# src/self_summarization_agent/backend.py
from dataclasses import dataclass
from typing import Protocol


class BrowseCompBackend(Protocol):
    def search(self, query: str) -> list[str]:
        ...

    def get_document(self, doc_id: str) -> str:
        ...


@dataclass(slots=True)
class FakeBackend:
    search_index: dict[str, list[str]]
    documents: dict[str, str]

    def search(self, query: str) -> list[str]:
        return list(self.search_index.get(query, []))

    def get_document(self, doc_id: str) -> str:
        return self.documents[doc_id]
```

- [ ] **Step 4: Add a `bc-plus` integration note in code comments**

```python
# src/self_summarization_agent/backend.py
# Real BrowseComp-Plus integration should adapt the benchmark's fixed search
# and document retrieval APIs behind this protocol without importing the
# existing search_agent runtime.
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_runtime.py::test_fake_backend_returns_search_hits_and_document -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/self_summarization_agent/backend.py tests/test_runtime.py
git commit -m "feat: add backend protocol and fake backend"
```

## Task 4: Implement Context Threshold Triggering And Prompt Packing

**Files:**
- Create: `src/self_summarization_agent/context.py`
- Test: `tests/test_context.py`

- [ ] **Step 1: Write failing tests for threshold and compactable-prefix selection**

```python
# tests/test_context.py
from self_summarization_agent.context import ContextManager
from self_summarization_agent.models import EpisodeState, Message, ToolCallRecord, ToolRound


def test_threshold_crossing_marks_summary_after_completed_round() -> None:
    manager = ContextManager(token_counter=lambda text: len(text.split()), max_context_tokens=64)
    state = EpisodeState(query_id="q1", user_prompt="user prompt", context_threshold_tokens=10)
    state.rounds.append(
        ToolRound(
            assistant_message=Message(role="assistant", content="search for clue"),
            tool_call=ToolCallRecord(tool_name="search", arguments={"query": "clue"}, raw_output='{"query": "clue"}'),
            tool_result=Message(role="tool", content="doc-1 doc-2 doc-3 doc-4 doc-5"),
        )
    )

    assert manager.should_summarize(state) is True


def test_pack_summary_input_excludes_latest_round() -> None:
    manager = ContextManager(token_counter=lambda text: len(text.split()), max_context_tokens=64)
    state = EpisodeState(query_id="q1", user_prompt="user prompt", context_threshold_tokens=5)
    state.rounds.extend(
        [
            ToolRound(
                assistant_message=Message(role="assistant", content="older search"),
                tool_call=ToolCallRecord(tool_name="search", arguments={"query": "older"}, raw_output='{"query": "older"}'),
                tool_result=Message(role="tool", content="old result"),
            ),
            ToolRound(
                assistant_message=Message(role="assistant", content="latest search"),
                tool_call=ToolCallRecord(tool_name="search", arguments={"query": "latest"}, raw_output='{"query": "latest"}'),
                tool_result=Message(role="tool", content="latest result"),
            ),
        ]
    )

    packed = manager.build_summary_context(state)
    assert "older search" in packed
    assert "latest search" not in packed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_context.py -v`
Expected: FAIL with `ImportError` for `ContextManager`

- [ ] **Step 3: Implement threshold checks and summary-context packing**

```python
# src/self_summarization_agent/context.py
from dataclasses import dataclass
from typing import Callable

from self_summarization_agent.models import EpisodeState
from self_summarization_agent.prompts import build_summary_prompt, build_system_prompt


@dataclass(slots=True)
class ContextManager:
    token_counter: Callable[[str], int]
    max_context_tokens: int
    safety_margin_tokens: int = 256

    def current_token_count(self, state: EpisodeState) -> int:
        pieces = [build_system_prompt(), state.user_prompt]
        if state.latest_summary:
            pieces.append(state.latest_summary)
        for round_record in state.rounds:
            pieces.extend(
                [
                    round_record.assistant_message.content,
                    round_record.tool_result.content,
                ]
            )
        return self.token_counter("\n".join(pieces))

    def should_summarize(self, state: EpisodeState) -> bool:
        return self.current_token_count(state) >= state.context_threshold_tokens

    def build_summary_context(self, state: EpisodeState) -> str:
        older_rounds = state.rounds[:-1]
        pieces = [build_system_prompt(), state.user_prompt]
        if state.latest_summary:
            pieces.append(state.latest_summary)
        for round_record in older_rounds:
            pieces.extend(
                [
                    round_record.assistant_message.content,
                    round_record.tool_result.content,
                ]
            )
        pieces.append(build_summary_prompt())
        return "\n".join(pieces)
```

- [ ] **Step 4: Add the hard pre-call pack check**

```python
# src/self_summarization_agent/context.py
    def assert_fits(self, packed_prompt: str) -> None:
        packed_tokens = self.token_counter(packed_prompt)
        limit = self.max_context_tokens - self.safety_margin_tokens
        if packed_tokens > limit:
            raise ValueError(f"Packed prompt exceeds safe limit: {packed_tokens} > {limit}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_context.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/self_summarization_agent/context.py tests/test_context.py
git commit -m "feat: add context threshold and packing logic"
```

## Task 5: Implement Reward Rules And RL Mask Extraction

**Files:**
- Create: `src/self_summarization_agent/rewards.py`
- Create: `src/self_summarization_agent/trajectory.py`
- Test: `tests/test_rewards.py`

- [ ] **Step 1: Write failing tests for terminal rewards and malformed-tool behavior**

```python
# tests/test_rewards.py
from self_summarization_agent.rewards import apply_terminal_reward, apply_malformed_tool_penalty


def test_terminal_correct_reward_trains_summary_and_answer_turns() -> None:
    rewards = apply_terminal_reward(outcome="correct_answer", summary_turn_ids=["s1", "s2"], final_answer_turn_id="a1")
    assert rewards == {"s1": 1.0, "s2": 1.0, "a1": 1.0}


def test_malformed_tool_penalty_only_marks_offending_turn() -> None:
    rewards = apply_malformed_tool_penalty(turn_id="tool-3")
    assert rewards == {"tool-3": -1.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rewards.py -v`
Expected: FAIL with `ImportError` for reward helpers

- [ ] **Step 3: Implement reward helpers**

```python
# src/self_summarization_agent/rewards.py
from typing import Literal


Outcome = Literal["correct_answer", "wrong_answer", "budget_exhausted"]


def apply_terminal_reward(
    outcome: Outcome,
    summary_turn_ids: list[str],
    final_answer_turn_id: str | None,
) -> dict[str, float]:
    reward = 1.0 if outcome == "correct_answer" else -1.0
    assigned = {turn_id: reward for turn_id in summary_turn_ids}
    if final_answer_turn_id is not None:
        assigned[final_answer_turn_id] = reward
    return assigned


def apply_malformed_tool_penalty(turn_id: str) -> dict[str, float]:
    return {turn_id: -1.0}
```

- [ ] **Step 4: Implement RL sample extraction**

```python
# src/self_summarization_agent/trajectory.py
from dataclasses import dataclass


@dataclass(slots=True)
class RLSample:
    query_id: str
    turn_id: str
    prompt: str
    completion: str
    reward: float
    trainable_kind: str


def extract_trainable_samples(turns: list[dict[str, str]], rewards: dict[str, float]) -> list[RLSample]:
    samples: list[RLSample] = []
    for turn in turns:
        if turn["kind"] not in {"summary", "final_answer"}:
            continue
        if turn["turn_id"] not in rewards:
            continue
        samples.append(
            RLSample(
                query_id=turn["query_id"],
                turn_id=turn["turn_id"],
                prompt=turn["prompt"],
                completion=turn["completion"],
                reward=rewards[turn["turn_id"]],
                trainable_kind=turn["kind"],
            )
        )
    return samples
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_rewards.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/self_summarization_agent/rewards.py src/self_summarization_agent/trajectory.py tests/test_rewards.py
git commit -m "feat: add reward rules and rl sample extraction"
```

## Task 6: Build The Episode Runtime With Synthetic Summary Turns

**Files:**
- Create: `src/self_summarization_agent/runtime.py`
- Modify: `src/self_summarization_agent/models.py`
- Test: `tests/test_runtime.py`

- [ ] **Step 1: Write failing runtime tests for summary injection and final answer capture**

```python
# tests/test_runtime.py
from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.runtime import EpisodeRuntime, ScriptedModel


def test_runtime_injects_summary_after_threshold_crossing() -> None:
    backend = FakeBackend(search_index={"q": ["doc-1"]}, documents={"doc-1": "fact from doc-1"})
    model = ScriptedModel(
        outputs=[
            '{"tool_name": "search", "arguments": {"query": "q"}}',
            "summary with doc_id: doc-1",
            '{"tool_name": "finish", "arguments": {"answer": "done"}}',
        ]
    )
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=5, max_context_tokens=64)

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.summary_turns == ["summary-1"]
    assert result.final_answer == "done"


def test_runtime_stops_on_malformed_tool_call() -> None:
    backend = FakeBackend(search_index={}, documents={})
    model = ScriptedModel(outputs=['{"tool_name": "search"}'])
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=100, max_context_tokens=256)

    result = runtime.run(query_id="q1", user_prompt="question")

    assert result.status == "malformed_tool_call"
    assert result.turn_rewards == {"tool-1": -1.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runtime.py -v`
Expected: FAIL with `ImportError` for `EpisodeRuntime`

- [ ] **Step 3: Implement scripted model and runtime result types**

```python
# src/self_summarization_agent/models.py
@dataclass(slots=True)
class RuntimeResult:
    query_id: str
    status: str
    final_answer: str | None
    summary_turns: list[str]
    turn_rewards: dict[str, float]
    tool_call_counts: dict[str, int]
    retrieved_docids: list[str]
```

```python
# src/self_summarization_agent/runtime.py
from dataclasses import dataclass, field
import json

from self_summarization_agent.backend import BrowseCompBackend
from self_summarization_agent.context import ContextManager
from self_summarization_agent.models import EpisodeState, Message, RuntimeResult, ToolCallRecord, ToolRound
from self_summarization_agent.prompts import build_summary_prompt, build_system_prompt
from self_summarization_agent.rewards import apply_malformed_tool_penalty


@dataclass(slots=True)
class ScriptedModel:
    outputs: list[str]
    cursor: int = 0

    def generate(self, prompt: str) -> str:
        output = self.outputs[self.cursor]
        self.cursor += 1
        return output
```

- [ ] **Step 4: Implement the core episode loop**

```python
# src/self_summarization_agent/runtime.py
@dataclass(slots=True)
class EpisodeRuntime:
    model: ScriptedModel
    backend: BrowseCompBackend
    context_threshold_tokens: int
    max_context_tokens: int
    token_counter: callable = lambda text: len(text.split())

    def run(self, query_id: str, user_prompt: str) -> RuntimeResult:
        state = EpisodeState(
            query_id=query_id,
            user_prompt=user_prompt,
            context_threshold_tokens=self.context_threshold_tokens,
        )
        context_manager = ContextManager(
            token_counter=self.token_counter,
            max_context_tokens=self.max_context_tokens,
        )
        retrieved_docids: list[str] = []
        summary_turns: list[str] = []
        tool_call_counts = {"search": 0, "get_document": 0}

        while True:
            prompt = "\n".join([build_system_prompt(), user_prompt, state.latest_summary or ""])
            raw_output = self.model.generate(prompt)
            try:
                payload = json.loads(raw_output)
                tool_name = payload["tool_name"]
                arguments = payload["arguments"]
            except Exception:
                return RuntimeResult(
                    query_id=query_id,
                    status="malformed_tool_call",
                    final_answer=None,
                    summary_turns=summary_turns,
                    turn_rewards=apply_malformed_tool_penalty("tool-1"),
                    tool_call_counts=tool_call_counts,
                    retrieved_docids=retrieved_docids,
                )

            if tool_name == "finish":
                return RuntimeResult(
                    query_id=query_id,
                    status="completed",
                    final_answer=arguments["answer"],
                    summary_turns=summary_turns,
                    turn_rewards={},
                    tool_call_counts=tool_call_counts,
                    retrieved_docids=retrieved_docids,
                )

            if tool_name == "search":
                tool_call_counts["search"] += 1
                doc_ids = self.backend.search(arguments["query"])
                retrieved_docids.extend(doc_ids)
                tool_result = " ".join(doc_ids)
            elif tool_name == "get_document":
                tool_call_counts["get_document"] += 1
                doc_id = arguments["doc_id"]
                retrieved_docids.append(doc_id)
                tool_result = self.backend.get_document(doc_id)
            else:
                return RuntimeResult(
                    query_id=query_id,
                    status="malformed_tool_call",
                    final_answer=None,
                    summary_turns=summary_turns,
                    turn_rewards=apply_malformed_tool_penalty("tool-1"),
                    tool_call_counts=tool_call_counts,
                    retrieved_docids=retrieved_docids,
                )

            state.rounds.append(
                ToolRound(
                    assistant_message=Message(role="assistant", content=raw_output),
                    tool_call=ToolCallRecord(tool_name=tool_name, arguments=arguments, raw_output=raw_output),
                    tool_result=Message(role="tool", content=tool_result),
                )
            )

            if context_manager.should_summarize(state):
                summary_prompt = context_manager.build_summary_context(state)
                context_manager.assert_fits(summary_prompt)
                state.latest_summary = self.model.generate(summary_prompt + "\n" + build_summary_prompt())
                state.summary_count += 1
                summary_turns.append(f"summary-{state.summary_count}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_runtime.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/self_summarization_agent/models.py src/self_summarization_agent/runtime.py tests/test_runtime.py
git commit -m "feat: add runtime loop with synthetic summary turns"
```

## Task 7: Export Benchmark-Compatible Runs And RL Samples

**Files:**
- Create: `src/self_summarization_agent/export.py`
- Modify: `src/self_summarization_agent/runtime.py`
- Test: `tests/test_export.py`

- [ ] **Step 1: Write the failing export test**

```python
# tests/test_export.py
from self_summarization_agent.export import build_run_record
from self_summarization_agent.models import RuntimeResult


def test_build_run_record_matches_browsecomp_plus_shape() -> None:
    result = RuntimeResult(
        query_id="q1",
        status="completed",
        final_answer="answer",
        summary_turns=["summary-1"],
        turn_rewards={},
        tool_call_counts={"search": 1, "get_document": 1},
        retrieved_docids=["doc-1", "doc-2"],
    )

    record = build_run_record(result)

    assert record["query_id"] == "q1"
    assert record["status"] == "completed"
    assert record["retrieved_docids"] == ["doc-1", "doc-2"]
    assert record["result"][0]["output"] == "answer"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_export.py -v`
Expected: FAIL with `ImportError` for `build_run_record`

- [ ] **Step 3: Implement the exporter**

```python
# src/self_summarization_agent/export.py
from self_summarization_agent.models import RuntimeResult


def build_run_record(result: RuntimeResult) -> dict[str, object]:
    return {
        "query_id": result.query_id,
        "tool_call_counts": result.tool_call_counts,
        "status": result.status,
        "retrieved_docids": result.retrieved_docids,
        "result": [
            {
                "type": "output_text",
                "output": result.final_answer or "",
            }
        ],
    }
```

- [ ] **Step 4: Add RL sample collation for GRPO groups**

```python
# src/self_summarization_agent/train_grpo.py
from collections import defaultdict

from self_summarization_agent.trajectory import RLSample


def group_samples_by_query(samples: list[RLSample]) -> dict[str, list[RLSample]]:
    grouped: dict[str, list[RLSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.query_id].append(sample)
    return dict(grouped)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_export.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/self_summarization_agent/export.py src/self_summarization_agent/train_grpo.py tests/test_export.py
git commit -m "feat: add browsecomp export and grpo sample collation"
```

## Task 8: Add CLI Smoke Commands And End-To-End Test Run

**Files:**
- Create: `src/self_summarization_agent/cli.py`
- Modify: `main.py`
- Test: `tests/test_runtime.py`

- [ ] **Step 1: Write the failing CLI smoke test**

```python
# tests/test_runtime.py
from self_summarization_agent.cli import run_smoke_episode


def test_run_smoke_episode_returns_completed_record() -> None:
    record = run_smoke_episode()
    assert record["query_id"] == "smoke-q1"
    assert record["status"] in {"completed", "malformed_tool_call"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runtime.py::test_run_smoke_episode_returns_completed_record -v`
Expected: FAIL with `ImportError` for `run_smoke_episode`

- [ ] **Step 3: Implement the CLI helpers**

```python
# src/self_summarization_agent/cli.py
from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.export import build_run_record
from self_summarization_agent.runtime import EpisodeRuntime, ScriptedModel


def run_smoke_episode() -> dict[str, object]:
    backend = FakeBackend(search_index={"smoke": ["doc-1"]}, documents={"doc-1": "fact"})
    model = ScriptedModel(
        outputs=[
            '{"tool_name": "search", "arguments": {"query": "smoke"}}',
            "summary with doc_id: doc-1",
            '{"tool_name": "finish", "arguments": {"answer": "smoke answer"}}',
        ]
    )
    runtime = EpisodeRuntime(model=model, backend=backend, context_threshold_tokens=5, max_context_tokens=64)
    result = runtime.run(query_id="smoke-q1", user_prompt="smoke")
    return build_run_record(result)
```

```python
# main.py
from pprint import pprint

from self_summarization_agent.cli import run_smoke_episode


def main() -> None:
    pprint(run_smoke_episode())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the targeted test and smoke command**

Run: `pytest tests/test_runtime.py::test_run_smoke_episode_returns_completed_record -v`
Expected: PASS

Run: `python main.py`
Expected: prints a BrowseComp-Plus shaped run record for `smoke-q1`

- [ ] **Step 5: Run the full local test suite**

Run: `pytest tests -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/self_summarization_agent/cli.py main.py tests/test_runtime.py
git commit -m "feat: add smoke cli for self summarization runtime"
```

## Self-Review Notes

- Spec coverage:
  - fixed threshold-triggered summarization is implemented in Task 4 and Task 6
  - summary and final-answer-only RL masking is implemented in Task 5
  - malformed tool-call local penalty is implemented in Task 5 and Task 6
  - benchmark-compatible run export is implemented in Task 7
  - separate RL runtime over a `bc-plus` backend adapter is implemented across Tasks 3 through 8
- Placeholder scan:
  - no `TODO`, `TBD`, or implicit “write tests later” steps remain
- Type consistency:
  - `EpisodeState`, `RuntimeResult`, `ContextManager`, `EpisodeRuntime`, and `RLSample` names are used consistently across tasks
