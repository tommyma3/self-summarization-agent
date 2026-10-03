import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_tito import TokenModel, Tokenizer, SUMMARY
from self_summarization_agent.benchmarks.terminal_bench.scaffold import TerminusKiraScaffold
from self_summarization_agent.benchmarks.terminal_bench.dataset import task_manifest
from self_summarization_agent.benchmarks.terminal_bench.verification import apply_verifier_reward
from self_summarization_agent.config import BenchmarkConfig, load_train_config
from self_summarization_agent.collection_contract import collection_profile_id
from self_summarization_agent.runtime import EpisodeRuntime
from self_summarization_agent.trajectory import extract_trainable_samples, build_rollout_native_training_cache, _extract_collection_tokens, ProviderHistoryRewriteError


def call(name, arguments=None):
    parameters = "".join(f"<parameter={k}>\n{json.dumps(v) if not isinstance(v, str) else v}\n</parameter>\n"
                         for k, v in (arguments or {}).items())
    return f"</think>\n<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call><|im_end|>"


EXECUTE = call("execute_commands", {"analysis": "inspect", "plan": "write", "commands": [
    {"keystrokes": "printf 'hello' > result.txt\n", "duration": 0.1}]})
COMPLETE = call("task_complete")


class Backend:
    def __init__(self):
        self.calls = []

    def execute(self, name, arguments, *, query_id):
        self.calls.append((query_id, name, deepcopy(arguments)))
        return "result.txt now contains hello"


def runtime(outputs, **kwargs):
    model = TokenModel(outputs)
    backend = Backend()
    scaffold = TerminusKiraScaffold(backend, image_enabled=kwargs.pop("image_enabled", False))
    rt = EpisodeRuntime(model=model, backend=backend, scaffold=scaffold,
        context_threshold_tokens=kwargs.pop("context_threshold_tokens", 45000),
        max_context_tokens=kwargs.pop("max_context_tokens", 60000),
        max_summary_tokens=kwargs.pop("max_summary_tokens", 100), token_counter=model.count_tokens, **kwargs)
    return rt, model, backend


@pytest.mark.parametrize("compactions", [0, 1, 2])
def test_terminal_compaction_preserves_exact_tokens_shell_and_confirmation(compactions):
    outputs = [EXECUTE, COMPLETE, COMPLETE] if not compactions else [EXECUTE, SUMMARY, COMPLETE, SUMMARY, COMPLETE]
    rt, model, backend = runtime(outputs, context_threshold_tokens=1 if compactions else 45000)
    if compactions == 1:
        # Compact after execution, then keep confirmation in the successor.
        model.outputs = iter([EXECUTE, SUMMARY, COMPLETE, COMPLETE])
        active = rt._new_active_episode("task", "Write hello to result.txt.\n")
        rt._advance_active_episodes([active])
        active.state.context_threshold_tokens = 45000
        while active.result is None:
            rt._advance_active_episodes([active])
        result = active.result
    else:
        result = rt.run("task", "Write hello to result.txt.\n")
    assert result.status == "completed"
    assert len(backend.calls) == 1
    assert len(result.trajectory_records) == compactions + 1
    assert model.tokenizer.template_calls == compactions + 1
    first_prefix = result.trajectory_records[0]["messages"][:2]
    for index, record in enumerate(result.trajectory_records):
        assert record["messages"][:2] == first_prefix
        if index:
            assert record["messages"][2]["content"].startswith("<summary>")
            assert record["messages"][3]["role"] == "assistant"
        payload = record["collection_tokens"]
        _extract_collection_tokens(record, turn_id=record["turn_id"])
        assert 999999 in payload["full_token_ids"]
        generations = payload["generations"]
        for previous, following in zip(generations, generations[1:]):
            assert following["prompt_token_ids"][:len(previous["full_token_ids"])] == previous["full_token_ids"]
        cache = build_rollout_native_training_cache(payload)
        assert cache["input_ids"] == payload["full_token_ids"][:-1]
        assert sum(cache["completion_mask"]) == sum(payload["assistant_token_mask"])
        calls = {tc["id"] for m in record["messages"] for tc in m.get("tool_calls", [])}
        assert all(m["tool_call_id"] in calls for m in record["messages"] if m["role"] == "tool")
    assert len(extract_trainable_samples(result.trajectory_records, result.turn_rewards)) == compactions + 1


@pytest.mark.parametrize("summary,status", [
    ("</think>bad<|im_end|>", "malformed_tool_call"),
    ("</think><summary></summary><|im_end|>", "empty_summary"),
    ("</think><summary>" + "x" * 101 + "</summary><|im_end|>", "summary_length_exceeded"),
    ("</thinking><summary>good</summary><|im_end|>", "malformed_tool_call"),
    ("</think><summary>good</summary>", "malformed_tool_call"),
])
def test_failed_summary_is_retained_never_installed(summary, status):
    rt, model, _ = runtime([EXECUTE, summary], context_threshold_tokens=1)
    result = rt.run("task", "instruction")
    assert result.status == status
    assert len(result.trajectory_records) == model.tokenizer.template_calls == 1
    assert result.trajectory_records[0]["completion"].endswith(summary)
    assert set(result.turn_rewards.values()) == {-1.0}


def test_forced_completion_preempts_summary_and_executes_no_more_commands():
    rt, model, backend = runtime([EXECUTE, COMPLETE], context_threshold_tokens=1, generated_token_budget=1)
    result = rt.run("task", "instruction")
    assert result.status == "completed"
    assert len(backend.calls) == 1
    assert model.requests[-1].generation_kind == "forced_answer"
    assert "task_complete" in model.requests[-1].response_regex
    assert result.trajectory_records[-1]["termination_kind"] == "forced_answer"
    assert model.tokenizer.template_calls == 1
    prompt = model.tokenizer.decode(model.requests[-1].prompt_token_ids)
    assert prompt.count("<forced_answer_request>") == 1


@pytest.mark.parametrize("output", [COMPLETE.replace("</think>", "</thinking>"), COMPLETE.replace("<|im_end|>", ""), EXECUTE+EXECUTE])
def test_malformed_action_does_not_execute(output):
    rt, _, backend = runtime([output])
    result = rt.run("task", "instruction")
    assert result.status == "malformed_tool_call"
    assert not backend.calls
    assert result.trajectory_records[0]["completion"].endswith(output)


def test_parser_preserves_command_bytes_and_rejects_bad_types():
    scaffold = TerminusKiraScaffold(None)
    keys = "  printf '%s' '$HOME `literal`'\n\n"
    args = dict(analysis="a", plan="p", commands=[dict(keystrokes=keys, duration=1.5)])
    message = scaffold.parse(call("execute_commands", args), call_id="linked", thinking=True)
    assert message.tool_calls[0].arguments["commands"][0]["keystrokes"] == keys
    assert message.tool_calls[0].id == "linked"
    duplicate = EXECUTE.replace('"duration": 0.1', '"duration": 0.1, "duration": 0.2')
    assert scaffold.parse(duplicate, call_id="linked", thinking=True) is None
    for duration in (-1, True, "1", float("nan"), float("inf")):
        args["commands"][0]["duration"] = duration
        assert scaffold.parse(call("execute_commands", args), call_id="linked", thinking=True) is None
    assert scaffold.parse(call("image_read", dict(file_path="/x.png", image_read_instruction="read")), call_id="x", thinking=True) is None


def test_image_helper_output_is_conditioning_only():
    rt, _, backend = runtime([call("image_read", dict(file_path="/x.png", image_read_instruction="read")), COMPLETE, COMPLETE], image_enabled=True)
    result = rt.run("task", "instruction")
    assert backend.calls[0][1] == "image_read"
    tokens = result.trajectory_records[0]["collection_tokens"]
    for span in tokens["spans"]:
        if span["kind"] == "tool_result":
            assert not any(tokens["assistant_token_mask"][span["start"]:span["end"]])


def test_rewrite_is_discarded_and_missing_ids_abort():
    rt, model, _ = runtime([EXECUTE, COMPLETE])
    generate = model.generate_token_batch
    def corrupt(requests):
        results = generate(requests)
        if len(model.requests) > 1:
            return [replace(results[0], prompt_token_ids=[0] + results[0].prompt_token_ids[1:])]
        return results
    model.generate_token_batch = corrupt
    result = rt.run("task", "instruction")
    assert result.status == "history_rewrite_detected"
    assert len(result.trajectory_records[0]["collection_tokens"]["generations"]) == 1
    rt, model, _ = runtime([EXECUTE])
    generate = model.generate_token_batch
    model.generate_token_batch = lambda requests: [replace(generate(requests)[0], completion_token_ids=None)]
    with pytest.raises(RuntimeError, match="exact prompt and completion"):
        rt.run("task", "instruction")


def test_verifier_rewards_shared_across_intervals_and_infrastructure_excluded():
    rt, _, _ = runtime([EXECUTE, SUMMARY, COMPLETE, SUMMARY, COMPLETE], context_threshold_tokens=1)
    result = rt.run("task", "instruction")
    row = dict(query_id="task", rollout_index=0, status=result.status,
        trajectory_records=result.trajectory_records, benchmark_verifier=dict(rewards={"reward": 1}, exception=None))
    judged = apply_verifier_reward(row)
    assert set(judged["turn_rewards"].values()) == {1.0}
    assert judged["trainable_sample_count"] == 3
    row["status"] = "malformed_tool_call"
    assert set(apply_verifier_reward(row)["turn_rewards"].values()) == {-1.0}
    row["benchmark_verifier"]["exception"] = {"exception_type": "VerifierTimeout"}
    assert apply_verifier_reward(row)["trainable_sample_count"] == 0
    assert not apply_verifier_reward(row)["training_eligible"]


def test_pinned_dataset_and_training_opt_in():
    config = BenchmarkConfig(name="terminal-bench")
    tasks = task_manifest(config, split="eval")
    assert len(tasks) == 89
    assert len({t["git_commit_id"] for t in tasks}) == 1
    with pytest.raises(ValueError, match="evaluation-only"):
        task_manifest(config, split="train")
    with pytest.raises(ValueError, match="revision"):
        task_manifest(replace(config, dataset_revision="wrong"), split="eval")


def test_concurrent_terminal_episodes_keep_independent_ledgers_and_confirmation():
    rt, model, backend = runtime([EXECUTE, EXECUTE, SUMMARY, SUMMARY,
                                 COMPLETE, COMPLETE, SUMMARY, SUMMARY, COMPLETE, COMPLETE],
                                context_threshold_tokens=1)
    results = rt.run_many([("first", "first instruction"), ("second", "second instruction")])
    assert [r.status for r in results] == ["completed", "completed"]
    assert {c[0] for c in backend.calls} == {"first", "second"}
    for result in results:
        assert len(result.trajectory_records) == 3
        for record in result.trajectory_records:
            assert record["messages"][1]["content"] == result.query_id + " instruction"
            _extract_collection_tokens(record, turn_id=record["turn_id"])


class Session:
    def __init__(self):
        self.keys = []
    async def send_keys(self, keys, **kwargs):
        self.keys.append(keys)
    async def capture_pane(self):
        return ""
    async def get_incremental_output(self):
        return "new observation"


def test_terminal_execution_order_waiting_and_observation_artifact(tmp_path):
    from self_summarization_agent.benchmarks.terminal_bench.tools import TerminalBackend
    async def run():
        session = Session()
        backend = TerminalBackend(session, loop=asyncio.get_running_loop(), logs_dir=tmp_path)
        args = dict(commands=[dict(keystrokes="cd /tmp\n", duration=0),
                              dict(keystrokes="", duration=0), dict(keystrokes="C-c", duration=0)])
        result = await asyncio.to_thread(backend.execute, "execute_commands", args, query_id="q")
        assert result == "new observation"
        assert session.keys[0] == "cd /tmp\n"
        assert session.keys[-1] == "C-c"
        assert len(session.keys) == 3
        assert (tmp_path / "observation-00001.txt").read_text() == result
    asyncio.run(run())


def test_harbor_agent_collection_verification_cache_and_resume(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    from harbor.agents.factory import AgentFactory
    from harbor.models.agent.context import AgentContext
    import harbor.trial.trial as trials
    import self_summarization_agent.benchmarks.terminal_bench.harbor_agent as adapter
    from self_summarization_agent.benchmarks.terminal_bench.collection import collect_terminal_rollouts
    from self_summarization_agent.cache_step import run_cache_step
    from self_summarization_agent.judge_step import judge_rollouts

    class FixtureSession(Session):
        def __init__(self, *, environment, **kwargs):
            super().__init__()
            self.environment = environment
        async def start(self):
            pass
        async def stop(self):
            pass
        async def send_keys(self, keys, **kwargs):
            import subprocess
            self.keys.append(keys)
            subprocess.run(keys, shell=True, check=True, cwd=self.environment.root, capture_output=True)

    class FixtureTrial:
        count = 0
        def __init__(self, config):
            FixtureTrial.count += 1
            self.config = config
            self.trial_dir = config.trials_dir / config.trial_name
            self.environment = SimpleNamespace(root=self.trial_dir / "sandbox")
            self.environment.root.mkdir(parents=True)
            self.agent = AgentFactory.create_agent_from_config(config.agent, logs_dir=self.trial_dir / "agent")
        async def run(self):
            await self.agent.setup(self.environment)
            context = AgentContext()
            await self.agent.run(Path(self.config.task.path, "instruction.md").read_text(), self.environment, context)
            passed = (self.environment.root / "result.txt").read_text() == "hello"
            assert context.n_output_tokens > 0
            assert context.metadata["exact_token_records"] == "trajectory.records.jsonl"
            return SimpleNamespace(task_checksum="fixture", exception_info=None,
                                   verifier_result=SimpleNamespace(rewards={"reward": int(passed)}))

    monkeypatch.setattr(adapter, "TmuxSession", FixtureSession)
    monkeypatch.setattr(trials, "Trial", FixtureTrial)
    config = load_train_config("configs/eval/terminal_bench.yaml")
    config.benchmark.task_paths = ["tests/fixtures/terminal_bench/write-file"]
    config.runtime.context_threshold_tokens = 1
    config.runtime.max_context_tokens = 60000
    config.model.max_new_tokens = 1024
    config.rollout.max_model_len = 60000
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    raw, judged, cached = [tmp_path / f"{name}.jsonl" for name in ("raw", "judged", "cached")]
    model = TokenModel([EXECUTE, SUMMARY, COMPLETE, SUMMARY, COMPLETE])
    collect_terminal_rollouts(config, checkpoint_path=checkpoint, output_path=raw, generator=model,
                             judged_output_path=judged, split="eval")
    row = json.loads(raw.read_text())
    assert row["benchmark_passed"] and row["trainable_sample_count"] == 3
    assert FixtureTrial.count == 1
    judge_rollouts(config, rollout_path=raw, output_path=judged, checkpoint_path=checkpoint, split="eval")
    run_cache_step(config, checkpoint_path=checkpoint, rollout_path=judged, output_path=cached)
    cached_row = json.loads(cached.read_text())
    samples = extract_trainable_samples(cached_row["trajectory_records"], cached_row["turn_rewards"])
    assert len(samples) == 3 and all(s.has_training_cache for s in samples)
    collect_terminal_rollouts(config, checkpoint_path=checkpoint, output_path=raw, generator=None, resume=True, split="eval")
    assert FixtureTrial.count == 1
    config.benchmark.max_output_chars += 1
    with pytest.raises(ValueError, match="identity changed|contract changed"):
        collect_terminal_rollouts(config, checkpoint_path=checkpoint, output_path=raw, resume=True, split="eval")


def test_summary_control_is_identical_in_ledger_and_diagnostics():
    rt, model, _ = runtime([EXECUTE, SUMMARY, COMPLETE, SUMMARY, COMPLETE], context_threshold_tokens=1)
    result = rt.run("task", "instruction")
    record = result.trajectory_records[0]
    summary_request = model.requests[1]
    assert summary_request.response_regex.startswith(r"[^<]*</think>")
    assert record["messages"][-2]["content"] == rt.scaffold.summary_control
    prompt = model.tokenizer.decode(summary_request.prompt_token_ids)
    assert rt.scaffold.summary_control in prompt
    assert prompt.count("<summary_request>") == 2  # stable system mention and appended control


def test_local_vision_is_frozen_external_conditioning_and_records_provenance(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    import base64
    from hashlib import sha256
    from self_summarization_agent.benchmarks.terminal_bench.tools import FrozenVisionService
    from self_summarization_agent.benchmarks.terminal_bench.harbor_agent import make_vision
    model = tmp_path / "frozen"
    model.mkdir()
    service = FrozenVisionService(model_path=str(model))
    monkeypatch.setattr(service, "_analyze", lambda data, instruction: ("a chart", {"completion_tokens": 3}))
    class Environment:
        async def exec(self, command):
            assert "head -c" in command and "'image with space.png'" in command
            return SimpleNamespace(return_code=0, stdout=base64.b64encode(b"fixture image").decode(), stderr="")
    result = asyncio.run(service.read(Environment(), dict(file_path="image with space.png",
        image_read_instruction="describe"), tmp_path / "logs"))
    assert "a chart" in result
    record = json.loads(next((tmp_path / "logs").glob("vision-*.json")).read_text())
    assert record["image_sha256"] == sha256(b"fixture image").hexdigest()
    assert record["trainable"] is False
    with pytest.raises(ValueError, match="separate frozen"):
        make_vision(BenchmarkConfig(image_profile="local-vision", vision_model_path=str(model)), model)


def test_profile_tracks_local_task_content(tmp_path):
    config = load_train_config("configs/eval/terminal_bench.yaml")
    (tmp_path / "instruction.md").write_text("original")
    (tmp_path / "task.toml").write_text('version = "1.0"')
    config.benchmark.task_paths = [str(tmp_path)]
    before = collection_profile_id(config, tmp_path / "checkpoint")
    (tmp_path / "instruction.md").write_text("changed")
    assert collection_profile_id(config, tmp_path / "checkpoint") != before


def test_verification_script_detects_changed_generation_prefix():
    import runpy
    check_interval = runpy.run_path("scripts/verify_tito_records.py")["check_interval"]
    rt, _, _ = runtime([EXECUTE, COMPLETE, COMPLETE])
    record = rt.run("task", "instruction").trajectory_records[0]
    assert check_interval(record, 0, 2) == []
    damaged = deepcopy(record)
    generation = damaged["collection_tokens"]["generations"][1]
    generation["prompt_token_ids"][0] += 1
    generation["full_token_ids"][0] += 1
    assert any("append-only violation" in p for p in check_interval(damaged, 0, 2))
    with pytest.raises(ProviderHistoryRewriteError):
        _extract_collection_tokens(damaged, turn_id=damaged["turn_id"])


def test_verification_script_accepts_tool_schema_and_checks_full_successor_state():
    import runpy
    check = runpy.run_path("scripts/verify_tito_records.py")["check_successor_prefix"]
    rt, model, _ = runtime([EXECUTE, SUMMARY, COMPLETE, SUMMARY, COMPLETE], context_threshold_tokens=1)
    records = rt.run("task", "instruction").trajectory_records
    assert check(records, model.tokenizer) == []
    damaged = deepcopy(records)
    damaged[1]["messages"][2]["content"] = "<summary>\nwrong state\n</summary>"
    assert any("predecessor summary" in problem for problem in check(damaged, model.tokenizer))
    rt, model, _ = runtime([EXECUTE, SUMMARY, COMPLETE], context_threshold_tokens=1, max_tool_calls=1)
    records = rt.run("task", "instruction").trajectory_records
    assert records[-1]["termination_kind"] == "forced_answer"
    assert check(records, model.tokenizer) == []


@pytest.mark.parametrize("value_mode", [False, True])
def test_terminal_intervals_rescore_and_update_both_rl_objectives(value_mode):
    import torch
    from self_summarization_agent.generation import GenerationResult
    from self_summarization_agent.config import ModelConfig, TrainingConfig, CompactionValueConfig
    from self_summarization_agent.trainer import TransformersPolicyTrainer
    from self_summarization_agent.value_model import CompactionValueHead
    from self_summarization_agent.cache_step import _attach_training_caches

    class CompactTokenizer(Tokenizer):
        def __init__(self):
            super().__init__()
            self.unicode = {}
        def encode(self, text, **kwargs):
            ids = super().encode(text, **kwargs)
            for token in ids:
                if token > 255 and token not in self.unicode:
                    self.unicode[token] = 300 + len(self.unicode)
            return [self.unicode.get(token, token) for token in ids]
        def decode(self, ids, **kwargs):
            inverse = {v: k for k, v in self.unicode.items()}
            return super().decode([999999 if i == 3 else inverse.get(i, i) for i in ids], **kwargs)

    class PolicyOutputs(TokenModel):
        def __init__(self):
            super().__init__([EXECUTE, SUMMARY, COMPLETE, SUMMARY, COMPLETE])
            self.tokenizer = CompactTokenizer()
        def generate_token_batch(self, requests):
            outputs = []
            for request in requests:
                ids = [3] + self.tokenizer.encode(next(self.outputs))
                outputs.append(GenerationResult(text=self.tokenizer.decode(ids),
                    prompt_token_ids=list(request.prompt_token_ids), completion_token_ids=ids, finish_reason="stop"))
            return outputs

    model = PolicyOutputs()
    backend = Backend()
    rt = EpisodeRuntime(model=model, backend=backend, scaffold=TerminusKiraScaffold(backend),
        context_threshold_tokens=1, max_context_tokens=60000, max_summary_tokens=100,
        token_counter=model.count_tokens)
    result = rt.run("task", "Write result.txt")
    row = apply_verifier_reward(dict(query_id="task", rollout_index=0, status=result.status,
        trajectory_records=result.trajectory_records, benchmark_verifier={"rewards": {"reward": 1}}))

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(512, 4)
            self.lm_head = torch.nn.Linear(4, 512, bias=False)
        @property
        def device(self):
            return self.embedding.weight.device
        def get_output_embeddings(self):
            return self.lm_head
        def forward(self, input_ids, logits_to_keep=0, use_cache=False):
            hidden = self.embedding(input_ids).cumsum(dim=1) / 1000
            selected = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            return SimpleNamespace(logits=self.lm_head(hidden[:, selected]))

    trainer = TransformersPolicyTrainer.__new__(TransformersPolicyTrainer)
    trainer.model = TinyPolicy()
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)  # No tokenization fallback exists.
    trainer.model_config = ModelConfig(model_path="unused")
    trainer.training_config = TrainingConfig(update_epochs=1, minibatch_size=4,
        gradient_accumulation_microbatch_size=1, target_kl=None,
        advantage_estimator="compaction_mc_value" if value_mode else "group_relative",
        value=CompactionValueConfig(enabled=value_mode))
    trainer.value_head = CompactionValueHead(4, zero_initialize=True) if value_mode else None
    trainer.value_head_loaded = False
    parameters = list(trainer.model.parameters()) + (list(trainer.value_head.parameters()) if value_mode else [])
    trainer.optimizer = torch.optim.SGD(parameters, lr=0.01)
    samples = extract_trainable_samples(row["trajectory_records"], row["turn_rewards"], rollout_id="task:0")
    caches = trainer.cache_samples(samples)
    row = _attach_training_caches(row, cache_payloads=caches, checkpoint_id="tiny")
    samples = extract_trainable_samples(row["trajectory_records"], row["turn_rewards"], rollout_id="task:0")
    negative = deepcopy(samples[0])
    negative.rollout_id, negative.reward = "task:1", -1.0
    samples.append(negative)
    before = [p.detach().clone() for p in parameters]
    metrics = trainer.step({"task": samples})
    assert metrics.sample_count == 4 and metrics.optimizer_step_count == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, parameters))
    assert all(torch.isfinite(p).all() for p in parameters)
    if value_mode:
        assert metrics.extra_metrics["value/state_count"] == 4
        assert metrics.extra_metrics["value/rollout_count"] == 2
