from copy import deepcopy
import json
import re

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.chat_template import load_chat_template
from self_summarization_agent.generation import GenerationResult, VLLMGenerator, SGLangGenerator, OpenAICompatibleGenerator
from self_summarization_agent.models import Message
from self_summarization_agent.prompts import ConversationPrompt, build_initial_messages, format_tool_response
from self_summarization_agent.runtime import EpisodeRuntime
from self_summarization_agent.token_renderer import QwenAgentTokenRenderer, TEMPLATE_PATH, UnsupportedTokenBoundary, parse_native_completion
from self_summarization_agent.token_stream import IntervalTokenLedger, TokenRequest
from self_summarization_agent.trajectory import build_rollout_native_training_cache, extract_trainable_samples, _extract_collection_tokens


class Tokenizer:
    """Atomic ChatML seams plus a deliberately noncanonical sampled token."""
    chat_template = load_chat_template(TEMPLATE_PATH)
    special = {"<|im_start|>": 1, "<|im_end|>": 2}

    def __init__(self):
        self.template_calls = 0

    def encode(self, text, add_special_tokens=False):
        assert "sampled-only-reason" not in text, "Historical assistant content was re-encoded"
        ids = []
        for part in re.split(r"(<\|im_start\|>|<\|im_end\|>)", text):
            ids.extend([self.special[part]] if part in self.special else [ord(c) + 100 for c in part])
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join({1: "<|im_start|>", 2: "<|im_end|>", 999999: "sampled-only-reason"}.get(i, chr(i - 100) if 100 <= i < 999999 else "") for i in ids)

    def convert_ids_to_tokens(self, i):
        return {1: "<|im_start|>", 2: "<|im_end|>"}[i]

    def get_vocab(self):
        return self.special

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls += 1
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
        env.globals["raise_exception"] = lambda x: (_ for _ in ()).throw(ValueError(x))
        text = env.from_string(self.chat_template).render(messages=messages, **kwargs)
        return self.encode(text) if kwargs.get("tokenize") else text


class TokenModel:
    require_exact_token_ids = True
    max_new_tokens = 1024
    enable_thinking = True

    def __init__(self, outputs, *, native=False):
        self.outputs = iter(outputs)
        self.tokenizer = Tokenizer()
        self.supports_native_tools = native
        self.requests = []
        self.results = []

    def create_token_renderer(self):
        return QwenAgentTokenRenderer(self.tokenizer, enable_thinking=self.enable_thinking, native_tools=self.supports_native_tools)

    def generate_token_batch(self, requests):
        results = []
        for request in requests:
            assert isinstance(request, TokenRequest)
            self.requests.append(request)
            text = next(self.outputs)
            ids = [999999] + self.tokenizer.encode(text)
            result = GenerationResult(self.tokenizer.decode(ids), list(request.prompt_token_ids), ids,
                                      token_logprobs=[-0.25] * len(ids), token_logprobs_mode="raw_logprobs",
                                      message=parse_native_completion(self.tokenizer.decode(ids)) if self.supports_native_tools else None,
                                      finish_reason="stop")
            results.append(result)
            self.results.append(result)
        return results

    def count_tokens(self, text):
        return len(text)

    def count_prompt_tokens(self, prompt):
        raise AssertionError("TITO must count ledger IDs")


SEARCH = "</think>\n<search>q</search><|im_end|>"
SUMMARY = "</think>\n<summary>evidence</summary><|im_end|>"
FINISH = "</think>\n<answer>done</answer><|im_end|>"


def runtime(model, **kwargs):
    return EpisodeRuntime(model=model, backend=FakeBackend(search_index={"q": ["d"]}, documents={"d": "fact"}),
                          context_threshold_tokens=kwargs.pop("context_threshold_tokens", 15000),
                          max_context_tokens=kwargs.pop("max_context_tokens", 20000), max_summary_tokens=100, token_counter=model.count_tokens,
                          **kwargs)


def test_ledger_snapshots_and_duplicate_commit():
    ledger = IntervalTokenLedger([4, 5], fingerprint="a" * 64)
    snapshot = ledger.ids
    request = ledger.request([6], generation_kind="action")
    ledger.commit(request)
    with pytest.raises((RuntimeError, ValueError)):
        ledger.commit(request)
    ledger.append_sampled([999999, 2], expected_prompt=request.prompt_token_ids, kind="action")
    assert snapshot == (4, 5)
    payload = ledger.payload()
    payload["full_token_ids"].clear()
    assert ledger.ids == (4, 5, 6, 999999, 2)
    ledger.finalize()
    with pytest.raises(RuntimeError):
        ledger.append_external([7], kind="tool")


@pytest.mark.parametrize("compactions", [0, 1, 2])
def test_all_segments_keep_exact_sampled_tokens_and_native_cache(compactions):
    outputs = [SEARCH, FINISH] if compactions == 0 else [SEARCH, SUMMARY] * compactions + [FINISH]
    model = TokenModel(outputs)
    result = runtime(model, context_threshold_tokens=15000 if compactions == 0 else 1).run("q1", "question")
    assert result.status == "completed"
    assert len(result.trajectory_records) == compactions + 1
    assert model.tokenizer.template_calls == compactions + 1
    samples = extract_trainable_samples(result.trajectory_records, result.turn_rewards, rollout_id="q1:0")
    assert len(samples) == compactions + 1
    assert sum(sum(s.collection_assistant_token_mask) for s in samples) == sum(len(r.completion_token_ids) for r in model.results)
    for record in result.trajectory_records:
        payload = record["collection_tokens"]
        cache = build_rollout_native_training_cache(payload)
        assert cache is not None
        assert cache["input_ids"] == payload["full_token_ids"][:-1]
        assert sum(cache["completion_mask"]) == sum(payload["assistant_token_mask"])
        record["training_cache"] = cache
    assert len(extract_trainable_samples(result.trajectory_records, result.turn_rewards)) == compactions + 1
    for before, after in zip(model.requests, model.requests[1:]):
        if before.generation_kind != "summary":
            completion = model.results[model.requests.index(before)].completion_token_ids
            prefix = before.prompt_token_ids + tuple(completion)
            assert after.prompt_token_ids[:len(prefix)] == prefix


@pytest.mark.parametrize("summary,status", [
    ("</think>bad<|im_end|>", "malformed_tool_call"),
    ("</think><summary></summary><|im_end|>", "empty_summary"),
    ("</think><summary>" + "x" * 101 + "</summary><|im_end|>", "summary_length_exceeded"),
])
def test_failed_summary_retains_interval_and_never_creates_new_state(summary, status):
    model = TokenModel([SEARCH, summary])
    result = runtime(model, context_threshold_tokens=1).run("q", "question")
    assert result.status == status
    assert model.tokenizer.template_calls == 1
    assert len(result.trajectory_records) == 1
    assert result.trajectory_records[0]["termination_kind"] == "malformed"
    assert len(extract_trainable_samples(result.trajectory_records, result.turn_rewards)) == 1


def test_forced_answer_appends_control_without_rewriting_or_compacting():
    model = TokenModel([SEARCH, FINISH])
    result = runtime(model, max_tool_calls=1).run("q", "question")
    assert result.status == "completed"
    text = model.tokenizer.decode(model.requests[-1].prompt_token_ids)
    assert text.count("<forced_answer_request>") == 1
    assert text.count("<|im_start|>assistant") == 2
    assert model.requests[-1].generation_kind == "forced_answer"
    assert model.tokenizer.template_calls == 1


def test_overflow_retains_untrained_tool_tail():
    model = TokenModel([SEARCH])
    rt = runtime(model)
    active = rt._new_active_episode("q", "question")
    initial = len(active.token_ledger)
    rt.max_context_tokens = initial + 80
    result = rt.run("q", "question")
    assert result.status == "context_length_exceeded"
    record = result.trajectory_records[0]
    payload = record["collection_tokens"]
    assert len(payload["full_token_ids"]) > len(payload["generations"][-1]["full_token_ids"])
    assert payload["spans"][-1]["kind"] == "tool_result"
    assert len(extract_trainable_samples([record], result.turn_rewards)) == 1


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
def test_renderer_matches_template_for_canonical_tool_result(native, thinking):
    tokenizer = Tokenizer()
    renderer = QwenAgentTokenRenderer(tokenizer, enable_thinking=thinking, native_tools=native)
    tool = Message(role="tool", content="  fact 雪  ", tool_call_id="call-1") if native else Message(role="user", content=format_tool_response("fact 雪"))
    dummy = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<search>q</search>", "reasoning_content": "reason"}]
    rendered = tokenizer.apply_chat_template(dummy + [{"role": tool.role, "content": tool.content}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    sampled_end = rendered.index("<|im_end|>", rendered.index("<|im_start|>assistant")) + len("<|im_end|>")
    assert renderer.render_tool_result(tool, completion_ids=[2], finish_reason="stop") + renderer.header("action") == tuple(tokenizer.encode(rendered[sampled_end:]))
    with pytest.raises(UnsupportedTokenBoundary):
        renderer.render_tool_result(tool, completion_ids=[777], finish_reason="length")


@pytest.mark.parametrize("kind", ["summary", "forced_answer"])
def test_control_renderer_matches_terminal_template_suffix(kind):
    tokenizer = Tokenizer()
    renderer = QwenAgentTokenRenderer(tokenizer, enable_thinking=True)
    from self_summarization_agent.prompts import build_summary_prompt, build_forced_answer_prompt
    control = {"role": "user" if kind == "summary" else "system", "content": build_summary_prompt() if kind == "summary" else build_forced_answer_prompt()}
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "q"}, control], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    suffix = rendered[rendered.index("<|im_end|>") + len("<|im_end|>\n"):]
    assert renderer.header(kind) == tuple(tokenizer.encode(suffix))


@pytest.mark.parametrize("mutation", ["mask", "span", "prefix", "identity"])
def test_tito_extraction_rejects_corruption(mutation):
    result = runtime(TokenModel([SEARCH, FINISH])).run("q", "question")
    record = deepcopy(result.trajectory_records[0])
    p = record["collection_tokens"]
    if mutation == "mask":
        p["assistant_token_mask"][0] = True
    elif mutation == "span":
        p["spans"][0]["end"] -= 1
    elif mutation == "prefix":
        p["full_token_ids"][0] = 999
    else:
        p["contract"] = "old"
    with pytest.raises(ValueError):
        _extract_collection_tokens(record, turn_id="q")


def test_grpo_normalization_uses_rollouts_with_unequal_segment_counts():
    from self_summarization_agent.trainer import compute_group_advantages
    many = runtime(TokenModel([SEARCH, SUMMARY, FINISH]), context_threshold_tokens=1).run("q", "question")
    one = runtime(TokenModel([FINISH])).run("q", "question")
    samples = extract_trainable_samples(many.trajectory_records, many.turn_rewards, rollout_id="q:0")
    samples += extract_trainable_samples(one.trajectory_records, {"trajectory-1": -1.0}, rollout_id="q:1")
    assert compute_group_advantages(samples) == pytest.approx([1, 1, -1], abs=2e-6)


def test_parseable_thinking_typo_is_retained_but_unresolved_thinking_does_not_dispatch():
    model = TokenModel([SEARCH.replace("</think>", "</thinking>"), FINISH])
    result = runtime(model).run("q", "question")
    assert result.status == "completed"
    assert "</thinking>" in model.tokenizer.decode(model.requests[1].prompt_token_ids)
    assert model.tokenizer.template_calls == 1
    invalid = TokenModel([SEARCH.replace("</think>", "")])
    rejected = runtime(invalid).run("q", "question")
    assert rejected.status == "malformed_tool_call"
    assert rejected.tool_call_counts["search"] == 0
    assert len(extract_trainable_samples(rejected.trajectory_records, rejected.turn_rewards)) == 1


def test_batched_episodes_have_independent_compaction_ledgers():
    model = TokenModel([SEARCH, FINISH, SUMMARY, FINISH])
    results = runtime(model, context_threshold_tokens=1).run_many([("q1", "first"), ("q2", "second")])
    assert [r.status for r in results] == ["completed", "completed"]
    assert [len(r.trajectory_records) for r in results] == [2, 1]
    assert model.tokenizer.template_calls == 3
    assert "second" not in model.tokenizer.decode(model.requests[-1].prompt_token_ids)


def test_backend_modified_input_is_discarded_without_training_its_completion():
    class BadModel(TokenModel):
        def generate_token_batch(self, requests):
            results = super().generate_token_batch(requests)
            if len(self.requests) > 1:
                results[0].prompt_token_ids[0] = 999
            return results
    model = BadModel([SEARCH, FINISH])
    result = runtime(model).run("q", "question")
    assert result.status == "history_rewrite_detected"
    samples = extract_trainable_samples(result.trajectory_records, result.turn_rewards)
    assert len(samples) == 1
    assert sum(samples[0].collection_assistant_token_mask) == len(model.results[0].completion_token_ids)


def test_native_tool_round_and_summary_keep_linked_messages():
    def call(name, key, value):
        return f"</think><tool_call>\n<function={name}>\n<parameter={key}>\n{value}\n</parameter>\n</function>\n</tool_call><|im_end|>"
    model = TokenModel([call("search", "query", "q"), SUMMARY, call("finish", "answer", "done")], native=True)
    result = runtime(model, context_threshold_tokens=1).run("q", "question")
    assert result.status == "completed"
    messages = result.trajectory_records[0]["messages"]
    tool = next(m for m in messages if m["role"] == "tool")
    assistant = next(m for m in messages if m.get("tool_calls"))
    assert tool["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert len(extract_trainable_samples(result.trajectory_records, result.turn_rewards)) == 2


def test_vllm_adapter_submits_ids_and_preserves_raw_sampled_logprobs():
    from types import SimpleNamespace
    generator = VLLMGenerator.__new__(VLLMGenerator)
    generator.tokenizer = Tokenizer()
    generator.max_new_tokens = 99
    generator.temperature = 0.7
    generator.top_p = 0.9
    generator.do_sample = True
    generator.sampling_extra = {}
    generator._sampling_params_cls = lambda **kw: kw
    seen = []
    def generate(prompts, params):
        seen.append((prompts, params))
        completion = SimpleNamespace(token_ids=[999999, 2], logprobs=[{999999: SimpleNamespace(logprob=-0.2)}, {2: SimpleNamespace(logprob=-0.3)}], cumulative_logprob=-0.5, finish_reason="stop")
        return [SimpleNamespace(prompt_token_ids=prompts[0]["prompt_token_ids"], outputs=[completion])]
    generator.llm = SimpleNamespace(generate=generate)
    result = generator.generate_token_batch([TokenRequest((1, 999999, 2), max_new_tokens=8)])[0]
    assert seen[0][0] == [{"prompt_token_ids": [1, 999999, 2]}]
    assert seen[0][1][0]["max_tokens"] == 8
    assert result.completion_token_ids == [999999, 2]
    assert result.token_logprobs == [-0.2, -0.3]
    assert result.token_logprobs_mode == "raw_logprobs"
    assert result.text == "sampled-only-reason<|im_end|>"


def test_sglang_adapter_submits_ids_without_claiming_raw_logprobs():
    from types import SimpleNamespace
    generator = SGLangGenerator.__new__(SGLangGenerator)
    generator.tokenizer = Tokenizer()
    generator.max_new_tokens = 99
    generator.do_sample = False
    seen = []
    def generate(**kwargs):
        seen.append(kwargs)
        return [{"meta_info": {"output_token_logprobs": [[-0.2, 999999], [-0.3, 2]], "finish_reason": {"type": "stop"}}}]
    generator.engine = SimpleNamespace(generate=generate)
    result = generator.generate_token_batch([TokenRequest((1, 999999, 2), max_new_tokens=8)])[0]
    assert seen[0]["input_ids"] == [[1, 999999, 2]]
    assert "prompt" not in seen[0]
    assert result.prompt_token_ids == [1, 999999, 2]
    assert result.token_logprobs_mode is None


def test_api_adapter_uses_completions_tokens_and_controls():
    from types import SimpleNamespace
    generator = OpenAICompatibleGenerator.__new__(OpenAICompatibleGenerator)
    generator.tokenizer = Tokenizer()
    generator.api_extra_body = {}
    generator.api_model = "model"
    generator.max_new_tokens = 99
    generator.temperature = 0.7
    generator.do_sample = False
    generator.enable_thinking = True
    generator.api_max_concurrency = 1
    seen = []
    def create(**kwargs):
        seen.append(kwargs)
        return {"prompt_token_ids": kwargs["prompt"], "choices": [{"token_ids": generator.tokenizer.encode(SUMMARY), "finish_reason": "stop"}]}
    generator.client = SimpleNamespace(completions=SimpleNamespace(create=create))
    result = generator.generate_token_batch([TokenRequest((1, 999999, 2), generation_kind="summary", max_new_tokens=8)])[0]
    assert seen[0]["prompt"] == [1, 999999, 2]
    assert "messages" not in seen[0]
    assert seen[0]["extra_body"]["return_token_ids"] is True
    assert seen[0]["extra_body"]["add_special_tokens"] is False
    assert "<summary>" in seen[0]["extra_body"]["structured_outputs"]["regex"]
    assert result.message.content == "<summary>evidence</summary>"


def test_cache_identity_mismatch_is_rejected():
    result = runtime(TokenModel([FINISH])).run("q", "question")
    record = result.trajectory_records[0]
    record["training_cache"] = build_rollout_native_training_cache(record["collection_tokens"])
    record["training_cache"]["renderer_fingerprint"] = "b" * 64
    with pytest.raises(ValueError, match="identity"):
        extract_trainable_samples([record], result.turn_rewards)


def test_native_parser_requires_the_prompt_opened_thinking_block_to_close():
    from self_summarization_agent.token_renderer import parse_native_completion
    assert parse_native_completion("<summary>unfinished thought</summary><|im_end|>",
                                   require_thinking_close=True) is None
    raw = "</think><tool_call><function=search><parameter=query>same query</parameter></function></tool_call><|im_end|>"
    first = parse_native_completion(raw, require_thinking_close=True, call_id="round-1")
    second = parse_native_completion(raw, require_thinking_close=True, call_id="round-2")
    assert first.tool_calls[0].id != second.tool_calls[0].id
    assert first.tool_calls[0].arguments == second.tool_calls[0].arguments


def test_transformers_adapter_preserves_exact_inputs_and_sampled_stop():
    import torch
    from types import SimpleNamespace
    from self_summarization_agent.generation import TransformersGenerator
    seen = []
    def generate(**kwargs):
        seen.append(kwargs)
        return torch.cat([kwargs["input_ids"], torch.tensor([[999999, 2]])], dim=1)
    tokenizer = Tokenizer()
    adapter = SimpleNamespace(model=SimpleNamespace(device="cpu", generate=generate),
        tokenizer=SimpleNamespace(pad_token_id=0, decode=tokenizer.decode,
                                  convert_tokens_to_ids=lambda token: 2),
        max_new_tokens=99, do_sample=False)
    result = TransformersGenerator.generate_token_batch(adapter, [TokenRequest((1, 999999, 2), max_new_tokens=8)])[0]
    assert seen[0]["input_ids"].tolist() == [[1, 999999, 2]]
    assert seen[0]["max_new_tokens"] == 8
    assert result.completion_token_ids == [999999, 2]
    assert result.finish_reason == "stop"


def test_non_thinking_mode_compacts_without_a_sampled_thinking_close():
    model = TokenModel([SEARCH.replace("</think>", ""), SUMMARY.replace("</think>", ""), FINISH.replace("</think>", "")])
    model.enable_thinking = False
    result = runtime(model, context_threshold_tokens=1).run("q", "question")
    assert result.status == "completed"
    assert len(result.trajectory_records) == 2


def test_lineage_preflight_rejects_old_or_changed_artifacts(tmp_path):
    from types import SimpleNamespace
    from self_summarization_agent.collection_contract import collection_profile_id, validate_artifact_lineage
    from self_summarization_agent.token_stream import TITO_CONTRACT
    config = SimpleNamespace(model=SimpleNamespace(chat_template_path=TEMPLATE_PATH, enable_thinking=True),
                             rollout=SimpleNamespace(backend="vllm_offline"))
    result = runtime(TokenModel([FINISH])).run("q", "question")
    path = tmp_path / "raw.jsonl"
    row = {"collection_contract": TITO_CONTRACT, "collection_profile_id": collection_profile_id(config, tmp_path),
           "trajectory_records": result.trajectory_records}
    path.write_text(json.dumps(row), encoding="utf-8")
    validate_artifact_lineage([path], config=config, checkpoint=tmp_path)
    path.write_text(json.dumps({key: value for key, value in row.items() if key != "trajectory_records"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing trajectory records"):
        validate_artifact_lineage([path], config=config, checkpoint=tmp_path)
    path.write_text(json.dumps(row), encoding="utf-8")
    config.model.enable_thinking = False
    with pytest.raises(ValueError, match="fresh output lineage"):
        validate_artifact_lineage([path], config=config, checkpoint=tmp_path)


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_real_tokenizer_renderer_conformance(thinking, native):
    import os
    from transformers import AutoTokenizer
    from self_summarization_agent.chat_template import configure_tokenizer_chat_template
    from self_summarization_agent.prompts import ACTION_TOOLS
    path = os.environ.get("TITO_TEST_TOKENIZER_PATH")
    if not path:
        pytest.skip("Set TITO_TEST_TOKENIZER_PATH to the server's Qwen tokenizer/checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    configure_tokenizer_chat_template(tokenizer, TEMPLATE_PATH)
    renderer = QwenAgentTokenRenderer(tokenizer, enable_thinking=thinking, native_tools=native)
    initial = build_initial_messages("  Original query 雪  ", native_tools=native)
    prompt = ConversationPrompt(initial, tools=ACTION_TOOLS if native else None)
    initial_ids = renderer.render_initial_state(prompt)
    for result in ["fact", "  雪\n中文\n ", 'JSON {"b": false, "a": 1}', ""]:
        msg = Message(role="tool", content=result, tool_call_id="call-1") if native else Message(role="user", content=format_tool_response(result))
        messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<search>q</search>", "reasoning_content": "reason"}]
        full = tokenizer.apply_chat_template(messages + [{"role": msg.role, "content": msg.content}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=thinking)
        end = full.index("<|im_end|>", full.index("<|im_start|>assistant")) + len("<|im_end|>")
        suffix = renderer.render_tool_result(msg, completion_ids=[renderer.im_end], finish_reason="stop") + renderer.header("action")
        assert suffix == tuple(tokenizer.encode(full[end:], add_special_tokens=False))
        assert initial_ids + renderer.header("action") == tuple(tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in initial], tools=list(prompt.tools) or None,
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=thinking))


@pytest.mark.parametrize("value_mode", [False, True])
def test_exact_tito_segments_rescore_and_update_on_cpu(value_mode):
    import torch
    from types import SimpleNamespace
    from self_summarization_agent.config import ModelConfig, TrainingConfig, CompactionValueConfig
    from self_summarization_agent.trainer import TransformersPolicyTrainer
    from self_summarization_agent.value_model import CompactionValueHead
    from self_summarization_agent.cache_step import _attach_training_caches
    from self_summarization_agent.trajectory import TRAJECTORY_SCHEMA_VERSION

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.lm_head = torch.nn.Linear(4, 16, bias=False)

        @property
        def device(self):
            return self.embedding.weight.device

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids, logits_to_keep=0, use_cache=False):
            # A causal toy policy; no later token can affect an earlier value.
            hidden = self.embedding(input_ids).cumsum(dim=1)
            selected = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            return SimpleNamespace(logits=self.lm_head(hidden[:, selected]))

    def record(index, initial, sampled):
        ledger = IntervalTokenLedger(initial, fingerprint="a" * 64)
        request = ledger.request([3], generation_kind="action")
        ledger.commit(request)
        ledger.append_sampled(sampled, expected_prompt=request.prompt_token_ids, kind="action")
        payload = ledger.payload()
        payload["generations"] = [{"prompt_token_ids": list(request.prompt_token_ids),
                                  "completion_token_ids": sampled, "full_token_ids": list(ledger.ids)}]
        ledger.finalize()
        return {"schema_version": TRAJECTORY_SCHEMA_VERSION, "kind": "trajectory", "turn_id": f"trajectory-{index}",
                "query_id": "q", "termination_kind": "compaction" if index == 1 else "final_answer",
                "messages": [{"role": "system", "content": "policy"}, {"role": "user", "content": "state"}, {"role": "assistant", "content": "output"}],
                "collection_tokens": payload}

    trainer = TransformersPolicyTrainer.__new__(TransformersPolicyTrainer)
    trainer.model = TinyPolicy()
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)  # No encode/template method: rescoring must use IDs.
    trainer.model_config = ModelConfig(model_path="unused")
    trainer.training_config = TrainingConfig(update_epochs=1, minibatch_size=3,
        gradient_accumulation_microbatch_size=1, target_kl=None,
        advantage_estimator="compaction_mc_value" if value_mode else "group_relative",
        value=CompactionValueConfig(enabled=value_mode))
    trainer.value_head = CompactionValueHead(4, zero_initialize=True) if value_mode else None
    trainer.value_head_loaded = False
    parameters = list(trainer.model.parameters()) + (list(trainer.value_head.parameters()) if value_mode else [])
    trainer.optimizer = torch.optim.SGD(parameters, lr=0.01)
    records = [record(1, [1, 2], [4, 5]), record(2, [1, 6], [7, 5])]
    row = {"query_id": "q", "rollout_index": 0, "trajectory_records": records,
           "turn_rewards": {"trajectory-1": 1.0, "trajectory-2": 1.0}}
    samples = extract_trainable_samples(records, row["turn_rewards"], rollout_id="q:0")
    caches = trainer.cache_samples(samples)
    for collected, cache in zip(records, caches):
        generation = collected["collection_tokens"]["generations"][0]
        generation["completion_token_logprobs"] = cache["reference_logprobs"][len(generation["prompt_token_ids"]) - 1:]
        generation["logprobs_mode"] = "raw_logprobs"
        native = build_rollout_native_training_cache(collected["collection_tokens"])
        assert native is not None
        assert native["reference_logprobs"] == pytest.approx(cache["reference_logprobs"], abs=1e-6)
        assert native["completion_mask"] == cache["completion_mask"]
    row = _attach_training_caches(row, cache_payloads=caches, checkpoint_id="tiny")
    samples = extract_trainable_samples(row["trajectory_records"], row["turn_rewards"], rollout_id="q:0")
    negative = deepcopy(samples[0])
    negative.rollout_id, negative.reward = "q:1", -1.0
    samples.append(negative)
    before = [p.detach().clone() for p in parameters]
    metrics = trainer.step({"q": samples})
    assert metrics.sample_count == 3
    assert metrics.optimizer_step_count == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, parameters))
    if value_mode:
        assert metrics.mean_advantage == pytest.approx(1 / 3)
        assert metrics.extra_metrics["value/state_count"] == 3
        assert metrics.extra_metrics["value/rollout_count"] == 2
