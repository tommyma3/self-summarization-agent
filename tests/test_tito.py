from copy import deepcopy
import json
import re

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.chat_template import load_chat_template
from self_summarization_agent.generation import GenerationResult, VLLMGenerator, SGLangGenerator, OpenAICompatibleGenerator
from self_summarization_agent.models import Message
from self_summarization_agent.prompts import ConversationPrompt, build_initial_messages, format_compacted_summary, format_tool_response
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
PLAIN_ANSWER = "Explanation: Evidence supports the answer. [d]\nExact Answer: 42\nConfidence: 95%"
PLAIN_FINISH = "</think>\n" + PLAIN_ANSWER + "<|im_end|>"


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


@pytest.mark.parametrize("summary", [SUMMARY, "</think>bad<|im_end|>"])
def test_mixed_control_batch_preserves_individual_trajectories(summary):
    class BatchedModel(TokenModel):
        supports_mixed_control_batches = True

        def __init__(self):
            super().__init__([])
            self.batches = []
            self.short_actions = 0

        def generate_token_batch(self, requests):
            self.batches.append([r.generation_kind for r in requests])
            assert len(self.batches) <= 4
            outputs = []
            for request in requests:
                prompt = self.tokenizer.decode(request.prompt_token_ids)
                if request.generation_kind == "summary":
                    outputs.append(summary)
                elif "long-question" in prompt:
                    outputs.append(FINISH if format_compacted_summary("evidence") in prompt else SEARCH)
                else:
                    self.short_actions += 1
                    outputs.append(SEARCH if self.short_actions < 3 else FINISH)
            self.outputs = iter(outputs)
            return super().generate_token_batch(requests)

    episodes = [("long", "long-question" + "x" * 5000), ("short", "short-question")]
    baseline = BatchedModel()
    baseline.supports_mixed_control_batches = False
    mixed = BatchedModel()
    expected = runtime(baseline, context_threshold_tokens=5000).run_many(episodes)
    actual = runtime(mixed, context_threshold_tokens=5000).run_many(episodes)
    assert ["summary", "action"] in mixed.batches
    assert len(mixed.batches) == 3
    assert len(baseline.batches) == 4
    for before, after in zip(expected, actual):
        assert before.trajectory_records == after.trajectory_records
        assert before.status == after.status
        assert before.turn_rewards == after.turn_rewards


@pytest.mark.parametrize("compactions", [0, 1, 2])
@pytest.mark.parametrize("final_output", [FINISH, PLAIN_FINISH])
def test_all_segments_keep_exact_sampled_tokens_and_native_cache(compactions, final_output):
    outputs = [SEARCH, final_output] if compactions == 0 else [SEARCH, SUMMARY] * compactions + [final_output]
    model = TokenModel(outputs)
    result = runtime(model, context_threshold_tokens=15000 if compactions == 0 else 1).run("q1", "question")
    assert result.status == "completed"
    assert result.final_answer == (PLAIN_ANSWER if final_output == PLAIN_FINISH else "done")
    assert result.turn_records[-1]["completion"] == model.results[-1].text
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


@pytest.mark.parametrize("final_output", [FINISH, PLAIN_FINISH])
def test_forced_answer_appends_control_without_rewriting_or_compacting(final_output):
    model = TokenModel([SEARCH, final_output])
    result = runtime(model, max_tool_calls=1).run("q", "question")
    assert result.status == "completed"
    assert result.final_answer == (PLAIN_ANSWER if final_output == PLAIN_FINISH else "done")
    text = model.tokenizer.decode(model.requests[-1].prompt_token_ids)
    assert text.count("<forced_answer_request>") == 1
    assert text.count("<|im_start|>assistant") == 2
    assert model.requests[-1].generation_kind == "forced_answer"
    assert model.tokenizer.template_calls == 1


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("finish_reason", ["length", "stop"])
def test_unwrapped_final_answer_requires_sampled_end_of_turn(forced, finish_reason):
    class UnterminatedAnswerModel(TokenModel):
        def generate_token_batch(self, requests):
            results = super().generate_token_batch(requests)
            if len(self.requests) == 2:
                from dataclasses import replace
                results[0] = replace(results[0], finish_reason=finish_reason)
            return results

    # A length-limited generation is malformed even if it sampled an end token;
    # a stop without the sampled end token is malformed too.
    final = PLAIN_FINISH if finish_reason == "length" else PLAIN_FINISH.removesuffix("<|im_end|>")
    model = UnterminatedAnswerModel([SEARCH, final])
    result = runtime(model, max_tool_calls=1 if forced else None).run("q", "question")
    assert result.status == "malformed_tool_call"
    assert result.final_answer is None
    assert len(result.trajectory_records) == 1
    assert result.trajectory_records[0]["termination_kind"] == "malformed"


def test_overflow_retains_untrained_tool_tail():
    model = TokenModel([SEARCH])
    rt = runtime(model)
    active = rt._new_active_episode("q", "question")
    response = Message(role="user", content=format_tool_response('[{"docid": "d", "snippet": "fact"}]'))
    completion = [999999] + model.tokenizer.encode(SEARCH)
    response_ids = rt.token_renderer.render_tool_result(response, completion_ids=completion, finish_reason="stop")
    rt.max_context_tokens = (
        len(active.token_ledger) + len(rt.token_renderer.header("action")) + len(completion) + len(response_ids)
    )
    result = rt.run("q", "question")
    assert result.status == "context_length_exceeded"
    record = result.trajectory_records[0]
    payload = record["collection_tokens"]
    assert len(payload["full_token_ids"]) > len(payload["generations"][-1]["full_token_ids"])
    assert payload["spans"][-1]["kind"] == "tool_result"
    assert len(extract_trainable_samples([record], result.turn_rewards)) == 1


@pytest.mark.parametrize("field,value", [("split_seed", 1), ("train_limit", 2), ("eval_limit", 2),
                                         ("offset", 1), ("limit", 10)])
def test_lineage_preflight_rejects_changed_dataset_split(tmp_path, field, value):
    from types import SimpleNamespace
    from self_summarization_agent.collection_contract import collection_profile_id, validate_artifact_lineage
    from self_summarization_agent.config import DatasetConfig
    from self_summarization_agent.token_stream import TITO_CONTRACT

    config = SimpleNamespace(
        model=SimpleNamespace(chat_template_path=TEMPLATE_PATH, enable_thinking=True),
        rollout=SimpleNamespace(backend="vllm_offline"),
        dataset=DatasetConfig(train_limit=3, eval_limit=1),
        experiment=SimpleNamespace(name="grpo", seed=0),
    )
    result = runtime(TokenModel([FINISH])).run("q", "question")
    path = tmp_path / "raw.jsonl"
    identity = collection_profile_id(config, tmp_path)
    path.write_text(json.dumps({
        "collection_contract": TITO_CONTRACT, "collection_profile_id": identity,
        "trajectory_records": result.trajectory_records,
    }), encoding="utf-8")
    config.dataset.shuffle = True
    config.experiment.name = "value-mc"
    config.experiment.seed = 99
    assert collection_profile_id(config, tmp_path) == identity
    validate_artifact_lineage([path], config=config, checkpoint=tmp_path)
    setattr(config.dataset, field, value)
    with pytest.raises(ValueError, match="fresh output lineage"):
        validate_artifact_lineage([path], config=config, checkpoint=tmp_path)


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
    assert compute_group_advantages(samples) == pytest.approx([1, 1, -1], abs=3e-6)


def test_wrongly_closed_thinking_is_malformed_but_retained():
    model = TokenModel([SEARCH.replace("</think>", "</thinking>"), FINISH])
    result = runtime(model).run("q", "question")
    assert result.status == "malformed_tool_call"
    assert result.tool_call_counts["search"] == 0
    assert "</thinking>" in result.trajectory_records[0]["completion"]
    assert model.tokenizer.template_calls == 1
    anomalies = [t for t in result.turn_records if t.get("kind") == "format_anomaly"]
    assert [a["anomaly"] for a in anomalies] == ["wrongly_closed_thinking"]
    samples = extract_trainable_samples(result.trajectory_records, result.turn_rewards)
    assert len(samples) == 1


def test_unclosed_thinking_is_malformed_without_dispatch():
    model = TokenModel([SEARCH.replace("</think>", ""), FINISH])
    rejected = runtime(model).run("q", "question")
    assert rejected.status == "malformed_tool_call"
    assert rejected.tool_call_counts["search"] == 0
    anomalies = [t for t in rejected.turn_records if t.get("kind") == "format_anomaly"]
    assert [a["anomaly"] for a in anomalies] == ["unclosed_thinking"]
    assert len(extract_trainable_samples(rejected.trajectory_records, rejected.turn_rewards)) == 1


@pytest.mark.parametrize("variant", ["</thinking>", "</Think>", "</think >", "</thinkx>", "</THINK>"])
def test_wrong_thinking_closers_are_malformed(variant):
    model = TokenModel([SEARCH.replace("</think>", variant), FINISH])
    result = runtime(model).run("q", "question")
    assert result.status == "malformed_tool_call"
    assert result.tool_call_counts["search"] == 0


def test_finish_with_wrongly_closed_thinking_is_malformed():
    model = TokenModel([FINISH.replace("</think>", "</thinking>")])
    result = runtime(model).run("q", "question")
    assert result.status == "malformed_tool_call"
    assert result.final_answer is None


def test_forced_answer_with_wrongly_closed_thinking_is_malformed():
    model = TokenModel([SEARCH, FINISH.replace("</think>", "</thinking>")])
    result = runtime(model, generated_token_budget=1).run("q", "question")
    assert result.status == "malformed_tool_call"
    assert result.final_answer is None
    anomalies = [t for t in result.turn_records if t.get("kind") == "format_anomaly"]
    assert [a["anomaly"] for a in anomalies] == ["wrongly_closed_thinking"]


def test_summary_with_wrongly_closed_thinking_is_malformed_and_flagged():
    model = TokenModel([SEARCH, SUMMARY.replace("</think>", "</thinking>"), FINISH])
    result = runtime(model, context_threshold_tokens=1).run("q", "question")
    assert result.status == "malformed_tool_call"
    assert len(result.trajectory_records) == 1
    anomalies = [t for t in result.turn_records if t.get("kind") == "format_anomaly"]
    assert [a["anomaly"] for a in anomalies] == ["wrongly_closed_thinking"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("reasoning</think>\n<search>q</search>", None),
        ("reasoning</think>\n<answer>done</answer>", None),
        ("reasoning</thinking>\n<search>q</search>", "wrongly_closed_thinking"),
        ("reasoning</Think>\n<search>q</search>", "wrongly_closed_thinking"),
        ("reasoning</think >\n<search>q</search>", "wrongly_closed_thinking"),
        ("reasoning</thinkx>\n<search>q</search>", "wrongly_closed_thinking"),
        ("reasoning<search>q</search>", "unclosed_thinking"),
        ("reasoning that just ends", "unclosed_thinking"),
        ("</think>ok</think >", "wrongly_closed_thinking"),
    ],
)
def test_classify_thinking_closure(text, expected):
    from self_summarization_agent.runtime import classify_thinking_closure

    assert classify_thinking_closure(text, enable_thinking=True) == expected
    assert classify_thinking_closure(text, enable_thinking=False) is None


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


def native_call(name, key, value):
    return f"</think><tool_call>\n<function={name}>\n<parameter={key}>\n{value}\n</parameter>\n</function>\n</tool_call><|im_end|>"


def native_action(*calls):
    return "</think>\n" + "\n".join(
        native_call(*call).removeprefix("</think>").removesuffix("<|im_end|>") for call in calls
    ) + "<|im_end|>"


@pytest.mark.parametrize(("summary", "termination"), [
    (None, "final_answer"),
    (SUMMARY, "compaction"),
    ("</think>invalid summary<|im_end|>", "malformed"),
])
def test_multiple_native_calls_complete_one_round_before_the_next_generation(summary, termination):
    action = native_action(
        ("get_document", "docid", "d"), ("search", "query", "q"), ("search", "query", "q")
    )
    valid = summary in (None, SUMMARY)
    outputs = [action] + ([summary] if summary else [])
    if valid:
        outputs.append(native_call("finish", "answer", "done"))
    model = TokenModel(outputs, native=True)
    result = runtime(model, context_threshold_tokens=1 if summary else 15000).run("q", "question")

    assert result.status == ("completed" if valid else "malformed_tool_call")
    assert result.tool_call_counts == {"search": 2, "get_document": 1}
    assert result.token_usage["retired_round_count"] == (1 if summary == SUMMARY else 0)
    assert len([turn for turn in result.turn_records if turn["kind"] == "tool"]) == 1
    assert len(result.trajectory_records) == model.tokenizer.template_calls == (2 if summary == SUMMARY else 1)
    record = result.trajectory_records[0]
    assert record["termination_kind"] == termination
    assistant, *tools = record["messages"][2:6]
    assert assistant["role"] == "assistant"
    document_parameters = record["tools"][1]["function"]["parameters"]
    assert document_parameters["required"] == ["docid"]
    assert "docid" in document_parameters["properties"]
    assert [tool["role"] for tool in tools] == ["tool"] * 3
    call_ids = [call["id"] for call in assistant["tool_calls"]]
    assert len(set(call_ids)) == 3
    assert [tool["tool_call_id"] for tool in tools] == call_ids
    assert tools[0]["content"] == "fact"
    assert [json.loads(tool["content"]) for tool in tools[1:]] == [[{"docid": "d", "snippet": "fact"}]] * 2
    assert result.turn_records[0]["completion"] == model.results[0].text

    tokens = record["collection_tokens"]
    first, following = tokens["generations"]
    prefix = first["full_token_ids"]
    assert first["completion_token_ids"] == model.results[0].completion_token_ids
    assert following["prompt_token_ids"][:len(prefix)] == prefix
    result_spans = [span for span in tokens["spans"] if span["kind"] == "tool_result"]
    span, = result_spans
    assert span["start"] == len(prefix)
    assert not any(tokens["assistant_token_mask"][span["start"]:span["end"]])
    assert model.tokenizer.decode(tokens["full_token_ids"][span["start"]:span["end"]]) == (
        "\n<|im_start|>user\n"
        + "\n".join(f"<tool_response>\n{tool['content']}\n</tool_response>" for tool in tools)
        + "<|im_end|>\n"
    )
    if summary == SUMMARY:
        successor = result.trajectory_records[1]
        assert successor["messages"][:2] == record["messages"][:2]
        assert successor["messages"][2] == {"role": "user", "content": format_compacted_summary("evidence")}
    if not valid:
        assert not result.summary_turns
        assert record["completion"].endswith(model.results[-1].text)
    assert len(extract_trainable_samples(result.trajectory_records, result.turn_rewards)) == len(
        result.trajectory_records
    )


@pytest.mark.parametrize("max_compactions", [0, 1])
def test_oversized_native_tool_round_fits_before_interval_finalization(max_compactions):
    action = native_action(("get_document", "docid", "a"), ("get_document", "docid", "b"))
    outputs = [action, SUMMARY, native_call("finish", "answer", "done")] if max_compactions else [action]
    model = TokenModel(outputs, native=True)
    model.max_new_tokens = 512
    rt = runtime(model, context_threshold_tokens=1, max_compactions=max_compactions)
    reserve = max(
        len(rt.token_renderer.header("summary")) + rt.max_summary_tokens,
        len(rt.token_renderer.header("forced_answer")) + model.max_new_tokens,
    )
    document_length = reserve + 1000
    rt.backend = FakeBackend(search_index={}, documents={"a": "a" * document_length, "b": "b" * document_length})
    initial_length = len(rt._new_active_episode("q", "question").token_ledger)
    model.max_model_len = initial_length + reserve + 600

    result = rt.run("q", "question")

    assert result.status == ("completed" if max_compactions else "overlong")
    assert len(result.trajectory_records) == max_compactions + 1
    record = result.trajectory_records[0]
    assistant, first_response, second_response = record["messages"][2:5]
    assert [first_response["tool_call_id"], second_response["tool_call_id"]] == [
        call["id"] for call in assistant["tool_calls"]
    ]
    assert first_response["content"].startswith("a")
    assert second_response["content"].startswith("b")
    assert all("[Observation truncated" in tool["content"] for tool in (first_response, second_response))
    tokens = record["collection_tokens"]
    first = tokens["generations"][0]
    assert first["completion_token_ids"] == model.results[0].completion_token_ids
    assert tokens["full_token_ids"][:len(first["full_token_ids"])] == first["full_token_ids"]
    if max_compactions:
        assert tokens["generations"][1]["prompt_token_ids"][:len(first["full_token_ids"])] == first["full_token_ids"]
    assert len(tokens["full_token_ids"]) <= model.max_model_len
    tool_span, = [span for span in tokens["spans"] if span["kind"] == "tool_result"]
    assert not any(tokens["assistant_token_mask"][tool_span["start"]:tool_span["end"]])
    assert len(extract_trainable_samples(result.trajectory_records, result.turn_rewards)) == max_compactions + 1


def test_exact_token_runtime_stops_before_tools_when_response_wrappers_cannot_fit():
    action = native_call("search", "query", "q")
    model = TokenModel([action], native=True)
    rt = runtime(model)
    active = rt._new_active_episode("q", "question")
    prompt_length = len(active.token_ledger) + len(rt.token_renderer.header("action"))
    rt.max_context_tokens = prompt_length + 1 + len(model.tokenizer.encode(action))

    result = rt.run("q", "question")

    assert result.status == "context_length_exceeded"
    assert result.tool_call_counts == {"search": 0, "get_document": 0}
    record, = result.trajectory_records
    assert record["termination_kind"] == "context_overflow"
    assert record["messages"][-1]["role"] == "assistant"
    tokens = record["collection_tokens"]
    assert tokens["full_token_ids"] == tokens["generations"][0]["full_token_ids"]
    assert len(tokens["full_token_ids"]) == rt.max_context_tokens
    assert len(extract_trainable_samples(result.trajectory_records, result.turn_rewards)) == 1


@pytest.mark.parametrize("second_call", [("get_document", "doc_id", "d"), ("finish", "answer", "done")])
def test_invalid_multi_call_action_runs_no_tools(second_call):
    model = TokenModel([native_action(("search", "query", "q"), second_call)], native=True)
    result = runtime(model).run("q", "question")
    assert result.status == "malformed_tool_call"
    assert result.tool_call_counts == {"search": 0, "get_document": 0}
    record, = result.trajectory_records
    assert record["completion"] == model.results[0].text
    assert all(message["role"] != "tool" for message in record["messages"])


def test_multi_call_action_respects_remaining_tool_budget():
    action = native_action(("search", "query", "q"), ("get_document", "docid", "other"))
    model = TokenModel([action, native_call("finish", "answer", "done")], native=True)
    result = runtime(model, max_tool_calls=1).run("q", "question")
    assert result.status == "completed"
    assert result.tool_call_counts == {"search": 1, "get_document": 0}
    assert result.retrieved_docids == ["d"]
    record, = result.trajectory_records
    assert record["termination_kind"] == "forced_answer"
    skipped = record["messages"][4]
    assert skipped["tool_call_id"] == record["messages"][2]["tool_calls"][1]["id"]
    assert skipped["content"] == "Error: Tool call budget exhausted."


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("limit", [0, 2])
def test_compaction_limit_ends_at_threshold_with_zero_interval_rewards(native, limit):
    from self_summarization_agent.config import load_run_config, parse_cli_overrides
    from self_summarization_agent.dataset import QueryExample
    from self_summarization_agent.judge import RewardJudge
    from self_summarization_agent.judge_step import judge_rollout_rows
    from self_summarization_agent.launcher_utils import build_runtime, serialize_runtime_result
    from self_summarization_agent.rollout_collection import apply_judged_rewards
    from self_summarization_agent.segment_returns import ordered_rollout_chains
    from self_summarization_agent.token_value import validate_token_samples

    search = native_call("search", "query", "q") if native else SEARCH
    model = TokenModel([search, SUMMARY] * limit + [search], native=native)
    config = load_run_config("configs/run/default.yaml", parse_cli_overrides([
        f"runtime.max_compactions={limit}",
        "runtime.context_threshold_tokens=1",
        "runtime.max_summary_tokens=100",
        "runtime.tool_budget=0",
        "runtime.generated_token_budget=1",
    ]))
    backend = FakeBackend(search_index={"q": ["d"]}, documents={"d": "fact"})
    result = build_runtime(model, backend, config.runtime).run("q1", "question")

    assert result.status == "overlong"
    assert result.final_answer is None
    assert result.tool_call_counts["search"] == limit + 1
    assert result.token_usage["summary_count"] == limit
    assert result.token_usage["forced_answer_reasons"] == []
    assert [request.generation_kind for request in model.requests] == ["action", "summary"] * limit + ["action"]
    assert [record["termination_kind"] for record in result.trajectory_records] == ["compaction"] * limit + ["overlong"]
    rewards = {f"trajectory-{index + 1}": 0.0 for index in range(limit + 1)}
    assert result.turn_rewards == rewards
    terminal = result.trajectory_records[-1]
    tokens = terminal["collection_tokens"]
    assert terminal["completion"] == model.results[-1].text
    assert tokens["generations"][0]["completion_token_ids"] == model.results[-1].completion_token_ids
    assert tokens["spans"][-1]["kind"] == "tool_result"
    assert not tokens["spans"][-1]["sampled"]
    assert model.tokenizer.template_calls == limit + 1
    if native:
        assert terminal["messages"][-1]["tool_call_id"] == terminal["messages"][-2]["tool_calls"][0]["id"]

    row = serialize_runtime_result(result, query_text="question", include_rewards=False)
    row["rollout_index"] = 0
    judge = RewardJudge(model)
    judged, = judge_rollout_rows([row], judge=judge, examples_by_query_id={})
    assert judged["turn_rewards"] == rewards
    assert judged["judge"]["outcome"] == "overlong"
    assert judged["trainable_sample_count"] == limit + 1
    example = QueryExample(query_id="q1", query="question", answer="answer")
    assert apply_judged_rewards(result, example, judge)["outcome"] == "overlong"
    assert result.turn_rewards == rewards
    for record in judged["trajectory_records"]:
        record["training_cache"] = build_rollout_native_training_cache(record["collection_tokens"])
    samples = extract_trainable_samples(judged["trajectory_records"], rewards, rollout_id="q1:0")
    validate_token_samples(samples)
    chain, = ordered_rollout_chains([judged], samples)
    assert chain[-1]["terminal"]
    assert chain[-1]["reward"] == 0.0


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("limit", [0, 2])
def test_force_answer_at_compaction_limit_appends_one_terminal_control(native, limit):
    from self_summarization_agent.config import load_run_config, parse_cli_overrides
    from self_summarization_agent.launcher_utils import build_runtime

    search = native_call("search", "query", "q") if native else SEARCH
    finish = native_call("finish", "answer", "done") if native else FINISH
    model = TokenModel([search, SUMMARY] * limit + [search, finish], native=native)
    config = load_run_config("configs/run/default.yaml", parse_cli_overrides([
        f"runtime.max_compactions={limit}", "runtime.force_answer_at_compaction_limit=true",
        "runtime.context_threshold_tokens=1", "runtime.tool_budget=0", "runtime.generated_token_budget=1",
    ]))
    backend = FakeBackend(search_index={"q": ["d"]}, documents={"d": "fact"})
    result = build_runtime(model, backend, config.runtime).run("q1", "question")

    assert result.status == "completed"
    assert result.final_answer == "done"
    assert result.token_usage["forced_answer_reasons"] == ["max_compactions"]
    assert [r.generation_kind for r in model.requests] == ["action", "summary"] * limit + ["action", "forced_answer"]
    assert [r["termination_kind"] for r in result.trajectory_records] == ["compaction"] * limit + ["forced_answer"]
    terminal = result.trajectory_records[-1]
    control = terminal["messages"][-2]
    assert control["role"] == "system" and "<forced_answer_request>" in control["content"]
    if native:
        assert terminal["messages"][-3]["tool_call_id"] == terminal["messages"][-4]["tool_calls"][0]["id"]
    action, answer = terminal["collection_tokens"]["generations"]
    assert answer["prompt_token_ids"][:len(action["full_token_ids"])] == action["full_token_ids"]
    assert len(extract_trainable_samples(result.trajectory_records, result.turn_rewards)) == limit + 1


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("case", ["below_threshold", "answer_at_limit", "malformed_summary"])
def test_compaction_limit_still_allows_completion_and_rejects_malformed_summaries(native, case):
    search = native_call("search", "query", "q") if native else SEARCH
    finish = native_call("finish", "answer", "done") if native else FINISH
    if case == "below_threshold":
        outputs, limit, threshold, status = [search, finish], 0, 15000, "completed"
    elif case == "answer_at_limit":
        outputs, limit, threshold, status = [search, SUMMARY, finish], 1, 1, "completed"
    else:
        outputs, limit, threshold, status = [search, "</think><summary></summary><|im_end|>"], 1, 1, "empty_summary"

    model = TokenModel(outputs, native=native)
    result = runtime(model, context_threshold_tokens=threshold, max_compactions=limit,
                     force_answer_at_compaction_limit=True, max_tool_calls=0, generated_token_budget=1).run("q", "question")
    assert result.status == status
    assert result.token_usage["forced_answer_reasons"] == []
    if case == "malformed_summary":
        assert result.summary_turns == []
        assert result.turn_rewards == {"trajectory-1": 0.0}
        assert result.trajectory_records[-1]["completion"].endswith(model.results[-1].text)
    else:
        assert result.final_answer == "done"
        assert len(result.summary_turns) == limit


@pytest.mark.parametrize("compact", [False, True])
def test_native_tool_round_and_summary_keep_linked_messages(compact):
    actions = [native_call("search", "query", "q"), native_call("get_document", "docid", "d")]
    outputs = []
    for action in actions:
        outputs.append(action)
        if compact:
            outputs.append(SUMMARY)
    model = TokenModel([*outputs, native_call("finish", "answer", "done")], native=True)
    result = runtime(model, context_threshold_tokens=1 if compact else 15000).run("q", "question")
    assert result.status == "completed"
    assert result.final_answer == "done"
    assert result.tool_call_counts == {"search": 1, "get_document": 1}
    intervals = 3 if compact else 1
    assert model.tokenizer.template_calls == intervals
    assert len(extract_trainable_samples(result.trajectory_records, result.turn_rewards)) == intervals
    for record in result.trajectory_records:
        for assistant, tool in zip(record["messages"], record["messages"][1:]):
            if tool["role"] == "tool":
                assert tool["tool_call_id"] == assistant["tool_calls"][0]["id"]
        generations = record["collection_tokens"]["generations"]
        for previous, following in zip(generations, generations[1:]):
            prefix = previous["prompt_token_ids"] + previous["completion_token_ids"]
            assert following["prompt_token_ids"][:len(prefix)] == prefix
        for generation in generations:
            assert generation["completion_token_ids"][0] == 999999
    for successor in result.trajectory_records[1:]:
        assert successor["messages"][:2] == result.trajectory_records[0]["messages"][:2]
        assert successor["messages"][2] == {"role": "user", "content": format_compacted_summary("evidence")}


@pytest.mark.parametrize("summary", [
    SUMMARY.removesuffix("<|im_end|>"),
    "</thinking>" + SUMMARY,
    native_call("search", "query", "q"),
])
def test_native_malformed_summary_keeps_raw_sample_and_terminates(summary):
    model = TokenModel([native_call("search", "query", "q"), summary], native=True)
    result = runtime(model, context_threshold_tokens=1).run("q", "question")
    assert result.status == "malformed_tool_call"
    assert not result.summary_turns
    assert model.tokenizer.template_calls == 1
    record, = result.trajectory_records
    assert record["termination_kind"] == "malformed"
    assert record["collection_tokens"]["generations"][-1]["completion_token_ids"] == model.results[-1].completion_token_ids


def test_collection_trace_shows_raw_native_calls_and_recorded_inputs(tmp_path, monkeypatch):
    from pathlib import Path
    from self_summarization_agent.dataset import QueryExample

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from simulate_collection import trace_collection

    search = native_call("search", "query", "q")
    finish = native_call("finish", "answer", "done")
    model = TokenModel([search, SUMMARY, finish], native=True)
    output_path = tmp_path / "trace.txt"
    result = trace_collection(
        runtime=runtime(model, context_threshold_tokens=1),
        generator=model,
        example=QueryExample(query_id="q", query="question", answer="done"),
        sample_index=0,
        output_path=output_path,
        include_formatted_prompt=True,
    )

    assert result.status == "completed"
    assert model.tokenizer.template_calls == 2
    trace = output_path.read_text(encoding="utf-8")
    assert "native_tools: True" in trace
    assert '"tool_call_id": "' in trace
    exact_first_input = model.tokenizer.decode(model.requests[0].prompt_token_ids)
    assert "--- Recorded model inputs ---" in trace and exact_first_input in trace
    assert "--- Recorded model inputs ---\ntrajectory 2 (trajectory-2), generation 0" in trace
    assert "--- Trainable assistant completions ---\nsampled-only-reason" + search in trace
    assert "--- Trainable assistant completions ---\nsampled-only-reason" + finish in trace


@pytest.mark.parametrize("native", [False, True])
def test_vllm_adapter_submits_ids_and_preserves_raw_sampled_logprobs(native):
    from types import SimpleNamespace
    generator = VLLMGenerator.__new__(VLLMGenerator)
    generator.supports_native_tools = native
    generator.enable_thinking = True
    generator.tokenizer = Tokenizer()
    generator.max_new_tokens = 99
    generator.temperature = 0.7
    generator.top_p = 0.9
    generator.do_sample = True
    generator.sampling_extra = {}
    generator._sampling_params_cls = lambda **kw: kw
    seen = []
    ids = [999999] + generator.tokenizer.encode(native_call("search", "query", "q"))
    def generate(prompts, params):
        seen.append((prompts, params))
        completion = SimpleNamespace(token_ids=ids, logprobs=[{i: SimpleNamespace(logprob=-0.2)} for i in ids],
                                     cumulative_logprob=-0.2 * len(ids), finish_reason="stop")
        return [SimpleNamespace(prompt_token_ids=prompts[0]["prompt_token_ids"], outputs=[completion])]
    generator.llm = SimpleNamespace(generate=generate)
    result = generator.generate_token_batch([TokenRequest((1, 999999, 2), max_new_tokens=8)])[0]
    assert seen[0][0] == [{"prompt_token_ids": [1, 999999, 2]}]
    assert seen[0][1][0]["max_tokens"] == 8
    assert seen[0][1][0]["logprobs"] == 0
    assert seen[0][1][0]["detokenize"] is False
    generator.sampling_extra = {"stop": ["stop here"]}
    generator.generate_token_batch([TokenRequest((1,))])
    assert seen[1][1][0].get("detokenize", True)
    assert result.completion_token_ids == ids
    assert result.token_logprobs == [-0.2] * len(ids)
    assert result.token_logprobs_mode == "raw_logprobs"
    assert result.text == "sampled-only-reason" + native_call("search", "query", "q")
    if native:
        call, = result.message.tool_calls
        assert call.name == "search" and call.arguments == {"query": "q"} and call.id
    else:
        assert result.message is None


def test_offline_native_controls_apply_per_request(monkeypatch):
    from types import SimpleNamespace

    pytest.importorskip("vllm.sampling_params")
    monkeypatch.setattr(VLLMGenerator, "__post_init__", lambda self: None)
    generator = VLLMGenerator("model", 128, 0, 1, False, enable_thinking=True, supports_native_tools=True)
    generator._sampling_params_cls = dict
    generator.tokenizer = Tokenizer()
    settings = []

    def generate(prompts, params):
        settings.extend(params)
        completion = SimpleNamespace(token_ids=[999999, 2], finish_reason="stop")
        return [SimpleNamespace(prompt_token_ids=p["prompt_token_ids"], outputs=[completion]) for p in prompts]

    generator.llm = SimpleNamespace(generate=generate)
    generator.generate_token_batch([TokenRequest((1,), generation_kind=kind)
                                    for kind in ("action", "summary", "forced_answer")])
    assert "structured_outputs" not in settings[0]
    summary = SUMMARY.removesuffix("<|im_end|>")
    finish = native_call("finish", "answer", "done").removesuffix("<|im_end|>")
    search = native_call("search", "query", "q").removesuffix("<|im_end|>")
    for params, expected in zip(settings[1:], [summary, finish]):
        regex = params["structured_outputs"].regex
        assert re.fullmatch(regex, expected) and not re.fullmatch(regex, search)


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
    adapter = SimpleNamespace(model=SimpleNamespace(device="cpu", generate=generate,
        generation_config=SimpleNamespace(eos_token_id=7)),
        tokenizer=SimpleNamespace(pad_token_id=0, decode=tokenizer.decode,
                                  convert_tokens_to_ids=lambda token: 2, convert_ids_to_tokens=lambda _: "<|im_end|>"),
        max_new_tokens=99, do_sample=True, temperature=0.6, top_p=1.0,
        sampling_extra={"presence_penalty": 0.0})
    result = TransformersGenerator.generate_token_batch(adapter, [TokenRequest((1, 999999, 2), max_new_tokens=8)])[0]
    assert seen[0]["input_ids"].tolist() == [[1, 999999, 2]]
    assert seen[0]["max_new_tokens"] == 8
    assert seen[0]["eos_token_id"] == [7, 2]
    assert "presence_penalty" not in seen[0]
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
    from self_summarization_agent.config import RuntimeConfig
    from self_summarization_agent.token_stream import TITO_CONTRACT
    config = SimpleNamespace(model=SimpleNamespace(chat_template_path=TEMPLATE_PATH, enable_thinking=True),
                             rollout=SimpleNamespace(backend="vllm_offline"), runtime=RuntimeConfig())
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
    config.runtime.max_compactions = 0
    with pytest.raises(ValueError, match="fresh output lineage"):
        validate_artifact_lineage([path], config=config, checkpoint=tmp_path)
    capped_profile = collection_profile_id(config, tmp_path)
    config.runtime.force_answer_at_compaction_limit = True
    assert collection_profile_id(config, tmp_path) != capped_profile
    config.runtime.max_compactions = None
    validate_artifact_lineage([path], config=config, checkpoint=tmp_path)
    config.rollout.browsecomp_native_tools = True
    with pytest.raises(ValueError, match="fresh output lineage"):
        validate_artifact_lineage([path], config=config, checkpoint=tmp_path)
    config.rollout.browsecomp_native_tools = False
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
    negative.rollout_id, negative.reward = "q:1", 0.0
    samples.append(negative)
    before = [p.detach().clone() for p in parameters]
    metrics = trainer.step({"q": samples})
    assert metrics.sample_count == 3
    assert metrics.optimizer_step_count == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, parameters))
    if value_mode:
        assert metrics.mean_advantage == pytest.approx(1 / 6)
        assert metrics.extra_metrics["value/state_count"] == 3
        assert metrics.extra_metrics["value/rollout_count"] == 2


@pytest.mark.parametrize(("outputs", "limit", "outcome", "expected"), [
    ([FINISH], 0, "correct_answer", 1.0),
    ([SEARCH, SUMMARY, FINISH], 1, "correct_answer", 1.0),
    ([SEARCH, FINISH], 0, "correct_answer", 0.9),
    ([SEARCH, SUMMARY, SEARCH, FINISH], 1, "correct_answer", 0.9),
    ([SEARCH, FINISH], 0, "wrong_answer", 0.0),
    ([SEARCH, "</think><summary></summary><|im_end|>"], 1, "wrong_answer", 0.0),
])
def test_voluntary_stopping_judgment_through_segment_returns(outputs, limit, outcome, expected):
    from self_summarization_agent.config import load_train_config
    from self_summarization_agent.judge import JudgeDecision
    from self_summarization_agent.judge_step import apply_decision_to_rollout_row
    from self_summarization_agent.launcher_utils import serialize_runtime_result
    from self_summarization_agent.segment_returns import ordered_rollout_chains, segment_gae
    from self_summarization_agent.train_step import samples_from_rollout_rows

    config = load_train_config("configs/train/native_segment_gae_force_answer.yaml",
                               {"benchmark.reward_profile": "voluntary_stopping"})
    profile = config.training.value.reward_profile
    result = runtime(TokenModel(outputs), max_compactions=limit, context_threshold_tokens=1,
                     force_answer_at_compaction_limit=True).run("q", "question")
    raw = serialize_runtime_result(result, query_text="question", include_rewards=False)
    raw.update(policy_checkpoint_id="test", rollout_index=0)
    decision = JudgeDecision(outcome=outcome, judge_prompt=None, judge_response=None, parse_error=False)
    row = apply_decision_to_rollout_row(raw, decision, reward_profile=profile)
    assert set(row["turn_rewards"].values()) == {expected}
    assert row["judge"]["outcome"] == (outcome if result.status == "completed" else result.status)
    for record in row["trajectory_records"]:
        record["training_cache"] = build_rollout_native_training_cache(record["collection_tokens"])
    samples = samples_from_rollout_rows([row], expected_checkpoint_id="test", reward_profile=profile)
    chain, = ordered_rollout_chains([row], samples, profile)
    values = {tuple(item["key"]): 0.5 for item in chain}
    returns = segment_gae([chain], values, 1.0, profile)
    for item in chain:
        interval = returns[tuple(item["key"])]
        assert interval["transition_reward"] == (expected if item["terminal"] else 0)
        assert interval["target"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="reward profile"):
        samples_from_rollout_rows([row], expected_checkpoint_id="test")
