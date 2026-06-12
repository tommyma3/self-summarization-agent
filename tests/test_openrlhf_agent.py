import asyncio
from types import SimpleNamespace

from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.judge import JudgeDecision
import self_summarization_agent.openrlhf_agent as openrlhf_agent
from self_summarization_agent.openrlhf_agent import AgentExecutor, _AgentResources
from self_summarization_agent.runtime import EpisodeRuntime


class CharTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None):
        del add_special_tokens, return_tensors
        return {"input_ids": [SimpleTensor([ord(char) for char in text])]}

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class SimpleTensor(list):
    def tolist(self):
        return list(self)


class FakeOutput:
    def __init__(self, text: str) -> None:
        self.text = text
        self.token_ids = [ord(char) for char in text]
        self.logprobs = None


class FakeRequestOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [FakeOutput(text)]


class RecordingLLMEngine:
    def __init__(self, tokenizer: CharTokenizer, outputs: list[str]) -> None:
        self.tokenizer = tokenizer
        self.outputs = outputs
        self.prompts: list[str] = []

    async def generate(self, prompt_tokens, sampling_params):
        del sampling_params
        self.prompts.append(self.tokenizer.decode(prompt_tokens))
        return FakeRequestOutput(self.outputs.pop(0))


class FakeJudge:
    def evaluate(self, example, status: str, response: str) -> JudgeDecision:
        del example, response
        outcome = "correct_answer" if status == "completed" else "budget_exhausted"
        return JudgeDecision(outcome=outcome, judge_prompt="judge", judge_response="correct: yes", parse_error=False)


def tool_output(json_text: str, thinking: str = "thinking") -> str:
    return f"<think>{thinking}</think>\n{json_text}"


def install_resources(runtime: EpisodeRuntime) -> None:
    openrlhf_agent._RESOURCES = _AgentResources(
        config=SimpleNamespace(),
        backend=runtime.backend,
        runtime=runtime,
        local_judge=FakeJudge(),
        judge_url=None,
    )


def test_openrlhf_executor_keeps_tool_cot_out_of_next_prompt() -> None:
    tokenizer = CharTokenizer()
    llm = RecordingLLMEngine(
        tokenizer,
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "q"}}', thinking="secret search thought"),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}', thinking="secret final thought"),
        ],
    )
    runtime = EpisodeRuntime(
        model=SimpleNamespace(),
        backend=FakeBackend(search_index={"q": ["doc-1"]}, documents={"doc-1": "fact"}),
        context_threshold_tokens=1000,
        max_context_tokens=4096,
        token_counter=lambda text: len(text),
    )
    install_resources(runtime)

    result = asyncio.run(
        AgentExecutor().execute(
            "question",
            '{"query_id": "q1", "answer": "done"}',
            SimpleNamespace(logprobs=None, max_tokens=None),
            4096,
            tokenizer,
            llm,
        )
    )

    assert len(llm.prompts) == 2
    assert "secret search thought" not in llm.prompts[1]
    assert '<think>' not in llm.prompts[1]
    assert '### ASSISTANT_TOOL_CALL\n{"tool_name": "search", "arguments": {"query": "q"}}' in llm.prompts[1]
    assert '### TOOL_RESULT\n[{"docid": "doc-1", "snippet": "fact"}]' in llm.prompts[1]
    assert result["reward"] == 1.0
    assert len(result["action_ranges"]) == 2
    trained_text = tokenizer.decode(result["observation_tokens"])
    assert "secret search thought" not in trained_text
    assert "secret final thought" not in trained_text
    assert '{"tool_name": "search", "arguments": {"query": "q"}}' in trained_text
    assert '{"tool_name": "finish", "arguments": {"answer": "done"}}' in trained_text


def test_openrlhf_executor_trains_summary_but_context_uses_post_think_summary() -> None:
    tokenizer = CharTokenizer()
    llm = RecordingLLMEngine(
        tokenizer,
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "first"}}'),
            tool_output('{"tool_name": "search", "arguments": {"query": "second"}}'),
            "<think>summary reasoning</think>\nsummary body",
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ],
    )
    runtime = EpisodeRuntime(
        model=SimpleNamespace(),
        backend=FakeBackend(
            search_index={"first": ["old-doc"], "second": ["trigger-doc"]},
            documents={},
        ),
        context_threshold_tokens=1,
        max_context_tokens=4096,
        token_counter=lambda text: text.count("trigger-doc"),
    )
    install_resources(runtime)

    result = asyncio.run(
        AgentExecutor().execute(
            "question",
            {"query_id": "q1", "answer": "done"},
            SimpleNamespace(logprobs=None, max_tokens=None),
            4096,
            tokenizer,
            llm,
        )
    )

    assert len(llm.prompts) == 4
    final_prompt = llm.prompts[3]
    assert "### SUMMARY\nsummary body" in final_prompt
    assert "summary reasoning" not in final_prompt
    assert len(result["action_ranges"]) == 4
    trained_text = tokenizer.decode(result["observation_tokens"])
    assert "<think>summary reasoning</think>\nsummary body" in trained_text


def test_openrlhf_executor_trains_malformed_tool_call_as_negative_example() -> None:
    tokenizer = CharTokenizer()
    llm = RecordingLLMEngine(tokenizer, ['{"tool_name": "search"}'])
    runtime = EpisodeRuntime(
        model=SimpleNamespace(),
        backend=FakeBackend(search_index={}, documents={}),
        context_threshold_tokens=1000,
        max_context_tokens=4096,
    )
    install_resources(runtime)

    result = asyncio.run(
        AgentExecutor().execute(
            "question",
            {"query_id": "q1", "answer": "done"},
            SimpleNamespace(logprobs=None, max_tokens=None),
            4096,
            tokenizer,
            llm,
        )
    )

    assert result["reward"] == -1.0
    assert len(result["action_ranges"]) == 1
    assert '{"tool_name": "search"}' in tokenizer.decode(result["observation_tokens"])
    assert result["extra_logs"]["status"] == "malformed_tool_call"
