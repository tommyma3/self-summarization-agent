import sys
from types import SimpleNamespace

from self_summarization_agent.config import JudgeConfig, ModelConfig
from self_summarization_agent import generation
from self_summarization_agent.generation import (
    OpenAICompatibleGenerator,
    SGLangGenerator,
    VLLMGenerator,
    build_generator,
)
from self_summarization_agent.models import Message
from self_summarization_agent.prompts import ACTION_TOOLS, ConversationPrompt


class FakeTokenizer:
    chat_template = None


class FakeSamplingParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeLogprob:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


class FakeCompletion:
    def __init__(
        self,
        text: str,
        token_ids: list[int] | None = None,
        cumulative_logprob: float | None = None,
    ) -> None:
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = [{11: FakeLogprob(-0.75)}, {12: FakeLogprob(-1.25)}] if token_ids == [11, 12] else None


class FakeRequestOutput:
    def __init__(self, text: str, prompt_token_ids: list[int] | None = None) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.outputs = [FakeCompletion(text, token_ids=[11, 12], cumulative_logprob=-2.0)]


class FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.params: FakeSamplingParams | None = None

    def generate(self, prompts: list[str], params: FakeSamplingParams) -> list[FakeRequestOutput]:
        self.prompts = prompts
        self.params = params
        return [FakeRequestOutput(f"response:{prompt}") for prompt in prompts]


class FakeSGLangEngine:
    def __init__(self) -> None:
        self.input_ids: list[list[int]] = []
        self.params: dict | None = None
        self.return_logprob = False

    def generate(
        self,
        *,
        input_ids: list[list[int]],
        sampling_params: dict,
        return_logprob: bool = False,
    ) -> list[dict]:
        self.input_ids = input_ids
        self.params = sampling_params
        self.return_logprob = return_logprob
        return [
            {
                "text": f"response:{index}",
                "meta_info": {"output_token_logprobs": [[-0.75, 11], [-1.25, 12]]},
            }
            for index, _ in enumerate(input_ids)
        ]


def test_build_generator_accepts_vllm_offline_backend(monkeypatch) -> None:
    def fake_init(self) -> None:
        self.tokenizer = object()
        self.llm = object()
        self._sampling_params_cls = object()

    monkeypatch.setattr(VLLMGenerator, "__post_init__", fake_init)

    generator = build_generator(
        ModelConfig(
            backend="vllm_offline",
            model_path="/models/demo",
            tensor_parallel_size=2,
            attention_backend="FLASH_ATTN",
            enable_prefix_caching=True,
        ),
        sampling_extra={"top_k": 20},
    )

    assert isinstance(generator, VLLMGenerator)
    assert generator.model_path == "/models/demo"
    assert generator.tensor_parallel_size == 2
    assert generator.enable_prefix_caching is True
    assert generator.sampling_extra == {"top_k": 20}


def test_vllm_generator_enables_prefix_caching_in_offline_engine(monkeypatch) -> None:
    engine_kwargs: dict = {}

    class FakeEngine:
        def __init__(self, **kwargs) -> None:
            engine_kwargs.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeEngine, SamplingParams=FakeSamplingParams),
    )
    monkeypatch.setattr(generation, "_apply_vllm_subprocess_fix", lambda: None)
    monkeypatch.setattr(
        generation.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    VLLMGenerator(
        model_path="/models/demo",
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        enable_prefix_caching=True,
    )

    assert engine_kwargs["enable_prefix_caching"] is True
    assert engine_kwargs["logprobs_mode"] == "raw_logprobs"


def test_build_generator_accepts_sglang_backend(monkeypatch) -> None:
    def fake_init(self) -> None:
        self.tokenizer = object()
        self.engine = object()

    monkeypatch.setattr(SGLangGenerator, "__post_init__", fake_init)

    generator = build_generator(
        ModelConfig(
            backend="sglang",
            model_path="/models/demo",
            tensor_parallel_size=2,
            attention_backend="flashinfer",
            max_model_len=8192,
        )
    )

    assert isinstance(generator, SGLangGenerator)
    assert generator.model_path == "/models/demo"
    assert generator.tensor_parallel_size == 2
    assert generator.attention_backend == "flashinfer"
    assert generator.max_model_len == 8192
    assert generator.require_exact_token_ids is True


def test_sglang_generator_maps_model_options_to_offline_engine(monkeypatch) -> None:
    engine_kwargs: dict = {}

    class FakeEngine:
        def __init__(self, **kwargs) -> None:
            engine_kwargs.update(kwargs)

    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=FakeEngine))
    monkeypatch.setattr(
        generation.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    SGLangGenerator(
        model_path="/models/demo",
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        dtype="bfloat16",
        tensor_parallel_size=2,
        attention_backend="flashinfer",
        max_model_len=8192,
        trust_remote_code=True,
    )

    assert engine_kwargs == {
        "model_path": "/models/demo",
        "dtype": "bfloat16",
        "tp_size": 2,
        "attention_backend": "flashinfer",
        "context_length": 8192,
        "trust_remote_code": True,
    }


def test_build_generator_uses_judge_backend_overrides(monkeypatch) -> None:
    def fake_init(self) -> None:
        self.tokenizer = object()
        self.llm = object()
        self._sampling_params_cls = object()

    monkeypatch.setattr(VLLMGenerator, "__post_init__", fake_init)

    generator = build_generator(
        ModelConfig(
            backend="transformers",
            model_path="/models/policy",
            judge_model_path="/models/legacy-judge",
        ),
        judge_config=JudgeConfig(
            backend="vllm_offline",
            model_path="/models/judge",
            tensor_parallel_size=2,
            attention_backend="FLASH_ATTN",
            max_model_len=8192,
        ),
    )

    assert isinstance(generator, VLLMGenerator)
    assert generator.model_path == "/models/judge"
    assert generator.tensor_parallel_size == 2
    assert generator.attention_backend == "FLASH_ATTN"
    assert generator.max_model_len == 8192


def test_vllm_generator_batches_prompts(monkeypatch) -> None:
    def fake_init(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.llm = FakeLLM()
        self._sampling_params_cls = FakeSamplingParams

    monkeypatch.setattr(VLLMGenerator, "__post_init__", fake_init)

    generator = VLLMGenerator(
        model_path="/models/demo",
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        sampling_extra={"top_k": 20, "presence_penalty": 1.5},
    )

    outputs = generator.generate_batch(["first", "second"])

    assert outputs == ["response:first", "response:second"]
    assert generator.llm.prompts == ["first", "second"]
    assert generator.llm.params.kwargs == {
        "max_tokens": 16,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
    }


def test_vllm_generator_can_return_generation_metadata(monkeypatch) -> None:
    class MetadataLLM(FakeLLM):
        def generate(self, prompts: list[str], params: FakeSamplingParams) -> list[FakeRequestOutput]:
            self.prompts = prompts
            self.params = params
            return [FakeRequestOutput(f"response:{prompt}", prompt_token_ids=[1, 2]) for prompt in prompts]

    def fake_init(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.llm = MetadataLLM()
        self._sampling_params_cls = FakeSamplingParams

    monkeypatch.setattr(VLLMGenerator, "__post_init__", fake_init)
    generator = VLLMGenerator(
        model_path="/models/demo",
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
    )

    outputs = generator.generate_batch_with_metadata(["first"])

    assert outputs[0].text == "response:first"
    assert outputs[0].prompt_token_ids == [1, 2]
    assert outputs[0].completion_token_ids == [11, 12]
    assert outputs[0].cumulative_logprob == -2.0
    assert outputs[0].token_logprobs == [-0.75, -1.25]
    assert outputs[0].token_logprobs_mode == "raw_logprobs"
    assert generator.llm.params.kwargs == {
        "max_tokens": 16,
        "temperature": 0.7,
        "top_p": 0.95,
        "logprobs": 1,
    }


def test_sglang_generator_batches_prompts_and_returns_metadata(monkeypatch) -> None:
    class SGLangTokenizer(FakeTokenizer):
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [1, 2]

    def fake_init(self) -> None:
        self.tokenizer = SGLangTokenizer()
        self.engine = FakeSGLangEngine()

    monkeypatch.setattr(SGLangGenerator, "__post_init__", fake_init)
    generator = SGLangGenerator(
        model_path="/models/demo",
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
    )

    outputs = generator.generate_batch_with_metadata(["first", "second"])

    assert [output.text for output in outputs] == ["response:0", "response:1"]
    assert outputs[0].prompt_token_ids == [1, 2]
    assert outputs[0].completion_token_ids == [11, 12]
    assert outputs[0].cumulative_logprob == -2.0
    assert outputs[0].token_logprobs == [-0.75, -1.25]
    assert outputs[0].token_logprobs_mode == "raw_logprobs"
    assert outputs[0].reference_logprob_source == "sglang_raw_rollout"
    assert generator.engine.input_ids == [[1, 2], [1, 2]]
    assert generator.engine.params == {"max_new_tokens": 16, "temperature": 0.7, "top_p": 0.95}
    assert generator.engine.return_logprob is True


def test_openai_compatible_generator_preserves_native_tools_and_exact_ids(monkeypatch) -> None:
    requests: list[dict] = []

    class ApiTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return [1, 2]

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
            assert token_ids == [11, 12]
            assert skip_special_tokens is False
            return "<think>search</think><tool_call>search</tool_call>"

    class Completions:
        def create(self, **kwargs):
            requests.append(kwargs)
            return {
                "prompt_token_ids": [1, 2, 3],
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "token_ids": [11, 12],
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning": "search",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query": "q"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }

    def fake_init(self) -> None:
        self.tokenizer = ApiTokenizer()
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        self.chat_template = "custom agent template"

    monkeypatch.setattr(OpenAICompatibleGenerator, "__post_init__", fake_init)
    generator = OpenAICompatibleGenerator(
        model_path="/models/demo",
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        api_base_url="http://localhost:8000/v1",
    )
    prompt = ConversationPrompt(
        [Message(role="system", content="system"), Message(role="user", content="question")],
        tools=ACTION_TOOLS,
        tool_choice="auto",
    )

    result = generator.generate_batch_with_metadata([prompt])[0]

    assert result.prompt_token_ids == [1, 2, 3]
    assert result.completion_token_ids == [11, 12]
    assert result.message is not None
    assert result.message.reasoning_content == "search"
    assert result.message.tool_calls[0].name == "search"
    assert result.message.tool_calls[0].arguments == {"query": "q"}
    assert requests[0]["parallel_tool_calls"] is False
    assert requests[0]["extra_body"]["return_token_ids"] is True
    assert requests[0]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert requests[0]["extra_body"]["chat_template"] == "custom agent template"
