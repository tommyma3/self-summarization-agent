import sys
from types import SimpleNamespace

from self_summarization_agent.config import JudgeConfig, ModelConfig
from self_summarization_agent import generation
from self_summarization_agent.generation import SGLangGenerator, VLLMGenerator, build_generator


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
        self.prompts: list[str] = []
        self.params: dict | None = None
        self.return_logprob = False

    def generate(self, prompts: list[str], params: dict, *, return_logprob: bool = False) -> list[dict]:
        self.prompts = prompts
        self.params = params
        self.return_logprob = return_logprob
        return [
            {
                "text": f"response:{prompt}",
                "meta_info": {"output_token_logprobs": [[-0.75, 11], [-1.25, 12]]},
            }
            for prompt in prompts
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
        )
    )

    assert isinstance(generator, VLLMGenerator)
    assert generator.model_path == "/models/demo"
    assert generator.tensor_parallel_size == 2


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
    )

    outputs = generator.generate_batch(["first", "second"])

    assert outputs == ["response:first", "response:second"]
    assert generator.llm.prompts == ["first", "second"]
    assert generator.llm.params.kwargs == {"max_tokens": 16, "temperature": 0.7, "top_p": 0.95}


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

    assert [output.text for output in outputs] == ["response:first", "response:second"]
    assert outputs[0].prompt_token_ids == [1, 2]
    assert outputs[0].completion_token_ids == [11, 12]
    assert outputs[0].cumulative_logprob == -2.0
    assert outputs[0].token_logprobs == [-0.75, -1.25]
    assert generator.engine.prompts == ["first", "second"]
    assert generator.engine.params == {"max_new_tokens": 16, "temperature": 0.7, "top_p": 0.95}
    assert generator.engine.return_logprob is True
