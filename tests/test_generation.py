from self_summarization_agent.config import ModelConfig
from self_summarization_agent.generation import VLLMGenerator, build_generator


class FakeTokenizer:
    chat_template = None


class FakeSamplingParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeCompletion:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeRequestOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [FakeCompletion(text)]


class FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.params: FakeSamplingParams | None = None

    def generate(self, prompts: list[str], params: FakeSamplingParams) -> list[FakeRequestOutput]:
        self.prompts = prompts
        self.params = params
        return [FakeRequestOutput(f"response:{prompt}") for prompt in prompts]


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
            attention_backend="TORCH_SDPA",
        )
    )

    assert isinstance(generator, VLLMGenerator)
    assert generator.model_path == "/models/demo"
    assert generator.tensor_parallel_size == 2


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
