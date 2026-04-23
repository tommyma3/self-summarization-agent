from self_summarization_agent.config import ModelConfig
from self_summarization_agent.generation import VLLMGenerator, build_generator


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
