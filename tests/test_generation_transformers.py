import torch

from self_summarization_agent.generation import _load_transformers_model


class FakeModel:
    pass


def test_load_transformers_model_uses_multimodal_auto_class(monkeypatch) -> None:
    expected = FakeModel()

    def return_model(*args, **kwargs):
        return expected

    monkeypatch.setattr(
        "self_summarization_agent.generation.AutoModelForMultimodalLM.from_pretrained",
        return_model,
    )

    loaded = _load_transformers_model(
        "/models/qwen35",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )

    assert loaded is expected
