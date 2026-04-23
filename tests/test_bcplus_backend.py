from self_summarization_agent.bcplus_backend import RealBrowseCompBackend


class DummySearcher:
    def search(self, query: str, k: int = 5):
        return [{"docid": "doc-1", "snippet": "hello world"}]

    def get_document(self, doc_id: str):
        return {"text": "document text"}


def test_backend_disables_snippet_truncation_when_tokenizer_load_fails(monkeypatch) -> None:
    def raise_connect_error(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(
        "self_summarization_agent.bcplus_backend.AutoTokenizer.from_pretrained",
        raise_connect_error,
    )

    backend = RealBrowseCompBackend(
        searcher=DummySearcher(),
        snippet_max_tokens=16,
        snippet_tokenizer_path="/models/qwen-tokenizer",
    )

    assert backend.snippet_tokenizer is None
    assert backend.search("query")[0]["snippet"] == "hello world"
