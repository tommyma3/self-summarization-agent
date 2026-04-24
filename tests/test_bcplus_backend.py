from self_summarization_agent.bcplus_backend import RealBrowseCompBackend


class DummySearcher:
    def search(self, query: str, k: int = 5):
        return [{"docid": "doc-1", "snippet": "hello world"}]

    def get_document(self, doc_id: str):
        return {"text": "document text"}


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return text.split()

    def decode(self, tokens, skip_special_tokens: bool = True):
        del skip_special_tokens
        return " ".join(tokens)


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


def test_backend_truncates_full_documents_with_configured_token_cap(monkeypatch) -> None:
    class LongDocumentSearcher(DummySearcher):
        def get_document(self, doc_id: str):
            del doc_id
            return {"text": "one two three four"}

    monkeypatch.setattr(
        "self_summarization_agent.bcplus_backend.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    backend = RealBrowseCompBackend(
        searcher=LongDocumentSearcher(),
        snippet_max_tokens=16,
        document_max_tokens=2,
        snippet_tokenizer_path="/models/qwen-tokenizer",
    )

    assert backend.get_document("doc-1") == "one two"
