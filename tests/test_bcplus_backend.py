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


def test_backend_batches_and_caches_search_results(monkeypatch) -> None:
    class BatchSearcher(DummySearcher):
        def __init__(self) -> None:
            self.search_many_calls: list[tuple[list[str], int]] = []

        def search_many(self, queries: list[str], k: int = 5):
            self.search_many_calls.append((list(queries), k))
            return [
                [{"docid": f"doc-{query}", "score": 1.0, "text": f"text {query}"}]
                for query in queries
            ]

    monkeypatch.setattr(
        "self_summarization_agent.bcplus_backend.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )
    searcher = BatchSearcher()
    backend = RealBrowseCompBackend(
        searcher=searcher,
        top_k=7,
        snippet_max_tokens=16,
        snippet_tokenizer_path="/models/qwen-tokenizer",
    )

    first = backend.search_many(["a", "b", "a"])
    second = backend.search_many(["b", "c"])

    assert searcher.search_many_calls == [(["a", "b"], 7), (["c"], 7)]
    assert first[0][0]["docid"] == "doc-a"
    assert first[2][0]["docid"] == "doc-a"
    assert second[0][0]["docid"] == "doc-b"
    assert second[1][0]["docid"] == "doc-c"


def test_backend_caches_truncated_documents(monkeypatch) -> None:
    class CountingDocumentSearcher(DummySearcher):
        def __init__(self) -> None:
            self.document_calls = 0

        def get_document(self, doc_id: str):
            self.document_calls += 1
            return {"text": "one two three"}

    monkeypatch.setattr(
        "self_summarization_agent.bcplus_backend.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )
    searcher = CountingDocumentSearcher()
    backend = RealBrowseCompBackend(
        searcher=searcher,
        snippet_max_tokens=16,
        document_max_tokens=2,
        snippet_tokenizer_path="/models/qwen-tokenizer",
    )

    assert backend.get_document("doc-1") == "one two"
    assert backend.get_document("doc-1") == "one two"
    assert searcher.document_calls == 1
