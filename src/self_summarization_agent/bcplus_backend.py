from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
from urllib import error, request

from transformers import AutoTokenizer

from self_summarization_agent.backend import SearchResult
from self_summarization_agent.config import RetrievalConfig

LOGGER = logging.getLogger(__name__)


def _ensure_bc_plus_searcher_imports(bc_plus_root: str | Path) -> None:
    searcher_root = str(Path(bc_plus_root) / "searcher")
    if searcher_root not in sys.path:
        sys.path.insert(0, searcher_root)


def _build_searcher_args(retrieval_config: RetrievalConfig) -> argparse.Namespace:
    return argparse.Namespace(
        index_path=retrieval_config.index_path,
        model_name=retrieval_config.model_name,
        normalize=retrieval_config.normalize,
        pooling=retrieval_config.pooling,
        torch_dtype=retrieval_config.torch_dtype,
        dataset_name=retrieval_config.dataset_name,
        task_prefix=retrieval_config.task_prefix,
        max_length=retrieval_config.max_length,
    )


@dataclass(slots=True)
class RealBrowseCompBackend:
    searcher: Any
    top_k: int = 5
    snippet_max_tokens: int | None = 512
    document_max_tokens: int | None = 8192
    snippet_tokenizer_path: str | None = None
    snippet_tokenizer: Any = field(init=False, default=None)
    _search_cache: dict[str, list[SearchResult]] = field(init=False, default_factory=dict)
    _document_cache: dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        needs_tokenizer = (
            (self.snippet_max_tokens is not None and self.snippet_max_tokens > 0)
            or (self.document_max_tokens is not None and self.document_max_tokens > 0)
        )
        if needs_tokenizer:
            tokenizer_path = self.snippet_tokenizer_path or "Qwen/Qwen3-0.6B"
            tokenizer_kwargs: dict[str, Any] = {}
            if self.snippet_tokenizer_path:
                tokenizer_kwargs["local_files_only"] = True
            try:
                self.snippet_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to load snippet tokenizer from %s; snippet truncation is disabled. %s: %s",
                    tokenizer_path,
                    type(exc).__name__,
                    exc,
                )
                self.snippet_tokenizer = None

    def _truncate_text(self, text: str, max_tokens: int | None) -> str:
        if (
            not max_tokens
            or max_tokens <= 0
            or self.snippet_tokenizer is None
        ):
            return text
        tokens = self.snippet_tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text
        return self.snippet_tokenizer.decode(
            tokens[:max_tokens],
            skip_special_tokens=True,
        )

    def _build_snippet(self, candidate: dict[str, Any]) -> str:
        text = str(candidate.get("snippet") or candidate.get("text") or "")
        return self._truncate_text(text, self.snippet_max_tokens)

    def _format_search_results(self, candidates: list[dict[str, Any]]) -> list[SearchResult]:
        results: list[SearchResult] = []
        for candidate in candidates:
            result: SearchResult = {
                "docid": str(candidate["docid"]),
                "snippet": self._build_snippet(candidate),
            }
            if candidate.get("score") is not None:
                result["score"] = candidate["score"]
            results.append(result)
        return results

    def search(self, query: str) -> list[SearchResult]:
        return self.search_many([query])[0]

    def search_many(self, queries: list[str]) -> list[list[SearchResult]]:
        if not queries:
            return []

        results_by_query: dict[str, list[SearchResult]] = {}
        misses: list[str] = []
        for query in queries:
            cached = self._search_cache.get(query)
            if cached is None:
                if query not in results_by_query:
                    misses.append(query)
                    results_by_query[query] = []
                continue
            results_by_query[query] = [dict(result) for result in cached]

        if misses:
            search_many = getattr(self.searcher, "search_many", None)
            if search_many is None:
                candidate_batches = [self.searcher.search(query, k=self.top_k) for query in misses]
            else:
                candidate_batches = search_many(misses, k=self.top_k)
                if len(candidate_batches) != len(misses):
                    raise ValueError(
                        f"search_many returned {len(candidate_batches)} result batches for {len(misses)} queries"
                    )
            for query, candidates in zip(misses, candidate_batches):
                formatted = self._format_search_results(candidates)
                self._search_cache[query] = [dict(result) for result in formatted]
                results_by_query[query] = formatted

        return [[dict(result) for result in results_by_query[query]] for query in queries]

    def get_document(self, doc_id: str) -> str:
        cached = self._document_cache.get(doc_id)
        if cached is not None:
            return cached
        document = self.searcher.get_document(doc_id)
        if not document or "text" not in document:
            raise KeyError(f"Document not found: {doc_id}")
        truncated = self._truncate_text(str(document["text"]), self.document_max_tokens)
        self._document_cache[doc_id] = truncated
        return truncated


@dataclass(slots=True)
class RetrievalWorkerClient:
    base_url: str
    timeout_seconds: float = 300.0

    def _post_json(self, path: str, payload: dict[str, Any]) -> Any:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                response_payload = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Retrieval worker request failed: HTTP {exc.code}: {detail}") from exc
        return json.loads(response_payload) if response_payload else None

    def search(self, query: str) -> list[SearchResult]:
        return self.search_many([query])[0]

    def search_many(self, queries: list[str]) -> list[list[SearchResult]]:
        payload = self._post_json("/search_many", {"queries": queries})
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise RuntimeError("Retrieval worker returned invalid search_many payload")
        return results

    def get_document(self, doc_id: str) -> str:
        payload = self._post_json("/get_document", {"doc_id": doc_id})
        document = payload.get("document") if isinstance(payload, dict) else None
        if not isinstance(document, str):
            raise RuntimeError("Retrieval worker returned invalid get_document payload")
        return document


def build_direct_backend(bc_plus_root: str | Path, retrieval_config: RetrievalConfig) -> RealBrowseCompBackend:
    _ensure_bc_plus_searcher_imports(bc_plus_root)
    from searchers import SearcherType

    backend_name = retrieval_config.backend.lower()
    if backend_name not in {"faiss", "bm25"}:
        raise ValueError(f"Unsupported retrieval backend: {retrieval_config.backend}")
    searcher_class = SearcherType.get_searcher_class(backend_name)
    searcher = searcher_class(_build_searcher_args(retrieval_config))
    return RealBrowseCompBackend(
        searcher=searcher,
        top_k=retrieval_config.top_k,
        snippet_max_tokens=retrieval_config.snippet_max_tokens,
        document_max_tokens=retrieval_config.document_max_tokens,
        snippet_tokenizer_path=retrieval_config.snippet_tokenizer_path,
    )


def build_backend(
    bc_plus_root: str | Path,
    retrieval_config: RetrievalConfig,
    *,
    worker_url: str | None = None,
) -> RealBrowseCompBackend | RetrievalWorkerClient:
    if worker_url:
        return RetrievalWorkerClient(worker_url)
    return build_direct_backend(bc_plus_root, retrieval_config)
