from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

    def search(self, query: str) -> list[SearchResult]:
        candidates = self.searcher.search(query, k=self.top_k)
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

    def get_document(self, doc_id: str) -> str:
        document = self.searcher.get_document(doc_id)
        if not document or "text" not in document:
            raise KeyError(f"Document not found: {doc_id}")
        return self._truncate_text(str(document["text"]), self.document_max_tokens)


def build_backend(bc_plus_root: str | Path, retrieval_config: RetrievalConfig) -> RealBrowseCompBackend:
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
