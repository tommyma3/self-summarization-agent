from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from self_summarization_agent.backend import SearchResult
from self_summarization_agent.config import RetrievalConfig


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
    snippet_tokenizer: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.snippet_max_tokens and self.snippet_max_tokens > 0:
            self.snippet_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def _build_snippet(self, candidate: dict[str, Any]) -> str:
        text = str(candidate.get("snippet") or candidate.get("text") or "")
        if (
            not self.snippet_max_tokens
            or self.snippet_max_tokens <= 0
            or self.snippet_tokenizer is None
        ):
            return text
        tokens = self.snippet_tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.snippet_max_tokens:
            return text
        return self.snippet_tokenizer.decode(
            tokens[: self.snippet_max_tokens],
            skip_special_tokens=True,
        )

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
        return str(document["text"])


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
    )
