from dataclasses import dataclass
from typing import Any, Protocol


SearchResult = dict[str, Any]


# Real BrowseComp-Plus integration should adapt the benchmark's fixed search
# and document retrieval APIs behind this protocol without importing the
# existing search_agent runtime.
class BrowseCompBackend(Protocol):
    def search(self, query: str) -> list[SearchResult]:
        ...

    def search_many(self, queries: list[str]) -> list[list[SearchResult]]:
        ...

    def get_document(self, doc_id: str) -> str:
        ...


@dataclass(slots=True)
class FakeBackend:
    search_index: dict[str, list[str]]
    documents: dict[str, str]

    def search(self, query: str) -> list[SearchResult]:
        return [
            {"docid": doc_id, "snippet": self.documents.get(doc_id, "")}
            for doc_id in self.search_index.get(query, [])
        ]

    def search_many(self, queries: list[str]) -> list[list[SearchResult]]:
        return [self.search(query) for query in queries]

    def get_document(self, doc_id: str) -> str:
        return self.documents[doc_id]
