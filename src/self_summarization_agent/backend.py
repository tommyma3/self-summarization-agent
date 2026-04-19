from dataclasses import dataclass
from typing import Protocol


# Real BrowseComp-Plus integration should adapt the benchmark's fixed search
# and document retrieval APIs behind this protocol without importing the
# existing search_agent runtime.
class BrowseCompBackend(Protocol):
    def search(self, query: str) -> list[str]:
        ...

    def get_document(self, doc_id: str) -> str:
        ...


@dataclass(slots=True)
class FakeBackend:
    search_index: dict[str, list[str]]
    documents: dict[str, str]

    def search(self, query: str) -> list[str]:
        return list(self.search_index.get(query, []))

    def get_document(self, doc_id: str) -> str:
        return self.documents[doc_id]
