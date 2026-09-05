"""Append-only inference state. Text is never an input to this ledger."""
from dataclasses import dataclass
from typing import Any, Iterable


TITO_CONTRACT = "qwen-agent-tito-v1"
TITO_COLLECTION_VERSION = 3


def token_ids(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(values)
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in result):
        raise ValueError("Token IDs must be nonnegative integers")
    return result


@dataclass(frozen=True, slots=True)
class TokenRequest:
    prompt_token_ids: tuple[int, ...]
    generation_kind: str = "action"
    # Only conditioning tokens not yet committed to the ledger.
    suffix: tuple[int, ...] = ()
    ledger_version: int = 0
    max_new_tokens: int | None = None


class IntervalTokenLedger:
    def __init__(self, initial_ids: Iterable[int], *, fingerprint: str) -> None:
        self.__ids: list[int] = []
        self.__mask: list[bool] = []
        self.__spans: list[dict[str, Any]] = []
        self.__closed = False
        self.__version = 0
        self.fingerprint = fingerprint
        self.append_external(initial_ids, kind="initial_state")

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(self.__ids)

    @property
    def version(self) -> int:
        return self.__version

    @property
    def finalized(self) -> bool:
        return self.__closed

    def __len__(self) -> int:
        return len(self.__ids)

    def _append(self, values: Iterable[int], *, sampled: bool, kind: str,
                expected_version: int | None = None) -> None:
        if self.__closed:
            raise RuntimeError("Cannot append to a finalized interval")
        if expected_version is not None and expected_version != self.__version:
            raise RuntimeError("Stale or already committed token append")
        ids = token_ids(values)
        if ids:
            start = len(self.__ids)
            self.__ids.extend(ids)
            self.__mask.extend([sampled] * len(ids))
            self.__spans.append(dict(start=start, end=len(self.__ids), sampled=sampled, kind=kind))
        self.__version += 1

    def append_external(self, values: Iterable[int], *, kind: str,
                        expected_version: int | None = None) -> None:
        self._append(values, sampled=False, kind=kind, expected_version=expected_version)

    def append_sampled(self, values: Iterable[int], *, expected_prompt: Iterable[int],
                       kind: str) -> None:
        if token_ids(expected_prompt) != self.ids:
            raise ValueError("Inference input differs from the committed token ledger")
        self._append(values, sampled=True, kind=kind)

    def request(self, suffix: Iterable[int], *, generation_kind: str) -> TokenRequest:
        ids = token_ids(suffix)
        return TokenRequest(self.ids + ids, generation_kind, ids, self.version)

    def commit(self, request: TokenRequest) -> None:
        if request.prompt_token_ids != self.ids + request.suffix:
            raise ValueError("Request does not extend this interval")
        self.append_external(request.suffix, kind=request.generation_kind + "_header",
                             expected_version=request.ledger_version)

    def payload(self) -> dict[str, Any]:
        return dict(version=TITO_COLLECTION_VERSION, contract=TITO_CONTRACT,
                    renderer_fingerprint=self.fingerprint, full_token_ids=list(self.__ids),
                    assistant_token_mask=list(self.__mask), spans=[dict(s) for s in self.__spans])

    def finalize(self) -> None:
        if self.__closed:
            raise RuntimeError("Interval already finalized")
        self.__closed = True
