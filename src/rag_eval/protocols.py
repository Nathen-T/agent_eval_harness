from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_eval.task import Doc


@runtime_checkable
class Retriever(Protocol):
    """Contract for the retrieval half of a system under test.

    The harness only ever talks to this interface, never to a concrete retriever.
    Implement it in your own package (vector store, BM25 service, hybrid search,
    ...) and return at most ``k`` docs, most relevant first.
    """

    def retrieve(self, question: str, corpus: list[Doc], k: int) -> list[Doc]: ...


@runtime_checkable
class Generator(Protocol):
    """Contract for the generation half of a system under test.

    The harness only ever passes the retrieved docs, so groundedness stays
    meaningful. Implement it to plug in a real LLM/provider.
    """

    def generate(self, question: str, docs: list[Doc]) -> str: ...
