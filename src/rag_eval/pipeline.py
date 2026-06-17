from __future__ import annotations

from rag_eval.protocols import Generator, Retriever
from rag_eval.task import Doc


class RAGPipeline:
    """Minimal retrieve-then-generate system under test.

    This is the only seam between the harness and a concrete RAG system: it holds
    a corpus plus any objects satisfying the ``Retriever`` and ``Generator``
    protocols. The harness depends on this glue and those protocols only - never
    on a specific retriever or generator. Bring your own system by implementing
    the protocols (see the ``example_rag`` package for a worked example).
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        corpus: list[Doc],
        k: int = 5,
        name: str | None = None,
    ):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.retriever = retriever
        self.generator = generator
        self.corpus = corpus
        self.k = k
        self.name = name or f"{retriever.__class__.__name__}+{generator.__class__.__name__}"

    def retrieve(self, question: str) -> list[Doc]:
        return self.retriever.retrieve(question, self.corpus, self.k)

    def generate(self, question: str, docs: list[Doc]) -> str:
        return self.generator.generate(question, docs)
