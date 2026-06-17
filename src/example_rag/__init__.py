"""An example RAG system under test, kept separate from the ``rag_eval`` harness.

Everything here is swappable: the retrievers and generators implement the
``rag_eval`` ``Retriever``/``Generator`` protocols, and the harness never imports
this package. Replace these pieces (or this whole package) to evaluate your own
system.
"""

from example_rag.generators import (
    LLMGenerator,
    MockGenerator,
    MockGeneratorConfig,
)
from example_rag.retrievers import (
    BM25Retriever,
    EmbeddingRetriever,
    TfidfRetriever,
    build_retriever,
)

__all__ = [
    "BM25Retriever",
    "EmbeddingRetriever",
    "LLMGenerator",
    "MockGenerator",
    "MockGeneratorConfig",
    "TfidfRetriever",
    "build_retriever",
]
