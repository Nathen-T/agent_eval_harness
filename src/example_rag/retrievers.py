from __future__ import annotations

from collections import Counter
import math

import numpy as np
from rank_bm25 import BM25Okapi

from example_rag.text import tokenize
from rag_eval.task import Doc


class BM25Retriever:
    """Deterministic BM25 retriever over a tiny in-memory corpus."""

    def retrieve(self, question: str, corpus: list[Doc], k: int) -> list[Doc]:
        if k < 1:
            raise ValueError("k must be at least 1")
        if not corpus:
            return []

        tokenized_corpus = [tokenize(doc.text) for doc in corpus]
        model = BM25Okapi(tokenized_corpus)
        question_tokens = tokenize(question)
        bm25_scores = model.get_scores(question_tokens)
        question_token_set = set(question_tokens)
        scores = [
            float(score) + (1e-6 * len(question_token_set & set(doc_tokens)))
            for score, doc_tokens in zip(bm25_scores, tokenized_corpus, strict=True)
        ]
        return _top_k_by_score(corpus, scores, k)


class TfidfRetriever:
    """Small hand-rolled TF-IDF retriever to keep v1 dependency-light."""

    def retrieve(self, question: str, corpus: list[Doc], k: int) -> list[Doc]:
        if k < 1:
            raise ValueError("k must be at least 1")
        if not corpus:
            return []

        doc_tokens = [tokenize(doc.text) for doc in corpus]
        query_tokens = tokenize(question)
        vocabulary = sorted(set(query_tokens).union(*(set(tokens) for tokens in doc_tokens)))
        if not vocabulary:
            return corpus[:k]

        token_index = {token: index for index, token in enumerate(vocabulary)}
        document_frequency = Counter(
            token for tokens in doc_tokens for token in set(tokens)
        )
        doc_count = len(doc_tokens)
        idf = np.array(
            [
                math.log((1 + doc_count) / (1 + document_frequency[token])) + 1
                for token in vocabulary
            ]
        )

        query_vector = _tfidf_vector(query_tokens, token_index, idf)
        scores = [
            _cosine_similarity(query_vector, _tfidf_vector(tokens, token_index, idf))
            for tokens in doc_tokens
        ]
        return _top_k_by_score(corpus, scores, k)


class EmbeddingRetriever:
    """Adapter stub for a real embedding retriever or vector store."""

    def retrieve(self, question: str, corpus: list[Doc], k: int) -> list[Doc]:
        # TODO: Plug in embeddings and a vector store here. Keep the return type
        # as list[Doc] so this can drop into RAGPipeline without changing evals.
        raise NotImplementedError("Real embedding retriever is not implemented yet.")


def build_retriever(name: str):
    if name == "bm25":
        return BM25Retriever()
    if name == "tfidf":
        return TfidfRetriever()
    raise ValueError(f"Unsupported retriever {name!r}")


def _top_k_by_score(corpus: list[Doc], scores, k: int) -> list[Doc]:
    ranked = sorted(
        zip(corpus, scores, strict=True),
        key=lambda item: (-float(item[1]), item[0].id),
    )
    return [doc for doc, _ in ranked[:k]]


def _tfidf_vector(tokens: list[str], token_index: dict[str, int], idf: np.ndarray) -> np.ndarray:
    vector = np.zeros(len(token_index))
    counts = Counter(tokens)
    for token, count in counts.items():
        index = token_index.get(token)
        if index is not None:
            vector[index] = count * idf[index]
    return vector


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)
