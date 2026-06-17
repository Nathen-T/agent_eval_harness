from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import os
import re
import string

import numpy as np
from rank_bm25 import BM25Okapi

from rag_eval.task import Doc


class RAGPipeline:
    """Minimal retrieve-then-generate system under test."""

    def __init__(
        self,
        retriever: object,
        generator: object,
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


@dataclass(frozen=True)
class MockGeneratorConfig:
    # Minimum question/sentence content-word overlap before the generator will
    # commit to an answer. Below this it abstains instead of guessing, which is
    # how starved retrieval (small k) turns into ungrounded, wrong answers.
    min_content_overlap: int = 2
    max_span_tokens: int = 5
    abstain_answer: str = "no answer found in the retrieved context"


class MockGenerator:
    """Deterministic, context-only extractive generator for offline demos and tests.

    It finds the retrieved sentence most relevant to the question and extracts a
    short answer span from it. When no retrieved sentence is relevant enough it
    abstains rather than copying text, so groundedness and F1 fall when the gold
    paragraph is not retrieved.
    """

    def __init__(self, config: MockGeneratorConfig | None = None):
        self.config = config or MockGeneratorConfig()

    def generate(self, question: str, docs: list[Doc]) -> str:
        best_sentence, overlap = _best_retrieved_sentence(question, docs)
        if not best_sentence or overlap < self.config.min_content_overlap:
            return self.config.abstain_answer

        return _extract_span(question, best_sentence, self.config.max_span_tokens)


class EmbeddingRetriever:
    """Adapter stub for a real embedding retriever or vector store."""

    def retrieve(self, question: str, corpus: list[Doc], k: int) -> list[Doc]:
        # TODO: Plug in embeddings and a vector store here. Keep the return type
        # as list[Doc] so this can drop into RAGPipeline without changing evals.
        raise NotImplementedError("Real embedding retriever is not implemented yet.")


class LLMGenerator:
    """Adapter stub for a real LLM provider."""

    def generate(self, question: str, docs: list[Doc]) -> str:
        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY is not set. Configure your provider before using LLMGenerator."
            )

        # TODO: Call a real LLM API here with the question and retrieved docs.
        # Preserve the generate(question, docs) -> answer shape.
        raise NotImplementedError("Real LLM generator is not implemented yet.")


def tokenize(text: str) -> list[str]:
    translator = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translator).split()


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


def _best_retrieved_sentence(question: str, docs: list[Doc]) -> tuple[str | None, int]:
    question_tokens = set(content_tokens(question))
    if not question_tokens:
        return None, 0

    best_sentence: str | None = None
    best_overlap = 0
    for doc in docs:
        for sentence in _sentences(doc.text):
            overlap = len(question_tokens & set(content_tokens(sentence)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence

    return best_sentence, best_overlap


def _extract_span(question: str, sentence: str, max_span_tokens: int) -> str:
    """Pick a short answer span from a sentence with a simple, explainable rule.

    SQuAD answers are usually proper nouns or numbers, so we return the longest
    contiguous run of capitalized/numeric words that are not already in the
    question. This is deliberately crude (no ML) and readable; it falls back to
    the first novel content word when no such run exists.
    """

    question_words = {match.lower() for match in re.findall(r"[A-Za-z0-9]+", question)}
    tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", sentence)

    runs: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        salient = token[0].isupper() or token[0].isdigit()
        if salient and token.lower() not in question_words:
            current.append(token)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    if runs:
        best_run = max(runs, key=lambda run: len("".join(run)))
        return " ".join(best_run[:max_span_tokens])

    novel = [token for token in content_tokens(sentence) if token not in question_words]
    return novel[0] if novel else sentence.split()[0]


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def content_tokens(text: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "did",
        "does",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
    }
    return [token for token in tokenize(text) if token not in stopwords]
