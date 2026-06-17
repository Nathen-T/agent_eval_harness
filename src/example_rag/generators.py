from __future__ import annotations

from dataclasses import dataclass
import os
import re

from example_rag.text import tokenize
from rag_eval.task import Doc


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
