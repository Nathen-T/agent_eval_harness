from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
import re
import string

from rag_eval.task import Doc, Task


class Scorer(ABC):
    """Scores one RAG answer for one task."""

    name: str

    @abstractmethod
    def score(self, task: Task, answer: str, retrieved_docs: list[Doc]) -> dict[str, float]:
        """Return one or more normalized metrics in [0.0, 1.0]."""


class AnswerCorrectnessScorer(Scorer):
    name = "answer_correctness"

    def score(self, task: Task, answer: str, retrieved_docs: list[Doc]) -> dict[str, float]:
        return {
            "em": exact_match(answer, task.reference_answer),
            "f1": token_f1(answer, task.reference_answer),
        }


class GroundednessScorer(Scorer):
    name = "groundedness"

    def score(self, task: Task, answer: str, retrieved_docs: list[Doc]) -> dict[str, float]:
        answer_tokens = content_tokens(answer)
        if not answer_tokens:
            return {"groundedness": 0.0}

        context_tokens = set(content_tokens(" ".join(doc.text for doc in retrieved_docs)))
        supported = sum(1 for token in answer_tokens if token in context_tokens)
        return {"groundedness": supported / len(answer_tokens)}


class RetrievalHitRateScorer(Scorer):
    name = "retrieval_hit_rate"

    def score(self, task: Task, answer: str, retrieved_docs: list[Doc]) -> dict[str, float]:
        if not task.gold_doc_id:
            return {"retrieval_hit": 0.0}

        retrieved_doc_ids = {doc.id for doc in retrieved_docs}
        hit = task.gold_doc_id in retrieved_doc_ids
        return {"retrieval_hit": 1.0 if hit else 0.0}


class LLMJudgeScorer(Scorer):
    name = "llm_judge"

    def score(self, task: Task, answer: str, retrieved_docs: list[Doc]) -> dict[str, float]:
        # TODO: Wire this to an LLM-as-judge provider and return normalized
        # metrics, e.g. {"llm_correctness": 0.0-1.0, "llm_faithfulness": 0.0-1.0}.
        raise NotImplementedError("LLM-as-judge scoring is not implemented yet.")


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()

    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        return "".join(char for char in value if char not in set(string.punctuation))

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    return white_space_fix(remove_articles(remove_punctuation(text.lower())))


def content_tokens(text: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "by",
        "does",
        "for",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
    }
    return [token for token in normalize_answer(text).split() if token not in stopwords]
