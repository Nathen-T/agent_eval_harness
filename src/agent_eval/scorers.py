from __future__ import annotations

from abc import ABC, abstractmethod
import re
import string

from agent_eval.task import Task


class Scorer(ABC):
    """Scores one system output for one task."""

    name: str

    @abstractmethod
    def score(self, task: Task, output: str) -> float:
        """Return a score between 0.0 and 1.0."""


class ExactMatchScorer(Scorer):
    name = "exact_match"

    def score(self, task: Task, output: str) -> float:
        if task.reference is None:
            return 0.0
        return 1.0 if normalize_text(output) == normalize_text(task.reference) else 0.0


class KeywordScorer(Scorer):
    name = "keyword_contains"

    def score(self, task: Task, output: str) -> float:
        keywords = task.metadata.get("keywords")
        if not keywords and task.reference:
            keywords = [task.reference]
        if not keywords:
            return 0.0

        normalized_output = normalize_text(output)
        matches = sum(
            1 for keyword in keywords if normalize_text(str(keyword)) in normalized_output
        )
        return matches / len(keywords)


class LLMJudgeScorer(Scorer):
    name = "llm_judge"

    def score(self, task: Task, output: str) -> float:
        # TODO: Wire this up to an LLM-as-judge provider once API configuration
        # and a rubric format are chosen. Keep returning a normalized 0.0-1.0
        # score so it composes with the rest of the harness.
        raise NotImplementedError("LLM-as-judge scoring is not implemented yet.")


def normalize_text(value: str) -> str:
    """Lowercase and remove punctuation/extra whitespace for stable matching."""

    lowered = value.lower().strip()
    without_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", without_punctuation).strip()
