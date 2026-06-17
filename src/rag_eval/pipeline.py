from __future__ import annotations

from dataclasses import dataclass
import os
import re
import string

from rag_eval.task import Doc


class RAGPipeline:
    """Minimal retrieve-then-generate system under test."""

    def __init__(self, retriever: object, generator: object, name: str | None = None):
        self.retriever = retriever
        self.generator = generator
        self.name = name or f"{retriever.__class__.__name__}+{generator.__class__.__name__}"

    def retrieve(self, question: str) -> list[Doc]:
        return self.retriever.retrieve(question)

    def generate(self, question: str, docs: list[Doc]) -> str:
        return self.generator.generate(question, docs)


class MockRetriever:
    """Deterministic lexical retriever for offline demos and tests."""

    def __init__(self, corpus: list[Doc], k: int = 3):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.corpus = corpus
        self.k = k

    def retrieve(self, question: str) -> list[Doc]:
        question_tokens = set(tokenize(question))

        ranked = sorted(
            self.corpus,
            key=lambda doc: (
                -len(question_tokens & set(tokenize(doc.text))),
                doc.id,
            ),
        )
        return ranked[: self.k]


@dataclass(frozen=True)
class MockGeneratorConfig:
    use_context: bool = True


class MockGenerator:
    """Deterministic generator that can deliberately ignore context."""

    def __init__(self, config: MockGeneratorConfig | None = None):
        self.config = config or MockGeneratorConfig()

    def generate(self, question: str, docs: list[Doc]) -> str:
        if not self.config.use_context:
            return "The answer is probably quantum bananas."

        qa_answer = _answer_from_qa_blocks(question, docs)
        if qa_answer:
            return qa_answer

        combined_context = " ".join(doc.text for doc in docs)
        heuristic = _heuristic_short_answer(question, combined_context)
        if heuristic:
            return heuristic

        return _best_sentence(question, combined_context) or "I do not know."


class EmbeddingRetriever:
    """Adapter stub for a real embedding retriever or vector store."""

    def retrieve(self, question: str) -> list[Doc]:
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


def _answer_from_qa_blocks(question: str, docs: list[Doc]) -> str | None:
    question_tokens = set(tokenize(question))
    best_score = 0
    best_answer: str | None = None

    for doc in docs:
        lines = [line.strip() for line in doc.text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if not line.lower().startswith("question:"):
                continue
            if index + 1 >= len(lines) or not lines[index + 1].lower().startswith("answer:"):
                continue

            candidate_question = line.split(":", 1)[1]
            candidate_answer = lines[index + 1].split(":", 1)[1].strip()
            score = len(question_tokens & set(tokenize(candidate_question)))
            if score > best_score:
                best_score = score
                best_answer = candidate_answer.rstrip(".")

    return best_answer


def _heuristic_short_answer(question: str, context: str) -> str | None:
    lowered_question = question.lower()

    patterns = [
        ("normandy" in lowered_question and "country" in lowered_question, r"\bFrance\b"),
        ("normans" in lowered_question and "when" in lowered_question, r"10th and 11th centuries"),
        ("who" in lowered_question and "normandy" in lowered_question, r"The Normans"),
        ("city" in lowered_question and "eiffel" in lowered_question, r"\bParis\b"),
        ("material" in lowered_question and "eiffel" in lowered_question, r"wrought-iron"),
        ("agency" in lowered_question and "apollo" in lowered_question, r"\bNASA\b"),
        ("when" in lowered_question and "apollo" in lowered_question, r"July 16, 1969"),
        ("apollo" in lowered_question and "land" in lowered_question, r"the Moon"),
    ]

    for applies, pattern in patterns:
        if not applies:
            continue
        match = re.search(pattern, context, flags=re.IGNORECASE)
        if match:
            return match.group(0)

    return None


def _best_sentence(question: str, context: str) -> str | None:
    question_tokens = set(tokenize(question))
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", context)]
    if not sentences:
        return None

    best = max(
        sentences,
        key=lambda sentence: len(question_tokens & set(tokenize(sentence))),
    )
    return best.rstrip(".") if best else None
