from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any

from example_rag.text import tokenize
from rag_eval.task import Doc

LOCAL_OPENAI_BASE_URL = "http://localhost:1234/v1"
HOSTED_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LOCAL_API_KEY = "lm-studio"
DEFAULT_ABSTAIN_ANSWER = "no answer found in the retrieved context"
DEFAULT_EXTRACTIVE_QA_MODEL = "distilbert-base-cased-distilled-squad"


@dataclass(frozen=True)
class MockGeneratorConfig:
    # Minimum question/sentence content-word overlap before the generator will
    # commit to an answer. Below this it abstains instead of guessing, which is
    # how starved retrieval (small k) turns into ungrounded, wrong answers.
    min_content_overlap: int = 2
    max_span_tokens: int = 5
    abstain_answer: str = DEFAULT_ABSTAIN_ANSWER


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


@dataclass(frozen=True)
class LLMGeneratorConfig:
    model: str
    base_url: str = LOCAL_OPENAI_BASE_URL
    api_key: str = DEFAULT_LOCAL_API_KEY
    temperature: float = 0.0
    max_tokens: int = 64
    abstain_answer: str = DEFAULT_ABSTAIN_ANSWER


class LLMGenerator:
    """OpenAI-compatible, context-only generator for local or hosted models."""

    def __init__(self, config: LLMGeneratorConfig, client: Any | None = None):
        self.config = config
        self._client = client

    def generate(self, question: str, docs: list[Doc]) -> str:
        messages = _build_llm_messages(question, docs, self.config.abstain_answer)
        try:
            completion = self._openai_client().chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:
            self._raise_friendly_error(exc)

        return _clean_answer(_completion_text(completion), self.config.abstain_answer)

    def _openai_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency config issue
                raise RuntimeError(
                    "The openai package is not installed. Run `uv sync` or "
                    "`pip install -r requirements.txt` before using LLMGenerator."
                ) from exc

            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
            )
        return self._client

    def _raise_friendly_error(self, exc: Exception) -> None:
        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                AuthenticationError,
                OpenAIError,
            )
        except ImportError:  # pragma: no cover - handled earlier in real use
            raise exc

        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            raise RuntimeError(
                f"No model server reachable at {self.config.base_url}. Start LM Studio "
                "with a model loaded and the local server enabled, or run with "
                "--generator mock."
            ) from exc
        if isinstance(exc, AuthenticationError):
            raise RuntimeError(
                "OpenAI-compatible API authentication failed. Set a valid API key "
                "with --api-key or OPENAI_API_KEY, or run with --generator mock."
            ) from exc
        if isinstance(exc, OpenAIError):
            raise RuntimeError(f"OpenAI-compatible model call failed: {exc}") from exc
        raise exc


@dataclass(frozen=True)
class ExtractiveQAGeneratorConfig:
    model: str = DEFAULT_EXTRACTIVE_QA_MODEL
    # Minimum span probability before committing to an answer. Raise it above 0
    # to make the reader abstain on weak (likely wrong) spans, mirroring the
    # mock's abstain-when-unsure behavior; 0.0 keeps every predicted span.
    min_score: float = 0.0
    max_answer_len: int = 30
    max_seq_len: int = 384
    doc_stride: int = 128
    abstain_answer: str = DEFAULT_ABSTAIN_ANSWER


class ExtractiveQAGenerator:
    """Local, offline extractive reader backed by a HuggingFace QA pipeline.

    It runs a small SQuAD-style model (e.g. distilbert-base-cased-distilled-squad)
    on CPU with no server and no API token. The model only sees the retrieved
    docs, reads the best answer span out of them, and abstains when its span
    probability is below ``min_score`` - so answers degrade as retrieval worsens.
    """

    def __init__(self, config: ExtractiveQAGeneratorConfig, qa_pipeline: Any | None = None):
        self.config = config
        self._pipeline = qa_pipeline

    def generate(self, question: str, docs: list[Doc]) -> str:
        context = "\n\n".join(doc.text.strip() for doc in docs if doc.text.strip())
        if not context:
            return self.config.abstain_answer

        result = self._qa_pipeline()(
            question=question,
            context=context,
            max_answer_len=self.config.max_answer_len,
            max_seq_len=self.config.max_seq_len,
            doc_stride=self.config.doc_stride,
            handle_impossible_answer=False,
        )

        answer = str(result.get("answer", "")).strip()
        score = float(result.get("score", 0.0))
        if not answer or score < self.config.min_score:
            return self.config.abstain_answer
        return answer

    def _qa_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "transformers/torch are not installed. Run `uv sync` or "
                    "`pip install -r requirements.txt` before using --generator hf, "
                    "or run with --generator mock."
                ) from exc

            self._pipeline = pipeline("question-answering", model=self.config.model)
        return self._pipeline


def build_generator(
    name: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
):
    if name == "mock":
        return MockGenerator()

    if name == "hf":
        return ExtractiveQAGenerator(
            ExtractiveQAGeneratorConfig(model=model or DEFAULT_EXTRACTIVE_QA_MODEL)
        )

    if name not in {"local", "api"}:
        raise ValueError(f"Unsupported generator {name!r}")

    if not model:
        raise RuntimeError(
            f"--model is required when using --generator {name}. "
            "Use --generator mock for the zero-setup offline path."
        )

    if name == "local":
        resolved_base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or LOCAL_OPENAI_BASE_URL
        )
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or DEFAULT_LOCAL_API_KEY
    else:
        resolved_base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or HOSTED_OPENAI_BASE_URL
        )
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set OPENAI_API_KEY or pass --api-key "
                "when using --generator api, or run with --generator mock."
            )

    return LLMGenerator(
        LLMGeneratorConfig(
            model=model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
        )
    )


def _build_llm_messages(
    question: str,
    docs: list[Doc],
    abstain_answer: str,
) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{index}] {doc.text.strip()}"
        for index, doc in enumerate(docs, start=1)
        if doc.text.strip()
    )
    if not context:
        context = "(no retrieved context)"

    system_prompt = (
        "You are an extractive question-answering system. Answer using only the "
        "retrieved context. Do not use outside knowledge. Return the shortest "
        "supported answer span, not a full sentence. If the context does not "
        f"support an answer, reply exactly: {abstain_answer}"
    )
    user_prompt = (
        "Retrieved context:\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _completion_text(completion: Any) -> str:
    choices = getattr(completion, "choices", [])
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "") if message is not None else ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts)
    return str(content or "")


def _clean_answer(text: str, abstain_answer: str) -> str:
    answer = text.strip()
    if not answer:
        return abstain_answer

    answer = next((line.strip() for line in answer.splitlines() if line.strip()), "")
    answer = re.sub(r"^(?:final\s+)?answer\s*:\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(
        r"^(?:based on|from|according to) (?:the )?context,?\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(r"^the answer is\s+", "", answer, flags=re.IGNORECASE)
    answer = answer.strip(" \t\"'`")
    answer = re.sub(r"\s+", " ", answer)
    if answer.endswith(".") and len(answer.split()) <= 12:
        answer = answer[:-1].strip()
    if not answer:
        return abstain_answer

    tokens = answer.split()
    if len(tokens) > 32:
        answer = " ".join(tokens[:32])
    return answer


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
