from __future__ import annotations

from example_rag.generators import (
    DEFAULT_ABSTAIN_ANSWER,
    ExtractiveQAGenerator,
    ExtractiveQAGeneratorConfig,
)
from rag_eval.task import Doc


def test_extractive_generator_reads_span_from_retrieved_docs() -> None:
    qa_pipeline = _FakePipeline({"answer": "Paris", "score": 0.92})
    generator = ExtractiveQAGenerator(ExtractiveQAGeneratorConfig(), qa_pipeline=qa_pipeline)
    docs = [
        Doc(id="doc-1", text="The archive says Paris is the capital."),
        Doc(id="doc-2", text="An unrelated distractor passage."),
    ]

    answer = generator.generate("What is the capital?", docs)

    assert answer == "Paris"
    call = qa_pipeline.calls[0]
    assert call["question"] == "What is the capital?"
    assert "The archive says Paris is the capital." in call["context"]
    assert "An unrelated distractor passage." in call["context"]


def test_extractive_generator_abstains_below_min_score() -> None:
    qa_pipeline = _FakePipeline({"answer": "Lyon", "score": 0.02})
    generator = ExtractiveQAGenerator(
        ExtractiveQAGeneratorConfig(min_score=0.2),
        qa_pipeline=qa_pipeline,
    )

    answer = generator.generate(
        "What is the capital?",
        [Doc(id="doc-1", text="Some weakly related passage mentioning Lyon.")],
    )

    assert answer == DEFAULT_ABSTAIN_ANSWER


def test_extractive_generator_abstains_without_context() -> None:
    qa_pipeline = _FakePipeline({"answer": "Paris", "score": 0.99})
    generator = ExtractiveQAGenerator(ExtractiveQAGeneratorConfig(), qa_pipeline=qa_pipeline)

    answer = generator.generate("What is the capital?", [Doc(id="doc-1", text="   ")])

    assert answer == DEFAULT_ABSTAIN_ANSWER
    assert qa_pipeline.calls == []


class _FakePipeline:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result
