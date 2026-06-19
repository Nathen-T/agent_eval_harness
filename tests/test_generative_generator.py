from __future__ import annotations

from example_rag.generators import (
    GenerativeQAGenerator,
    GenerativeQAGeneratorConfig,
)
from rag_eval.task import Doc


def test_generative_generator_uses_retrieved_docs_and_greedy_decoding() -> None:
    pipeline = _FakePipeline([{"generated_text": "Answer: Paris."}])
    generator = GenerativeQAGenerator(
        GenerativeQAGeneratorConfig(model="tiny-test-model", max_new_tokens=16),
        text_pipeline=pipeline,
    )
    docs = [
        Doc(id="doc-1", text="The archive says Paris is the capital."),
        Doc(id="doc-2", text="An unrelated distractor passage."),
    ]

    answer = generator.generate("What is the capital?", docs)

    assert answer == "Paris"
    call = pipeline.calls[0]
    assert call["kwargs"]["max_new_tokens"] == 16
    assert call["kwargs"]["do_sample"] is False

    prompt = "\n".join(message["content"] for message in call["messages"])
    assert "The archive says Paris is the capital." in prompt
    assert "An unrelated distractor passage." in prompt


def test_generative_generator_reads_assistant_turn_from_chat_output() -> None:
    pipeline = _FakePipeline(
        [
            {
                "generated_text": [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "Lyon"},
                ]
            }
        ]
    )
    generator = GenerativeQAGenerator(
        GenerativeQAGeneratorConfig(model="tiny-test-model"),
        text_pipeline=pipeline,
    )

    answer = generator.generate(
        "What is the capital?",
        [Doc(id="doc-1", text="The archive says Lyon is the capital.")],
    )

    assert answer == "Lyon"


class _FakePipeline:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.outputs
