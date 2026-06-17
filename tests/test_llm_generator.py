from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from example_rag.generators import LLMGenerator, LLMGeneratorConfig
from rag_eval.task import Doc


def test_llm_generator_uses_retrieved_docs_and_greedy_completion() -> None:
    client = _FakeClient("Answer: Paris.\nI found this in the context.")
    generator = LLMGenerator(
        LLMGeneratorConfig(model="local-test-model"),
        client=client,
    )
    docs = [Doc(id="doc-1", text="The archive says Paris is the capital.")]

    answer = generator.generate("What is the capital?", docs)

    assert answer == "Paris"
    call = client.chat.completions.calls[0]
    assert call["model"] == "local-test-model"
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 64

    prompt = "\n".join(message["content"] for message in call["messages"])
    assert "The archive says Paris is the capital." in prompt
    assert "gold paragraph says Lyon is the answer" not in prompt


def test_llm_generator_unreachable_server_has_friendly_error() -> None:
    client = _FakeClient(
        APIConnectionError(
            message="connection refused",
            request=httpx.Request("POST", "http://localhost:1234/v1/chat/completions"),
        )
    )
    generator = LLMGenerator(
        LLMGeneratorConfig(
            model="local-test-model",
            base_url="http://localhost:1234/v1",
        ),
        client=client,
    )

    with pytest.raises(RuntimeError, match="No model server reachable at"):
        generator.generate("What is the capital?", [Doc(id="doc-1", text="Paris.")])


class _FakeClient:
    def __init__(self, response_or_error):
        self.chat = SimpleNamespace(completions=_FakeCompletions(response_or_error))


class _FakeCompletions:
    def __init__(self, response_or_error):
        self.response_or_error = response_or_error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response_or_error, Exception):
            raise self.response_or_error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.response_or_error),
                )
            ]
        )
