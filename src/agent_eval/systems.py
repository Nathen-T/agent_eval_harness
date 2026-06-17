from __future__ import annotations

import os
import re


MOCK_RESPONSES = {
    "what is the capital of france?": "Paris",
    "who wrote the hobbit?": "J.R.R. Tolkien",
    "what is 17 + 25?": "42",
    "what is 9 times 6?": "54",
    "extract the email address from: contact ada at ada@example.com for details.": "ada@example.com",
    "extract the order id from: order #a-12345 ships tomorrow.": "A-12345",
    "classify the sentiment as positive, neutral, or negative: the setup was fast and the docs were clear.": "positive",
    "from this sentence, extract the color: the small robot is painted bright blue.": "blue",
}


def mock_system(input_text: str) -> str:
    """Deterministic offline system under test for the bundled examples."""

    key = input_text.lower().strip()
    if key in MOCK_RESPONSES:
        return MOCK_RESPONSES[key]

    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", input_text)
    if email_match:
        return email_match.group(0)

    order_match = re.search(r"\b[A-Z]-\d{5}\b", input_text, flags=re.IGNORECASE)
    if order_match:
        return order_match.group(0).upper()

    return "I do not know."


def real_llm_adapter(input_text: str) -> str:
    """Adapter stub for swapping in a real model provider later."""

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is not set. Configure your provider before using the real LLM adapter."
        )

    # TODO: Plug in a real LLM API call here (for example, OpenAI, Anthropic,
    # or an internal model gateway). Keep this function shaped as
    # `str -> str` so it can be passed directly to run_eval().
    raise NotImplementedError("Real LLM adapter is not implemented yet.")
