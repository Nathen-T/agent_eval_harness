from __future__ import annotations

import string


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace.

    Shared by the retrievers and the generator so they tokenize identically.
    """

    translator = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translator).split()
