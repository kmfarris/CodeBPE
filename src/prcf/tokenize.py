"""Whitespace tokenizer with light punctuation stripping."""

from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Split on whitespace; strip leading/trailing punctuation per token."""
    words: list[str] = []
    for raw in text.split():
        w = _PUNCT_RE.sub("", raw)
        if w:
            words.append(w)
    return words
