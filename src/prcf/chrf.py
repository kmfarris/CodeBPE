"""chrF (character n-gram F-score) for reconstruction eval."""

from __future__ import annotations

from collections import Counter


def _char_ngrams(text: str, n: int) -> Counter[str]:
    t = text.lower()
    if len(t) < n:
        return Counter([t] if t else [])
    return Counter(t[i : i + n] for i in range(len(t) - n + 1))


def chrf(hypothesis: str, reference: str, n: int = 3, beta: float = 2.0) -> float:
    """
    Character n-gram F-score (Popović-style simplified single-n).
    Returns score in [0, 1].
    """
    hyp = _char_ngrams(hypothesis, n)
    ref = _char_ngrams(reference, n)
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    overlap = sum((hyp & ref).values())
    prec = overlap / max(sum(hyp.values()), 1)
    rec = overlap / max(sum(ref.values()), 1)
    if prec + rec == 0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * prec * rec / (beta2 * prec + rec)
