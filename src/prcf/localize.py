"""Localize a single-word edit from fingerprint diffs."""

from __future__ import annotations

from prcf.fingerprint import slot_shingle_sets, token_id_fingerprint
from prcf.tokenize import tokenize


def localize_edit(text_a: str, text_b: str) -> int | None:
    """
    Predict the word index that changed between text_a and text_b.

    Scores each position by how many of its covering shingle ids appear in
    the symmetric difference of the two texts' shingle multisets.
    Returns None if either side is empty or lengths differ with no signal.
    """
    words_a = tokenize(text_a)
    words_b = tokenize(text_b)
    if not words_a or not words_b:
        return None

    slots_a = slot_shingle_sets(text_a)
    slots_b = slot_shingle_sets(text_b)

    ids_a: set[int] = set()
    for s in slots_a:
        ids_a |= s
    ids_b: set[int] = set()
    for s in slots_b:
        ids_b |= s

    changed = ids_a.symmetric_difference(ids_b)
    if not changed:
        # Fallback: token sequence align for pure substitution of equal length
        return localize_by_token_diff(text_a, text_b)

    n = max(len(slots_a), len(slots_b))
    scores = [0.0] * n
    for i, slot in enumerate(slots_a):
        scores[i] += len(slot & changed)
    for i, slot in enumerate(slots_b):
        if i < len(scores):
            scores[i] += len(slot & changed)
        else:
            scores.append(float(len(slot & changed)))

    best = max(scores)
    if best <= 0:
        return localize_by_token_diff(text_a, text_b)

    # Prefer indices that are content-aligned; among ties, pick center of mass
    tied = [i for i, s in enumerate(scores) if s == best]
    if len(tied) == 1:
        return tied[0]
    return int(round(sum(tied) / len(tied)))


def localize_by_token_diff(text_a: str, text_b: str) -> int | None:
    """First index where token sequences differ (sanity / equal-length sub)."""
    a = token_id_fingerprint(text_a)
    b = token_id_fingerprint(text_b)
    if not a and not b:
        return None
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return limit
    return None
