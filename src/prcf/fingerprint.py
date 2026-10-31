"""Positioned word-trigram fingerprints."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

from prcf.tokenize import tokenize

BOS = "<BOS>"
EOS = "<EOS>"


def shingle_id(trigram: tuple[str, str, str]) -> int:
    payload = "\0".join(trigram).encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


@dataclass(frozen=True)
class ShingleHit:
    """One trigram anchored at the center word index."""

    anchor: int
    shingle_id: int
    trigram: tuple[str, str, str]


def _padded(words: list[str]) -> list[str]:
    return [BOS, BOS, *words, EOS, EOS]


def positioned_trigram_fingerprint(text: str) -> list[ShingleHit]:
    """
    Ordered list of trigrams covering the text.

    For each word index i, emit the three trigrams centered near i
    (left-aligned, center, right-aligned) that include w[i], via sliding
    windows of length 3 over a BOS/EOS-padded sequence. Deduped by
    (anchor=window_start_content_index, shingle_id) for stable ordering.
    """
    words = tokenize(text)
    if not words:
        return []

    padded = _padded(words)
    # padded indices: 0,1 = BOS; 2..n+1 = words; n+2,n+3 = EOS
    hits: list[ShingleHit] = []
    seen: set[tuple[int, int]] = set()

    # All length-3 windows over padded sequence
    for start in range(len(padded) - 2):
        tri = (padded[start], padded[start + 1], padded[start + 2])
        sid = shingle_id(tri)
        # Anchor = content word index of the middle token when it is a real word,
        # else nearest content word covered by this window.
        anchors: list[int] = []
        for offset in range(3):
            pidx = start + offset
            # content word at padded index pidx maps to word index pidx - 2
            if 2 <= pidx <= len(words) + 1:
                anchors.append(pidx - 2)
        if not anchors:
            continue
        # Record once per distinct content word covered, using min anchor as key
        # but emit one hit per window with primary anchor = middle content if any
        mid = start + 1
        if 2 <= mid <= len(words) + 1:
            primary = mid - 2
        else:
            primary = anchors[0]
        key = (primary, sid)
        if key in seen:
            continue
        seen.add(key)
        hits.append(ShingleHit(anchor=primary, shingle_id=sid, trigram=tri))

    hits.sort(key=lambda h: (h.anchor, h.shingle_id))
    return hits


def slot_shingle_sets(text: str) -> list[set[int]]:
    """For each word index, set of shingle ids whose trigram covers that word."""
    words = tokenize(text)
    n = len(words)
    slots: list[set[int]] = [set() for _ in range(n)]
    if n == 0:
        return slots

    padded = _padded(words)
    for start in range(len(padded) - 2):
        tri = (padded[start], padded[start + 1], padded[start + 2])
        sid = shingle_id(tri)
        for offset in range(3):
            pidx = start + offset
            if 2 <= pidx <= n + 1:
                slots[pidx - 2].add(sid)
    return slots


def token_id_fingerprint(text: str) -> list[str]:
    """Identity fingerprint: ordered word tokens (sanity upper bound)."""
    return tokenize(text)
