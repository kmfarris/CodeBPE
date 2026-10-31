"""Collapsed SimHash — negative control (cannot localize edits)."""

from __future__ import annotations

import hashlib

from prcf.tokenize import tokenize


def _feature_hashes(text: str) -> list[int]:
    words = [w.lower() for w in tokenize(text)]
    feats: list[str] = []
    feats.extend(words)
    for i in range(len(words) - 1):
        feats.append(f"{words[i]}_{words[i + 1]}")
    out: list[int] = []
    for f in feats:
        digest = hashlib.sha256(f.encode("utf-8")).digest()
        out.append(int.from_bytes(digest[:8], "big"))
    return out


def simhash64(text: str) -> int:
    """64-bit SimHash over word unigrams + bigrams."""
    feats = _feature_hashes(text)
    if not feats:
        return 0
    bits = [0] * 64
    for h in feats:
        for b in range(64):
            if (h >> b) & 1:
                bits[b] += 1
            else:
                bits[b] -= 1
    result = 0
    for b in range(64):
        if bits[b] > 0:
            result |= 1 << b
    return result


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()
