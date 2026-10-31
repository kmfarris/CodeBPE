"""Character n-gram hashed features for tokens (context-free)."""

from __future__ import annotations

import math
import zlib


def token_feature(word: str, dim: int = 32) -> list[float]:
    """Fixed-dim feature via signed hashed char trigrams."""
    w = f"#{word.lower()}#"
    vec = [0.0] * dim
    if len(w) < 3:
        w = w + "#" * (3 - len(w))
    for i in range(len(w) - 2):
        tri = w[i : i + 3]
        h = zlib.crc32(tri.encode("utf-8")) & 0xFFFFFFFF
        idx = h % dim
        sign = 1.0 if (h & 1) == 0 else -1.0
        vec[idx] += sign
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
