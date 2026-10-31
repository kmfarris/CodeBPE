"""Equal-compute n-gram LMs for comparing substrates."""

from __future__ import annotations

import math
from collections import Counter, defaultdict


class TrigramLM:
    """Add-k smoothed trigram LM over integer token ids."""

    def __init__(self, vocab_size: int, k: float = 0.1):
        self.vocab_size = vocab_size
        self.k = k
        self.uni: Counter[int] = Counter()
        self.bi: dict[int, Counter[int]] = defaultdict(Counter)
        self.tri: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        self.n_tokens = 0

    def fit(self, sequences: list[list[int]]) -> None:
        for seq in sequences:
            padded = [1, 1] + seq + [2]  # bos=1, eos=2; 0 reserved unk
            self.n_tokens += len(seq) + 1
            for i, tok in enumerate(padded):
                self.uni[tok] += 1
                if i >= 1:
                    self.bi[padded[i - 1]][tok] += 1
                if i >= 2:
                    self.tri[(padded[i - 2], padded[i - 1])][tok] += 1

    def logprob(self, tok: int, h2: int, h1: int) -> float:
        """log P(tok | h2, h1) with add-k trigram, backoff to bi/uni."""
        tri_counts = self.tri[(h2, h1)]
        tri_den = sum(tri_counts.values())
        if tri_den > 0:
            return math.log((tri_counts[tok] + self.k) / (tri_den + self.k * self.vocab_size))
        bi_counts = self.bi[h1]
        bi_den = sum(bi_counts.values())
        if bi_den > 0:
            return math.log((bi_counts[tok] + self.k) / (bi_den + self.k * self.vocab_size))
        uni_den = sum(self.uni.values()) or 1
        return math.log((self.uni[tok] + self.k) / (uni_den + self.k * self.vocab_size))

    def sequence_nll(self, seq: list[int]) -> float:
        """Total negative log-likelihood in nats (includes eos)."""
        padded = [1, 1] + seq + [2]
        nll = 0.0
        for i in range(2, len(padded)):
            nll -= self.logprob(padded[i], padded[i - 2], padded[i - 1])
        return nll


def bits_per_char(nll_nats: float, n_chars: int) -> float:
    if n_chars <= 0:
        return float("inf")
    return (nll_nats / math.log(2)) / n_chars
