"""Positioned Residual Code Fingerprint model (Phase B)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prcf.features import token_feature
from prcf.rq import ResidualVQ
from prcf.tokenize import tokenize


@dataclass
class PRCFModel:
    dim: int
    rq: ResidualVQ
    code_to_word: dict[tuple[int, ...], str]
    word_to_codes: dict[str, tuple[int, ...]]
    word_vectors: dict[str, list[float]]

    def encode_tokens(self, words: list[str]) -> list[tuple[int, ...]]:
        if not words:
            return []
        vecs = np.array([self._vec(w) for w in words], dtype=np.float64)
        codes = self.rq.encode(vecs)
        return [tuple(int(c) for c in row) for row in codes]

    def encode_text(self, text: str) -> list[tuple[int, ...]]:
        return self.encode_tokens(tokenize(text))

    def decode_codes(self, codes: list[tuple[int, ...]]) -> list[str]:
        words: list[str] = []
        for code in codes:
            if code in self.code_to_word:
                words.append(self.code_to_word[code])
            else:
                words.append(self._nearest_word(code))
        return words

    def decode_text(self, codes: list[tuple[int, ...]]) -> str:
        return " ".join(self.decode_codes(codes))

    def reconstruct(self, text: str) -> str:
        return self.decode_text(self.encode_text(text))

    def _vec(self, word: str) -> np.ndarray:
        key = word.lower()
        if key in self.word_vectors:
            return np.array(self.word_vectors[key], dtype=np.float64)
        return np.array(token_feature(word, self.dim), dtype=np.float64)

    def _nearest_word(self, code: tuple[int, ...]) -> str:
        target = self.rq.decode_vectors(np.array([code], dtype=np.int32))[0]
        best_w = "<unk>"
        best_d = float("inf")
        for w, vec in self.word_vectors.items():
            v = np.array(vec, dtype=np.float64)
            d = float(np.sum((v - target) ** 2))
            if d < best_d:
                best_d = d
                best_w = w
        return best_w

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.dim,
            "n_codebooks": self.rq.n_codebooks,
            "codebook_size": self.rq.codebook_size,
            "codebooks": [cb.tolist() for cb in self.rq.codebooks],
            "code_to_word": {",".join(map(str, k)): v for k, v in self.code_to_word.items()},
            "word_to_codes": {k: list(v) for k, v in self.word_to_codes.items()},
            "word_vectors": self.word_vectors,
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> PRCFModel:
        payload = json.loads(Path(path).read_text())
        codebooks = [np.array(cb, dtype=np.float64) for cb in payload["codebooks"]]
        rq = ResidualVQ(
            n_codebooks=payload["n_codebooks"],
            codebook_size=payload["codebook_size"],
            dim=payload["dim"],
            codebooks=codebooks,
            rng=np.random.default_rng(0),
        )
        code_to_word = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in payload["code_to_word"].items()
        }
        word_to_codes = {k: tuple(v) for k, v in payload["word_to_codes"].items()}
        return cls(
            dim=payload["dim"],
            rq=rq,
            code_to_word=code_to_word,
            word_to_codes=word_to_codes,
            word_vectors=payload["word_vectors"],
        )


def train_prcf(
    texts: list[str],
    dim: int = 32,
    n_codebooks: int = 2,
    codebook_size: int = 128,
    seed: int = 0,
) -> PRCFModel:
    """Fit RQ on unique token features; build code→word decode table."""
    word_vectors: dict[str, list[float]] = {}
    for text in texts:
        for w in tokenize(text):
            key = w.lower()
            if key not in word_vectors:
                # Preserve first-seen casing as display form via separate map later
                word_vectors[key] = token_feature(w, dim)

    # Prefer surface forms from corpus (first occurrence casing)
    surface: dict[str, str] = {}
    for text in texts:
        for w in tokenize(text):
            surface.setdefault(w.lower(), w)

    keys = sorted(word_vectors.keys())
    mat = np.array([word_vectors[k] for k in keys], dtype=np.float64)
    rq = ResidualVQ.fit(
        mat,
        n_codebooks=n_codebooks,
        codebook_size=codebook_size,
        seed=seed,
    )
    codes = rq.encode(mat)
    code_to_words: dict[tuple[int, ...], Counter[str]] = defaultdict(Counter)
    word_to_codes: dict[str, tuple[int, ...]] = {}
    for i, key in enumerate(keys):
        code = tuple(int(c) for c in codes[i])
        word_to_codes[key] = code
        code_to_words[code][surface[key]] += 1

    code_to_word = {c: ctr.most_common(1)[0][0] for c, ctr in code_to_words.items()}
    return PRCFModel(
        dim=dim,
        rq=rq,
        code_to_word=code_to_word,
        word_to_codes=word_to_codes,
        word_vectors=word_vectors,
    )


def localize_by_codes(
    codes_a: list[tuple[int, ...]],
    codes_b: list[tuple[int, ...]],
) -> int | None:
    """Predict edit index from differing RQ code slots."""
    n = max(len(codes_a), len(codes_b))
    if n == 0:
        return None
    scores = [0.0] * n
    for i in range(min(len(codes_a), len(codes_b))):
        if codes_a[i] != codes_b[i]:
            scores[i] += 1.0
    # length mismatch: score the extra region
    for i in range(min(len(codes_a), len(codes_b)), n):
        scores[i] += 1.0
    best = max(scores)
    if best <= 0:
        return None
    tied = [i for i, s in enumerate(scores) if s == best]
    return int(round(sum(tied) / len(tied)))
