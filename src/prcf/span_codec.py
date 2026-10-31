"""Span-level residual VQ codec (Phase C)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from prcf.features import token_feature
from prcf.rq import ResidualVQ
from prcf.tokenize import tokenize


def chunk_spans(words: list[str], span_size: int) -> list[list[str]]:
    if not words:
        return []
    spans = []
    for i in range(0, len(words), span_size):
        spans.append(words[i : i + span_size])
    return spans


def span_feature(words: list[str], dim: int) -> np.ndarray:
    if not words:
        return np.zeros(dim, dtype=np.float64)
    mat = np.array([token_feature(w, dim) for w in words], dtype=np.float64)
    v = mat.mean(axis=0)
    n = np.linalg.norm(v) or 1.0
    return v / n


@dataclass
class SpanCodec:
    span_size: int
    dim: int
    rq: ResidualVQ
    code_to_span: dict[tuple[int, ...], tuple[str, ...]]
    span_to_code: dict[tuple[str, ...], tuple[int, ...]]

    def encode_text(self, text: str) -> list[tuple[int, ...]]:
        words = tokenize(text)
        spans = chunk_spans(words, self.span_size)
        if not spans:
            return []
        vecs = np.array([span_feature(s, self.dim) for s in spans], dtype=np.float64)
        codes = self.rq.encode(vecs)
        return [tuple(int(c) for c in row) for row in codes]

    def decode_codes(self, codes: list[tuple[int, ...]]) -> list[str]:
        words: list[str] = []
        for code in codes:
            if code in self.code_to_span:
                words.extend(self.code_to_span[code])
            else:
                words.extend(self._nearest_span(code))
        return words

    def decode_text(self, codes: list[tuple[int, ...]]) -> str:
        return " ".join(self.decode_codes(codes))

    def reconstruct(self, text: str) -> str:
        return self.decode_text(self.encode_text(text))

    def code_id_sequence(self, text: str, code_vocab: dict[tuple[int, ...], int]) -> list[int]:
        """Map RQ tuples to dense ids for LM; unknown → 0 (unk)."""
        unk = 0
        out = []
        for code in self.encode_text(text):
            out.append(code_vocab.get(code, unk))
        return out

    def _nearest_span(self, code: tuple[int, ...]) -> list[str]:
        # Avoid O(n_spans) search in benches; unknown codes become unk pads.
        return ["<unk>"] * self.span_size


def train_span_codec(
    texts: list[str],
    span_size: int = 3,
    dim: int = 32,
    n_codebooks: int = 2,
    codebook_size: int = 256,
    seed: int = 0,
) -> SpanCodec:
    span_vecs: dict[tuple[str, ...], np.ndarray] = {}
    span_counts: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        words = tokenize(text)
        for span in chunk_spans(words, span_size):
            key = tuple(w.lower() for w in span)
            surface = tuple(span)
            span_counts[surface] += 1
            if key not in span_vecs:
                span_vecs[key] = span_feature(list(span), dim)

    # Prefer most common surface form per lower-key
    key_to_surface: dict[tuple[str, ...], tuple[str, ...]] = {}
    surface_by_key: dict[tuple[str, ...], Counter[tuple[str, ...]]] = defaultdict(Counter)
    for text in texts:
        for span in chunk_spans(tokenize(text), span_size):
            key = tuple(w.lower() for w in span)
            surface_by_key[key][tuple(span)] += 1
    for key, ctr in surface_by_key.items():
        key_to_surface[key] = ctr.most_common(1)[0][0]

    keys = sorted(span_vecs.keys())
    mat = np.array([span_vecs[k] for k in keys], dtype=np.float64)
    rq = ResidualVQ.fit(
        mat,
        n_codebooks=n_codebooks,
        codebook_size=codebook_size,
        seed=seed,
    )
    codes = rq.encode(mat)
    code_to_spans: dict[tuple[int, ...], Counter[tuple[str, ...]]] = defaultdict(Counter)
    span_to_code: dict[tuple[str, ...], tuple[int, ...]] = {}
    for i, key in enumerate(keys):
        code = tuple(int(c) for c in codes[i])
        surface = key_to_surface[key]
        span_to_code[surface] = code
        span_to_code[key] = code
        code_to_spans[code][surface] += span_counts[surface]

    code_to_span = {c: ctr.most_common(1)[0][0] for c, ctr in code_to_spans.items()}
    return SpanCodec(
        span_size=span_size,
        dim=dim,
        rq=rq,
        code_to_span=code_to_span,
        span_to_code=span_to_code,
    )


def localize_span_edit(
    codes_a: list[tuple[int, ...]],
    codes_b: list[tuple[int, ...]],
) -> int | None:
    n = max(len(codes_a), len(codes_b))
    if n == 0:
        return None
    scores = [0.0] * n
    for i in range(min(len(codes_a), len(codes_b))):
        if codes_a[i] != codes_b[i]:
            scores[i] += 1.0
    for i in range(min(len(codes_a), len(codes_b)), n):
        scores[i] += 1.0
    best = max(scores)
    if best <= 0:
        return None
    tied = [i for i, s in enumerate(scores) if s == best]
    return int(round(sum(tied) / len(tied)))


def phrase_swap_success(
    codec: SpanCodec,
    text_a: str,
    text_b: str,
    span_index: int,
) -> bool:
    """Replace one span's codes from B into A; check decoded span matches B's phrase."""
    ca = codec.encode_text(text_a)
    cb = codec.encode_text(text_b)
    if span_index >= len(ca) or span_index >= len(cb):
        return False
    mixed = list(ca)
    mixed[span_index] = cb[span_index]
    decoded = codec.decode_text(mixed)
    target_span = chunk_spans(tokenize(text_b), codec.span_size)[span_index]
    return " ".join(target_span).lower() in decoded.lower()
