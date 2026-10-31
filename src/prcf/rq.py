"""Residual vector quantization (numpy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResidualVQ:
    n_codebooks: int
    codebook_size: int
    dim: int
    codebooks: list[np.ndarray]  # each (K, dim)
    rng: np.random.Generator

    @classmethod
    def fit(
        cls,
        vectors: np.ndarray,
        n_codebooks: int = 2,
        codebook_size: int = 128,
        seed: int = 0,
        kmeans_iters: int = 25,
    ) -> ResidualVQ:
        """Fit residual VQ by successive k-means on residuals."""
        assert vectors.ndim == 2
        n, dim = vectors.shape
        rng = np.random.default_rng(seed)
        codebooks: list[np.ndarray] = []
        residual = vectors.astype(np.float64).copy()
        for _ in range(n_codebooks):
            cb = _kmeans(residual, codebook_size, rng, kmeans_iters)
            codebooks.append(cb)
            # Assign and subtract
            idx = _nearest_indices(residual, cb)
            residual = residual - cb[idx]
        return cls(
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            dim=dim,
            codebooks=codebooks,
            rng=rng,
        )

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode (n, dim) -> (n, n_codebooks) int codes."""
        residual = vectors.astype(np.float64).copy()
        codes = np.zeros((len(vectors), self.n_codebooks), dtype=np.int32)
        for d, cb in enumerate(self.codebooks):
            idx = _nearest_indices(residual, cb)
            codes[:, d] = idx
            residual = residual - cb[idx]
        return codes

    def decode_vectors(self, codes: np.ndarray) -> np.ndarray:
        """codes (n, D) -> reconstructed vectors (n, dim)."""
        out = np.zeros((len(codes), self.dim), dtype=np.float64)
        for d, cb in enumerate(self.codebooks):
            out += cb[codes[:, d]]
        return out

    def encode_one(self, vector: np.ndarray) -> tuple[int, ...]:
        codes = self.encode(vector.reshape(1, -1))[0]
        return tuple(int(c) for c in codes)

    def codebook_usage(self, codes: np.ndarray) -> list[float]:
        """Fraction of codes used per depth."""
        usages = []
        for d in range(self.n_codebooks):
            used = len(np.unique(codes[:, d]))
            usages.append(used / self.codebook_size)
        return usages


def _nearest_indices(x: np.ndarray, cb: np.ndarray) -> np.ndarray:
    # (n, dim) vs (K, dim) -> (n,)
    # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
    x2 = np.sum(x * x, axis=1, keepdims=True)
    c2 = np.sum(cb * cb, axis=1)
    dots = x @ cb.T
    dist = x2 + c2 - 2 * dots
    return np.argmin(dist, axis=1)


def _kmeans(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    iters: int,
) -> np.ndarray:
    n = len(x)
    k = min(k, n)
    # init: random distinct points
    choices = rng.choice(n, size=k, replace=False)
    centers = x[choices].copy()
    for _ in range(iters):
        idx = _nearest_indices(x, centers)
        new_centers = centers.copy()
        for j in range(k):
            members = x[idx == j]
            if len(members) == 0:
                new_centers[j] = x[rng.integers(0, n)]
            else:
                new_centers[j] = members.mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers
