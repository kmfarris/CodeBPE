"""Hybrid codec: PRCF RQ for in-vocab words; ESC+chars+END for OOV (exact spelling)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from prcf.model import PRCFModel, train_prcf
from prcf.tokenize import tokenize


# Specials occupy low ids; RQ codes and chars use disjoint ranges above.
ID_PAD = 0
ID_BOS = 1
ID_EOS = 2
ID_ESC = 3
ID_ESC_END = 4
N_SPECIAL = 5


def _pack_rq(code: tuple[int, ...], codebook_size: int) -> int:
    """Pack RQ code tuple into a single non-negative int (base-K)."""
    v = 0
    for c in code:
        v = v * codebook_size + int(c)
    return v


def _unpack_rq(packed: int, n_codebooks: int, codebook_size: int) -> tuple[int, ...]:
    codes = [0] * n_codebooks
    v = packed
    for i in range(n_codebooks - 1, -1, -1):
        codes[i] = v % codebook_size
        v //= codebook_size
    return tuple(codes)


@dataclass
class HybridCodec:
    """
    Mixed symbol stream:

    - In-vocab word → one RQ-packed id in [rq_base, char_base)
    - OOV/rare → ESC, char ids in [char_base, ...), ESC_END

    Decode never uses nearest-word guessing.
    """

    prcf: PRCFModel
    in_vocab: set[str]
    surface: dict[str, str]
    rq_base: int
    char_base: int
    codebook_size: int
    n_codebooks: int
    min_count: int = 2

    @classmethod
    def train(
        cls,
        texts: list[str],
        min_count: int = 2,
        dim: int = 32,
        n_codebooks: int = 2,
        codebook_size: int = 256,
        seed: int = 0,
    ) -> HybridCodec:
        counts: Counter[str] = Counter()
        surface: dict[str, str] = {}
        for t in texts:
            for w in tokenize(t):
                key = w.lower()
                counts[key] += 1
                surface.setdefault(key, w)

        in_vocab = {w for w, n in counts.items() if n >= min_count}
        # Fit PRCF on full train text (same as before); vocab gate is separate.
        prcf = train_prcf(
            texts,
            dim=dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            seed=seed,
        )
        rq_base = N_SPECIAL
        # Max packed RQ value for D codebooks of size K is K^D - 1
        rq_span = codebook_size**n_codebooks
        char_base = rq_base + rq_span
        return cls(
            prcf=prcf,
            in_vocab=in_vocab,
            surface=surface,
            rq_base=rq_base,
            char_base=char_base,
            codebook_size=codebook_size,
            n_codebooks=n_codebooks,
            min_count=min_count,
        )

    def _rq_id(self, code: tuple[int, ...]) -> int:
        return self.rq_base + _pack_rq(code, self.codebook_size)

    def _is_rq_id(self, i: int) -> bool:
        return self.rq_base <= i < self.char_base

    def _is_char_id(self, i: int) -> bool:
        return i >= self.char_base

    def encode_text(self, text: str) -> list[int]:
        ids: list[int] = []
        # Whitespace split preserves punctuation attached to tokens (needed for chrF vs raw text).
        for w in text.split():
            core = tokenize(w)
            use_rq = False
            if len(core) == 1 and core[0] == w:
                key = w.lower()
                if key in self.in_vocab:
                    code = self.prcf.word_to_codes.get(key)
                    if code is None:
                        code = self.prcf.encode_tokens([w])[0]
                    canon = self.prcf.code_to_word.get(code, "")
                    if canon.lower() == key:
                        ids.append(self._rq_id(code))
                        use_rq = True
            if not use_rq:
                ids.append(ID_ESC)
                for ch in w:
                    ids.append(self.char_base + ord(ch))
                ids.append(ID_ESC_END)
        return ids

    def decode_ids(self, ids: list[int]) -> str:
        words: list[str] = []
        i = 0
        while i < len(ids):
            tok = ids[i]
            if tok == ID_ESC:
                chars: list[str] = []
                i += 1
                while i < len(ids) and ids[i] != ID_ESC_END:
                    if self._is_char_id(ids[i]):
                        chars.append(chr(ids[i] - self.char_base))
                    i += 1
                words.append("".join(chars))
                if i < len(ids) and ids[i] == ID_ESC_END:
                    i += 1
                continue
            if self._is_rq_id(tok):
                packed = tok - self.rq_base
                code = _unpack_rq(packed, self.n_codebooks, self.codebook_size)
                if code in self.prcf.code_to_word:
                    words.append(self.prcf.code_to_word[code])
                else:
                    # Unknown RQ id: empty placeholder (should be rare)
                    words.append("<unk>")
                i += 1
                continue
            # skip pad/bos/eos/unknown specials
            i += 1
        return " ".join(words)

    def reconstruct(self, text: str) -> str:
        return self.decode_ids(self.encode_text(text))

    def is_oov_word(self, word: str) -> bool:
        return word.lower() not in self.in_vocab

    @property
    def vocab_size_estimate(self) -> int:
        """Upper bound for embedding table (BMP chars + RQ span + specials)."""
        return self.char_base + 0x10000

    def symbols_per_char(self, texts: list[str]) -> float:
        n_syms = sum(len(self.encode_text(t)) for t in texts)
        n_chars = sum(max(len(t), 1) for t in texts)
        return n_syms / max(n_chars, 1)
