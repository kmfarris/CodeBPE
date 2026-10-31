"""Bucket+slot codec: RQ bucket + unique slot (injective in-vocab); ESC for OOV/punct."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from prcf.model import PRCFModel, train_prcf
from prcf.tokenize import tokenize

ID_PAD = 0
ID_BOS = 1
ID_EOS = 2
ID_ESC = 3
ID_ESC_END = 4
N_SPECIAL = 5


def _pack_rq(code: tuple[int, ...], codebook_size: int) -> int:
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
class BucketSlotCodec:
    """
    In-vocab clean token → one packed id = rq_base + pack(bucket)*S + slot.
    Slot is unique within each RQ bucket (freq desc, then alpha).
    OOV / rare / punct → ESC + chars + END.
    """

    prcf: PRCFModel
    in_vocab: set[str]
    surface: dict[str, str]
    key_to_bs: dict[str, tuple[tuple[int, ...], int]]
    bs_to_surface: dict[tuple[tuple[int, ...], int], str]
    slot_stride: int  # S
    rq_base: int
    char_base: int
    codebook_size: int
    n_codebooks: int
    min_count: int = 2
    max_slot: int = 0
    n_collision_buckets: int = 0
    _extra: dict = field(default_factory=dict)

    @classmethod
    def train(
        cls,
        texts: list[str],
        min_count: int = 2,
        dim: int = 32,
        n_codebooks: int = 2,
        codebook_size: int = 256,
        seed: int = 0,
    ) -> BucketSlotCodec:
        counts: Counter[str] = Counter()
        surface: dict[str, str] = {}
        for t in texts:
            for w in tokenize(t):
                key = w.lower()
                counts[key] += 1
                surface.setdefault(key, w)

        in_vocab = {w for w, n in counts.items() if n >= min_count}
        prcf = train_prcf(
            texts,
            dim=dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            seed=seed,
        )

        # Group in-vocab keys by RQ bucket
        buckets: dict[tuple[int, ...], list[str]] = defaultdict(list)
        for key in in_vocab:
            code = prcf.word_to_codes.get(key)
            if code is None:
                code = prcf.encode_tokens([surface[key]])[0]
            buckets[code].append(key)

        key_to_bs: dict[str, tuple[tuple[int, ...], int]] = {}
        bs_to_surface: dict[tuple[tuple[int, ...], int], str] = {}
        max_occ = 1
        n_coll = 0
        slot_sum = 0
        n_assigned = 0

        for code, keys in buckets.items():
            keys_sorted = sorted(keys, key=lambda k: (-counts[k], k))
            if len(keys_sorted) > 1:
                n_coll += 1
            max_occ = max(max_occ, len(keys_sorted))
            for slot, key in enumerate(keys_sorted):
                key_to_bs[key] = (code, slot)
                bs_to_surface[(code, slot)] = surface[key]
                slot_sum += slot
                n_assigned += 1

        # S = 1 + max occupancy so pack(bucket)*S + slot is unique
        S = max_occ
        rq_base = N_SPECIAL
        n_buckets_span = codebook_size**n_codebooks
        char_base = rq_base + n_buckets_span * S

        # Injectivity assert
        packed_ids = set()
        for key, (code, slot) in key_to_bs.items():
            pid = rq_base + _pack_rq(code, codebook_size) * S + slot
            assert pid not in packed_ids, f"collision on packed id for {key}"
            packed_ids.add(pid)
        assert len(packed_ids) == len(key_to_bs)

        return cls(
            prcf=prcf,
            in_vocab=in_vocab,
            surface=surface,
            key_to_bs=key_to_bs,
            bs_to_surface=bs_to_surface,
            slot_stride=S,
            rq_base=rq_base,
            char_base=char_base,
            codebook_size=codebook_size,
            n_codebooks=n_codebooks,
            min_count=min_count,
            max_slot=max_occ - 1,
            n_collision_buckets=n_coll,
            _extra={
                "mean_slot": slot_sum / max(n_assigned, 1),
                "n_in_vocab": len(in_vocab),
                "n_buckets_used": len(buckets),
            },
        )

    def _pack_id(self, code: tuple[int, ...], slot: int) -> int:
        return self.rq_base + _pack_rq(code, self.codebook_size) * self.slot_stride + slot

    def _unpack_id(self, tok: int) -> tuple[tuple[int, ...], int] | None:
        if not (self.rq_base <= tok < self.char_base):
            return None
        v = tok - self.rq_base
        slot = v % self.slot_stride
        bucket_packed = v // self.slot_stride
        code = _unpack_rq(bucket_packed, self.n_codebooks, self.codebook_size)
        return code, slot

    def _is_bucket_id(self, i: int) -> bool:
        return self.rq_base <= i < self.char_base

    def _is_char_id(self, i: int) -> bool:
        return i >= self.char_base

    def encode_text(self, text: str) -> list[int]:
        ids: list[int] = []
        for w in text.split():
            core = tokenize(w)
            use_bs = False
            if len(core) == 1 and core[0] == w:
                key = w.lower()
                if key in self.key_to_bs:
                    code, slot = self.key_to_bs[key]
                    ids.append(self._pack_id(code, slot))
                    use_bs = True
            if not use_bs:
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
            if self._is_bucket_id(tok):
                parsed = self._unpack_id(tok)
                if parsed is None:
                    words.append("<unk>")
                else:
                    code, slot = parsed
                    words.append(self.bs_to_surface.get((code, slot), "<unk>"))
                i += 1
                continue
            i += 1
        return " ".join(words)

    def reconstruct(self, text: str) -> str:
        return self.decode_ids(self.encode_text(text))

    def is_oov_word(self, word: str) -> bool:
        return word.lower() not in self.in_vocab

    def path_fractions(self, texts: list[str]) -> dict:
        n_bs = 0
        n_esc = 0
        for t in texts:
            for w in t.split():
                core = tokenize(w)
                if len(core) == 1 and core[0] == w and w.lower() in self.key_to_bs:
                    n_bs += 1
                else:
                    n_esc += 1
        total = n_bs + n_esc
        return {
            "n_bucket_path_words": n_bs,
            "n_esc_path_words": n_esc,
            "frac_words_bucket_path": n_bs / max(total, 1),
        }

    def symbols_per_char(self, texts: list[str]) -> float:
        n_syms = sum(len(self.encode_text(t)) for t in texts)
        n_chars = sum(max(len(t), 1) for t in texts)
        return n_syms / max(n_chars, 1)

    def stats(self) -> dict:
        return {
            "slot_stride": self.slot_stride,
            "max_slot": self.max_slot,
            "mean_slot": self._extra.get("mean_slot", 0.0),
            "n_collision_buckets": self.n_collision_buckets,
            "n_in_vocab": self._extra.get("n_in_vocab", len(self.in_vocab)),
            "n_buckets_used": self._extra.get("n_buckets_used", 0),
            "char_base": self.char_base,
        }
