"""CodeBPE over BucketSlotCodec streams: merge adjacent B-atoms only (never ESC spans)."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

from prcf.bucket_slot import (
    ID_ESC,
    ID_ESC_END,
    BucketSlotCodec,
)
from prcf.code_bpe import _apply_merge_incremental

# LM-local specials / char range (compact; not codec.char_base)
LM_UNK = 0
LM_BOS = 1
LM_EOS = 2
LM_ESC = 3
LM_ESC_END = 4
LM_N_SPECIAL = 5
CHAR_VOCAB = 256  # latin-1 ordinals for ESC payloads in the LM stream


def _is_b_path(sym: str) -> bool:
    return sym.startswith("B") or sym.startswith("M")


def _is_e_atom(sym: str) -> bool:
    return sym.startswith("E:")


def segment_ids(ids: list[int], codec: BucketSlotCodec) -> list[str]:
    """Bucket ids → B{id}; ESC…END → one opaque E:ord,ord,… atom."""
    atoms: list[str] = []
    i = 0
    while i < len(ids):
        tok = ids[i]
        if tok == ID_ESC:
            ords: list[str] = []
            i += 1
            while i < len(ids) and ids[i] != ID_ESC_END:
                if codec._is_char_id(ids[i]):
                    ords.append(str(ids[i] - codec.char_base))
                i += 1
            atoms.append("E:" + ",".join(ords))
            if i < len(ids) and ids[i] == ID_ESC_END:
                i += 1
            continue
        if codec._is_bucket_id(tok):
            atoms.append(f"B{tok}")
            i += 1
            continue
        i += 1
    return atoms


def atoms_to_ids(atoms: list[str], codec: BucketSlotCodec) -> list[int]:
    """Expand base atoms back to BucketSlotCodec int stream."""
    out: list[int] = []
    for a in atoms:
        if a.startswith("B"):
            out.append(int(a[1:]))
        elif a.startswith("E:"):
            out.append(ID_ESC)
            payload = a[2:]
            if payload:
                for part in payload.split(","):
                    out.append(codec.char_base + int(part))
            out.append(ID_ESC_END)
        else:
            raise ValueError(f"unexpected base atom {a!r}")
    return out


def _best_bb_pair(
    pairs: Counter[tuple[str, str]], min_freq: int
) -> tuple[tuple[str, str], int] | None:
    best: tuple[str, str] | None = None
    best_f = -1
    for p, f in pairs.items():
        if f < min_freq:
            continue
        a, b = p
        if not (_is_b_path(a) and _is_b_path(b)):
            continue
        if f > best_f or (f == best_f and (best is None or p < best)):
            best = p
            best_f = f
    if best is None:
        return None
    return best, best_f


def _b_mean_unit_len(
    corpus: list[list[str]],
    sym_to_expansion: dict[str, list[str]],
) -> float:
    total_leaves = 0
    total_syms = 0
    for syms in corpus:
        for s in syms:
            if not _is_b_path(s):
                continue
            total_syms += 1
            total_leaves += len(sym_to_expansion[s])
    if total_syms == 0:
        return 1.0
    return total_leaves / total_syms


@dataclass
class BucketCodeBPE:
    """
    Merges frequent adjacent bucket+slot atoms only.
    ESC spans are opaque during merge; LM encode expands them to ESC+latin1+END.
    """

    codec: BucketSlotCodec
    merges: list[tuple[str, str]]
    token_to_id: dict[str, int]
    id_to_atom: dict[int, str]
    sym_to_expansion: dict[str, list[str]]
    bm_vocab_end: int  # exclusive end of B/M id range; char base follows specials layout
    n_merges_freq2: int = 0
    n_merges_freq1: int = 0
    mean_b_unit_len: float = 1.0
    train_wall_s: float = 0.0
    merge_freq_hist: dict[str, int] = field(default_factory=dict)

    @classmethod
    def train(
        cls,
        codec: BucketSlotCodec,
        texts: list[str],
        target_mean_len: float = 1.5,
        max_merges: int = 20000,
        min_freq: int = 2,
        allow_freq1: bool = False,
    ) -> BucketCodeBPE:
        t0 = time.perf_counter()
        corpus: list[list[str]] = []
        sym_to_expansion: dict[str, list[str]] = {}

        for text in texts:
            ids = codec.encode_text(text)
            atoms = segment_ids(ids, codec)
            if not atoms:
                continue
            corpus.append(atoms)
            for a in atoms:
                sym_to_expansion.setdefault(a, [a])

        merges: list[tuple[str, str]] = []
        next_id = 0
        n_merges_freq2 = 0
        n_merges_freq1 = 0
        freq_hist: Counter[int] = Counter()

        full_pairs: Counter[tuple[str, str]] = Counter()
        for syms in corpus:
            for i in range(len(syms) - 1):
                full_pairs[(syms[i], syms[i + 1])] += 1

        current_min_freq = min_freq
        while len(merges) < max_merges:
            mean_len = _b_mean_unit_len(corpus, sym_to_expansion)
            if mean_len >= target_mean_len:
                break

            if len(merges) % 500 == 0 and len(merges) > 0:
                print(
                    f"  BucketCodeBPE merge {len(merges)}/{max_merges} "
                    f"mean_b_len={mean_len:.3f} min_freq={current_min_freq}",
                    flush=True,
                )

            picked = _best_bb_pair(full_pairs, current_min_freq)
            if picked is None:
                if current_min_freq > 1 and allow_freq1:
                    current_min_freq = 1
                    continue
                break

            best, freq = picked
            a, b = best
            merges.append(best)
            freq_hist[freq] += 1
            if freq >= 2:
                n_merges_freq2 += 1
            else:
                n_merges_freq1 += 1
            merged = f"M{next_id}"
            next_id += 1
            sym_to_expansion[merged] = sym_to_expansion[a] + sym_to_expansion[b]
            _apply_merge_incremental(corpus, full_pairs, a, b, merged)

        # Vocab: only B/M atoms (E expanded at encode time)
        bm_syms = sorted(
            {s for syms in corpus for s in syms if _is_b_path(s)}
            | {s for s in sym_to_expansion if _is_b_path(s)}
        )
        token_to_id = {
            "<unk>": LM_UNK,
            "<bos>": LM_BOS,
            "<eos>": LM_EOS,
            "<esc>": LM_ESC,
            "<esc_end>": LM_ESC_END,
        }
        for i, sym in enumerate(bm_syms, start=LM_N_SPECIAL):
            token_to_id[sym] = i
        bm_vocab_end = LM_N_SPECIAL + len(bm_syms)
        id_to_atom = {i: s for s, i in token_to_id.items() if s.startswith("B") or s.startswith("M")}

        mean_b = _b_mean_unit_len(corpus, sym_to_expansion)
        wall = time.perf_counter() - t0
        hist_buckets = {
            "freq_1": sum(c for f, c in freq_hist.items() if f == 1),
            "freq_2_4": sum(c for f, c in freq_hist.items() if 2 <= f <= 4),
            "freq_5_9": sum(c for f, c in freq_hist.items() if 5 <= f <= 9),
            "freq_10_plus": sum(c for f, c in freq_hist.items() if f >= 10),
        }

        return cls(
            codec=codec,
            merges=merges,
            token_to_id=token_to_id,
            id_to_atom=id_to_atom,
            sym_to_expansion=sym_to_expansion,
            bm_vocab_end=bm_vocab_end,
            n_merges_freq2=n_merges_freq2,
            n_merges_freq1=n_merges_freq1,
            mean_b_unit_len=mean_b,
            train_wall_s=wall,
            merge_freq_hist=hist_buckets,
        )

    @property
    def char_lm_base(self) -> int:
        return self.bm_vocab_end

    def _apply_merges(self, atoms: list[str]) -> list[str]:
        syms = atoms
        for mi, (a, b) in enumerate(self.merges):
            merged = f"M{mi}"
            out: list[str] = []
            i = 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            syms = out
        return syms

    def _expand_atom(self, sym: str) -> list[str]:
        if sym in self.sym_to_expansion:
            return list(self.sym_to_expansion[sym])
        return [sym]

    def encode_text(self, text: str) -> list[int]:
        """LM ids: B/M dense ids; E spans → ESC + latin-1 chars + ESC_END."""
        atoms = self._apply_merges(segment_ids(self.codec.encode_text(text), self.codec))
        ids: list[int] = []
        for s in atoms:
            if _is_e_atom(s):
                ids.append(LM_ESC)
                payload = s[2:]
                if payload:
                    for part in payload.split(","):
                        o = int(part)
                        ids.append(self.char_lm_base + (o if 0 <= o < CHAR_VOCAB else 0))
                ids.append(LM_ESC_END)
            elif _is_b_path(s):
                ids.append(self.token_to_id.get(s, LM_UNK))
            else:
                ids.append(LM_UNK)
        return ids

    def decode_ids(self, ids: list[int]) -> str:
        """LM ids → codec stream → text."""
        base_atoms: list[str] = []
        i = 0
        while i < len(ids):
            tok = ids[i]
            if tok == LM_ESC:
                ords: list[str] = []
                i += 1
                while i < len(ids) and ids[i] != LM_ESC_END:
                    if ids[i] >= self.char_lm_base:
                        ords.append(str(ids[i] - self.char_lm_base))
                    i += 1
                base_atoms.append("E:" + ",".join(ords))
                if i < len(ids) and ids[i] == LM_ESC_END:
                    i += 1
                continue
            if tok in self.id_to_atom:
                base_atoms.extend(self._expand_atom(self.id_to_atom[tok]))
                i += 1
                continue
            i += 1
        stream = atoms_to_ids(base_atoms, self.codec)
        return self.codec.decode_ids(stream)

    def reconstruct(self, text: str) -> str:
        """Lossless round-trip via atoms (E payloads kept; no UNK drop)."""
        atoms = self._apply_merges(segment_ids(self.codec.encode_text(text), self.codec))
        base: list[str] = []
        for s in atoms:
            base.extend(self._expand_atom(s))
        return self.codec.decode_ids(atoms_to_ids(base, self.codec))

    @property
    def vocab_size(self) -> int:
        return self.char_lm_base + CHAR_VOCAB

    def symbols_per_char(self, texts: list[str]) -> float:
        n_syms = sum(len(self.encode_text(t)) for t in texts)
        n_chars = sum(max(len(t), 1) for t in texts)
        return n_syms / max(n_chars, 1)

    def merge_stats(self) -> dict:
        return {
            "n_merges": len(self.merges),
            "n_merges_freq2": self.n_merges_freq2,
            "n_merges_freq1": self.n_merges_freq1,
            "vocab_size": self.vocab_size,
            "mean_b_unit_len": self.mean_b_unit_len,
            "train_wall_s": self.train_wall_s,
            "merge_freq_hist": dict(self.merge_freq_hist),
        }
