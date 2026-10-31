"""BPE-style merges over PRCF token-code sequences (variable-length units)."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

from prcf.model import PRCFModel
from prcf.tokenize import tokenize


def code_to_sym(code: tuple[int, ...]) -> str:
    return ":".join(map(str, code))


def sym_to_code(sym: str) -> tuple[int, ...]:
    return tuple(int(x) for x in sym.split(":"))


def _corpus_mean_unit_len(
    corpus: list[list[str]],
    sym_to_expansion: dict[str, list[tuple[int, ...]]],
) -> float:
    total_codes = 0
    total_syms = 0
    for syms in corpus:
        for s in syms:
            total_codes += len(sym_to_expansion[s])
            total_syms += 1
    if total_syms == 0:
        return 1.0
    return total_codes / total_syms


def _symbols_per_word(corpus: list[list[str]], n_words: int) -> float:
    n_syms = sum(len(s) for s in corpus)
    return n_syms / max(n_words, 1)


def _count_pairs(corpus: list[list[str]]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for syms in corpus:
        for i in range(len(syms) - 1):
            pairs[(syms[i], syms[i + 1])] += 1
    return pairs


def _best_pair(
    pairs: Counter[tuple[str, str]], min_freq: int
) -> tuple[tuple[str, str], int] | None:
    best: tuple[str, str] | None = None
    best_f = -1
    for p, f in pairs.items():
        if f < min_freq:
            continue
        if f > best_f or (f == best_f and (best is None or p < best)):
            best = p
            best_f = f
    if best is None:
        return None
    return best, best_f


def _apply_merge_incremental(
    corpus: list[list[str]],
    pairs: Counter[tuple[str, str]],
    a: str,
    b: str,
    merged: str,
) -> int:
    """Apply (a,b)->merged across corpus; update pair counts in place. Returns #merges."""
    n_applied = 0
    for doc_i, syms in enumerate(corpus):
        if len(syms) < 2:
            continue
        out: list[str] = []
        i = 0
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                left = out[-1] if out else None
                right = syms[i + 2] if i + 2 < len(syms) else None
                # remove old pairs involving (a,b)
                pairs[(a, b)] -= 1
                if left is not None:
                    pairs[(left, a)] -= 1
                if right is not None:
                    pairs[(b, right)] -= 1
                # add new pairs involving merged
                if left is not None:
                    pairs[(left, merged)] += 1
                if right is not None:
                    pairs[(merged, right)] += 1
                out.append(merged)
                i += 2
                n_applied += 1
            else:
                out.append(syms[i])
                i += 1
        corpus[doc_i] = out
    # clean zero/negative clutter
    dead = [p for p, f in pairs.items() if f <= 0]
    for p in dead:
        del pairs[p]
    return n_applied


@dataclass
class CodeBPE:
    """
    Variable-length units from merging frequent adjacent token-RQ codes.

    Encoding → LM ids is reversible to the base RQ code sequence via
    id_to_expansion, then PRCF decodes to text.
    """

    merges: list[tuple[str, str]]
    token_to_id: dict[str, int]
    id_to_expansion: dict[int, list[tuple[int, ...]]]
    sym_to_expansion: dict[str, list[tuple[int, ...]]]
    prcf: PRCFModel
    n_merges_freq2: int = 0
    n_merges_freq1: int = 0
    corpus_mean_unit_len: float = 1.0
    symbols_per_word: float = 1.0
    train_wall_s: float = 0.0
    merge_freq_hist: dict[str, int] = field(default_factory=dict)
    _extra_stats: dict = field(default_factory=dict)

    @classmethod
    def train(
        cls,
        prcf: PRCFModel,
        texts: list[str],
        vocab_size: int = 1500,
        n_merges: int | None = None,
        target_mean_len: float | None = None,
        max_merges: int = 5000,
        min_freq: int = 2,
        allow_freq1: bool = True,
    ) -> CodeBPE:
        t0 = time.perf_counter()
        corpus: list[list[str]] = []
        sym_to_expansion: dict[str, list[tuple[int, ...]]] = {}
        n_words = 0
        total_codes = 0
        total_syms = 0

        for text in texts:
            codes = prcf.encode_tokens(tokenize(text))
            n_words += len(codes)
            syms = [code_to_sym(c) for c in codes]
            if not syms:
                continue
            corpus.append(syms)
            total_syms += len(syms)
            total_codes += len(syms)  # each base sym expands to 1 code tuple
            for c, s in zip(codes, syms):
                sym_to_expansion.setdefault(s, [c])

        merges: list[tuple[str, str]] = []
        next_id = 0
        n_merges_freq2 = 0
        n_merges_freq1 = 0
        freq_hist: Counter[int] = Counter()

        # Legacy path: fixed merge count when no coarseness target
        if target_mean_len is None:
            if n_merges is None:
                n_merges = max(0, vocab_size - len(sym_to_expansion) - 3)
                if n_merges == 0:
                    n_merges = min(800, max(64, len(sym_to_expansion) // 4))
            max_merges = n_merges
            target_mean_len = float("inf")  # only stop on max_merges / no pairs

        pairs = _count_pairs(corpus)
        current_min_freq = min_freq

        while len(merges) < max_merges:
            mean_len = total_codes / max(total_syms, 1)
            if mean_len >= target_mean_len:
                break

            if len(merges) % 500 == 0 and len(merges) > 0:
                print(
                    f"  CodeBPE merge {len(merges)}/{max_merges} "
                    f"mean_len={mean_len:.3f} min_freq={current_min_freq}",
                    flush=True,
                )

            picked = _best_pair(pairs, current_min_freq)
            if picked is None:
                if current_min_freq > 1 and allow_freq1:
                    print(
                        f"  lowering min_freq 2→1 at merge {len(merges)} "
                        f"mean_len={mean_len:.3f}",
                        flush=True,
                    )
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
            n_applied = _apply_merge_incremental(corpus, pairs, a, b, merged)
            total_syms -= n_applied

        final_syms: set[str] = set()
        for syms in corpus:
            final_syms.update(syms)

        token_to_id = {"<unk>": 0, "<bos>": 1, "<eos>": 2}
        for i, sym in enumerate(sorted(final_syms), start=3):
            token_to_id[sym] = i

        id_to_expansion = {
            idx: sym_to_expansion[sym]
            for sym, idx in token_to_id.items()
            if idx >= 3 and sym in sym_to_expansion
        }

        corpus_mean = total_codes / max(total_syms, 1)
        spw = _symbols_per_word(corpus, n_words)
        wall = time.perf_counter() - t0

        # Compact freq histogram buckets for logging
        hist_buckets = {
            "freq_1": sum(c for f, c in freq_hist.items() if f == 1),
            "freq_2_4": sum(c for f, c in freq_hist.items() if 2 <= f <= 4),
            "freq_5_9": sum(c for f, c in freq_hist.items() if 5 <= f <= 9),
            "freq_10_plus": sum(c for f, c in freq_hist.items() if f >= 10),
        }

        return cls(
            merges=merges,
            token_to_id=token_to_id,
            id_to_expansion=id_to_expansion,
            sym_to_expansion=sym_to_expansion,
            prcf=prcf,
            n_merges_freq2=n_merges_freq2,
            n_merges_freq1=n_merges_freq1,
            corpus_mean_unit_len=corpus_mean,
            symbols_per_word=spw,
            train_wall_s=wall,
            merge_freq_hist=hist_buckets,
        )

    def encode_text(self, text: str) -> list[int]:
        codes = self.prcf.encode_tokens(tokenize(text))
        syms = [code_to_sym(c) for c in codes]
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
        return [self.token_to_id.get(s, 0) for s in syms]

    def expand_ids(self, ids: list[int]) -> list[tuple[int, ...]]:
        codes: list[tuple[int, ...]] = []
        for i in ids:
            if i in self.id_to_expansion:
                codes.extend(self.id_to_expansion[i])
        return codes

    def decode_ids(self, ids: list[int]) -> str:
        return self.prcf.decode_text(self.expand_ids(ids))

    def reconstruct(self, text: str) -> str:
        return self.decode_ids(self.encode_text(text))

    @property
    def vocab_size(self) -> int:
        return max(self.token_to_id.values()) + 1

    def merge_stats(self) -> dict:
        lengths = [len(exp) for exp in self.sym_to_expansion.values()]
        return {
            "n_merges": len(self.merges),
            "n_merges_freq2": self.n_merges_freq2,
            "n_merges_freq1": self.n_merges_freq1,
            "vocab_size": self.vocab_size,
            "mean_unit_len_codes": self.corpus_mean_unit_len,
            "mean_expansion_table_len": sum(lengths) / max(len(lengths), 1),
            "max_unit_len_codes": max(
                (len(self.sym_to_expansion[s]) for s in self.token_to_id if s in self.sym_to_expansion),
                default=0,
            ),
            "symbols_per_word": self.symbols_per_word,
            "train_wall_s": self.train_wall_s,
            "merge_freq_hist": dict(self.merge_freq_hist),
        }
