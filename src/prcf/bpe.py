"""Minimal BPE trainer/encoder for baseline comparison."""

from __future__ import annotations

from collections import Counter


class BPEEncoder:
    def __init__(self, token_to_id: dict[str, int], merges: list[tuple[str, str]]):
        self.token_to_id = token_to_id
        self.merges = merges

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 500) -> BPEEncoder:
        word_freqs: Counter[str] = Counter()
        for text in texts:
            for w in text.split():
                body = " ".join(list(w)) + " </w>"
                word_freqs[body] += 1
        merges: list[tuple[str, str]] = []
        vocab: set[str] = set()
        for word in word_freqs:
            vocab.update(word.split())
        target_merges = max(0, vocab_size - len(vocab) - 3)
        for _ in range(target_merges):
            pairs: Counter[tuple[str, str]] = Counter()
            for word, freq in word_freqs.items():
                syms = word.split()
                for i in range(len(syms) - 1):
                    pairs[(syms[i], syms[i + 1])] += freq
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            merges.append(best)
            a, b = best
            merged = a + b
            vocab.add(merged)
            new_freqs: Counter[str] = Counter()
            for word, freq in word_freqs.items():
                syms = word.split()
                out: list[str] = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                new_freqs[" ".join(out)] += freq
            word_freqs = new_freqs
        token_to_id = {"<unk>": 0, "<bos>": 1, "<eos>": 2}
        for i, tok in enumerate(sorted(vocab), start=3):
            token_to_id[tok] = i
        return cls(token_to_id, merges)

    def encode_word(self, word: str) -> list[int]:
        syms = list(word) + ["</w>"]
        for a, b in self.merges:
            merged = a + b
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

    def encode_text(self, text: str) -> list[int]:
        ids: list[int] = []
        for w in text.split():
            ids.extend(self.encode_word(w))
        return ids

    @property
    def vocab_size(self) -> int:
        return max(self.token_to_id.values()) + 1
