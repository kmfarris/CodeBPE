"""Phase C: span-grouped token RQ vs BPE under equal-compute trigram LMs."""

from __future__ import annotations

import json
import random
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.phase_a.build_pairs import load_paragraphs
from prcf.bpe import BPEEncoder
from prcf.chrf import chrf
from prcf.model import train_prcf
from prcf.ngram_lm import TrigramLM, bits_per_char
from prcf.tokenize import tokenize

OUT_DIR = Path(__file__).resolve().parent

GATE_BPC_RATIO = 1.05
GATE_CHRF = 0.80
GATE_PHRASE_LOCALIZE = 0.70
GATE_COMPRESSION = 0.55
SPAN_SIZE = 3
SPAN_LM_VOCAB = 4096  # hashed span ids for LM


def chunk_spans(words: list[str], span_size: int) -> list[list[str]]:
    return [words[i : i + span_size] for i in range(0, len(words), span_size)] if words else []


def span_lm_id(token_codes: list[tuple[int, ...]]) -> int:
    """Hash a span's token-code tuples into LM id space [3, 3+SPAN_LM_VOCAB)."""
    payload = "|".join(",".join(map(str, c)) for c in token_codes).encode()
    return 3 + (zlib.crc32(payload) % SPAN_LM_VOCAB)


def encode_span_lm_seq(model, text: str) -> list[int]:
    words = tokenize(text)
    codes = model.encode_tokens(words)
    spans = []
    for i in range(0, len(codes), SPAN_SIZE):
        spans.append(span_lm_id(codes[i : i + SPAN_SIZE]))
    return spans


def encode_span_code_seqs(model, text: str) -> list[list[tuple[int, ...]]]:
    words = tokenize(text)
    codes = model.encode_tokens(words)
    return [codes[i : i + SPAN_SIZE] for i in range(0, len(codes), SPAN_SIZE)]


def reconstruct_via_token_codes(model, text: str) -> str:
    return model.reconstruct(text)


def localize_span(model, original: str, edited: str) -> int | None:
    a = encode_span_code_seqs(model, original)
    b = encode_span_code_seqs(model, edited)
    n = max(len(a), len(b))
    if n == 0:
        return None
    scores = [0.0] * n
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            scores[i] += 1.0
    for i in range(min(len(a), len(b)), n):
        scores[i] += 1.0
    best = max(scores)
    if best <= 0:
        return None
    tied = [i for i, s in enumerate(scores) if s == best]
    return int(round(sum(tied) / len(tied)))


def phrase_swap(model, text_a: str, text_b: str, span_index: int) -> bool:
    wa, wb = tokenize(text_a), tokenize(text_b)
    ca, cb = model.encode_tokens(wa), model.encode_tokens(wb)
    spans_a = [ca[i : i + SPAN_SIZE] for i in range(0, len(ca), SPAN_SIZE)]
    spans_b = [cb[i : i + SPAN_SIZE] for i in range(0, len(cb), SPAN_SIZE)]
    if span_index >= len(spans_a) or span_index >= len(spans_b):
        return False
    mixed = list(ca)
    start = span_index * SPAN_SIZE
    repl = spans_b[span_index]
    mixed[start : start + len(spans_a[span_index])] = repl
    decoded = model.decode_codes(mixed)
    target = chunk_spans(wb, SPAN_SIZE)[span_index]
    return " ".join(target).lower() in " ".join(decoded).lower()


def eval_lm_bpc(lm: TrigramLM, sequences: list[list[int]], texts: list[str]) -> float:
    total_nll = 0.0
    total_chars = 0
    for seq, text in zip(sequences, texts):
        total_nll += lm.sequence_nll(seq)
        total_chars += max(len(text), 1)
    return bits_per_char(total_nll, total_chars)


def symbols_per_char(sequences: list[list[int]], texts: list[str]) -> float:
    n_sym = sum(len(s) for s in sequences)
    n_char = sum(max(len(t), 1) for t in texts)
    return n_sym / n_char


def make_phrase_edit_pairs(paragraphs: list[str], n: int, seed: int):
    rng = random.Random(seed)
    pairs = []
    pool = [p for p in paragraphs if len(tokenize(p)) >= SPAN_SIZE * 2]
    while len(pairs) < n and pool:
        a = rng.choice(pool)
        b = rng.choice(pool)
        wa, wb = tokenize(a), tokenize(b)
        spans_a = chunk_spans(wa, SPAN_SIZE)
        spans_b = chunk_spans(wb, SPAN_SIZE)
        if len(spans_a) < 2 or len(spans_b) < 2:
            continue
        idx = rng.randrange(min(len(spans_a), len(spans_b)))
        start = idx * SPAN_SIZE
        new_words = wa[:start] + spans_b[idx] + wa[start + len(spans_a[idx]) :]
        pairs.append(
            {
                "original": a,
                "edited": " ".join(new_words),
                "gold_span": idx,
                "donor": b,
            }
        )
    return pairs


def run() -> dict:
    paragraphs = load_paragraphs()
    rng = random.Random(42)
    shuffled = list(paragraphs)
    rng.shuffle(shuffled)
    split = max(1, int(0.15 * len(shuffled)))
    test = shuffled[:split]
    train = shuffled[split:]

    print("training token PRCF on full corpus...", flush=True)
    model = train_prcf(
        paragraphs,
        dim=32,
        n_codebooks=2,
        codebook_size=256,
        seed=0,
    )

    print("training BPE...", flush=True)
    bpe = BPEEncoder.train(train, vocab_size=500)
    print(f"bpe vocab={bpe.vocab_size}", flush=True)

    train_span = [encode_span_lm_seq(model, t) for t in train]
    test_span = [encode_span_lm_seq(model, t) for t in test]
    train_bpe = [bpe.encode_text(t) for t in train]
    test_bpe = [bpe.encode_text(t) for t in test]

    print("fitting trigram LMs...", flush=True)
    span_lm = TrigramLM(vocab_size=3 + SPAN_LM_VOCAB, k=0.1)
    span_lm.fit(train_span)
    bpe_lm = TrigramLM(vocab_size=bpe.vocab_size, k=0.1)
    bpe_lm.fit(train_bpe)

    span_bpc = eval_lm_bpc(span_lm, test_span, test)
    bpe_bpc = eval_lm_bpc(bpe_lm, test_bpe, test)
    span_spc = symbols_per_char(test_span, test)
    bpe_spc = symbols_per_char(test_bpe, test)

    chrf_mean = sum(chrf(reconstruct_via_token_codes(model, t), t) for t in test) / len(test)

    edit_pairs = make_phrase_edit_pairs(test, n=min(300, len(test) * 2), seed=9)
    loc_hits = sum(
        1
        for p in edit_pairs
        if (pred := localize_span(model, p["original"], p["edited"])) is not None
        and abs(pred - p["gold_span"]) <= 1
    )
    loc_acc = loc_hits / len(edit_pairs) if edit_pairs else 0.0

    swap_n = min(100, len(edit_pairs))
    swap_ok = sum(
        1
        for p in edit_pairs[:swap_n]
        if phrase_swap(model, p["original"], p["donor"], p["gold_span"])
    )
    swap_acc = swap_ok / swap_n if swap_n else 0.0

    bpc_ratio = span_bpc / bpe_bpc if bpe_bpc > 0 else float("inf")
    compression_ratio = span_spc / bpe_spc if bpe_spc > 0 else float("inf")

    match_modeling = bpc_ratio <= GATE_BPC_RATIO and chrf_mean >= GATE_CHRF
    edit_compress_win = (
        compression_ratio <= GATE_COMPRESSION
        and chrf_mean >= GATE_CHRF
        and loc_acc >= GATE_PHRASE_LOCALIZE
    )
    passed = match_modeling or edit_compress_win

    results = {
        "passed": passed,
        "pass_via": (
            "modeling_match"
            if match_modeling
            else ("edit_compress_win" if edit_compress_win else "none")
        ),
        "gates": {
            "bpc_ratio_max": GATE_BPC_RATIO,
            "chrf_min": GATE_CHRF,
            "phrase_localize_min": GATE_PHRASE_LOCALIZE,
            "compression_ratio_max": GATE_COMPRESSION,
        },
        "span": {
            "bpc": span_bpc,
            "symbols_per_char": span_spc,
            "lm_vocab_size": 3 + SPAN_LM_VOCAB,
            "span_size": SPAN_SIZE,
            "chrf_reconstruct": chrf_mean,
            "phrase_localize_within1": loc_acc,
            "phrase_swap_acc": swap_acc,
            "n_edit_pairs": len(edit_pairs),
            "note": "LM atoms = hashed groups of token-RQ codes; decode uses token RQ",
        },
        "bpe": {
            "bpc": bpe_bpc,
            "symbols_per_char": bpe_spc,
            "vocab_size": bpe.vocab_size,
        },
        "comparisons": {
            "bpc_ratio_span_over_bpe": bpc_ratio,
            "compression_ratio_span_over_bpe": compression_ratio,
        },
        "n_train": len(train),
        "n_test": len(test),
    }

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    md = _md(results)
    (OUT_DIR / "results.md").write_text(md)
    print(md, flush=True)
    return results


def _md(r: dict) -> str:
    status = "PASS" if r["passed"] else "FAIL"
    return "\n".join(
        [
            f"# Phase C results — **{status}**",
            "",
            f"- Pass via: `{r['pass_via']}`",
            f"- Train/test paragraphs: {r['n_train']}/{r['n_test']}",
            f"- Design: {r['span']['note']}",
            "",
            "## Modeling (equal-compute trigram LM)",
            "",
            "| substrate | BPC ↓ | symbols/char ↓ | vocab |",
            "|---|---|---|---|",
            f"| span-grouped token RQ (size={r['span']['span_size']}) | {r['span']['bpc']:.4f} | "
            f"{r['span']['symbols_per_char']:.4f} | {r['span']['lm_vocab_size']} |",
            f"| BPE | {r['bpe']['bpc']:.4f} | {r['bpe']['symbols_per_char']:.4f} | "
            f"{r['bpe']['vocab_size']} |",
            "",
            f"- BPC ratio (span/BPE): {r['comparisons']['bpc_ratio_span_over_bpe']:.4f} "
            f"(gate ≤ {r['gates']['bpc_ratio_max']}, requires chrF)",
            f"- Compression ratio (span/BPE symbols): "
            f"{r['comparisons']['compression_ratio_span_over_bpe']:.4f} "
            f"(gate ≤ {r['gates']['compression_ratio_max']})",
            "",
            "## Quality / editability",
            "",
            "| metric | value | gate |",
            "|---|---|---|",
            f"| reconstruct chrF | {r['span']['chrf_reconstruct']:.4f} | ≥ {r['gates']['chrf_min']} |",
            f"| phrase localize ±1 | {r['span']['phrase_localize_within1']:.4f} | "
            f"≥ {r['gates']['phrase_localize_min']} |",
            f"| phrase swap acc | {r['span']['phrase_swap_acc']:.4f} | (informational) |",
            "",
        ]
    )


if __name__ == "__main__":
    run()
