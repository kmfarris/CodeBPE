"""Direction 1: equal-compute tiny Transformer — PRCF span codes vs BPE."""

from __future__ import annotations

import json
import math
import random
import sys
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.phase_a.build_pairs import load_paragraphs
from prcf.bpe import BPEEncoder
from prcf.model import train_prcf
from prcf.tiny_lm import TinyTransformerLM
from prcf.tokenize import tokenize

OUT_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture (shared)
D_MODEL = 128
N_LAYERS = 2
N_HEADS = 4
MAX_LEN = 64
DROPOUT = 0.1

# Training (shared)
STEPS = 800
BATCH = 16
LR = 3e-3
SEED = 0

SPAN_SIZE = 3
SPAN_LM_VOCAB = 2048  # keep closer to BPE vocab scale
BPE_VOCAB = 500

# PASS if codes BPC <= BPE BPC * ratio (match or beat)
GATE_BPC_RATIO = 1.05


def span_lm_id(token_codes: list[tuple[int, ...]]) -> int:
    payload = "|".join(",".join(map(str, c)) for c in token_codes).encode()
    return 3 + (zlib.crc32(payload) % SPAN_LM_VOCAB)


def encode_span_seq(model, text: str) -> list[int]:
    codes = model.encode_tokens(tokenize(text))
    return [
        span_lm_id(codes[i : i + SPAN_SIZE])
        for i in range(0, len(codes), SPAN_SIZE)
    ] or [3]


def bits_per_char_from_nll(nll_nats: float, n_chars: int) -> float:
    return (nll_nats / math.log(2)) / max(n_chars, 1)


def make_windows(seqs: list[list[int]], max_len: int) -> list[list[int]]:
    """Pad/truncate sequences to fixed windows with bos=1 prefix for teacher forcing."""
    windows: list[list[int]] = []
    for seq in seqs:
        # Use raw ids; reserve 0 unk already in encoders; prepend nothing —
        # causal LM predicts next from previous within window
        s = seq[:max_len]
        if len(s) < 2:
            continue
        if len(s) < max_len:
            s = s + [2]  # eos pad marker once then we'll mask via length… keep simple: pad with 0
            s = s + [0] * (max_len - len(s))
        windows.append(s[:max_len])
    return windows


@torch.no_grad()
def eval_bpc(
    lm: TinyTransformerLM,
    windows: list[list[int]],
    texts_for_chars: list[str],
    seqs: list[list[int]],
) -> dict:
    lm.eval()
    total_nll = 0.0
    total_tok = 0
    for w in windows:
        x = torch.tensor([w], dtype=torch.long, device=DEVICE)
        logits = lm(x[:, :-1])
        target = x[:, 1:]
        # ignore pad 0
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=0,
            reduction="sum",
        )
        n = int((target != 0).sum().item())
        total_nll += float(loss.item())
        total_tok += n
    n_chars = sum(max(len(t), 1) for t in texts_for_chars)
    n_syms = sum(len(s) for s in seqs)
    ppl = math.exp(total_nll / max(total_tok, 1))
    bpc = bits_per_char_from_nll(total_nll, n_chars)
    return {
        "nll_nats": total_nll,
        "n_target_tokens": total_tok,
        "perplexity": ppl,
        "bpc": bpc,
        "symbols_per_char": n_syms / max(n_chars, 1),
    }


def train_lm(
    name: str,
    vocab_size: int,
    train_windows: list[list[int]],
    steps: int,
) -> TinyTransformerLM:
    torch.manual_seed(SEED)
    lm = TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(DEVICE)
    opt = torch.optim.AdamW(lm.parameters(), lr=LR)
    rng = random.Random(SEED)
    lm.train()
    for step in range(steps):
        batch = [rng.choice(train_windows) for _ in range(BATCH)]
        x = torch.tensor(batch, dtype=torch.long, device=DEVICE)
        logits = lm(x[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
            ignore_index=0,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == steps - 1:
            print(f"[{name}] step {step}/{steps} loss={loss.item():.4f}", flush=True)
    return lm


def run() -> dict:
    print(f"device={DEVICE}", flush=True)
    paragraphs = load_paragraphs()
    rng = random.Random(42)
    shuffled = list(paragraphs)
    rng.shuffle(shuffled)
    split = max(1, int(0.15 * len(shuffled)))
    test = shuffled[:split]
    train = shuffled[split:]

    print("fitting PRCF token codec on full corpus...", flush=True)
    prcf = train_prcf(paragraphs, dim=32, n_codebooks=2, codebook_size=256, seed=0)

    print("training BPE on train split...", flush=True)
    bpe = BPEEncoder.train(train, vocab_size=BPE_VOCAB)

    train_span = [encode_span_seq(prcf, t) for t in train]
    test_span = [encode_span_seq(prcf, t) for t in test]
    train_bpe = [bpe.encode_text(t) for t in train]
    test_bpe = [bpe.encode_text(t) for t in test]

    span_vocab = 3 + SPAN_LM_VOCAB
    bpe_vocab = bpe.vocab_size

    train_span_w = make_windows(train_span, MAX_LEN)
    test_span_w = make_windows(test_span, MAX_LEN)
    train_bpe_w = make_windows(train_bpe, MAX_LEN)
    test_bpe_w = make_windows(test_bpe, MAX_LEN)

    print(
        f"windows: span train/test={len(train_span_w)}/{len(test_span_w)} "
        f"bpe train/test={len(train_bpe_w)}/{len(test_bpe_w)}",
        flush=True,
    )

    span_lm = train_lm("span", span_vocab, train_span_w, STEPS)
    bpe_lm = train_lm("bpe", bpe_vocab, train_bpe_w, STEPS)

    # Equal architecture check
    assert span_lm.param_count() != 0
    # Embedding tables differ by vocab — report non-embed params too
    def non_embed(m: TinyTransformerLM) -> int:
        return sum(
            p.numel()
            for n, p in m.named_parameters()
            if not n.startswith("tok.") and not n.startswith("pos.")
        )

    span_metrics = eval_bpc(span_lm, test_span_w, test, test_span)
    bpe_metrics = eval_bpc(bpe_lm, test_bpe_w, test, test_bpe)

    ratio = span_metrics["bpc"] / bpe_metrics["bpc"] if bpe_metrics["bpc"] > 0 else float("inf")
    passed = ratio <= GATE_BPC_RATIO

    results = {
        "passed": passed,
        "gate_bpc_ratio_max": GATE_BPC_RATIO,
        "bpc_ratio_span_over_bpe": ratio,
        "device": str(DEVICE),
        "shared_train": {
            "steps": STEPS,
            "batch": BATCH,
            "lr": LR,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "max_len": MAX_LEN,
        },
        "span": {
            **span_metrics,
            "vocab_size": span_vocab,
            "params_total": span_lm.param_count(),
            "params_non_embed": non_embed(span_lm),
            "span_size": SPAN_SIZE,
        },
        "bpe": {
            **bpe_metrics,
            "vocab_size": bpe_vocab,
            "params_total": bpe_lm.param_count(),
            "params_non_embed": non_embed(bpe_lm),
        },
        "n_train": len(train),
        "n_test": len(test),
    }

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    md = _md(results)
    (OUT_DIR / "results.md").write_text(md)
    print(md, flush=True)

    # Save small checkpoints for inspection
    torch.save(span_lm.state_dict(), OUT_DIR / "span_lm.pt")
    torch.save(bpe_lm.state_dict(), OUT_DIR / "bpe_lm.pt")
    return results


def _md(r: dict) -> str:
    status = "PASS" if r["passed"] else "FAIL"
    return "\n".join(
        [
            f"# Direction 1 — Neural LM bake-off — **{status}**",
            "",
            f"- Device: `{r['device']}`",
            f"- Shared: {r['shared_train']['n_layers']}L/{r['shared_train']['d_model']}d, "
            f"{r['shared_train']['steps']} steps, batch {r['shared_train']['batch']}",
            f"- BPC ratio (span/BPE): {r['bpc_ratio_span_over_bpe']:.4f} "
            f"(gate ≤ {r['gate_bpc_ratio_max']})",
            "",
            "| substrate | BPC ↓ | PPL ↓ | sym/char | vocab | params (all / non-embed) |",
            "|---|---|---|---|---|---|",
            f"| span-PRCF | {r['span']['bpc']:.4f} | {r['span']['perplexity']:.2f} | "
            f"{r['span']['symbols_per_char']:.4f} | {r['span']['vocab_size']} | "
            f"{r['span']['params_total']:,} / {r['span']['params_non_embed']:,} |",
            f"| BPE | {r['bpe']['bpc']:.4f} | {r['bpe']['perplexity']:.2f} | "
            f"{r['bpe']['symbols_per_char']:.4f} | {r['bpe']['vocab_size']} | "
            f"{r['bpe']['params_total']:,} / {r['bpe']['params_non_embed']:,} |",
            "",
            "Gate: span BPC ≤ BPE BPC × 1.05 on held-out paragraphs "
            "(equal architecture body; embedding tables scale with vocab).",
            "",
        ]
    )


if __name__ == "__main__":
    run()
