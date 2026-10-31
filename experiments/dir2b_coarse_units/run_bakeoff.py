"""Dir2b: coarse CodeBPE (target mean len ≥ 2) + moderate GPU bake-off vs text BPE."""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.phase_a.build_pairs import load_paragraphs
from prcf.bpe import BPEEncoder
from prcf.chrf import chrf
from prcf.code_bpe import CodeBPE
from prcf.model import train_prcf
from prcf.tiny_lm import TinyTransformerLM
from prcf.tokenize import tokenize

OUT_DIR = Path(__file__).resolve().parent

REQUIRE_GPU = os.environ.get("PRCF_REQUIRE_GPU", "1") == "1"
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.45, device=0)
    torch.set_float32_matmul_precision("high")
else:
    if REQUIRE_GPU:
        raise SystemExit(
            "GPU required but torch.cuda.is_available() is False. "
            "Run outside sandbox / check nvidia-smi."
        )
    DEVICE = torch.device("cpu")

D_MODEL = 160
N_LAYERS = 3
N_HEADS = 4
MAX_LEN = 64
DROPOUT = 0.1
STEPS = 1000
BATCH = 8
LR = 3e-3
SEED = 0
LOG_EVERY = 50

TARGET_MEAN_LEN = 2.0
MAX_MERGES = 20000
TEXT_BPE_VOCAB = 500

GATE_MEAN_LEN = 2.0
GATE_SYMBOLS_PER_WORD = 0.65
GATE_CHRF = 0.90
GATE_BPC_RATIO = 1.05


def make_windows(seqs: list[list[int]], max_len: int) -> list[list[int]]:
    windows: list[list[int]] = []
    for seq in seqs:
        s = seq[:max_len]
        if len(s) < 2:
            continue
        if len(s) < max_len:
            s = s + [2] + [0] * (max_len - len(s) - 1)
            s = s[:max_len]
        windows.append(s)
    return windows


@torch.no_grad()
def eval_bpc(
    lm: TinyTransformerLM,
    windows: list[list[int]],
    texts: list[str],
    seqs: list[list[int]],
) -> dict:
    lm.eval()
    total_nll = 0.0
    total_tok = 0
    for w in windows:
        x = torch.tensor([w], dtype=torch.long, device=DEVICE)
        logits = lm(x[:, :-1])
        target = x[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=0,
            reduction="sum",
        )
        n = int((target != 0).sum().item())
        total_nll += float(loss.item())
        total_tok += n
    n_chars = sum(max(len(t), 1) for t in texts)
    n_syms = sum(len(s) for s in seqs)
    return {
        "nll_nats": total_nll,
        "n_target_tokens": total_tok,
        "perplexity": math.exp(total_nll / max(total_tok, 1)),
        "bpc": (total_nll / math.log(2)) / max(n_chars, 1),
        "symbols_per_char": n_syms / max(n_chars, 1),
    }


def non_embed_params(m: TinyTransformerLM) -> int:
    return sum(
        p.numel()
        for n, p in m.named_parameters()
        if not n.startswith("tok.") and not n.startswith("pos.")
    )


def train_lm(name: str, vocab_size: int, train_windows: list[list[int]]) -> TinyTransformerLM:
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
    for step in range(STEPS):
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
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            mem = ""
            if DEVICE.type == "cuda":
                mem = f" alloc={torch.cuda.memory_allocated()/1e6:.0f}MB"
            print(
                f"[{name}] step {step}/{STEPS} loss={loss.item():.4f}{mem}",
                flush=True,
            )
    return lm


def write_results(results: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    md = _md(results)
    (OUT_DIR / "results.md").write_text(md)
    print(md, flush=True)


def run() -> dict:
    print(f"device={DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(
            f"gpu={torch.cuda.get_device_name(0)} "
            f"mem_fraction=0.45 batch={BATCH} steps={STEPS}",
            flush=True,
        )
    torch.set_num_threads(min(4, os.cpu_count() or 4))

    paragraphs = load_paragraphs()
    rng = random.Random(42)
    shuffled = list(paragraphs)
    rng.shuffle(shuffled)
    split = max(1, int(0.15 * len(shuffled)))
    test = shuffled[:split]
    train = shuffled[split:]

    print("fitting PRCF codec...", flush=True)
    prcf = train_prcf(paragraphs, dim=32, n_codebooks=2, codebook_size=256, seed=0)

    print(
        f"training coarse CodeBPE target_mean_len={TARGET_MEAN_LEN} "
        f"max_merges={MAX_MERGES}...",
        flush=True,
    )
    code_bpe = CodeBPE.train(
        prcf,
        paragraphs,
        target_mean_len=TARGET_MEAN_LEN,
        max_merges=MAX_MERGES,
        min_freq=2,
    )
    stats = code_bpe.merge_stats()
    print(f"code_bpe stats={stats}", flush=True)

    chrf_train = sum(chrf(code_bpe.reconstruct(t), t) for t in train[:200]) / min(
        200, len(train)
    )
    chrf_test = sum(chrf(code_bpe.reconstruct(t), t) for t in test) / len(test)
    print(f"code_bpe chrF train/test={chrf_train:.4f}/{chrf_test:.4f}", flush=True)

    train_code = [code_bpe.encode_text(t) for t in train]
    n_words_train = sum(len(tokenize(t)) for t in train)
    n_syms_train = sum(len(s) for s in train_code)
    symbols_per_word = n_syms_train / max(n_words_train, 1)

    unit_gates = {
        "mean_unit_len_ok": stats["mean_unit_len_codes"] >= GATE_MEAN_LEN,
        "symbols_per_word_ok": symbols_per_word <= GATE_SYMBOLS_PER_WORD,
        "chrf_ok": chrf_test >= GATE_CHRF,
    }
    units_passed = all(unit_gates.values())

    base_results = {
        "device": str(DEVICE),
        "unit_gates": {
            **unit_gates,
            "gates": {
                "mean_unit_len_min": GATE_MEAN_LEN,
                "symbols_per_word_max": GATE_SYMBOLS_PER_WORD,
                "chrf_min": GATE_CHRF,
            },
            "measured": {
                "mean_unit_len_codes": stats["mean_unit_len_codes"],
                "symbols_per_word": symbols_per_word,
                "chrf_test": chrf_test,
                "chrf_train": chrf_train,
            },
        },
        "units": {
            "type": "code_bpe_coarse",
            **stats,
            "symbols_per_word_train_encode": symbols_per_word,
            "chrf_train": chrf_train,
            "chrf_test": chrf_test,
        },
        "n_train": len(train),
        "n_test": len(test),
        "lm_ran": False,
    }

    if not units_passed:
        results = {
            **base_results,
            "passed": False,
            "pass_via": "none",
            "stopped_reason": "unit_gates_failed",
        }
        write_results(results)
        return results

    print("unit gates PASS — training text BPE + LMs...", flush=True)
    text_bpe = BPEEncoder.train(train, vocab_size=TEXT_BPE_VOCAB)

    test_code = [code_bpe.encode_text(t) for t in test]
    train_tbpe = [text_bpe.encode_text(t) for t in train]
    test_tbpe = [text_bpe.encode_text(t) for t in test]

    print(
        f"avg LM symbols/doc: code={sum(len(s) for s in train_code)/len(train_code):.2f} "
        f"words≈{n_words_train/len(train):.2f} "
        f"text_bpe={sum(len(s) for s in train_tbpe)/len(train_tbpe):.2f}",
        flush=True,
    )

    train_code_w = make_windows(train_code, MAX_LEN)
    test_code_w = make_windows(test_code, MAX_LEN)
    train_tbpe_w = make_windows(train_tbpe, MAX_LEN)
    test_tbpe_w = make_windows(test_tbpe, MAX_LEN)

    code_lm = train_lm("code_bpe", code_bpe.vocab_size, train_code_w)
    bpe_lm = train_lm("text_bpe", text_bpe.vocab_size, train_tbpe_w)

    code_m = eval_bpc(code_lm, test_code_w, test, test_code)
    bpe_m = eval_bpc(bpe_lm, test_tbpe_w, test, test_tbpe)
    ratio = code_m["bpc"] / bpe_m["bpc"] if bpe_m["bpc"] > 0 else float("inf")
    lm_passed = ratio <= GATE_BPC_RATIO

    results = {
        **base_results,
        "passed": lm_passed,
        "pass_via": "lm_and_units" if lm_passed else "units_only_lm_fail",
        "lm_ran": True,
        "gate_bpc_ratio_max": GATE_BPC_RATIO,
        "bpc_ratio_code_over_textbpe": ratio,
        "shared_train": {
            "steps": STEPS,
            "batch": BATCH,
            "lr": LR,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "max_len": MAX_LEN,
        },
        "code_bpe": {
            **code_m,
            "vocab_size": code_bpe.vocab_size,
            "params_total": code_lm.param_count(),
            "params_non_embed": non_embed_params(code_lm),
        },
        "text_bpe": {
            **bpe_m,
            "vocab_size": text_bpe.vocab_size,
            "params_total": bpe_lm.param_count(),
            "params_non_embed": non_embed_params(bpe_lm),
        },
    }
    write_results(results)
    torch.save(code_lm.state_dict(), OUT_DIR / "code_bpe_lm.pt")
    torch.save(bpe_lm.state_dict(), OUT_DIR / "text_bpe_lm.pt")
    return results


def _md(r: dict) -> str:
    status = "PASS" if r["passed"] else "FAIL"
    u = r["units"]
    ug = r["unit_gates"]
    m = ug["measured"]
    lines = [
        f"# Dir2b coarse CodeBPE — **{status}**",
        "",
        f"- Device: `{r['device']}`",
        f"- Stopped: `{r.get('stopped_reason', 'completed')}`",
        f"- Pass via: `{r.get('pass_via', 'n/a')}`",
        "",
        "## Unit gates",
        "",
        "| check | measured | gate | ok |",
        "|---|---|---|---|",
        f"| mean unit len | {m['mean_unit_len_codes']:.4f} | "
        f"≥ {ug['gates']['mean_unit_len_min']} | {ug['mean_unit_len_ok']} |",
        f"| symbols/word | {m['symbols_per_word']:.4f} | "
        f"≤ {ug['gates']['symbols_per_word_max']} | {ug['symbols_per_word_ok']} |",
        f"| chrF test | {m['chrf_test']:.4f} | ≥ {ug['gates']['chrf_min']} | {ug['chrf_ok']} |",
        "",
        f"- Merges: {u['n_merges']} (freq≥2: {u['n_merges_freq2']}, freq=1: {u['n_merges_freq1']})",
        f"- Vocab: {u['vocab_size']}, max unit len: {u['max_unit_len_codes']}",
        "",
    ]
    if r.get("lm_ran"):
        lines.extend(
            [
                "## LM bake-off",
                "",
                f"- Model: {r['shared_train']['n_layers']}L/"
                f"{r['shared_train']['d_model']}d, {r['shared_train']['steps']} steps",
                f"- BPC ratio (code/textBPE): {r['bpc_ratio_code_over_textbpe']:.4f} "
                f"(gate ≤ {r['gate_bpc_ratio_max']})",
                "",
                "| substrate | BPC ↓ | PPL ↓ | sym/char | vocab | params (all / non-embed) |",
                "|---|---|---|---|---|---|",
                f"| CodeBPE | {r['code_bpe']['bpc']:.4f} | {r['code_bpe']['perplexity']:.2f} | "
                f"{r['code_bpe']['symbols_per_char']:.4f} | {r['code_bpe']['vocab_size']} | "
                f"{r['code_bpe']['params_total']:,} / {r['code_bpe']['params_non_embed']:,} |",
                f"| text BPE | {r['text_bpe']['bpc']:.4f} | {r['text_bpe']['perplexity']:.2f} | "
                f"{r['text_bpe']['symbols_per_char']:.4f} | {r['text_bpe']['vocab_size']} | "
                f"{r['text_bpe']['params_total']:,} / {r['text_bpe']['params_non_embed']:,} |",
                "",
            ]
        )
    else:
        lines.append("LM bake-off skipped (unit gates failed).\n")
    return "\n".join(lines)


if __name__ == "__main__":
    run()
