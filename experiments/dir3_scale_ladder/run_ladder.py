"""Dir3: time-gated scale ladder on WikiText-2 slices (tokenization benefits only)."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.dir3_scale_ladder.prepare_data import load_slice, prepare_slices
from experiments.phase_a.build_pairs import substitute_word
from prcf.bpe import BPEEncoder
from prcf.chrf import chrf
from prcf.code_bpe import CodeBPE
from prcf.model import train_prcf
from prcf.tiny_lm import TinyTransformerLM
from prcf.tokenize import tokenize

OUT_DIR = Path(__file__).resolve().parent

# LM defaults (budget 30); budget 60 stretches steps / layers
D_MODEL = 160
N_LAYERS = 3
N_HEADS = 4
MAX_LEN = 64
DROPOUT = 0.1
LR = 3e-3
SEED = 0
LOG_EVERY = 100

TEXT_BPE_VOCAB = 2000
TARGET_MEAN_LEN = 2.0
MAX_MERGES = 20000
MIN_FREQ = 2

GATE_CHRF = 0.85
GATE_BPC_RATIO = 1.05
GATE_LOCALIZE = 0.80


def _require_device():
    import torch

    require_gpu = os.environ.get("PRCF_REQUIRE_GPU", "1") == "1"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(0.45, device=0)
        torch.set_float32_matmul_precision("high")
    else:
        if require_gpu:
            raise SystemExit(
                "GPU required but torch.cuda.is_available() is False. "
                "Run outside sandbox / check nvidia-smi."
            )
        device = torch.device("cpu")
    return torch, device


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


def non_embed_params(m) -> int:
    return sum(
        p.numel()
        for n, p in m.named_parameters()
        if not n.startswith("tok.") and not n.startswith("pos.")
    )


def train_lm(torch, device, name, vocab_size, train_windows, steps, batch, n_layers, d_model):
    torch.manual_seed(SEED)
    lm = TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=N_HEADS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(device)
    opt = torch.optim.AdamW(lm.parameters(), lr=LR)
    rng = random.Random(SEED)
    import torch.nn.functional as F

    lm.train()
    for step in range(steps):
        batch_seqs = [rng.choice(train_windows) for _ in range(batch)]
        x = torch.tensor(batch_seqs, dtype=torch.long, device=device)
        logits = lm(x[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
            ignore_index=0,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % LOG_EVERY == 0 or step == steps - 1:
            mem = ""
            if device.type == "cuda":
                mem = f" alloc={torch.cuda.memory_allocated()/1e6:.0f}MB"
            print(f"[{name}] step {step}/{steps} loss={loss.item():.4f}{mem}", flush=True)
    return lm


def _eval_bpc_impl(torch, device, lm, windows, texts, seqs):
    import torch.nn.functional as F

    lm.eval()
    total_nll = 0.0
    total_tok = 0
    with torch.no_grad():
        for w in windows:
            x = torch.tensor([w], dtype=torch.long, device=device)
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


def localize_by_codes(code_bpe: CodeBPE, text_a: str, text_b: str) -> int | None:
    """Word index of first base-code mismatch after CodeBPE encode→expand."""
    ca = code_bpe.expand_ids(code_bpe.encode_text(text_a))
    cb = code_bpe.expand_ids(code_bpe.encode_text(text_b))
    n = min(len(ca), len(cb))
    for i in range(n):
        if ca[i] != cb[i]:
            return i
    if len(ca) != len(cb):
        return n
    return None


def edit_locality(code_bpe: CodeBPE, texts: list[str], n_pairs: int = 200) -> dict:
    rng = random.Random(0)
    exact = 0
    within2 = 0
    total = 0
    for _ in range(n_pairs * 3):
        if total >= n_pairs:
            break
        t = rng.choice(texts)
        sub = substitute_word(t, rng)
        if sub is None:
            continue
        edited, gold = sub
        # rebuild gold index on tokenized forms used by codec
        words = tokenize(t)
        if gold >= len(words):
            continue
        pred = localize_by_codes(code_bpe, t, edited)
        if pred is None:
            continue
        total += 1
        if pred == gold:
            exact += 1
        if abs(pred - gold) <= 2:
            within2 += 1
    return {
        "n": total,
        "exact_acc": exact / max(total, 1),
        "within2_acc": within2 / max(total, 1),
    }


def run(budget: int, train_words: int, valid_words: int) -> dict:
    t_start = time.perf_counter()
    torch, device = _require_device()
    print(f"device={device} budget={budget}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)} mem_fraction=0.45", flush=True)
    torch.set_num_threads(min(4, os.cpu_count() or 4))

    # --- data ---
    t0 = time.perf_counter()
    train_path, valid_path, data_meta = prepare_slices(train_words, valid_words)
    train = load_slice(train_path)
    valid = load_slice(valid_path)
    data_wall = time.perf_counter() - t0
    print(f"data ready in {data_wall:.1f}s: {data_meta}", flush=True)

    # --- PRCF ---
    t0 = time.perf_counter()
    print("fitting PRCF codec on train slice...", flush=True)
    prcf = train_prcf(train, dim=32, n_codebooks=2, codebook_size=256, seed=0)
    rq_wall = time.perf_counter() - t0
    print(f"PRCF done in {rq_wall:.1f}s", flush=True)

    # --- CodeBPE (freq-stable) ---
    t0 = time.perf_counter()
    print(
        f"training CodeBPE allow_freq1=False target_mean_len={TARGET_MEAN_LEN} "
        f"max_merges={MAX_MERGES}...",
        flush=True,
    )
    code_bpe = CodeBPE.train(
        prcf,
        train,
        target_mean_len=TARGET_MEAN_LEN,
        max_merges=MAX_MERGES,
        min_freq=MIN_FREQ,
        allow_freq1=False,
    )
    cb_stats = code_bpe.merge_stats()
    print(f"code_bpe stats={cb_stats}", flush=True)

    # text BPE for packing comparison
    text_bpe = BPEEncoder.train(train, vocab_size=TEXT_BPE_VOCAB)

    # metrics
    chrf_n = min(200, len(valid))
    chrf_valid = sum(chrf(code_bpe.reconstruct(t), t) for t in valid[:chrf_n]) / max(chrf_n, 1)

    train_code = [code_bpe.encode_text(t) for t in train]
    train_tbpe = [text_bpe.encode_text(t) for t in train]
    n_chars_train = sum(max(len(t), 1) for t in train)
    n_words_train = sum(len(tokenize(t)) for t in train)
    code_spc = sum(len(s) for s in train_code) / n_chars_train
    tbpe_spc = sum(len(s) for s in train_tbpe) / n_chars_train
    code_spw = sum(len(s) for s in train_code) / max(n_words_train, 1)

    packing_win = code_spc <= tbpe_spc
    chrf_ok = chrf_valid >= GATE_CHRF
    # Stage-15 pass: reconstructable AND packing win, OR publish honest coarseness fail
    stage15_pass = chrf_ok and packing_win
    stage15_note = "packing_and_chrf" if stage15_pass else (
        "chrf_fail" if not chrf_ok else "no_packing_win"
    )

    stage15 = {
        "passed": stage15_pass,
        "pass_via": stage15_note,
        "gates": {"chrf_min": GATE_CHRF, "symbols_per_char_leq_text_bpe": True},
        "measured": {
            "chrf_valid": chrf_valid,
            "code_symbols_per_char": code_spc,
            "text_bpe_symbols_per_char": tbpe_spc,
            "code_symbols_per_word": code_spw,
            "mean_unit_len_codes": cb_stats["mean_unit_len_codes"],
            "n_merges": cb_stats["n_merges"],
            "n_merges_freq1": cb_stats["n_merges_freq1"],
            "n_merges_freq2": cb_stats["n_merges_freq2"],
            "merge_freq_hist": cb_stats["merge_freq_hist"],
            "code_vocab": cb_stats["vocab_size"],
            "text_bpe_vocab": text_bpe.vocab_size,
        },
        "walls_s": {
            "data": data_wall,
            "prcf": rq_wall,
            "code_bpe": cb_stats["train_wall_s"],
            "stage15_total": time.perf_counter() - t_start,
        },
    }
    results = {
        "budget": budget,
        "device": str(device),
        "data": data_meta,
        "stage15": stage15,
        "units": cb_stats,
    }
    _write(results, budget=15)
    print(f"=== stage15 {'PASS' if stage15_pass else 'FAIL'} ({stage15_note}) ===", flush=True)

    if budget <= 15:
        results["passed"] = stage15_pass
        results["stopped_reason"] = "budget_15"
        _write(results, budget=15)
        return results

    # --- stage 30: LM bake-off ---
    steps = 2000
    batch = 8
    n_layers = N_LAYERS
    d_model = D_MODEL
    if budget >= 60:
        steps = 5000
        n_layers = 4
        d_model = 192

    print(f"LM bake-off steps={steps} layers={n_layers} d={d_model} batch={batch}", flush=True)
    valid_code = [code_bpe.encode_text(t) for t in valid]
    valid_tbpe = [text_bpe.encode_text(t) for t in valid]
    train_code_w = make_windows(train_code, MAX_LEN)
    valid_code_w = make_windows(valid_code, MAX_LEN)
    train_tbpe_w = make_windows(train_tbpe, MAX_LEN)
    valid_tbpe_w = make_windows(valid_tbpe, MAX_LEN)

    t0 = time.perf_counter()
    code_lm = train_lm(
        torch, device, "code_bpe", code_bpe.vocab_size, train_code_w, steps, batch, n_layers, d_model
    )
    bpe_lm = train_lm(
        torch, device, "text_bpe", text_bpe.vocab_size, train_tbpe_w, steps, batch, n_layers, d_model
    )
    lm_wall = time.perf_counter() - t0

    code_m = _eval_bpc_impl(torch, device, code_lm, valid_code_w, valid, valid_code)
    bpe_m = _eval_bpc_impl(torch, device, bpe_lm, valid_tbpe_w, valid, valid_tbpe)
    ratio = code_m["bpc"] / bpe_m["bpc"] if bpe_m["bpc"] > 0 else float("inf")
    lm_pass = ratio <= GATE_BPC_RATIO

    stage30 = {
        "passed": lm_pass and stage15_pass,
        "lm_bpc_pass": lm_pass,
        "bpc_ratio_code_over_textbpe": ratio,
        "gate_bpc_ratio_max": GATE_BPC_RATIO,
        "shared_train": {
            "steps": steps,
            "batch": batch,
            "lr": LR,
            "d_model": d_model,
            "n_layers": n_layers,
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
        "lm_wall_s": lm_wall,
        "elapsed_s": time.perf_counter() - t_start,
    }
    results["stage30"] = stage30
    results["passed"] = stage30["passed"]
    _write(results, budget=30)
    print(
        f"=== stage30 {'PASS' if stage30['passed'] else 'FAIL'} "
        f"bpc_ratio={ratio:.4f} ===",
        flush=True,
    )

    if budget <= 30:
        results["stopped_reason"] = "budget_30"
        _write(results, budget=30)
        return results

    # --- stage 60: edit locality ---
    loc = edit_locality(code_bpe, valid, n_pairs=200)
    loc_pass = loc["within2_acc"] >= GATE_LOCALIZE
    stage60 = {
        "passed": loc_pass and stage30["passed"],
        "localize": loc,
        "gate_within2": GATE_LOCALIZE,
        "elapsed_s": time.perf_counter() - t_start,
    }
    results["stage60"] = stage60
    results["passed"] = stage60["passed"]
    results["stopped_reason"] = "budget_60"
    _write(results, budget=60)
    print(
        f"=== stage60 {'PASS' if stage60['passed'] else 'FAIL'} "
        f"localize±2={loc['within2_acc']:.3f} ===",
        flush=True,
    )
    return results


def _write(results: dict, budget: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    md = _md(results)
    # always refresh latest + budget snapshot
    (OUT_DIR / "results.md").write_text(md)
    (OUT_DIR / f"results_{budget}.md").write_text(md)
    print(md, flush=True)


def _md(r: dict) -> str:
    s15 = r["stage15"]
    m = s15["measured"]
    status = "PASS" if r.get("passed", s15["passed"]) else "FAIL"
    lines = [
        f"# Dir3 scale ladder — **{status}**",
        "",
        f"- Budget: `{r.get('budget')}`",
        f"- Device: `{r.get('device')}`",
        f"- Stopped: `{r.get('stopped_reason', 'in_progress')}`",
        f"- Data: train_words={r['data']['train_words']} valid_words={r['data']['valid_words']}",
        "",
        "## Stage 15 — tokenizer (freq-stable CodeBPE)",
        "",
        f"- Result: **{'PASS' if s15['passed'] else 'FAIL'}** (`{s15['pass_via']}`)",
        f"- chrF valid: {m['chrf_valid']:.4f} (gate ≥ {s15['gates']['chrf_min']})",
        f"- symbols/char: CodeBPE {m['code_symbols_per_char']:.4f} vs text BPE {m['text_bpe_symbols_per_char']:.4f}",
        f"- mean unit len: {m['mean_unit_len_codes']:.4f}",
        f"- merges: {m['n_merges']} (freq≥2: {m['n_merges_freq2']}, freq=1: {m['n_merges_freq1']})",
        f"- merge hist: {m['merge_freq_hist']}",
        f"- walls: {s15['walls_s']}",
        "",
    ]
    if "stage30" in r:
        s30 = r["stage30"]
        lines.extend(
            [
                "## Stage 30 — LM bake-off",
                "",
                f"- Result: **{'PASS' if s30['passed'] else 'FAIL'}** "
                f"(BPC ratio {s30['bpc_ratio_code_over_textbpe']:.4f}, "
                f"gate ≤ {s30['gate_bpc_ratio_max']})",
                f"- Model: {s30['shared_train']['n_layers']}L/"
                f"{s30['shared_train']['d_model']}d, {s30['shared_train']['steps']} steps",
                f"- LM wall: {s30['lm_wall_s']:.1f}s, elapsed: {s30['elapsed_s']:.1f}s",
                "",
                "| substrate | BPC ↓ | PPL ↓ | sym/char | vocab | params (all / non-embed) |",
                "|---|---|---|---|---|---|",
                f"| CodeBPE | {s30['code_bpe']['bpc']:.4f} | {s30['code_bpe']['perplexity']:.2f} | "
                f"{s30['code_bpe']['symbols_per_char']:.4f} | {s30['code_bpe']['vocab_size']} | "
                f"{s30['code_bpe']['params_total']:,} / {s30['code_bpe']['params_non_embed']:,} |",
                f"| text BPE | {s30['text_bpe']['bpc']:.4f} | {s30['text_bpe']['perplexity']:.2f} | "
                f"{s30['text_bpe']['symbols_per_char']:.4f} | {s30['text_bpe']['vocab_size']} | "
                f"{s30['text_bpe']['params_total']:,} / {s30['text_bpe']['params_non_embed']:,} |",
                "",
            ]
        )
    if "stage60" in r:
        s60 = r["stage60"]
        loc = s60["localize"]
        lines.extend(
            [
                "## Stage 60 — edit locality",
                "",
                f"- Result: **{'PASS' if s60['passed'] else 'FAIL'}**",
                f"- localize exact / ±2: {loc['exact_acc']:.3f} / {loc['within2_acc']:.3f} "
                f"(n={loc['n']}, gate ±2 ≥ {s60['gate_within2']})",
                f"- elapsed: {s60['elapsed_s']:.1f}s",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, choices=[15, 30, 60], default=30)
    ap.add_argument("--train-words", type=int, default=100_000)
    ap.add_argument("--valid-words", type=int, default=20_000)
    args = ap.parse_args()
    run(budget=args.budget, train_words=args.train_words, valid_words=args.valid_words)


if __name__ == "__main__":
    main()
