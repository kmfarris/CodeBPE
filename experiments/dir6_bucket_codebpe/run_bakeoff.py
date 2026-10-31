"""Dir6: BucketCodeBPE (BB merges only) + GPU LM bake-off vs text BPE."""

from __future__ import annotations

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

from experiments.dir3_scale_ladder.prepare_data import load_slice
from prcf.bpe import BPEEncoder
from prcf.bucket_code_bpe import BucketCodeBPE
from prcf.bucket_slot import BucketSlotCodec
from prcf.chrf import chrf
from prcf.tiny_lm import TinyTransformerLM
from prcf.tokenize import tokenize

OUT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TRAIN = DATA / "wikitext2_train_slice.txt"
VALID = DATA / "wikitext2_valid_slice.txt"

D_MODEL = 160
N_LAYERS = 3
N_HEADS = 4
MAX_LEN = 64
DROPOUT = 0.1
STEPS = 2000
BATCH = 8
LR = 3e-3
SEED = 0
LOG_EVERY = 100

TEXT_BPE_VOCAB = 2000
TARGET_MEAN_LEN = 1.5
MAX_MERGES = 20000
MIN_COUNT = 2

GATE_CHRF = 0.90
GATE_OOV_EXACT = 0.95
GATE_BPC_RATIO = 1.05


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


def oov_exact_acc(reconstructor, texts: list[str], in_vocab: set[str]) -> dict:
    n_oov = 0
    n_ok = 0
    for t in texts:
        words = t.split()
        hyp_words = reconstructor(t).split()
        n = min(len(words), len(hyp_words))
        for i in range(n):
            core = tokenize(words[i])
            key_oov = (not core) or (core[0].lower() not in in_vocab)
            if key_oov:
                n_oov += 1
                if words[i] == hyp_words[i]:
                    n_ok += 1
        for i in range(n, len(words)):
            core = tokenize(words[i])
            if not core or core[0].lower() not in in_vocab:
                n_oov += 1
    return {"n_oov": n_oov, "n_exact": n_ok, "exact_acc": n_ok / max(n_oov, 1)}


def train_lm(torch, device, name, vocab_size, train_windows):
    import torch.nn.functional as F

    torch.manual_seed(SEED)
    lm = TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(device)
    opt = torch.optim.AdamW(lm.parameters(), lr=LR)
    rng = random.Random(SEED)
    lm.train()
    for step in range(STEPS):
        batch = [rng.choice(train_windows) for _ in range(BATCH)]
        x = torch.tensor(batch, dtype=torch.long, device=device)
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
            if device.type == "cuda":
                mem = f" alloc={torch.cuda.memory_allocated()/1e6:.0f}MB"
            print(f"[{name}] step {step}/{STEPS} loss={loss.item():.4f}{mem}", flush=True)
    return lm


def eval_bpc(torch, device, lm, windows, texts, seqs):
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


def run() -> dict:
    t_start = time.perf_counter()
    torch, device = _require_device()
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)} mem_fraction=0.45", flush=True)
    torch.set_num_threads(min(4, os.cpu_count() or 4))

    if not TRAIN.exists() or not VALID.exists():
        raise SystemExit(f"Missing slices. Expected {TRAIN} and {VALID}")

    train = load_slice(TRAIN)
    valid = load_slice(VALID)

    print("training BucketSlotCodec...", flush=True)
    codec = BucketSlotCodec.train(
        train, min_count=MIN_COUNT, dim=32, n_codebooks=2, codebook_size=256, seed=0
    )
    raw_spc = codec.symbols_per_char(train)

    print(
        f"training BucketCodeBPE target_mean_len={TARGET_MEAN_LEN} "
        f"allow_freq1=False...",
        flush=True,
    )
    bcb = BucketCodeBPE.train(
        codec,
        train,
        target_mean_len=TARGET_MEAN_LEN,
        max_merges=MAX_MERGES,
        min_freq=2,
        allow_freq1=False,
    )
    stats = bcb.merge_stats()
    print(f"bucket_codebpe stats={stats}", flush=True)

    chrf_scores = [chrf(bcb.reconstruct(t), t) for t in valid]
    chrf_mean = sum(chrf_scores) / max(len(chrf_scores), 1)
    oov = oov_exact_acc(bcb.reconstruct, valid, codec.in_vocab)
    bcb_spc = bcb.symbols_per_char(train)

    text_bpe = BPEEncoder.train(train, vocab_size=TEXT_BPE_VOCAB)
    n_chars = sum(max(len(t), 1) for t in train)
    tbpe_spc = sum(len(text_bpe.encode_text(t)) for t in train) / max(n_chars, 1)

    unit_gates = {
        "chrf_ok": chrf_mean >= GATE_CHRF,
        "oov_exact_ok": oov["exact_acc"] >= GATE_OOV_EXACT,
        "packing_vs_raw_ok": bcb_spc <= raw_spc + 1e-12,
        "no_freq1_ok": stats["n_merges_freq1"] == 0,
    }
    units_passed = all(unit_gates.values())

    results = {
        "device": str(device),
        "unit_gates": {
            **unit_gates,
            "gates": {
                "chrf_min": GATE_CHRF,
                "oov_exact_min": GATE_OOV_EXACT,
                "symbols_per_char_leq_raw_bucket_slot": True,
                "n_merges_freq1_max": 0,
            },
            "measured": {
                "chrf_valid": chrf_mean,
                "oov_exact_acc": oov["exact_acc"],
                "n_oov": oov["n_oov"],
                "bucket_codebpe_symbols_per_char": bcb_spc,
                "raw_bucket_slot_symbols_per_char": raw_spc,
                "text_bpe_symbols_per_char": tbpe_spc,
                **stats,
            },
        },
        "lm_ran": False,
        "elapsed_s": time.perf_counter() - t_start,
    }

    if not units_passed:
        results["passed"] = False
        results["pass_via"] = "unit_gates_failed"
        results["stopped_reason"] = "unit_gates_failed"
        _write(results)
        return results

    print("unit gates PASS — training LMs...", flush=True)
    train_code = [bcb.encode_text(t) for t in train]
    valid_code = [bcb.encode_text(t) for t in valid]
    train_tbpe = [text_bpe.encode_text(t) for t in train]
    valid_tbpe = [text_bpe.encode_text(t) for t in valid]

    train_code_w = make_windows(train_code, MAX_LEN)
    valid_code_w = make_windows(valid_code, MAX_LEN)
    train_tbpe_w = make_windows(train_tbpe, MAX_LEN)
    valid_tbpe_w = make_windows(valid_tbpe, MAX_LEN)

    t0 = time.perf_counter()
    code_lm = train_lm(torch, device, "bucket_codebpe", bcb.vocab_size, train_code_w)
    bpe_lm = train_lm(torch, device, "text_bpe", text_bpe.vocab_size, train_tbpe_w)
    lm_wall = time.perf_counter() - t0

    code_m = eval_bpc(torch, device, code_lm, valid_code_w, valid, valid_code)
    bpe_m = eval_bpc(torch, device, bpe_lm, valid_tbpe_w, valid, valid_tbpe)
    ratio = code_m["bpc"] / bpe_m["bpc"] if bpe_m["bpc"] > 0 else float("inf")
    lm_pass = ratio <= GATE_BPC_RATIO

    results.update(
        {
            "passed": lm_pass,
            "pass_via": "lm_and_units" if lm_pass else "units_only_lm_fail",
            "stopped_reason": "completed",
            "lm_ran": True,
            "gate_bpc_ratio_max": GATE_BPC_RATIO,
            "bpc_ratio_code_over_textbpe": ratio,
            "lm_wall_s": lm_wall,
            "elapsed_s": time.perf_counter() - t_start,
            "shared_train": {
                "steps": STEPS,
                "batch": BATCH,
                "lr": LR,
                "d_model": D_MODEL,
                "n_layers": N_LAYERS,
                "n_heads": N_HEADS,
                "max_len": MAX_LEN,
            },
            "bucket_codebpe": {
                **code_m,
                "vocab_size": bcb.vocab_size,
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
    )
    _write(results)
    return results


def _write(results: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    md = _md(results)
    (OUT / "results.md").write_text(md)
    print(md, flush=True)


def _md(r: dict) -> str:
    status = "PASS" if r.get("passed") else "FAIL"
    ug = r["unit_gates"]
    m = ug["measured"]
    lines = [
        f"# Dir6 BucketCodeBPE + LM — **{status}**",
        "",
        f"- Device: `{r.get('device')}`",
        f"- Stopped: `{r.get('stopped_reason', 'n/a')}`",
        f"- Pass via: `{r.get('pass_via', 'n/a')}`",
        f"- Elapsed: {r.get('elapsed_s', 0):.1f}s",
        "",
        "## Unit gates",
        "",
        f"- chrF: {m['chrf_valid']:.4f} ok={ug['chrf_ok']}",
        f"- OOV exact: {m['oov_exact_acc']:.4f} (n={m['n_oov']}) ok={ug['oov_exact_ok']}",
        f"- symbols/char: codebpe {m['bucket_codebpe_symbols_per_char']:.4f} | "
        f"raw bucket+slot {m['raw_bucket_slot_symbols_per_char']:.4f} | "
        f"text BPE {m['text_bpe_symbols_per_char']:.4f} "
        f"ok={ug['packing_vs_raw_ok']}",
        f"- merges: {m['n_merges']} (freq≥2: {m['n_merges_freq2']}, "
        f"freq=1: {m['n_merges_freq1']}) ok={ug['no_freq1_ok']}",
        f"- mean B unit len: {m['mean_b_unit_len']:.4f}",
        f"- merge hist: {m['merge_freq_hist']}",
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
                f"- LM wall: {r['lm_wall_s']:.1f}s",
                "",
                "| substrate | BPC ↓ | PPL ↓ | sym/char | vocab | params (all / non-embed) |",
                "|---|---|---|---|---|---|",
                f"| BucketCodeBPE | {r['bucket_codebpe']['bpc']:.4f} | "
                f"{r['bucket_codebpe']['perplexity']:.2f} | "
                f"{r['bucket_codebpe']['symbols_per_char']:.4f} | "
                f"{r['bucket_codebpe']['vocab_size']} | "
                f"{r['bucket_codebpe']['params_total']:,} / "
                f"{r['bucket_codebpe']['params_non_embed']:,} |",
                f"| text BPE | {r['text_bpe']['bpc']:.4f} | "
                f"{r['text_bpe']['perplexity']:.2f} | "
                f"{r['text_bpe']['symbols_per_char']:.4f} | "
                f"{r['text_bpe']['vocab_size']} | "
                f"{r['text_bpe']['params_total']:,} / "
                f"{r['text_bpe']['params_non_embed']:,} |",
                "",
            ]
        )
    else:
        lines.append("LM bake-off skipped (unit gates failed).\n")
    return "\n".join(lines)


if __name__ == "__main__":
    run()
