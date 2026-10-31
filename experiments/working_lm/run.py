"""Working-scale BucketCodeBPE LM on full WikiText-2 (single pipeline)."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.dir3_scale_ladder.prepare_data import load_slice, prepare_full
from experiments.phase_a.build_pairs import substitute_word
from prcf.bpe import BPEEncoder
from prcf.bucket_code_bpe import BucketCodeBPE, LM_EOS, segment_ids
from prcf.bucket_slot import BucketSlotCodec
from prcf.chrf import chrf
from prcf.tiny_lm import TinyTransformerLM
from prcf.tokenize import tokenize

OUT = Path(__file__).resolve().parent
ART = OUT / "artifacts"
ENC = OUT / "encoded"

D_MODEL = 384
N_LAYERS = 6
N_HEADS = 6
MAX_LEN = 256
DROPOUT = 0.1
STEPS = 20000
BATCH = 8
LR = 3e-4
SEED = 0
LOG_EVERY = 200
TEXT_BPE_VOCAB = 10000
TARGET_MEAN_LEN = 1.5
MAX_MERGES = 20000
MIN_COUNT = 2
GATE_CHRF = 0.90
GATE_OOV_EXACT = 0.95
GATE_BPC_RATIO = 1.05
GATE_LOCALIZE = 0.80


def _require_device():
    import torch

    if os.environ.get("PRCF_REQUIRE_GPU", "1") == "1" and not torch.cuda.is_available():
        raise SystemExit("GPU required but CUDA unavailable.")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(0.45, device=0)
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")
    return torch, device


def make_windows(seqs: list[list[int]], max_len: int) -> list[list[int]]:
    windows: list[list[int]] = []
    for seq in seqs:
        if len(seq) < 2:
            continue
        if len(seq) <= max_len:
            s = seq + [LM_EOS] * (max_len - len(seq))
            windows.append(s[:max_len])
            continue
        for start in range(0, len(seq) - 1, max_len):
            chunk = seq[start : start + max_len]
            if len(chunk) < 2:
                continue
            if len(chunk) < max_len:
                chunk = chunk + [LM_EOS] * (max_len - len(chunk))
            windows.append(chunk[:max_len])
    return windows


def non_embed_params(m) -> int:
    return sum(
        p.numel()
        for n, p in m.named_parameters()
        if not n.startswith("tok.") and not n.startswith("pos.")
    )


def oov_exact(reconstructor, texts, in_vocab):
    n_oov = n_ok = 0
    for t in texts:
        words = t.split()
        hyp = reconstructor(t).split()
        n = min(len(words), len(hyp))
        for i in range(n):
            core = tokenize(words[i])
            if not core or core[0].lower() not in in_vocab:
                n_oov += 1
                if words[i] == hyp[i]:
                    n_ok += 1
        for i in range(n, len(words)):
            core = tokenize(words[i])
            if not core or core[0].lower() not in in_vocab:
                n_oov += 1
    return n_ok / max(n_oov, 1), n_oov


def words_from_bcb(bcb: BucketCodeBPE, text: str) -> list[str]:
    """Codec word stream: one surface per whitespace token (matches encode_text)."""
    atoms = bcb._apply_merges(segment_ids(bcb.codec.encode_text(text), bcb.codec))
    words: list[str] = []
    for s in atoms:
        for base in bcb._expand_atom(s):
            if base.startswith("B"):
                tok = int(base[1:])
                parsed = bcb.codec._unpack_id(tok)
                if parsed is None:
                    words.append("<unk>")
                else:
                    words.append(bcb.codec.bs_to_surface.get(parsed, "<unk>"))
            elif base.startswith("E:"):
                payload = base[2:]
                if not payload:
                    words.append("")
                else:
                    words.append("".join(chr(int(x)) for x in payload.split(",")))
    return words


def tokenize_to_split_index(text: str, tok_idx: int) -> int | None:
    """Map tokenize() index → text.split() index (codec word alignment)."""
    mapping: list[int] = []
    for si, w in enumerate(text.split()):
        for _ in tokenize(w):
            mapping.append(si)
    if tok_idx < 0 or tok_idx >= len(mapping):
        return None
    return mapping[tok_idx]


def locality_bench(bcb: BucketCodeBPE, texts: list[str], n_pairs: int = 300) -> dict:
    """Localize one-word edits in codec/split index space (not bare tokenize())."""
    rng = random.Random(0)
    exact = within2 = total = inv = 0
    for _ in range(n_pairs * 5):
        if total >= n_pairs:
            break
        t = rng.choice(texts)
        sub = substitute_word(t, rng)
        if sub is None:
            continue
        edited, gold_tok = sub
        core = tokenize(t)
        if gold_tok >= len(core):
            continue
        # in-vocab edits only for primary gate
        if core[gold_tok].lower() not in bcb.codec.in_vocab:
            inv += 1
            continue
        gold = tokenize_to_split_index(t, gold_tok)
        if gold is None:
            continue
        wa = words_from_bcb(bcb, t)
        wb = words_from_bcb(bcb, edited)
        pred = None
        n = min(len(wa), len(wb))
        for i in range(n):
            if wa[i].lower() != wb[i].lower():
                pred = i
                break
        if pred is None and len(wa) != len(wb):
            pred = n
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
        "skipped_oov_edits": inv,
        "index_space": "split_codec",
    }


def train_lm(torch, device, name, vocab_size, windows, steps, batch):
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
    for step in range(steps):
        batch_seqs = [rng.choice(windows) for _ in range(batch)]
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


def eval_bpc(torch, device, lm, windows, texts, seqs):
    import torch.nn.functional as F

    lm.eval()
    total_nll = 0.0
    total_tok = 0
    # Cap eval windows for wall time
    eval_w = windows[: min(len(windows), 2000)]
    with torch.no_grad():
        for w in eval_w:
            x = torch.tensor([w], dtype=torch.long, device=device)
            logits = lm(x[:, :-1])
            target = x[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=0,
                reduction="sum",
            )
            total_nll += float(loss.item())
            total_tok += int((target != 0).sum().item())
    n_chars = sum(max(len(t), 1) for t in texts)
    n_syms = sum(len(s) for s in seqs)
    return {
        "bpc": (total_nll / math.log(2)) / max(n_chars, 1),
        "perplexity": math.exp(total_nll / max(total_tok, 1)),
        "symbols_per_char": n_syms / max(n_chars, 1),
        "n_eval_windows": len(eval_w),
    }


def generate_samples(torch, device, lm, bcb: BucketCodeBPE, prompts: list[str], max_new: int = 40):
    import torch.nn.functional as F

    lm.eval()
    out_lines: list[str] = []
    with torch.no_grad():
        for p in prompts:
            ids = bcb.encode_text(p)[: MAX_LEN - max_new]
            if len(ids) < 1:
                ids = [LM_EOS]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            for _ in range(max_new):
                if x.size(1) >= MAX_LEN:
                    break
                logits = lm(x)[:, -1, :]
                nxt = int(torch.argmax(logits, dim=-1).item())
                if nxt == LM_EOS:
                    break
                x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
            gen_ids = x[0].tolist()
            text = bcb.decode_ids(gen_ids)
            out_lines.append(f"PROMPT: {p}\nGEN: {text}\n")
    return "\n".join(out_lines)


def encode_corpus(encoder, texts: list[str], label: str) -> list[list[int]]:
    seqs: list[list[int]] = []
    for i, t in enumerate(texts):
        if i % 500 == 0:
            print(f"  encode {label} {i}/{len(texts)}", flush=True)
        seqs.append(encoder.encode_text(t))
    return seqs


def run(skip_data: bool, skip_encode: bool, skip_train: bool) -> dict:
    t0 = time.perf_counter()
    torch, device = _require_device()
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    ENC.mkdir(parents=True, exist_ok=True)

    # --- data ---
    if not skip_data:
        _, _, data_meta = prepare_full()
    else:
        _, _, data_meta = prepare_full()
    train = load_slice(Path(data_meta["train_path"]))
    valid = load_slice(Path(data_meta["valid_path"]))
    print(f"data: train_words={data_meta['train_words']} valid_words={data_meta['valid_words']}", flush=True)

    codec_path = ART / "bucket_codebpe.pkl"
    tbpe_path = ART / "text_bpe.pkl"

    # --- fit ---
    if not skip_encode or not codec_path.exists():
        print("fitting BucketSlotCodec on full train...", flush=True)
        codec = BucketSlotCodec.train(
            train, min_count=MIN_COUNT, dim=32, n_codebooks=2, codebook_size=256, seed=0
        )
        print("fitting BucketCodeBPE...", flush=True)
        bcb = BucketCodeBPE.train(
            codec,
            train,
            target_mean_len=TARGET_MEAN_LEN,
            max_merges=MAX_MERGES,
            min_freq=2,
            allow_freq1=False,
        )
        print(f"codebpe stats={bcb.merge_stats()}", flush=True)
        text_bpe = BPEEncoder.train(train, vocab_size=TEXT_BPE_VOCAB)
        with open(codec_path, "wb") as f:
            pickle.dump(bcb, f)
        with open(tbpe_path, "wb") as f:
            pickle.dump(text_bpe, f)
    else:
        print("loading pickled codecs...", flush=True)
        with open(codec_path, "rb") as f:
            bcb = pickle.load(f)
        with open(tbpe_path, "rb") as f:
            text_bpe = pickle.load(f)

    # invertibility smoke
    smoke_n = min(100, len(valid))
    chrf_m = sum(chrf(bcb.reconstruct(t), t) for t in valid[:smoke_n]) / smoke_n
    oov_acc, n_oov = oov_exact(bcb.reconstruct, valid[:smoke_n], bcb.codec.in_vocab)
    print(f"invertibility smoke chrF={chrf_m:.4f} oov_exact={oov_acc:.4f}", flush=True)
    if chrf_m < GATE_CHRF or oov_acc < GATE_OOV_EXACT:
        raise SystemExit(f"invertibility smoke FAIL chrF={chrf_m} oov={oov_acc}")

    # --- encode ---
    code_train_path = ENC / "code_train.pkl"
    code_valid_path = ENC / "code_valid.pkl"
    bpe_train_path = ENC / "bpe_train.pkl"
    bpe_valid_path = ENC / "bpe_valid.pkl"
    if not skip_encode or not code_train_path.exists():
        print("encoding corpora to disk...", flush=True)
        code_train = encode_corpus(bcb, train, "code_train")
        code_valid = encode_corpus(bcb, valid, "code_valid")
        bpe_train = encode_corpus(text_bpe, train, "bpe_train")
        bpe_valid = encode_corpus(text_bpe, valid, "bpe_valid")
        for path, obj in [
            (code_train_path, code_train),
            (code_valid_path, code_valid),
            (bpe_train_path, bpe_train),
            (bpe_valid_path, bpe_valid),
        ]:
            with open(path, "wb") as f:
                pickle.dump(obj, f)
    else:
        print("loading encoded sequences...", flush=True)
        with open(code_train_path, "rb") as f:
            code_train = pickle.load(f)
        with open(code_valid_path, "rb") as f:
            code_valid = pickle.load(f)
        with open(bpe_train_path, "rb") as f:
            bpe_train = pickle.load(f)
        with open(bpe_valid_path, "rb") as f:
            bpe_valid = pickle.load(f)

    n_chars_tr = sum(max(len(t), 1) for t in train)
    n_chars_va = sum(max(len(t), 1) for t in valid)
    packing = {
        "code_train_spc": sum(len(s) for s in code_train) / max(n_chars_tr, 1),
        "code_valid_spc": sum(len(s) for s in code_valid) / max(n_chars_va, 1),
        "bpe_train_spc": sum(len(s) for s in bpe_train) / max(n_chars_tr, 1),
        "bpe_valid_spc": sum(len(s) for s in bpe_valid) / max(n_chars_va, 1),
    }
    print(f"packing={packing}", flush=True)

    results = {
        "data": data_meta,
        "device": str(device),
        "packing": packing,
        "codec": bcb.merge_stats(),
        "invertibility_smoke": {"chrf": chrf_m, "oov_exact": oov_acc, "n_oov": n_oov},
        "lm_ran": False,
    }

    if skip_train:
        results["stopped_reason"] = "skip_train"
        _write(results)
        return results

    # --- train ---
    batch = BATCH
    print(
        f"training LMs {N_LAYERS}L/{D_MODEL}d steps={STEPS} batch={batch} max_len={MAX_LEN}",
        flush=True,
    )
    code_w = make_windows(code_train, MAX_LEN)
    bpe_w = make_windows(bpe_train, MAX_LEN)
    valid_code_w = make_windows(code_valid, MAX_LEN)
    valid_bpe_w = make_windows(bpe_valid, MAX_LEN)
    print(f"windows train code/bpe={len(code_w)}/{len(bpe_w)}", flush=True)

    try:
        t_lm = time.perf_counter()
        code_lm = train_lm(torch, device, "bucket_codebpe", bcb.vocab_size, code_w, STEPS, batch)
        bpe_lm = train_lm(torch, device, "text_bpe", text_bpe.vocab_size, bpe_w, STEPS, batch)
        lm_wall = time.perf_counter() - t_lm
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and batch > 1:
            torch.cuda.empty_cache()
            batch = max(1, batch // 2)
            print(f"OOM — retrying with batch={batch}", flush=True)
            t_lm = time.perf_counter()
            code_lm = train_lm(torch, device, "bucket_codebpe", bcb.vocab_size, code_w, STEPS, batch)
            bpe_lm = train_lm(torch, device, "text_bpe", text_bpe.vocab_size, bpe_w, STEPS, batch)
            lm_wall = time.perf_counter() - t_lm
        else:
            raise

    torch.save(code_lm.state_dict(), OUT / "bucket_codebpe_lm.pt")
    torch.save(bpe_lm.state_dict(), OUT / "text_bpe_lm.pt")

    code_m = eval_bpc(torch, device, code_lm, valid_code_w, valid, code_valid)
    bpe_m = eval_bpc(torch, device, bpe_lm, valid_bpe_w, valid, bpe_valid)
    ratio = code_m["bpc"] / bpe_m["bpc"] if bpe_m["bpc"] > 0 else float("inf")
    bpc_ok = ratio <= GATE_BPC_RATIO

    print("locality bench...", flush=True)
    loc = locality_bench(bcb, valid, n_pairs=300)
    loc_ok = loc["within2_acc"] >= GATE_LOCALIZE

    prompts = [
        "The history of",
        "In mathematics,",
        "According to the",
        "The film was",
        "He was born in",
        "The company",
        "During the war",
        "It is located",
    ]
    samples = generate_samples(torch, device, code_lm, bcb, prompts)
    (OUT / "samples.txt").write_text(samples)

    passed = bpc_ok and loc_ok
    results.update(
        {
            "passed": passed,
            "lm_ran": True,
            "lm_wall_s": lm_wall,
            "elapsed_s": time.perf_counter() - t0,
            "batch_used": batch,
            "shared_train": {
                "steps": STEPS,
                "batch": batch,
                "lr": LR,
                "d_model": D_MODEL,
                "n_layers": N_LAYERS,
                "n_heads": N_HEADS,
                "max_len": MAX_LEN,
            },
            "bpc_ratio_code_over_textbpe": ratio,
            "gate_bpc_ratio_max": GATE_BPC_RATIO,
            "bpc_ok": bpc_ok,
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
            "locality": loc,
            "gate_localize_within2": GATE_LOCALIZE,
            "locality_ok": loc_ok,
        }
    )
    meta = {
        "data": data_meta,
        "shared_train": results["shared_train"],
        "packing": packing,
        "passed": passed,
    }
    (OUT / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    _write(results)
    return results


def _write(r: dict) -> None:
    (OUT / "results.json").write_text(json.dumps(r, indent=2) + "\n")
    lines = [
        f"# Working-scale BucketCodeBPE LM — **{'PASS' if r.get('passed') else 'FAIL'}**",
        "",
        f"- Device: `{r.get('device')}`",
        f"- Data train/valid words: {r['data'].get('train_words')} / {r['data'].get('valid_words')}",
        f"- Elapsed: {r.get('elapsed_s', 0):.1f}s",
        "",
        "## Packing",
        "",
        f"- CodeBPE train/valid sym/char: {r['packing']['code_train_spc']:.4f} / "
        f"{r['packing']['code_valid_spc']:.4f}",
        f"- text BPE train/valid sym/char: {r['packing']['bpe_train_spc']:.4f} / "
        f"{r['packing']['bpe_valid_spc']:.4f}",
        "",
        f"## Invertibility smoke: chrF={r['invertibility_smoke']['chrf']:.4f} "
        f"oov_exact={r['invertibility_smoke']['oov_exact']:.4f}",
        "",
    ]
    if r.get("lm_ran"):
        lines.extend(
            [
                "## LM bake-off",
                "",
                f"- Model: {r['shared_train']['n_layers']}L/{r['shared_train']['d_model']}d, "
                f"{r['shared_train']['steps']} steps, batch={r['batch_used']}",
                f"- BPC ratio: {r['bpc_ratio_code_over_textbpe']:.4f} "
                f"(gate ≤ {r['gate_bpc_ratio_max']}) ok={r['bpc_ok']}",
                f"- LM wall: {r['lm_wall_s']:.1f}s",
                "",
                "| substrate | BPC ↓ | PPL ↓ | valid sym/char | vocab | params (all / non-embed) |",
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
                "## Locality",
                "",
                f"- ±2 acc: {r['locality']['within2_acc']:.4f} "
                f"(n={r['locality']['n']}, gate ≥ {r['gate_localize_within2']}) "
                f"ok={r['locality_ok']}",
                f"- exact: {r['locality']['exact_acc']:.4f}",
                f"- index space: {r['locality'].get('index_space', 'split_codec')} "
                f"(gold aligned to codec whitespace tokens)",
                "",
                "Samples: `samples.txt`",
                "",
            ]
        )
    (OUT / "results.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def rescore_locality() -> dict:
    """Re-run locality only; merge into existing results.json (no retrain)."""
    codec_path = ART / "bucket_codebpe.pkl"
    results_path = OUT / "results.json"
    if not codec_path.exists() or not results_path.exists():
        raise SystemExit("need artifacts/bucket_codebpe.pkl and results.json")
    r = json.loads(results_path.read_text())
    valid = load_slice(Path(r["data"]["valid_path"]))
    with open(codec_path, "rb") as f:
        bcb = pickle.load(f)
    print("rescoring locality (split/codec gold)...", flush=True)
    loc = locality_bench(bcb, valid, n_pairs=300)
    loc_ok = loc["within2_acc"] >= GATE_LOCALIZE
    bpc_ok = bool(r.get("bpc_ok", False))
    r["locality"] = loc
    r["gate_localize_within2"] = GATE_LOCALIZE
    r["locality_ok"] = loc_ok
    r["passed"] = bpc_ok and loc_ok
    r["locality_rescored"] = True
    _write(r)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-encode", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--rescore-locality", action="store_true")
    args = ap.parse_args()
    if args.rescore_locality:
        rescore_locality()
        return
    run(skip_data=args.skip_data, skip_encode=args.skip_encode, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
