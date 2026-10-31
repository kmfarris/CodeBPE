"""Dir4 M1: HybridCodec gates — chrF, OOV exact, packing vs text BPE / legacy PRCF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.dir3_scale_ladder.prepare_data import load_slice
from prcf.bpe import BPEEncoder
from prcf.chrf import chrf
from prcf.hybrid_codec import HybridCodec
from prcf.model import train_prcf
from prcf.tokenize import tokenize

OUT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TRAIN = DATA / "wikitext2_train_slice.txt"
VALID = DATA / "wikitext2_valid_slice.txt"

GATE_CHRF = 0.90
GATE_OOV_EXACT = 0.95
TEXT_BPE_VOCAB = 2000
MIN_COUNT = 2


def oov_exact_acc(codec: HybridCodec, texts: list[str]) -> dict:
    n_oov = 0
    n_ok = 0
    for t in texts:
        words = t.split()
        hyp_words = codec.reconstruct(t).split()
        n = min(len(words), len(hyp_words))
        for i in range(n):
            core = tokenize(words[i])
            key_oov = (not core) or (core[0].lower() not in codec.in_vocab)
            if key_oov:
                n_oov += 1
                if words[i] == hyp_words[i]:
                    n_ok += 1
        for i in range(n, len(words)):
            core = tokenize(words[i])
            if not core or core[0].lower() not in codec.in_vocab:
                n_oov += 1
    return {
        "n_oov": n_oov,
        "n_exact": n_ok,
        "exact_acc": n_ok / max(n_oov, 1),
    }


def run() -> dict:
    if not TRAIN.exists() or not VALID.exists():
        raise SystemExit(f"Missing slices. Expected {TRAIN} and {VALID}")

    train = load_slice(TRAIN)
    valid = load_slice(VALID)

    print("training HybridCodec...", flush=True)
    hybrid = HybridCodec.train(
        train,
        min_count=MIN_COUNT,
        dim=32,
        n_codebooks=2,
        codebook_size=256,
        seed=0,
    )

    print("fitting legacy PRCF for packing baseline...", flush=True)
    legacy = train_prcf(train, dim=32, n_codebooks=2, codebook_size=256, seed=0)
    text_bpe = BPEEncoder.train(train, vocab_size=TEXT_BPE_VOCAB)

    chrf_scores = [chrf(hybrid.reconstruct(t), t) for t in valid]
    chrf_mean = sum(chrf_scores) / max(len(chrf_scores), 1)
    oov = oov_exact_acc(hybrid, valid)

    hybrid_spc = hybrid.symbols_per_char(train)
    legacy_n_syms = sum(len(legacy.encode_text(t)) for t in train)
    legacy_n_chars = sum(max(len(t), 1) for t in train)
    legacy_spc = legacy_n_syms / max(legacy_n_chars, 1)
    bpe_n_syms = sum(len(text_bpe.encode_text(t)) for t in train)
    bpe_spc = bpe_n_syms / max(legacy_n_chars, 1)

    chrf_ok = chrf_mean >= GATE_CHRF
    oov_ok = oov["exact_acc"] >= GATE_OOV_EXACT
    passed = chrf_ok and oov_ok

    results = {
        "passed": passed,
        "gates": {"chrf_min": GATE_CHRF, "oov_exact_min": GATE_OOV_EXACT},
        "measured": {
            "chrf_valid": chrf_mean,
            "oov_exact_acc": oov["exact_acc"],
            "n_oov": oov["n_oov"],
            "n_oov_exact": oov["n_exact"],
            "hybrid_symbols_per_char": hybrid_spc,
            "legacy_prcf_symbols_per_char": legacy_spc,
            "text_bpe_symbols_per_char": bpe_spc,
            "n_in_vocab": len(hybrid.in_vocab),
            "min_count": MIN_COUNT,
        },
        "gate_checks": {"chrf_ok": chrf_ok, "oov_exact_ok": oov_ok},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results_m1.json").write_text(json.dumps(results, indent=2) + "\n")
    md = _md(results)
    (OUT / "results_m1.md").write_text(md)
    print(md, flush=True)
    return results


def _md(r: dict) -> str:
    status = "PASS" if r["passed"] else "FAIL"
    m = r["measured"]
    g = r["gates"]
    return "\n".join(
        [
            f"# Dir4 M1 — HybridCodec — **{status}**",
            "",
            f"- chrF valid: {m['chrf_valid']:.4f} (gate ≥ {g['chrf_min']}) "
            f"ok={r['gate_checks']['chrf_ok']}",
            f"- OOV exact: {m['oov_exact_acc']:.4f} "
            f"(n={m['n_oov']}, gate ≥ {g['oov_exact_min']}) "
            f"ok={r['gate_checks']['oov_exact_ok']}",
            f"- symbols/char: hybrid {m['hybrid_symbols_per_char']:.4f} | "
            f"legacy PRCF {m['legacy_prcf_symbols_per_char']:.4f} | "
            f"text BPE {m['text_bpe_symbols_per_char']:.4f}",
            f"- in-vocab types (count ≥ {m['min_count']}): {m['n_in_vocab']}",
            "",
        ]
    )


if __name__ == "__main__":
    run()
