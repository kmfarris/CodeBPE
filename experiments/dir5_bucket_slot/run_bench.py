"""Dir5: BucketSlotCodec vs HybridCodec vs text BPE on WikiText slices."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.dir3_scale_ladder.prepare_data import load_slice
from prcf.bpe import BPEEncoder
from prcf.bucket_slot import BucketSlotCodec
from prcf.chrf import chrf
from prcf.hybrid_codec import HybridCodec
from prcf.tokenize import tokenize

OUT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TRAIN = DATA / "wikitext2_train_slice.txt"
VALID = DATA / "wikitext2_valid_slice.txt"

GATE_CHRF = 0.90
GATE_OOV_EXACT = 0.95
TEXT_BPE_VOCAB = 2000
MIN_COUNT = 2


def oov_exact_acc(codec, texts: list[str], in_vocab: set[str]) -> dict:
    n_oov = 0
    n_ok = 0
    for t in texts:
        words = t.split()
        hyp_words = codec.reconstruct(t).split()
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

    print("training BucketSlotCodec...", flush=True)
    bs = BucketSlotCodec.train(
        train,
        min_count=MIN_COUNT,
        dim=32,
        n_codebooks=2,
        codebook_size=256,
        seed=0,
    )
    # Injectivity already asserted in train; re-check packed ids
    ids = {bs._pack_id(c, s) for (c, s) in bs.bs_to_surface}
    assert len(ids) == len(bs.bs_to_surface)
    collision_rate = 0.0

    print("training HybridCodec (baseline)...", flush=True)
    hybrid = HybridCodec.train(
        train,
        min_count=MIN_COUNT,
        dim=32,
        n_codebooks=2,
        codebook_size=256,
        seed=0,
    )
    text_bpe = BPEEncoder.train(train, vocab_size=TEXT_BPE_VOCAB)

    chrf_scores = [chrf(bs.reconstruct(t), t) for t in valid]
    chrf_mean = sum(chrf_scores) / max(len(chrf_scores), 1)
    oov = oov_exact_acc(bs, valid, bs.in_vocab)

    bs_spc = bs.symbols_per_char(train)
    hy_spc = hybrid.symbols_per_char(train)
    n_chars = sum(max(len(t), 1) for t in train)
    bpe_spc = sum(len(text_bpe.encode_text(t)) for t in train) / max(n_chars, 1)

    paths_train = bs.path_fractions(train)
    paths_valid = bs.path_fractions(valid)

    chrf_ok = chrf_mean >= GATE_CHRF
    oov_ok = oov["exact_acc"] >= GATE_OOV_EXACT
    packing_vs_hybrid_ok = bs_spc < hy_spc
    passed = chrf_ok and oov_ok and packing_vs_hybrid_ok and collision_rate == 0.0

    results = {
        "passed": passed,
        "gates": {
            "chrf_min": GATE_CHRF,
            "oov_exact_min": GATE_OOV_EXACT,
            "symbols_per_char_lt_hybrid": True,
            "collision_rate_max": 0.0,
        },
        "gate_checks": {
            "chrf_ok": chrf_ok,
            "oov_exact_ok": oov_ok,
            "packing_vs_hybrid_ok": packing_vs_hybrid_ok,
            "collision_ok": collision_rate == 0.0,
        },
        "measured": {
            "chrf_valid": chrf_mean,
            "oov_exact_acc": oov["exact_acc"],
            "n_oov": oov["n_oov"],
            "collision_rate": collision_rate,
            "bucket_slot_symbols_per_char": bs_spc,
            "hybrid_symbols_per_char": hy_spc,
            "text_bpe_symbols_per_char": bpe_spc,
            "frac_words_bucket_path_train": paths_train["frac_words_bucket_path"],
            "frac_words_bucket_path_valid": paths_valid["frac_words_bucket_path"],
            **bs.stats(),
            **{f"train_{k}": v for k, v in paths_train.items()},
            **{f"valid_{k}": v for k, v in paths_valid.items()},
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    md = _md(results)
    (OUT / "results.md").write_text(md)
    print(md, flush=True)
    return results


def _md(r: dict) -> str:
    status = "PASS" if r["passed"] else "FAIL"
    m = r["measured"]
    g = r["gates"]
    return "\n".join(
        [
            f"# Dir5 Bucket+slot — **{status}**",
            "",
            f"- chrF valid: {m['chrf_valid']:.4f} (gate ≥ {g['chrf_min']}) "
            f"ok={r['gate_checks']['chrf_ok']}",
            f"- OOV exact: {m['oov_exact_acc']:.4f} (n={m['n_oov']}, "
            f"gate ≥ {g['oov_exact_min']}) ok={r['gate_checks']['oov_exact_ok']}",
            f"- collision rate: {m['collision_rate']:.4f} "
            f"ok={r['gate_checks']['collision_ok']}",
            f"- symbols/char: bucket+slot {m['bucket_slot_symbols_per_char']:.4f} | "
            f"hybrid {m['hybrid_symbols_per_char']:.4f} | "
            f"text BPE {m['text_bpe_symbols_per_char']:.4f} "
            f"(vs hybrid ok={r['gate_checks']['packing_vs_hybrid_ok']})",
            f"- bucket-path frac train/valid: "
            f"{m['frac_words_bucket_path_train']:.4f} / "
            f"{m['frac_words_bucket_path_valid']:.4f}",
            f"- slot_stride={m['slot_stride']} max_slot={m['max_slot']} "
            f"collision_buckets={m['n_collision_buckets']} "
            f"in_vocab={m['n_in_vocab']}",
            "",
        ]
    )


if __name__ == "__main__":
    run()
