"""Dir4 M0: diagnose OOV rate vs chrF under legacy PRCF nearest-word decode."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.dir3_scale_ladder.prepare_data import load_slice
from prcf.chrf import chrf
from prcf.model import train_prcf
from prcf.tokenize import tokenize

OUT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TRAIN = DATA / "wikitext2_train_slice.txt"
VALID = DATA / "wikitext2_valid_slice.txt"
MIN_COUNT = 2


def train_vocab_counts(texts: list[str]) -> Counter[str]:
    c: Counter[str] = Counter()
    for t in texts:
        for w in tokenize(t):
            c[w.lower()] += 1
    return c


def run() -> dict:
    if not TRAIN.exists() or not VALID.exists():
        raise SystemExit(f"Missing slices. Expected {TRAIN} and {VALID}")

    train = load_slice(TRAIN)
    valid = load_slice(VALID)
    counts = train_vocab_counts(train)
    in_vocab = {w for w, n in counts.items() if n >= MIN_COUNT}

    print(f"fitting PRCF on {len(train)} train lines...", flush=True)
    prcf = train_prcf(train, dim=32, n_codebooks=2, codebook_size=256, seed=0)

    n_words = 0
    n_oov = 0
    chrf_all: list[float] = []
    chrf_clean: list[float] = []
    chrf_oov_lines: list[float] = []

    for t in valid:
        words = tokenize(t)
        oov_flags = [w.lower() not in in_vocab for w in words]
        n_words += len(words)
        n_oov += sum(oov_flags)
        hyp = prcf.reconstruct(t)
        score = chrf(hyp, t)
        chrf_all.append(score)
        if any(oov_flags):
            chrf_oov_lines.append(score)
        else:
            chrf_clean.append(score)

    results = {
        "min_count": MIN_COUNT,
        "n_train_lines": len(train),
        "n_valid_lines": len(valid),
        "n_valid_words": n_words,
        "n_oov_words": n_oov,
        "oov_word_rate": n_oov / max(n_words, 1),
        "n_in_vocab_types": len(in_vocab),
        "chrf_valid_mean": sum(chrf_all) / max(len(chrf_all), 1),
        "chrf_zero_oov_lines_mean": sum(chrf_clean) / max(len(chrf_clean), 1),
        "chrf_with_oov_lines_mean": sum(chrf_oov_lines) / max(len(chrf_oov_lines), 1),
        "n_zero_oov_lines": len(chrf_clean),
        "n_with_oov_lines": len(chrf_oov_lines),
        "decode_path": "code_to_word_or_nearest_word",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results_m0.json").write_text(json.dumps(results, indent=2) + "\n")
    md = _md(results)
    (OUT / "results_m0.md").write_text(md)
    print(md, flush=True)
    return results


def _md(r: dict) -> str:
    return "\n".join(
        [
            "# Dir4 M0 — OOV vs chrF (legacy PRCF)",
            "",
            f"- Train/valid lines: {r['n_train_lines']} / {r['n_valid_lines']}",
            f"- In-vocab types (count ≥ {r['min_count']}): {r['n_in_vocab_types']}",
            f"- Valid OOV word rate: **{r['oov_word_rate']:.4f}** "
            f"({r['n_oov_words']}/{r['n_valid_words']})",
            f"- chrF all valid: **{r['chrf_valid_mean']:.4f}**",
            f"- chrF zero-OOV lines (n={r['n_zero_oov_lines']}): "
            f"**{r['chrf_zero_oov_lines_mean']:.4f}**",
            f"- chrF lines with OOV (n={r['n_with_oov_lines']}): "
            f"**{r['chrf_with_oov_lines_mean']:.4f}**",
            f"- Decode path: `{r['decode_path']}`",
            "",
        ]
    )


if __name__ == "__main__":
    run()
