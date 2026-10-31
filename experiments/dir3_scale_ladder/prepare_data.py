"""Download WikiText-2 raw and write word-capped train/valid paragraph slices."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def _lines_to_paragraphs(lines: list[str]) -> list[str]:
    """Keep non-empty, non-header lines as paragraph units."""
    paras: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("=") and s.endswith("="):
            continue
        if len(s.split()) < 5:
            continue
        paras.append(s)
    return paras


def _cap_by_words(paras: list[str], max_words: int) -> list[str]:
    out: list[str] = []
    n = 0
    for p in paras:
        w = len(p.split())
        if n >= max_words:
            break
        out.append(p)
        n += w
    return out


def _word_count(paras: list[str]) -> int:
    return sum(len(p.split()) for p in paras)


def _load_wikitext_split(split: str) -> list[str]:
    """Load wikitext-2-raw-v1 via HuggingFace datasets (cached after first pull)."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return [row["text"] for row in ds]


def prepare_slices(
    train_words: int = 100_000,
    valid_words: int = 20_000,
) -> tuple[Path, Path, dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    print("loading WikiText-2 raw via datasets...", flush=True)
    train_paras = _cap_by_words(_lines_to_paragraphs(_load_wikitext_split("train")), train_words)
    valid_paras = _cap_by_words(_lines_to_paragraphs(_load_wikitext_split("validation")), valid_words)

    train_out = DATA / "wikitext2_train_slice.txt"
    valid_out = DATA / "wikitext2_valid_slice.txt"
    train_out.write_text("\n".join(train_paras) + "\n", encoding="utf-8")
    valid_out.write_text("\n".join(valid_paras) + "\n", encoding="utf-8")

    meta = {
        "train_path": str(train_out),
        "valid_path": str(valid_out),
        "train_paragraphs": len(train_paras),
        "valid_paragraphs": len(valid_paras),
        "train_words": _word_count(train_paras),
        "valid_words": _word_count(valid_paras),
        "train_words_cap": train_words,
        "valid_words_cap": valid_words,
    }
    print(f"wrote slices: {meta}", flush=True)
    return train_out, valid_out, meta


def prepare_full() -> tuple[Path, Path, dict]:
    """Full WikiText-2 train/valid (no word cap) after paragraph filter."""
    DATA.mkdir(parents=True, exist_ok=True)
    train_out = DATA / "wikitext2_train_full.txt"
    valid_out = DATA / "wikitext2_valid_full.txt"
    if train_out.exists() and valid_out.exists() and train_out.stat().st_size > 0:
        train_paras = load_slice(train_out)
        valid_paras = load_slice(valid_out)
        meta = {
            "train_path": str(train_out),
            "valid_path": str(valid_out),
            "train_paragraphs": len(train_paras),
            "valid_paragraphs": len(valid_paras),
            "train_words": _word_count(train_paras),
            "valid_words": _word_count(valid_paras),
            "cached": True,
        }
        print(f"cached full splits: {meta}", flush=True)
        return train_out, valid_out, meta

    print("loading full WikiText-2 raw via datasets...", flush=True)
    train_paras = _lines_to_paragraphs(_load_wikitext_split("train"))
    valid_paras = _lines_to_paragraphs(_load_wikitext_split("validation"))
    train_out.write_text("\n".join(train_paras) + "\n", encoding="utf-8")
    valid_out.write_text("\n".join(valid_paras) + "\n", encoding="utf-8")
    meta = {
        "train_path": str(train_out),
        "valid_path": str(valid_out),
        "train_paragraphs": len(train_paras),
        "valid_paragraphs": len(valid_paras),
        "train_words": _word_count(train_paras),
        "valid_words": _word_count(valid_paras),
        "cached": False,
    }
    print(f"wrote full splits: {meta}", flush=True)
    return train_out, valid_out, meta


def load_slice(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-words", type=int, default=100_000)
    ap.add_argument("--valid-words", type=int, default=20_000)
    args = ap.parse_args()
    prepare_slices(train_words=args.train_words, valid_words=args.valid_words)


if __name__ == "__main__":
    main()
