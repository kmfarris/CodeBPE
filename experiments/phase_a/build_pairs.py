"""Build (paragraph, edited, gold_index) pairs for Phase A."""

from __future__ import annotations

import random
import re
from pathlib import Path

from prcf.tokenize import tokenize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "paragraphs.txt"

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "as", "by", "with", "from", "into", "over", "after", "before",
    "than", "then", "that", "this", "these", "those", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must",
    "can", "not", "no", "nor", "so", "if", "when", "while", "during",
    "each", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "too", "very", "just", "about", "up", "out", "off",
    "their", "its", "his", "her", "our", "your", "my", "they", "them",
    "he", "she", "it", "we", "you", "i", "who", "whom", "which", "what",
    "where", "how", "why", "all", "any", "both", "between", "through",
    "against", "above", "below", "under", "again", "further", "once",
    "here", "there", "also", "until", "because", "although", "though",
}

REPLACEMENTS = [
    "ancient", "bright", "careful", "dense", "eager", "faint", "gentle",
    "hidden", "intact", "jumbled", "keen", "luminous", "modest", "narrow",
    "open", "precise", "quiet", "rapid", "subtle", "tender", "uneven",
    "vivid", "wooden", "yearly", "zinc", "amber", "bronze", "copper",
    "delta", "echo", "frost", "grain", "harbor", "island", "jasper",
    "kernel", "ladder", "marble", "nickel", "orbit", "pebble", "quartz",
    "ridge", "silver", "timber", "umbra", "valley", "willow", "xenon",
]


def load_paragraphs(path: Path = DATA) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def content_indices(words: list[str]) -> list[int]:
    out = []
    for i, w in enumerate(words):
        if w.lower() not in STOPWORDS and len(w) >= 3 and re.search(r"[A-Za-z]", w):
            out.append(i)
    return out


def substitute_word(paragraph: str, rng: random.Random) -> tuple[str, int] | None:
    words = tokenize(paragraph)
    candidates = content_indices(words)
    if not candidates:
        return None
    idx = rng.choice(candidates)
    original = words[idx]
    options = [r for r in REPLACEMENTS if r.lower() != original.lower()]
    if not options:
        return None
    replacement = rng.choice(options)
    # Rebuild with simple whitespace join (punctuation already stripped by tokenize)
    # Prefer surgically replacing in the original string to keep surrounding punctuation.
    pattern = re.compile(rf"\b{re.escape(original)}\b")
    matches = list(pattern.finditer(paragraph))
    if not matches:
        # Fallback: reconstruct from tokens
        new_words = list(words)
        new_words[idx] = replacement
        return " ".join(new_words), idx
    # Map token index to occurrence among identical words
    occurrence = sum(1 for j in range(idx) if words[j] == original)
    if occurrence >= len(matches):
        new_words = list(words)
        new_words[idx] = replacement
        return " ".join(new_words), idx
    m = matches[occurrence]
    edited = paragraph[: m.start()] + replacement + paragraph[m.end() :]
    # Verify token index still matches
    edited_words = tokenize(edited)
    if len(edited_words) != len(words):
        new_words = list(words)
        new_words[idx] = replacement
        return " ".join(new_words), idx
    return edited, idx


def build_pairs(
    paragraphs: list[str] | None = None,
    variants_per: int = 3,
    seed: int = 42,
) -> list[dict]:
    paragraphs = paragraphs if paragraphs is not None else load_paragraphs()
    rng = random.Random(seed)
    pairs: list[dict] = []
    for p_idx, paragraph in enumerate(paragraphs):
        for v in range(variants_per):
            result = substitute_word(paragraph, rng)
            if result is None:
                continue
            edited, gold = result
            pairs.append(
                {
                    "paragraph_id": p_idx,
                    "variant": v,
                    "original": paragraph,
                    "edited": edited,
                    "gold_index": gold,
                    "n_tokens": len(tokenize(paragraph)),
                }
            )
    return pairs


if __name__ == "__main__":
    pairs = build_pairs()
    print(f"built {len(pairs)} pairs from {len(load_paragraphs())} paragraphs")
