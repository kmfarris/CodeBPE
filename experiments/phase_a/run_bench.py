"""Phase A localization bench."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.phase_a.build_pairs import build_pairs, load_paragraphs
from prcf.localize import localize_by_token_diff, localize_edit
from prcf.simhash import hamming64, simhash64

OUT_DIR = Path(__file__).resolve().parent
GATE = 0.95


def length_bucket(n: int) -> str:
    if n < 12:
        return "short_<12"
    if n < 20:
        return "mid_12_19"
    return "long_20+"


def run() -> dict:
    paragraphs = load_paragraphs()
    pairs = build_pairs(paragraphs, variants_per=3, seed=42)

    trigram_exact = 0
    trigram_within1 = 0
    trigram_abs = 0.0
    token_exact = 0
    by_bucket: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "exact": 0, "within1": 0}
    )

    simhash_changed = 0
    for pair in pairs:
        pred = localize_edit(pair["original"], pair["edited"])
        gold = pair["gold_index"]
        if pred is None:
            pred = -10_000
        err = abs(pred - gold)
        trigram_abs += err
        if pred == gold:
            trigram_exact += 1
        if err <= 1:
            trigram_within1 += 1

        tok_pred = localize_by_token_diff(pair["original"], pair["edited"])
        if tok_pred == gold:
            token_exact += 1

        bucket = length_bucket(pair["n_tokens"])
        by_bucket[bucket]["n"] += 1
        if pred == gold:
            by_bucket[bucket]["exact"] += 1
        if err <= 1:
            by_bucket[bucket]["within1"] += 1

        ha = simhash64(pair["original"])
        hb = simhash64(pair["edited"])
        if ha != hb:
            simhash_changed += 1

    n = len(pairs)
    exact_acc = trigram_exact / n if n else 0.0
    within1_acc = trigram_within1 / n if n else 0.0
    mean_abs_err = trigram_abs / n if n else 0.0
    token_acc = token_exact / n if n else 0.0
    passed = exact_acc > GATE

    bucket_report = {}
    for name, stats in sorted(by_bucket.items()):
        nn = int(stats["n"]) or 1
        bucket_report[name] = {
            "n": nn,
            "exact_acc": stats["exact"] / nn,
            "within1_acc": stats["within1"] / nn,
        }

    results = {
        "n_paragraphs": len(paragraphs),
        "n_pairs": n,
        "gate": GATE,
        "passed": passed,
        "positioned_trigram": {
            "exact_acc": exact_acc,
            "within1_acc": within1_acc,
            "mean_abs_err": mean_abs_err,
        },
        "token_sequence_diff": {
            "exact_acc": token_acc,
            "note": "sanity upper bound for equal-length single substitutions",
        },
        "simhash_negative_control": {
            "fraction_hashes_differ": simhash_changed / n if n else 0.0,
            "localization_acc": 0.0,
            "note": "collapsed 64-bit SimHash cannot localize; reported as 0%",
        },
        "by_length_bucket": bucket_report,
    }

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    md = _format_md(results)
    (OUT_DIR / "results.md").write_text(md)
    print(md)
    return results


def _format_md(r: dict) -> str:
    status = "PASS" if r["passed"] else "FAIL"
    pt = r["positioned_trigram"]
    lines = [
        f"# Phase A results — **{status}**",
        "",
        f"- Pairs: {r['n_pairs']} from {r['n_paragraphs']} paragraphs",
        f"- Gate: exact_acc > {r['gate']}",
        "",
        "## Positioned word-trigram fingerprint",
        "",
        f"| metric | value |",
        f"|---|---|",
        f"| exact_acc | {pt['exact_acc']:.4f} |",
        f"| within1_acc | {pt['within1_acc']:.4f} |",
        f"| mean_abs_err | {pt['mean_abs_err']:.4f} |",
        "",
        "## Controls",
        "",
        f"- Token-sequence diff exact_acc: {r['token_sequence_diff']['exact_acc']:.4f}",
        f"- SimHash localization_acc: {r['simhash_negative_control']['localization_acc']:.4f} "
        f"(hashes differ on {r['simhash_negative_control']['fraction_hashes_differ']:.2%} of pairs)",
        "",
        "## By length bucket",
        "",
        "| bucket | n | exact_acc | within1_acc |",
        "|---|---|---|---|",
    ]
    for name, s in r["by_length_bucket"].items():
        lines.append(
            f"| {name} | {s['n']} | {s['exact_acc']:.4f} | {s['within1_acc']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    run()
