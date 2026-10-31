"""Path 2 unit tests for CodeBPE."""

from prcf.chrf import chrf
from prcf.code_bpe import CodeBPE
from prcf.model import train_prcf


def test_code_bpe_roundtrip():
    texts = [
        "Alice visited Paris yesterday morning",
        "Bob visited London yesterday evening",
        "Alice visited Paris before noon",
        "the quick brown fox jumps over",
        "the quick brown dog jumps over",
    ]
    prcf = train_prcf(texts, codebook_size=64, n_codebooks=2, seed=0)
    cb = CodeBPE.train(prcf, texts, n_merges=40)
    for t in texts:
        rec = cb.reconstruct(t)
        assert chrf(rec, t) > 0.5
    assert cb.merge_stats()["n_merges"] >= 1


def test_code_bpe_target_mean_len():
    texts = [
        "Alice visited Paris yesterday morning after breakfast",
        "Bob visited London yesterday evening after dinner",
        "Alice visited Paris before noon yesterday morning",
        "the quick brown fox jumps over the lazy dog today",
        "the quick brown dog jumps over the lazy fox today",
        "Alice visited Paris yesterday morning after lunch",
    ] * 3
    prcf = train_prcf(texts, codebook_size=64, n_codebooks=2, seed=0)
    cb = CodeBPE.train(prcf, texts, target_mean_len=2.0, max_merges=2000)
    stats = cb.merge_stats()
    assert stats["mean_unit_len_codes"] >= 1.5
    assert "n_merges_freq2" in stats
