"""Phase B unit smoke tests."""

from prcf.chrf import chrf
from prcf.model import localize_by_codes, train_prcf


def test_chrf_identical():
    assert chrf("hello world", "hello world") == 1.0


def test_train_and_reconstruct():
    texts = [
        "Alice visited Paris yesterday",
        "Bob visited London yesterday",
        "the quick brown fox jumps",
    ]
    model = train_prcf(texts, codebook_size=32, n_codebooks=2, seed=0)
    rec = model.reconstruct("Alice visited Paris yesterday")
    assert "Alice" in rec or "alice" in rec.lower()


def test_localize_codes():
    a = [(1, 2), (3, 4), (5, 6)]
    b = [(1, 2), (9, 9), (5, 6)]
    assert localize_by_codes(a, b) == 1
