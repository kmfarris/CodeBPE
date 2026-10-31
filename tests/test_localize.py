"""Unit tests for Phase A localization."""

from prcf.fingerprint import slot_shingle_sets
from prcf.localize import localize_by_token_diff, localize_edit
from prcf.tokenize import tokenize


def test_tokenize_strips_punct():
    assert tokenize("Hello, world!") == ["Hello", "world"]


def test_localize_known_swap():
    a = "the quick brown fox jumps"
    b = "the quick red fox jumps"
    assert localize_edit(a, b) == 2
    assert localize_by_token_diff(a, b) == 2


def test_localize_first_word():
    a = "alpha beta gamma delta"
    b = "omega beta gamma delta"
    assert localize_edit(a, b) == 0


def test_localize_last_word():
    a = "alpha beta gamma delta"
    b = "alpha beta gamma epsilon"
    assert localize_edit(a, b) == 3


def test_slot_sets_nonempty():
    slots = slot_shingle_sets("one two three four five")
    assert len(slots) == 5
    assert all(len(s) > 0 for s in slots)
