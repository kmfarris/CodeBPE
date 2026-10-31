"""PRCF Phase A: classical positioned fingerprints."""

from prcf.fingerprint import positioned_trigram_fingerprint, slot_shingle_sets
from prcf.localize import localize_edit
from prcf.tokenize import tokenize

__all__ = [
    "tokenize",
    "positioned_trigram_fingerprint",
    "slot_shingle_sets",
    "localize_edit",
]
