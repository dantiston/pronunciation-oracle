"""Word normalization and tokenization shared by ASR, alignment, and search."""

from __future__ import annotations

import re
import unicodedata

_PUNCT_RE = re.compile(r"[^\w'-]+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_word(word: str) -> str:
    """Fold a word down to a canonical form for matching across ASR/subtitle sources.

    Strips punctuation (keeping internal apostrophes/hyphens), casefolds, and
    normalizes unicode so e.g. "Pikachu!" / "pikachu" / "PIKACHU," all match.
    """
    word = unicodedata.normalize("NFKC", word)
    word = word.casefold()
    word = _PUNCT_RE.sub("", word)
    return word.strip("'-")


def tokenize(text: str) -> list[str]:
    """Split free text into whitespace-delimited tokens, dropping empties."""
    text = _WHITESPACE_RE.sub(" ", text.strip())
    return [tok for tok in text.split(" ") if tok]
