"""A deterministic ASR backend for tests and offline demos (no model download).

Also useful for wiring the pipeline/corpus/clipper up against a known ground
truth when validating the alignment algorithm.
"""

from __future__ import annotations

from pathlib import Path

from ..transcript import WordTiming
from .base import ASRBackend, ASRResult


class FakeASR(ASRBackend):
    def __init__(self, result: ASRResult | None = None, words: list[WordTiming] | None = None):
        if result is not None and words is not None:
            raise ValueError("pass either result= or words=, not both")
        if result is not None:
            self._result = result
        elif words is not None:
            self._result = ASRResult(text=" ".join(w.word for w in words), words=words, language="en")
        else:
            self._result = ASRResult(text="", words=[], language="en")

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> ASRResult:
        return self._result
