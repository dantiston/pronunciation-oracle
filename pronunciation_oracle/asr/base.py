"""ASR backend interface.

Any backend just needs to turn an audio file into word-level timestamps.
The pipeline depends only on this interface, so alternative engines (whisper.cpp,
a cloud ASR API, ...) can be dropped in without touching alignment/corpus/clip code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..transcript import WordTiming


@dataclass
class ASRResult:
    text: str
    words: list[WordTiming] = field(default_factory=list)
    language: str | None = None


class ASRBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str | Path, language: str | None = None) -> ASRResult:
        """Transcribe a mono WAV file and return word-level timestamps (seconds)."""
        raise NotImplementedError
