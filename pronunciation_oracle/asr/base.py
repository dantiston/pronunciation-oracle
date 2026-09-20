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
    """The raw output of one ASR transcription pass."""

    text: str
    words: list[WordTiming] = field(default_factory=list)
    language: str | None = None


class ASRBackend(ABC):
    """Interface for turning an audio file into word-level timestamps.

    Any backend just needs to implement `transcribe`; the pipeline depends only
    on this interface, so alternative engines (whisper.cpp, a cloud ASR API, ...)
    can be dropped in without touching alignment/corpus/clip code.
    """

    @abstractmethod
    def transcribe(self, audio_path: str | Path, language: str | None = None) -> ASRResult:
        """Transcribe a mono WAV file and return word-level timestamps.

        Args:
            audio_path: Path to a mono WAV file (see `audio.extract_audio`).
            language: Force a language code (e.g. "en"); None auto-detects.

        Returns:
            The decoded text plus one WordTiming per recognized word, in seconds.
        """
        raise NotImplementedError
