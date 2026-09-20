"""Forced-alignment interface: reconcile ASR-derived timing with reference text.

When subtitles are supplied they are treated as the authoritative wording (they're
usually cleaner than raw ASR output) but they rarely carry per-word timing -- only
coarse per-line cue windows. An Aligner's job is to produce per-word timestamps for
the reference text that are anchored to the actual audio, using the ASR's
word-level timestamps as evidence of *where in time* each bit of speech happened.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..transcript import WordTiming


@dataclass
class ReferenceSegment:
    """A chunk of reference text with an optional known time window.

    For subtitle-derived references, (start, end) is the cue's on-screen window.
    For ASR-derived references there may be a single segment with no bounds,
    since the ASR words already carry their own timing.
    """

    text: str
    start: float | None = None
    end: float | None = None


class Aligner(ABC):
    """Interface for reconciling ASR-derived word timing with reference text.

    Implementations decide how to turn `reference` text plus the ASR's own
    word timestamps into one final, audio-anchored timestamp per reference
    token. See `SequenceAligner` for the default, dependency-light
    implementation, and `TorchaudioCTCAligner` for a true forced-alignment
    alternative.
    """

    @abstractmethod
    def align(
        self,
        asr_words: list[WordTiming],
        reference: list[ReferenceSegment],
        audio_duration: float | None = None,
        audio_path: str | None = None,
    ) -> list[WordTiming]:
        """Return one WordTiming per token in `reference`, in order, time-aligned to audio.

        `audio_path`, when given, points at the mono WAV extracted for this media
        file. ASR-anchor-based aligners can ignore it; aligners that re-derive
        timing straight from the waveform (e.g. a CTC forced aligner) need it.
        """
        raise NotImplementedError
