"""Orchestrates step (1) of the system: media in, time-aligned Transcript out.

audio/video --ffmpeg--> 16kHz mono wav --ASR--> word timestamps
                                                    |
                    (optional) subtitle file --parse--> reference text
                                                    |
                                                 Aligner --> Transcript
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .align.base import Aligner, ReferenceSegment
from .align.sequence_aligner import SequenceAligner
from .asr.base import ASRBackend
from .asr.faster_whisper_backend import FasterWhisperASR
from .audio import extract_audio, ffprobe_duration
from .subtitles import parse_subtitles
from .transcript import Transcript


def transcribe_and_align(
    media_path: str | Path,
    subtitles_path: str | Path | None = None,
    asr: ASRBackend | None = None,
    aligner: Aligner | None = None,
    language: str | None = None,
) -> Transcript:
    """Run ASR (and, if subtitles are given, alignment against them) on one media file.

    - No subtitles: the ASR's own word timestamps are the transcript. faster-whisper
      already derives these via forced alignment of its decoded text against audio
      (cross-attention + DTW), so no further alignment step is needed.
    - Subtitles given: they're treated as the authoritative wording. The Aligner
      reconciles ASR word timing with the subtitle text so the output carries the
      subtitle's (cleaner, human-authored) words with audio-accurate timestamps.
    """
    media_path = Path(media_path)
    asr = asr or FasterWhisperASR()
    aligner = aligner or SequenceAligner()

    with tempfile.TemporaryDirectory(prefix="pronunciation-oracle-") as tmp_dir:
        wav_path = extract_audio(media_path, Path(tmp_dir) / "audio.wav")
        duration = ffprobe_duration(media_path)
        asr_result = asr.transcribe(wav_path, language=language)

        if subtitles_path is not None:
            lines = parse_subtitles(subtitles_path)
            reference = [ReferenceSegment(text=line.text, start=line.start, end=line.end) for line in lines]
            words = aligner.align(
                asr_result.words, reference, audio_duration=duration, audio_path=str(wav_path)
            )
            text_origin = "subtitles"
        else:
            words = asr_result.words
            text_origin = "asr"

    return Transcript(
        source_path=str(media_path),
        duration=duration,
        language=asr_result.language,
        words=words,
        text_origin=text_origin,
    )
