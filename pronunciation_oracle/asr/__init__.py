"""ASR backends: turn an audio file into word-level timestamps."""

from .base import ASRBackend, ASRResult
from .fake import FakeASR
from .faster_whisper_backend import FasterWhisperASR
from .mlx_whisper_backend import MlxWhisperASR

__all__ = ["ASRBackend", "ASRResult", "FakeASR", "FasterWhisperASR", "MlxWhisperASR"]
