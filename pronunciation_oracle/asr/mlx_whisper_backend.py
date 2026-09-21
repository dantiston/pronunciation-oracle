"""ASR backend using mlx-whisper (Apple's MLX framework, Metal-accelerated).

Apple-Silicon-only -- MLX doesn't run on Intel Macs, Linux, or Windows.
Requires the `mlx` extra: `pip install pronunciation-oracle[mlx]`. Selected
automatically by `--asr-backend auto` (the CLI default) on a machine where
`mlx_available()` and `mlx_whisper_installed()` are both true; explicit
`--asr-backend faster-whisper` always overrides that.

Real measured numbers on an Apple M5, "small" model, 5-minute real clip:
3.9x faster than FasterWhisperASR's CPU int8 "small" (6.6s vs 25.9s). But a
real accuracy caveat found in that same validation, not just a speed win:
on a battle-scene segment where FasterWhisperASR's "small" correctly
recognized all 21 real spoken instances of a repeated proper noun,
mlx-whisper's "small" recognized only 11 (misheard several as short
fragments), and mlx-whisper's "medium" -- 8x slower -- did even worse (9/21).
Bigger isn't a fix here; this looks specific to the MLX conversion/decoding,
not model capacity. `--asr-backend auto` accepts this recall cost as a
knowing default (the user's explicit call after seeing these numbers); pass
`--asr-backend faster-whisper` if that tradeoff isn't right for your content.
"""

from __future__ import annotations

import importlib.util
import platform
from pathlib import Path

from ..transcript import WordTiming
from .base import ASRBackend, ASRResult

_MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


def mlx_available() -> bool:
    """Whether this machine can plausibly run mlx-whisper (Apple Silicon macOS).

    Checks the platform only, not whether the `mlx` extra is actually
    installed -- see `mlx_whisper_installed()` for that.
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def mlx_whisper_installed() -> bool:
    """Whether the `mlx_whisper` package is importable, without importing it.

    Used by `--asr-backend auto` to fall back to FasterWhisperASR when this
    machine is Apple Silicon but the `mlx` extra hasn't been installed,
    rather than picking mlx-whisper and only failing once transcription runs.
    """
    return importlib.util.find_spec("mlx_whisper") is not None


class MlxWhisperASR(ASRBackend):
    """ASR backend using mlx-whisper (Apple's MLX framework, Metal-accelerated).

    Apple-Silicon-only. See the module docstring for a real measured
    speed/accuracy tradeoff versus FasterWhisperASR -- validate against your
    own content before relying on it for search recall.
    """

    def __init__(self, model_size: str = "small") -> None:
        """Configure the backend.

        Args:
            model_size: A key in tiny/base/small/medium/large-v3, mapped to
                an mlx-community/whisper-<size>-mlx repo, or any other
                "org/repo" string naming a specific MLX-converted model.
        """
        self._path_or_hf_repo = _MODEL_REPOS.get(model_size, model_size)

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> ASRResult:
        """Transcribe a mono WAV file with mlx-whisper.

        Args:
            audio_path: Path to a mono WAV file (see `audio.extract_audio`).
            language: Force a language code (e.g. "en"); None auto-detects.

        Returns:
            The decoded text plus one WordTiming per recognized word.

        Raises:
            ImportError: If this isn't Apple Silicon, or mlx-whisper isn't installed.
        """
        if not mlx_available():
            raise ImportError(
                "mlx-whisper requires Apple Silicon (arm64 macOS); this machine is "
                f"{platform.system()}/{platform.machine()}. Use --asr-backend faster-whisper instead."
            )
        try:
            import mlx_whisper  # noqa: PLC0415 -- deliberately lazy: Apple-Silicon-only optional extra  # ty: ignore[unresolved-import]
        except ImportError as exc:
            raise ImportError(
                "mlx-whisper is required for MlxWhisperASR. "
                "Install it with `pip install pronunciation-oracle[mlx]`."
            ) from exc

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self._path_or_hf_repo,
            language=language,
            word_timestamps=True,
        )
        words: list[WordTiming] = []
        text_parts: list[str] = []
        for segment in result.get("segments", []):
            segment_text = (segment.get("text") or "").strip()
            if segment_text:
                text_parts.append(segment_text)
            for w in segment.get("words") or []:
                word_text = (w.get("word") or "").strip()
                if not word_text:
                    continue
                probability = w.get("probability")
                words.append(
                    WordTiming(
                        word=word_text,
                        start=float(w["start"]),
                        end=float(w["end"]),
                        confidence=float(probability) if probability is not None else None,
                    )
                )
        return ASRResult(text=" ".join(text_parts), words=words, language=result.get("language", language))
