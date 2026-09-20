"""ASR backend built on faster-whisper (CTranslate2 Whisper inference).

Chosen as the default real backend because it ships word-level timestamps out
of the box (via cross-attention + DTW), needs no torch/GPU, and is fast on CPU
with int8 quantization -- a good fit for batch-processing a media corpus.
"""

from __future__ import annotations

from pathlib import Path

from ..transcript import WordTiming
from .base import ASRBackend, ASRResult


class FasterWhisperASR(ASRBackend):
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        **model_kwargs,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model_kwargs = model_kwargs
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - exercised only when extra missing
                raise ImportError(
                    "faster-whisper is required for FasterWhisperASR. "
                    "Install it with `pip install pronunciation-oracle[asr]`."
                ) from exc
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                **self._model_kwargs,
            )
        return self._model

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> ASRResult:
        model = self._load_model()
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
        )
        words: list[WordTiming] = []
        text_parts: list[str] = []
        for segment in segments:
            segment_text = (segment.text or "").strip()
            if segment_text:
                text_parts.append(segment_text)
            for w in segment.words or []:
                word_text = (w.word or "").strip()
                if not word_text:
                    continue
                words.append(
                    WordTiming(
                        word=word_text,
                        start=float(w.start),
                        end=float(w.end),
                        confidence=float(w.probability) if w.probability is not None else None,
                    )
                )
        return ASRResult(text=" ".join(text_parts), words=words, language=getattr(info, "language", language))
