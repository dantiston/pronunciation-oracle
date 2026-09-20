"""Optional true forced aligner using torchaudio's CTC (MMS) alignment API.

This gives phoneme/word-level timing directly from an acoustic model instead
of interpolating around ASR anchors (see SequenceAligner), at the cost of a
torch/torchaudio dependency and a model download. Requires the `align-ctc`
extra: `pip install pronunciation-oracle[align-ctc]`.

Not wired in as the default because it's heavy; select it explicitly via
`--align-backend ctc` on the CLI, or `TorchaudioCTCAligner()` in code.
"""

from __future__ import annotations

from typing import Any

from ..text_norm import normalize_word, tokenize
from ..transcript import WordTiming
from .base import Aligner, ReferenceSegment


class TorchaudioCTCAligner(Aligner):
    """Forced-aligns reference text directly to the waveform via torchaudio MMS_FA.

    Unlike SequenceAligner, this ignores the ASR's word timestamps entirely and
    re-derives timing from the acoustic model's frame-level CTC alignment of
    the reference text against the raw audio -- a proper forced alignment.
    """

    def __init__(self, device: str = "cpu"):
        self._device = device
        # Populated lazily by `_load()`; typed Any since torch/torchaudio are an
        # optional dependency (align-ctc extra) and may not be installed/resolvable.
        self._torch: Any = None
        self._torchaudio: Any = None
        self._bundle: Any = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._aligner: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415 -- deliberately lazy: torch is an optional extra  # ty: ignore[unresolved-import]
            import torchaudio  # noqa: PLC0415 -- ditto  # ty: ignore[unresolved-import]
        except ImportError as exc:  # pragma: no cover - exercised only when extra missing
            raise ImportError(
                "torch/torchaudio are required for TorchaudioCTCAligner. "
                "Install them with `pip install pronunciation-oracle[align-ctc]`."
            ) from exc
        self._torch = torch
        self._torchaudio = torchaudio
        self._bundle = torchaudio.pipelines.MMS_FA
        self._model = self._bundle.get_model().to(self._device)
        self._tokenizer = self._bundle.get_tokenizer()
        self._aligner = self._bundle.get_aligner()

    def align(
        self,
        asr_words: list[WordTiming],
        reference: list[ReferenceSegment],
        audio_duration: float | None = None,
        audio_path: str | None = None,
    ) -> list[WordTiming]:
        """Forced-align reference text to the waveform at `audio_path` via CTC.

        Args:
            asr_words: Unused by this aligner (kept for interface compatibility);
                timing is re-derived from the waveform instead of ASR anchors.
            reference: The authoritative text to time-align, as one or more segments.
            audio_duration: Unused by this aligner.
            audio_path: Path to the mono WAV to align against. Required.

        Returns:
            One WordTiming per token in `reference`, in order, with a CTC
            per-token confidence score.

        Raises:
            ValueError: If `audio_path` is not given.
        """
        del asr_words, audio_duration  # unused: this aligner derives timing from the waveform directly
        if audio_path is None:
            raise ValueError("TorchaudioCTCAligner.align requires audio_path=<wav path>")
        self._load()
        torch = self._torch
        torchaudio = self._torchaudio

        raw_tokens: list[str] = []
        for seg in reference:
            raw_tokens.extend(tokenize(seg.text))
        norm_tokens = [normalize_word(t) or t for t in raw_tokens]
        if not norm_tokens:
            return []

        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self._bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self._bundle.sample_rate)
        waveform = waveform.mean(dim=0, keepdim=True).to(self._device)

        with torch.inference_mode():
            emission, _ = self._model(waveform)
            token_spans = self._aligner(emission[0], self._tokenizer(norm_tokens))

        num_frames = emission.shape[1]
        ratio = waveform.shape[1] / num_frames / self._bundle.sample_rate

        words: list[WordTiming] = []
        for raw, spans in zip(raw_tokens, token_spans, strict=True):
            start = spans[0].start * ratio
            end = spans[-1].end * ratio
            score = sum(s.score for s in spans) / max(len(spans), 1)
            words.append(WordTiming(word=raw, start=float(start), end=float(end), confidence=float(score)))
        return words
