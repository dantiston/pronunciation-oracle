"""Optional true forced aligner using torchaudio's CTC (MMS) alignment API.

This gives phoneme/word-level timing directly from an acoustic model instead
of interpolating around ASR anchors (see SequenceAligner), at the cost of a
torch/torchaudio dependency and a model download. Requires the `align-ctc`
extra: `pip install pronunciation-oracle[align-ctc]`.

Not wired in as the default because it's heavy -- and unlike swapping the ASR
backend (see asr.mlx_whisper_backend), "heavy" isn't buying more accuracy
here to trade against. On the same battle-scene test used throughout this
project (weak `tiny`-model ASR anchors for SequenceAligner, forcing it to
actually reconcile mismatched text, vs this aligner's forced alignment
straight from text+audio, ignoring ASR entirely): SequenceAligner landed
15/21 word timestamps within 0.5s of ground truth in ~0.00s; this aligner
landed 14/21 in 45.5s. No accuracy edge found, and roughly 1000x slower.
Select it explicitly via `--align-backend ctc` on the CLI, or
`TorchaudioCTCAligner()` in code, but there's no evidence yet that you
should prefer it over the default for content like this.
"""

from __future__ import annotations

import os
from typing import Any

from ..text_norm import normalize_word, tokenize
from ..transcript import WordTiming
from .base import Aligner, ReferenceSegment


def _resolve_auto_device(torch: Any) -> str:
    """Pick the best available torch device: cuda, then mps, then cpu.

    Same underlying model and math regardless of device -- this only changes
    where the computation runs, not what it computes, so unlike swapping ASR
    models there's no accuracy tradeoff to weigh here. Verified: cpu and mps
    produce matching word timestamps on the same input. mps does need one
    thing handled for correctness, not just speed -- see `_load()` -- because
    torchaudio's forced_align op isn't implemented for MPS as of current torch
    and crashes without it; measured real speedup even with that op's forced
    CPU fallback was modest (~30-40% on a short clip), not dramatic, since
    that fallback affects however much of the run forced_align accounts for.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchaudioCTCAligner(Aligner):
    """Forced-aligns reference text directly to the waveform via torchaudio MMS_FA.

    Unlike SequenceAligner, this ignores the ASR's word timestamps entirely and
    re-derives timing from the acoustic model's frame-level CTC alignment of
    the reference text against the raw audio -- a proper forced alignment.
    """

    def __init__(self, device: str = "auto"):
        """Configure the aligner.

        Args:
            device: "auto" (default) picks cuda, then mps (Apple Silicon GPU
                via Metal), then cpu -- whichever is actually available on
                this machine. Or force one explicitly ("cpu"/"cuda"/"mps").
                Resolved lazily on first use, since checking availability
                requires importing torch.
        """
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
        # torchaudio's forced_align op isn't implemented for MPS as of current torch
        # (crashes without this); with it set, that one op falls back to CPU while
        # the rest (the acoustic model's forward pass) still runs on MPS. PyTorch
        # reads this at `import torch` time, so it must be set before that, not
        # merely before the first MPS op -- setting it later is silently too late.
        # Harmless to set unconditionally even when the resolved device isn't mps.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            import torch  # noqa: PLC0415 -- deliberately lazy: torch is an optional extra  # ty: ignore[unresolved-import]
            import torchaudio  # noqa: PLC0415 -- ditto  # ty: ignore[unresolved-import]
        except ImportError as exc:  # pragma: no cover - exercised only when extra missing
            raise ImportError(
                "torch/torchaudio are required for TorchaudioCTCAligner. "
                "Install them with `pip install pronunciation-oracle[align-ctc]`."
            ) from exc
        if self._device == "auto":
            self._device = _resolve_auto_device(torch)
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
