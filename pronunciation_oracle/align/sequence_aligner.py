"""Default Aligner: reconcile ASR timing with reference text via sequence matching.

This is a dependency-light forced-alignment approximation. It does not need a
phoneme model (wav2vec2/CTC) -- it treats the ASR's own word-level timestamps
as noisy but locally-accurate "anchors" in time, and stretches the reference
text onto those anchors. A true CTC forced aligner can be dropped in behind
the same `Aligner` interface for higher-precision phoneme-level timing; see
align/base.py.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from ..text_norm import normalize_word, tokenize
from ..transcript import WordTiming
from .base import Aligner, ReferenceSegment


@dataclass
class _RefToken:
    raw: str
    norm: str
    seg_index: int


class SequenceAligner(Aligner):
    """Anchors reference-text words to ASR word timestamps by fuzzy sequence match.

    - Words that appear in both the ASR output and the reference text (same
      normalized spelling, in order) inherit the ASR word's exact start/end
      and confidence.
    - Words present only in the reference text (e.g. a proper noun the ASR
      misheard) get an interpolated timestamp, spread proportionally to word
      length across the gap between the nearest matched neighbors. When a
      gap touches a segment boundary (e.g. a subtitle cue) with no matched
      neighbor on that side, the segment's own start/end bounds it instead,
      so interpolation never drifts into a neighboring line.
    """

    def align(
        self,
        asr_words: list[WordTiming],
        reference: list[ReferenceSegment],
        audio_duration: float | None = None,
        audio_path: str | None = None,
    ) -> list[WordTiming]:
        ref_tokens: list[_RefToken] = []
        for seg_index, seg in enumerate(reference):
            for tok in tokenize(seg.text):
                ref_tokens.append(_RefToken(raw=tok, norm=normalize_word(tok), seg_index=seg_index))

        if not ref_tokens:
            return []

        asr_norm = [normalize_word(w.word) for w in asr_words]
        ref_norm = [t.norm for t in ref_tokens]

        matcher = difflib.SequenceMatcher(a=asr_norm, b=ref_norm, autojunk=False)
        result: list[WordTiming | None] = [None] * len(ref_tokens)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                continue
            for offset in range(i2 - i1):
                asr_word = asr_words[i1 + offset]
                ref_tok = ref_tokens[j1 + offset]
                result[j1 + offset] = WordTiming(
                    word=ref_tok.raw,
                    start=asr_word.start,
                    end=asr_word.end,
                    confidence=asr_word.confidence,
                )

        self._fill_gaps(result, ref_tokens, reference, audio_duration)
        return [w for w in result if w is not None]  # type: ignore[misc]

    def _fill_gaps(
        self,
        result: list[WordTiming | None],
        ref_tokens: list[_RefToken],
        reference: list[ReferenceSegment],
        audio_duration: float | None,
    ) -> None:
        n = len(result)
        i = 0
        while i < n:
            if result[i] is not None:
                i += 1
                continue
            j = i
            while j < n and result[j] is None:
                j += 1
            # Gap of unmatched reference tokens spans [i, j).
            prev_time = result[i - 1].end if i > 0 else None
            next_time = result[j].start if j < n else None
            if prev_time is None:
                prev_time = reference[ref_tokens[i].seg_index].start
            if next_time is None:
                next_time = reference[ref_tokens[j - 1].seg_index].end

            gap_len = j - i
            if prev_time is None and next_time is None:
                prev_time = 0.0
                next_time = audio_duration if audio_duration is not None else float(gap_len)
            elif prev_time is None:
                prev_time = max(0.0, next_time - 0.4 * gap_len)
            elif next_time is None:
                next_time = prev_time + 0.4 * gap_len
            if next_time < prev_time:
                next_time = prev_time

            self._distribute(result, ref_tokens, i, j, prev_time, next_time)
            i = j

    @staticmethod
    def _distribute(
        result: list[WordTiming | None],
        ref_tokens: list[_RefToken],
        i: int,
        j: int,
        window_start: float,
        window_end: float,
    ) -> None:
        span = max(window_end - window_start, 1e-6)
        weights = [max(len(ref_tokens[k].norm), 1) for k in range(i, j)]
        total_weight = sum(weights)
        cursor = window_start
        for offset, k in enumerate(range(i, j)):
            share = span * (weights[offset] / total_weight)
            start = cursor
            end = min(window_end, start + share) if k != j - 1 else window_end
            end = max(end, start)
            result[k] = WordTiming(word=ref_tokens[k].raw, start=start, end=end, confidence=None)
            cursor = end
