from itertools import pairwise

from pronunciation_oracle.align.base import ReferenceSegment
from pronunciation_oracle.align.sequence_aligner import SequenceAligner
from pronunciation_oracle.transcript import WordTiming


def _asr_words():
    return [
        WordTiming("hey", 1.0, 1.2, 0.9),
        WordTiming("pikachu", 1.3, 1.8, 0.95),
        WordTiming("use", 2.6, 2.8, 0.9),
        WordTiming("thunder", 2.9, 3.2, 0.6),
        WordTiming("bolt", 3.2, 3.5, 0.6),
    ]


def test_matched_words_inherit_exact_asr_timing():
    aligner = SequenceAligner()
    reference = [
        ReferenceSegment(text="Hey, Pikachu!", start=1.0, end=2.0),
        ReferenceSegment(text="Use Thunderbolt now!", start=2.5, end=4.0),
    ]
    result = aligner.align(_asr_words(), reference)

    words = {w.word: w for w in result}
    assert words["Hey,"].start == 1.0
    assert words["Hey,"].end == 1.2
    assert words["Hey,"].confidence == 0.9
    assert words["Pikachu!"].start == 1.3
    assert words["Pikachu!"].confidence == 0.95
    assert words["Use"].start == 2.6
    assert words["Use"].end == 2.8


def test_unmatched_subtitle_word_is_interpolated_within_bounds():
    aligner = SequenceAligner()
    reference = [
        ReferenceSegment(text="Hey, Pikachu!", start=1.0, end=2.0),
        ReferenceSegment(text="Use Thunderbolt now!", start=2.5, end=4.0),
    ]
    result = aligner.align(_asr_words(), reference)
    words = {w.word: w for w in result}

    # "Thunderbolt"/"now!" have no exact ASR match (ASR heard "thunder"+"bolt"),
    # so they must be interpolated -- bounded by the last matched anchor ("Use"
    # ends at 2.8) and the enclosing subtitle line's own end (4.0).
    thunderbolt = words["Thunderbolt"]
    now = words["now!"]
    assert thunderbolt.confidence is None
    assert now.confidence is None
    assert thunderbolt.start >= 2.8
    assert thunderbolt.start < thunderbolt.end
    assert now.end == 4.0
    assert thunderbolt.end <= now.start


def test_preserves_reference_word_order():
    aligner = SequenceAligner()
    reference = [ReferenceSegment(text="Hey, Pikachu!", start=1.0, end=2.0)]
    result = aligner.align(_asr_words(), reference)
    assert [w.word for w in result] == ["Hey,", "Pikachu!"]


def test_falls_back_to_audio_duration_when_nothing_matches():
    aligner = SequenceAligner()
    asr_words = [WordTiming("static", 0.0, 3.0, 0.5)]
    reference = [ReferenceSegment(text="completely different words", start=None, end=None)]
    result = aligner.align(asr_words, reference, audio_duration=6.0)
    assert len(result) == 3
    assert result[0].start >= 0.0
    assert result[-1].end <= 6.0
    # Monotonic non-decreasing timeline.
    for a, b in pairwise(result):
        assert a.end <= b.start + 1e-9


def test_empty_reference_returns_empty():
    aligner = SequenceAligner()
    result = aligner.align(_asr_words(), [])
    assert result == []
