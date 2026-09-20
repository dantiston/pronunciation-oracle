"""End-to-end integration test against the real ASR model (no fakes/mocks).

Unlike the rest of the suite (which uses FakeASR + an ffmpeg sine tone to test
plumbing without needing a model download), this test:

  1. Synthesizes real speech with espeak-ng (offline TTS, no network needed).
  2. Runs the real FasterWhisperASR "small" model on it.
  3. Runs the real SequenceAligner against a subtitle file whose wording
     deliberately differs from what was spoken in places, to exercise actual
     gap interpolation against real (not synthetic) ASR timings.
  4. Indexes into a real Corpus, searches for a word, and extracts real clips
     with ffmpeg.
  5. Re-transcribes each extracted clip in isolation as a self-consistency
     check that the clip genuinely contains the target word, not just that a
     file of plausible length exists.

This requires downloading model weights from huggingface.co, which most CI/
sandbox environments block by policy. It is opt-in and skipped by default:

    RUN_REAL_ASR_TESTS=1 pytest tests/test_integration_real_asr.py -v -s
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from pronunciation_oracle.align.sequence_aligner import SequenceAligner
from pronunciation_oracle.asr.faster_whisper_backend import FasterWhisperASR
from pronunciation_oracle.clipper import extract_clips
from pronunciation_oracle.corpus import Corpus
from pronunciation_oracle.pipeline import transcribe_and_align
from pronunciation_oracle.text_norm import normalize_word, tokenize

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_ASR_TESTS") != "1",
    reason="opt-in: downloads a real whisper model from huggingface.co. Set RUN_REAL_ASR_TESTS=1 to run.",
)

# A short scripted "episode" with the target word said by three different
# "lines" (stand-ins for different characters/moments), plus other dialogue
# so the corpus search has to actually discriminate.
SCRIPT = (
    "Hey, Pikachu! I choose you. "
    "The trainer shouted, use thunderbolt now! "
    "A wild Squirtle appeared in the tall grass. "
    "Pikachu jumped and used its electric attack. "
    "Everyone cheered for Pikachu after the battle."
)

# Subtitle text deliberately differs from the spoken script in a few places
# (synonyms / rewording), so the aligner has to reconcile mismatched wording
# against the real ASR's word timings, not just copy them through.
SUBTITLE_SRT = """1
00:00:00,000 --> 00:00:04,000
Hey, Pikachu! I choose you!

2
00:00:04,000 --> 00:00:08,000
The trainer yelled, use thunderbolt right now!

3
00:00:08,000 --> 00:00:12,500
A wild Squirtle showed up in the tall grass.

4
00:00:12,500 --> 00:00:17,000
Pikachu leapt and unleashed its electric attack.

5
00:00:17,000 --> 00:00:21,000
Everyone cheered for Pikachu after the battle.
"""


@pytest.fixture(scope="module")
def speech_wav(tmp_path_factory):
    if shutil.which("espeak-ng") is None:
        pytest.skip("espeak-ng not available for TTS")
    out_dir = tmp_path_factory.mktemp("real_asr")
    wav_path = out_dir / "episode.wav"
    subprocess.run(
        ["espeak-ng", "-s", "150", "-w", str(wav_path), SCRIPT],
        check=True,
        capture_output=True,
    )
    return wav_path


@pytest.fixture(scope="module")
def asr_model():
    return FasterWhisperASR(model_size="small", device="cpu", compute_type="int8")


def test_real_asr_transcribes_the_target_word(speech_wav, asr_model):
    transcript = transcribe_and_align(speech_wav, subtitles_path=None, asr=asr_model)

    assert transcript.text_origin == "asr"
    assert transcript.words, "ASR returned no words at all"

    norm_words = [normalize_word(w.word) for w in transcript.words]
    pikachu_count = norm_words.count("pikachu")
    print(f"\n[real ASR] decoded text: {transcript.text!r}")
    print(f"[real ASR] 'pikachu' recognized {pikachu_count} time(s) of 3 spoken")
    assert pikachu_count >= 2, f"expected 'pikachu' recognized at least twice, got {norm_words}"

    # Timestamps must be sane: monotonic-ish and within the audio's duration.
    for w in transcript.words:
        assert 0.0 <= w.start <= w.end <= transcript.duration + 0.5


def test_real_alignment_against_mismatched_subtitles(tmp_path, speech_wav, asr_model):
    srt_path = tmp_path / "episode.srt"
    srt_path.write_text(SUBTITLE_SRT, encoding="utf-8")

    transcript = transcribe_and_align(
        speech_wav,
        subtitles_path=srt_path,
        asr=asr_model,
        aligner=SequenceAligner(),
    )

    assert transcript.text_origin == "subtitles"
    # Output text must be exactly the subtitle wording (not the ASR's own words).
    assert "yelled" in transcript.text
    assert "leapt" in transcript.text
    assert "unleashed" in transcript.text

    words = transcript.words
    assert [normalize_word(w.word) for w in words].count("pikachu") == 3

    # Whole-transcript timeline must stay ordered and inside the audio.
    for a, b in zip(words, words[1:]):
        assert a.start <= a.end
        assert a.start <= b.start
    assert words[-1].end <= transcript.duration + 0.5

    # Words that matched the ASR exactly should carry a real confidence score;
    # only the reworded ("yelled"/"leapt"/"unleashed") ones should be interpolated (confidence=None).
    matched_confidences = [w.confidence for w in words if normalize_word(w.word) == "pikachu"]
    assert all(c is not None for c in matched_confidences), "exact word matches should inherit ASR confidence"


def test_search_and_clip_then_reverify_by_reasr(tmp_path, speech_wav, asr_model):
    transcript = transcribe_and_align(speech_wav, subtitles_path=None, asr=asr_model)

    with Corpus(tmp_path / "corpus.db") as corpus:
        corpus.add_transcript(transcript)
        hits = corpus.search("pikachu")

    assert len(hits) >= 2
    for hit in hits:
        print(f"[hit] {hit.start:.2f}-{hit.end:.2f}s  ...{hit.context_before} [{hit.word}] {hit.context_after}...")

    clips_dir = tmp_path / "clips"
    results = extract_clips(hits, clips_dir, pad=0.2, fmt="wav")
    assert len(results) == len(hits)

    # Self-consistency check: re-run real ASR on each extracted clip in
    # isolation and confirm the target word is actually audible in it, not
    # just that a plausibly-sized file exists at that offset.
    reheard = 0
    for clip in results:
        clip_result = asr_model.transcribe(clip.output_path)
        clip_norm_words = {normalize_word(tok) for tok in tokenize(clip_result.text)}
        print(f"[reverify] {clip.output_path} -> {clip_result.text!r}")
        if "pikachu" in clip_norm_words:
            reheard += 1
    assert reheard >= len(results) - 1, "most extracted clips should still be recognized as 'pikachu' in isolation"
