from pronunciation_oracle.asr.base import ASRResult
from pronunciation_oracle.asr.fake import FakeASR
from pronunciation_oracle.pipeline import transcribe_and_align
from pronunciation_oracle.transcript import WordTiming


def test_pipeline_without_subtitles_uses_asr_words_directly(tone_wav):
    asr_words = [
        WordTiming("hey", 0.5, 0.8, 0.9),
        WordTiming("pikachu", 0.9, 1.4, 0.95),
    ]
    asr = FakeASR(result=ASRResult(text="hey pikachu", words=asr_words, language="en"))

    transcript = transcribe_and_align(tone_wav, subtitles_path=None, asr=asr)

    assert transcript.text_origin == "asr"
    assert [w.word for w in transcript.words] == ["hey", "pikachu"]
    assert transcript.words[0].start == 0.5
    assert 2.9 <= transcript.duration <= 3.1
    assert transcript.source_path == str(tone_wav)


def test_pipeline_with_subtitles_reconciles_text_and_timing(tmp_path, tone_wav):
    asr_words = [
        WordTiming("hey", 0.5, 0.8, 0.9),
        WordTiming("pikachu", 0.9, 1.4, 0.95),
    ]
    asr = FakeASR(result=ASRResult(text="hey pikachu", words=asr_words, language="en"))

    srt_path = tmp_path / "sub.srt"
    srt_path.write_text(
        "1\n00:00:00,400 --> 00:00:01,500\nHey, Pikachu!\n",
        encoding="utf-8",
    )

    transcript = transcribe_and_align(tone_wav, subtitles_path=srt_path, asr=asr)

    assert transcript.text_origin == "subtitles"
    assert [w.word for w in transcript.words] == ["Hey,", "Pikachu!"]
    # Matched exactly against ASR anchors, so timing should be inherited verbatim.
    assert transcript.words[0].start == 0.5
    assert transcript.words[1].start == 0.9
