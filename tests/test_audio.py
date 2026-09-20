import wave

import pytest

from pronunciation_oracle.audio import cut_clip, extract_audio, ffprobe_duration


def test_ffprobe_duration(tone_wav):
    duration = ffprobe_duration(tone_wav)
    assert 2.9 <= duration <= 3.1


def test_extract_audio_resamples(tmp_path, tone_wav):
    out = extract_audio(tone_wav, tmp_path / "out.wav", sample_rate=8000, channels=1)
    assert out.exists()
    with wave.open(str(out), "rb") as wf:
        assert wf.getframerate() == 8000
        assert wf.getnchannels() == 1


def test_cut_clip_accurate_duration(tmp_path, tone_wav):
    out = cut_clip(tone_wav, 0.5, 1.5, tmp_path / "clip.wav")
    duration = ffprobe_duration(out)
    assert 0.9 <= duration <= 1.1


def test_cut_clip_rejects_bad_range(tmp_path, tone_wav):
    with pytest.raises(ValueError):
        cut_clip(tone_wav, 2.0, 1.0, tmp_path / "clip.wav")
