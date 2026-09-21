from unittest.mock import patch

import pytest

from pronunciation_oracle.asr.mlx_whisper_backend import MlxWhisperASR, mlx_available


def test_known_model_sizes_map_to_mlx_community_repos():
    assert MlxWhisperASR("small")._path_or_hf_repo == "mlx-community/whisper-small-mlx"
    assert MlxWhisperASR("tiny")._path_or_hf_repo == "mlx-community/whisper-tiny-mlx"
    assert MlxWhisperASR("large-v3")._path_or_hf_repo == "mlx-community/whisper-large-v3-mlx"


def test_unknown_model_size_passed_through_as_a_repo_id():
    custom = "mlx-community/whisper-large-v3-turbo"
    assert MlxWhisperASR(custom)._path_or_hf_repo == custom


def test_mlx_available_true_on_apple_silicon_macos():
    with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
        assert mlx_available() is True


@pytest.mark.parametrize(
    ("system", "machine"), [("Darwin", "x86_64"), ("Linux", "arm64"), ("Windows", "AMD64")]
)
def test_mlx_available_false_off_apple_silicon_macos(system, machine):
    with patch("platform.system", return_value=system), patch("platform.machine", return_value=machine):
        assert mlx_available() is False


def test_transcribe_raises_clear_error_when_not_apple_silicon():
    with patch("pronunciation_oracle.asr.mlx_whisper_backend.mlx_available", return_value=False):
        with pytest.raises(ImportError, match="Apple Silicon"):
            MlxWhisperASR().transcribe("audio.wav")
