import shutil
import subprocess

import pytest


@pytest.fixture
def tone_wav(tmp_path):
    """A real 3-second mono WAV, generated with ffmpeg, for exercising audio.py/pipeline.py."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    path = tmp_path / "tone.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=3",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return path
