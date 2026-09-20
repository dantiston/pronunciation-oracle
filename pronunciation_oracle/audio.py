"""ffmpeg/ffprobe wrappers for audio extraction, duration probing, and clip cutting.

Works on both video and audio containers -- ffmpeg is used to pull out (and,
for clips, re-encode) just the audio stream, so the rest of the pipeline never
has to care whether the source was an .mp4 or a .wav.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class FfmpegNotFoundError(RuntimeError):
    pass


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FfmpegNotFoundError(
            f"'{name}' was not found on PATH. Install ffmpeg (e.g. `apt-get install ffmpeg`)."
        )
    return path


def ffprobe_duration(path: str | Path) -> float:
    """Return the duration of a media file (audio or video) in seconds."""
    ffprobe = _require_binary("ffprobe")
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    duration = data.get("format", {}).get("duration")
    if duration is None:
        raise RuntimeError(f"ffprobe returned no duration for {path}: {proc.stdout}")
    return float(duration)


def extract_audio(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Extract (and resample/downmix) the audio track of a media file to a WAV file.

    Used to produce a clean, ASR-friendly mono 16 kHz WAV regardless of the
    source container/codec.
    """
    ffmpeg = _require_binary("ffmpeg")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-f", "wav",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed for {input_path}: {proc.stderr.strip()}")
    return output_path


def cut_clip(
    input_path: str | Path,
    start: float,
    end: float,
    output_path: str | Path,
    fmt: str = "wav",
    sample_rate: int | None = None,
) -> Path:
    """Cut a single accurate audio-only clip [start, end) (seconds) from a media file."""
    if end <= start:
        raise ValueError(f"end ({end}) must be greater than start ({start})")
    ffmpeg = _require_binary("ffmpeg")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start:.3f}",
        "-i", str(input_path),
        "-t", f"{duration:.3f}",
        "-vn",
    ]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    if fmt == "mp3":
        cmd += ["-codec:a", "libmp3lame", "-qscale:a", "2"]
    elif fmt == "flac":
        cmd += ["-codec:a", "flac"]
    # wav (and any other container) fall through to ffmpeg's default codec.
    cmd += [str(output_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg clip extraction failed for {input_path}: {proc.stderr.strip()}")
    return output_path
