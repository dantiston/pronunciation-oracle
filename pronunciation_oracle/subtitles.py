"""Parsers for SRT and WebVTT subtitle files.

Subtitles are optional reference text: when supplied, the pipeline treats them
as the authoritative transcript and uses the alignment step to snap ASR-derived
timing onto the subtitle words instead of trusting the ASR's own wording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TAG_RE = re.compile(r"<[^>]+>")
_ASS_TAG_RE = re.compile(r"\{[^}]*\}")
_SRT_TIME_RE = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})"
)


@dataclass
class SubtitleLine:
    """One cue from a subtitle file, in seconds."""

    index: int
    start: float
    end: float
    text: str


def _clean_text(raw_lines: list[str]) -> str:
    text = " ".join(line.strip() for line in raw_lines if line.strip())
    text = _TAG_RE.sub("", text)
    text = _ASS_TAG_RE.sub("", text)
    # Strip a leading "SPEAKER:" label if present, e.g. "ASH: Pikachu, go!"
    text = re.sub(r"^\s*[A-Z][A-Z0-9 _'-]{0,24}:\s+", "", text)
    return text.strip()


def _timestamp_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(text: str) -> list[SubtitleLine]:
    """Parse SRT-formatted subtitle text into a list of SubtitleLine."""
    lines: list[SubtitleLine] = []
    blocks = re.split(r"\r?\n\r?\n+", text.strip())
    auto_index = 0
    for block in blocks:
        block_lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if not block_lines:
            continue
        time_line_idx = None
        for i, ln in enumerate(block_lines):
            if _SRT_TIME_RE.search(ln):
                time_line_idx = i
                break
        if time_line_idx is None:
            continue
        match = _SRT_TIME_RE.search(block_lines[time_line_idx])
        assert match is not None
        start = _timestamp_to_seconds(*match.group(1, 2, 3, 4))
        end = _timestamp_to_seconds(*match.group(5, 6, 7, 8))
        raw_index = block_lines[0].strip() if time_line_idx == 1 else None
        text_lines = block_lines[time_line_idx + 1 :]
        cue_text = _clean_text(text_lines)
        if not cue_text:
            continue
        auto_index += 1
        try:
            index = int(raw_index) if raw_index is not None else auto_index
        except ValueError:
            index = auto_index
        lines.append(SubtitleLine(index=index, start=start, end=end, text=cue_text))
    return lines


def parse_vtt(text: str) -> list[SubtitleLine]:
    """Parse WebVTT-formatted subtitle text into a list of SubtitleLine."""
    text = re.sub(r"^﻿?WEBVTT.*?(\r?\n\r?\n|\r?\n(?=\d)|\r?\n(?=\d\d:))", "", text, count=1, flags=re.DOTALL)
    # Fall back: just strip the WEBVTT header line if the above didn't match.
    text = re.sub(r"^﻿?WEBVTT[^\n]*\n", "", text)
    blocks = re.split(r"\r?\n\r?\n+", text.strip())
    auto_index = 0
    lines: list[SubtitleLine] = []
    for block in blocks:
        block_lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if not block_lines:
            continue
        time_line_idx = None
        for i, ln in enumerate(block_lines):
            if "-->" in ln:
                time_line_idx = i
                break
        if time_line_idx is None:
            continue
        time_line = block_lines[time_line_idx].split("-->")
        if len(time_line) != 2:
            continue
        start = _parse_vtt_timestamp(time_line[0].strip())
        end_part = time_line[1].strip().split(" ")[0]
        end = _parse_vtt_timestamp(end_part)
        text_lines = block_lines[time_line_idx + 1 :]
        cue_text = _clean_text(text_lines)
        if not cue_text or start is None or end is None:
            continue
        auto_index += 1
        lines.append(SubtitleLine(index=auto_index, start=start, end=end, text=cue_text))
    return lines


def _parse_vtt_timestamp(value: str) -> float | None:
    parts = value.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = "0"
            m, s = parts
        else:
            return None
        sec, _, ms = s.partition(".")
        return int(h) * 3600 + int(m) * 60 + int(sec) + (int(ms.ljust(3, "0")[:3]) / 1000.0 if ms else 0.0)
    except ValueError:
        return None


def parse_subtitles(path: str | Path) -> list[SubtitleLine]:
    """Parse a subtitle file, dispatching on file extension (.srt / .vtt)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    suffix = path.suffix.lower()
    if suffix == ".vtt":
        return parse_vtt(text)
    if suffix == ".srt":
        return parse_srt(text)
    # Best-effort sniff: VTT files start with a WEBVTT header.
    if text.lstrip().startswith("WEBVTT"):
        return parse_vtt(text)
    return parse_srt(text)
