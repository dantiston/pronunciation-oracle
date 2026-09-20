"""The pipeline's output data model: per-word timings for one media file."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WordTiming:
    """A single spoken word, its time span in seconds, and (if known) confidence."""

    word: str
    start: float
    end: float
    confidence: float | None = None


@dataclass
class Transcript:
    """The final, time-aligned transcript for one media file."""

    source_path: str
    duration: float
    words: list[WordTiming] = field(default_factory=list)
    language: str | None = None
    text_origin: str = "asr"  # "asr" or "subtitles"

    @property
    def text(self) -> str:
        """The transcript's full text, as its words joined by spaces."""
        return " ".join(w.word for w in self.words)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (see `from_dict` for the inverse)."""
        return {
            "source_path": self.source_path,
            "duration": self.duration,
            "language": self.language,
            "text_origin": self.text_origin,
            "words": [asdict(w) for w in self.words],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        """Build a Transcript from a dict produced by `to_dict`."""
        return cls(
            source_path=data["source_path"],
            duration=data["duration"],
            language=data.get("language"),
            text_origin=data.get("text_origin", "asr"),
            words=[WordTiming(**w) for w in data.get("words", [])],
        )

    def save(self, path: str | Path) -> None:
        """Write this transcript as JSON to `path`."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Transcript:
        """Load a transcript previously written by `save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
