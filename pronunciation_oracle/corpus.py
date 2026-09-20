"""Step (2)'s index: a SQLite-backed corpus of every word spoken in every ingested file.

Search is a simple normalized-word lookup (optionally substring/"contains"),
returning every occurrence with its timestamps so the clipper can cut each one out.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .text_norm import normalize_word
from .transcript import Transcript

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    duration REAL NOT NULL,
    language TEXT,
    text_origin TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    word_index INTEGER NOT NULL,
    word TEXT NOT NULL,
    norm_word TEXT NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    confidence REAL
);

CREATE INDEX IF NOT EXISTS idx_words_norm_word ON words(norm_word);
CREATE INDEX IF NOT EXISTS idx_words_file_id ON words(file_id, word_index);
"""


@dataclass
class SearchHit:
    file_path: str
    word: str
    start: float
    end: float
    confidence: float | None
    context_before: str
    context_after: str


@dataclass
class FileRow:
    id: int
    path: str
    duration: float
    language: str | None
    text_origin: str
    word_count: int


@dataclass
class CorpusStats:
    num_files: int
    num_words: int
    total_duration: float


class Corpus:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Corpus":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def add_transcript(self, transcript: Transcript, replace: bool = True) -> int:
        """Ingest a Transcript's words into the corpus, keyed by source_path."""
        cur = self._conn.cursor()
        existing = cur.execute(
            "SELECT id FROM files WHERE path = ?", (transcript.source_path,)
        ).fetchone()
        if existing is not None:
            if not replace:
                raise ValueError(f"{transcript.source_path} is already in the corpus (replace=False)")
            file_id = existing[0]
            cur.execute("DELETE FROM words WHERE file_id = ?", (file_id,))
            cur.execute(
                "UPDATE files SET duration = ?, language = ?, text_origin = ?, ingested_at = ? WHERE id = ?",
                (
                    transcript.duration,
                    transcript.language,
                    transcript.text_origin,
                    datetime.now(timezone.utc).isoformat(),
                    file_id,
                ),
            )
        else:
            cur.execute(
                "INSERT INTO files (path, duration, language, text_origin, ingested_at) VALUES (?, ?, ?, ?, ?)",
                (
                    transcript.source_path,
                    transcript.duration,
                    transcript.language,
                    transcript.text_origin,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            file_id = cur.lastrowid

        cur.executemany(
            "INSERT INTO words (file_id, word_index, word, norm_word, start, end, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (file_id, idx, w.word, normalize_word(w.word), w.start, w.end, w.confidence)
                for idx, w in enumerate(transcript.words)
            ],
        )
        self._conn.commit()
        return file_id

    def remove_file(self, path: str) -> None:
        self._conn.execute("DELETE FROM files WHERE path = ?", (str(path),))
        self._conn.commit()

    def search(
        self,
        word: str,
        contains: bool = False,
        min_confidence: float | None = None,
        context_words: int = 3,
        limit: int | None = None,
    ) -> list[SearchHit]:
        norm = normalize_word(word)
        cur = self._conn.cursor()
        if contains:
            rows = cur.execute(
                "SELECT w.id, f.path, w.word_index, w.word, w.start, w.end, w.confidence, w.file_id "
                "FROM words w JOIN files f ON f.id = w.file_id "
                "WHERE w.norm_word LIKE ? ESCAPE '\\' "
                "ORDER BY f.path, w.start",
                (f"%{_escape_like(norm)}%",),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT w.id, f.path, w.word_index, w.word, w.start, w.end, w.confidence, w.file_id "
                "FROM words w JOIN files f ON f.id = w.file_id "
                "WHERE w.norm_word = ? "
                "ORDER BY f.path, w.start",
                (norm,),
            ).fetchall()

        hits: list[SearchHit] = []
        for _id, path, word_index, matched_word, start, end, confidence, file_id in rows:
            if min_confidence is not None and confidence is not None and confidence < min_confidence:
                continue
            context_before, context_after = self._context(cur, file_id, word_index, context_words)
            hits.append(
                SearchHit(
                    file_path=path,
                    word=matched_word,
                    start=start,
                    end=end,
                    confidence=confidence,
                    context_before=context_before,
                    context_after=context_after,
                )
            )
            if limit is not None and len(hits) >= limit:
                break
        return hits

    def _context(self, cur: sqlite3.Cursor, file_id: int, word_index: int, n: int) -> tuple[str, str]:
        if n <= 0:
            return "", ""
        before_rows = cur.execute(
            "SELECT word FROM words WHERE file_id = ? AND word_index BETWEEN ? AND ? ORDER BY word_index",
            (file_id, word_index - n, word_index - 1),
        ).fetchall()
        after_rows = cur.execute(
            "SELECT word FROM words WHERE file_id = ? AND word_index BETWEEN ? AND ? ORDER BY word_index",
            (file_id, word_index + 1, word_index + n),
        ).fetchall()
        return " ".join(r[0] for r in before_rows), " ".join(r[0] for r in after_rows)

    def files(self) -> list[FileRow]:
        rows = self._conn.execute(
            "SELECT f.id, f.path, f.duration, f.language, f.text_origin, COUNT(w.id) "
            "FROM files f LEFT JOIN words w ON w.file_id = f.id "
            "GROUP BY f.id ORDER BY f.path"
        ).fetchall()
        return [FileRow(*row) for row in rows]

    def stats(self) -> CorpusStats:
        num_files, total_duration = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(duration), 0) FROM files"
        ).fetchone()
        (num_words,) = self._conn.execute("SELECT COUNT(*) FROM words").fetchone()
        return CorpusStats(num_files=num_files, num_words=num_words, total_duration=total_duration)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
