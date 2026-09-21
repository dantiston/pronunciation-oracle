"""Step (2)'s index: a SQLite-backed corpus of every word spoken in every ingested file.

Search is a normalized-word lookup (optionally substring/"contains") over one word
or a whole phrase, returning every occurrence with its timestamps so the clipper
can cut each one out. A phrase query is matched as a contiguous run of words in a
single file -- there's no fuzzy/skip-word matching.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .text_norm import normalize_word, tokenize
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
    """One occurrence of a searched word/phrase: where, when, and how confidently."""

    file_path: str
    word: str
    start: float
    end: float
    confidence: float | None
    context_before: str
    context_after: str


@dataclass
class FileRow:
    """Summary of one ingested file, as returned by `Corpus.files()`."""

    id: int
    path: str
    duration: float
    language: str | None
    text_origin: str
    word_count: int


@dataclass
class CorpusStats:
    """Aggregate corpus size, as returned by `Corpus.stats()`."""

    num_files: int
    num_words: int
    total_duration: float


class Corpus:
    """A SQLite-backed index of every word spoken in every ingested file."""

    def __init__(self, db_path: str | Path):
        """Open (creating if needed) the corpus database at `db_path`."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> Corpus:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def add_transcript(self, transcript: Transcript, replace: bool = True) -> int:
        """Ingest a Transcript's words into the corpus, keyed by source_path.

        Args:
            transcript: The time-aligned transcript to index.
            replace: If a file at `transcript.source_path` is already in the
                corpus, replace its words with these instead of raising.

        Returns:
            The corpus-internal file id for `transcript.source_path`.

        Raises:
            ValueError: If the path is already ingested and `replace=False`.
        """
        cur = self._conn.cursor()
        existing = cur.execute("SELECT id FROM files WHERE path = ?", (transcript.source_path,)).fetchone()
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
                "INSERT INTO files (path, duration, language, text_origin, ingested_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    transcript.source_path,
                    transcript.duration,
                    transcript.language,
                    transcript.text_origin,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            file_id = cur.lastrowid
            if file_id is None:
                raise RuntimeError(f"INSERT into files did not produce a rowid for {transcript.source_path}")

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
        """Delete a file and all of its indexed words from the corpus."""
        self._conn.execute("DELETE FROM files WHERE path = ?", (str(path),))
        self._conn.commit()

    def ingested_paths(self) -> set[str]:
        """Return every source path already in the corpus, for resuming a batch ingest.

        Cheap existence check (no join for word counts, unlike `files()`) --
        used to skip files a previous `ingest-dir` run already committed.
        """
        rows = self._conn.execute("SELECT path FROM files").fetchall()
        return {row[0] for row in rows}

    def search(
        self,
        query: str,
        contains: bool = False,
        min_confidence: float | None = None,
        context_words: int = 3,
        limit: int | None = None,
    ) -> list[SearchHit]:
        """Search for one word or a whole phrase (e.g. "pikachu" or "I choose you").

        A multi-word query must appear as consecutive words in a single file;
        each token is matched the same way a single-word query would be
        (`contains`/`min_confidence` apply per token).

        Args:
            query: A word, or a whole phrase (quoted words matched in order).
            contains: Substring match instead of exact match.
            min_confidence: Drop hits below this ASR word-confidence.
            context_words: How many words of context to include on each side.
            limit: Cap the number of hits returned.

        Returns:
            Every matching occurrence, ordered by file path then start time.
        """
        tokens = [t for t in (normalize_word(tok) for tok in tokenize(query)) if t]
        if not tokens:
            return []
        if len(tokens) == 1:
            return self._search_word(tokens[0], contains, min_confidence, context_words, limit)
        return self._search_phrase(tokens, contains, min_confidence, context_words, limit)

    def _search_word(
        self,
        norm: str,
        contains: bool,
        min_confidence: float | None,
        context_words: int,
        limit: int | None,
    ) -> list[SearchHit]:
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
            context_before, context_after = self._context(cur, file_id, word_index, word_index, context_words)
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

    def _search_phrase(
        self,
        tokens: list[str],
        contains: bool,
        min_confidence: float | None,
        context_words: int,
        limit: int | None,
    ) -> list[SearchHit]:
        cur = self._conn.cursor()
        first = tokens[0]
        if contains:
            anchors = cur.execute(
                "SELECT f.path, w.file_id, w.word_index FROM words w JOIN files f ON f.id = w.file_id "
                "WHERE w.norm_word LIKE ? ESCAPE '\\' ORDER BY f.path, w.word_index",
                (f"%{_escape_like(first)}%",),
            ).fetchall()
        else:
            anchors = cur.execute(
                "SELECT f.path, w.file_id, w.word_index FROM words w JOIN files f ON f.id = w.file_id "
                "WHERE w.norm_word = ? ORDER BY f.path, w.word_index",
                (first,),
            ).fetchall()

        hits: list[SearchHit] = []
        for path, file_id, word_index in anchors:
            span = cur.execute(
                "SELECT word_index, word, norm_word, start, end, confidence FROM words "
                "WHERE file_id = ? AND word_index BETWEEN ? AND ? ORDER BY word_index",
                (file_id, word_index, word_index + len(tokens) - 1),
            ).fetchall()
            if len(span) != len(tokens):
                continue  # phrase would run past the end of the file
            if not all(
                (tok in norm_word) if contains else (tok == norm_word)
                for tok, (_idx, _word, norm_word, _s, _e, _c) in zip(tokens, span, strict=True)
            ):
                continue
            confidences = [c for *_rest, c in span if c is not None]
            if min_confidence is not None and confidences and min(confidences) < min_confidence:
                continue

            first_idx, last_idx = span[0][0], span[-1][0]
            context_before, context_after = self._context(cur, file_id, first_idx, last_idx, context_words)
            hits.append(
                SearchHit(
                    file_path=path,
                    word=" ".join(row[1] for row in span),
                    start=span[0][3],
                    end=span[-1][4],
                    confidence=min(confidences) if confidences else None,
                    context_before=context_before,
                    context_after=context_after,
                )
            )
            if limit is not None and len(hits) >= limit:
                break
        return hits

    def _context(
        self, cur: sqlite3.Cursor, file_id: int, before_index: int, after_index: int, n: int
    ) -> tuple[str, str]:
        """Return the `n` words on each side of a match.

        `before_index`/`after_index` bound the match itself: equal for a
        single word, or the first/last word_index for a phrase.
        """
        if n <= 0:
            return "", ""
        before_rows = cur.execute(
            "SELECT word FROM words WHERE file_id = ? AND word_index BETWEEN ? AND ? ORDER BY word_index",
            (file_id, before_index - n, before_index - 1),
        ).fetchall()
        after_rows = cur.execute(
            "SELECT word FROM words WHERE file_id = ? AND word_index BETWEEN ? AND ? ORDER BY word_index",
            (file_id, after_index + 1, after_index + n),
        ).fetchall()
        return " ".join(r[0] for r in before_rows), " ".join(r[0] for r in after_rows)

    def files(self) -> list[FileRow]:
        """List every ingested file with its word count, ordered by path."""
        rows = self._conn.execute(
            "SELECT f.id, f.path, f.duration, f.language, f.text_origin, COUNT(w.id) "
            "FROM files f LEFT JOIN words w ON w.file_id = f.id "
            "GROUP BY f.id ORDER BY f.path"
        ).fetchall()
        return [FileRow(*row) for row in rows]

    def stats(self) -> CorpusStats:
        """Return aggregate corpus size: file count, word count, total duration."""
        num_files, total_duration = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(duration), 0) FROM files"
        ).fetchone()
        (num_words,) = self._conn.execute("SELECT COUNT(*) FROM words").fetchone()
        return CorpusStats(num_files=num_files, num_words=num_words, total_duration=total_duration)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
