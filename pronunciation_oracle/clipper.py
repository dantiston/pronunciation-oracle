"""Step (2)'s output: cut every search hit into its own audio clip file."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .audio import cut_clip, ffprobe_duration
from .corpus import SearchHit
from .text_norm import normalize_word, tokenize

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(value: str) -> str:
    slug = _SLUG_RE.sub("_", value).strip("_")
    return slug or "clip"


def _group_key(word: str) -> str:
    """Folder name for a matched word/phrase, punctuation- and case-insensitive.

    Real transcripts spell the same word differently depending on sentence
    position ("Pikachu." vs "Pikachu," vs "Pikachu!"), so grouping by the raw
    matched text would scatter one search's hits across several sibling
    folders. Group by normalized tokens instead so they all land together.
    """
    norm_tokens = [normalize_word(tok) for tok in tokenize(word)]
    key = "_".join(t for t in norm_tokens if t)
    return key or "clip"


@dataclass
class ClipResult:
    """One extracted clip: its source, exact word/phrase span, and padded clip span."""

    word: str
    source_path: str
    word_start: float
    word_end: float
    clip_start: float
    clip_end: float
    confidence: float | None
    context_before: str
    context_after: str
    output_path: str


def extract_clips(
    hits: list[SearchHit],
    output_dir: str | Path,
    pad: float = 0.15,
    fmt: str = "wav",
    write_manifest: bool = True,
) -> list[ClipResult]:
    """Cut one audio clip per search hit into `output_dir`, padded by `pad` seconds.

    Clips are named `<word>/<source-stem>_<start_ms>-<end_ms>.<fmt>`, deduplicated
    if two hits would collide. A manifest.json/.csv listing every clip's source
    file, exact word timing, and surrounding context is written alongside them.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    duration_cache: dict[str, float] = {}
    used_names: set[str] = set()
    results: list[ClipResult] = []

    for hit in hits:
        if hit.file_path not in duration_cache:
            duration_cache[hit.file_path] = ffprobe_duration(hit.file_path)
        file_duration = duration_cache[hit.file_path]

        clip_start = max(0.0, hit.start - pad)
        clip_end = min(file_duration, hit.end + pad)
        if clip_end <= clip_start:
            clip_end = min(file_duration, clip_start + 0.05)

        word_dir = output_dir / _group_key(hit.word)
        word_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(hit.file_path).stem
        base_name = f"{_slugify(stem)}_{int(hit.start * 1000)}-{int(hit.end * 1000)}.{fmt}"
        name = base_name
        counter = 2
        rel_key = str(word_dir / name)
        while rel_key in used_names:
            name = f"{base_name[: -len(fmt) - 1]}_{counter}.{fmt}"
            rel_key = str(word_dir / name)
            counter += 1
        used_names.add(rel_key)

        output_path = word_dir / name
        cut_clip(hit.file_path, clip_start, clip_end, output_path, fmt=fmt)

        results.append(
            ClipResult(
                word=hit.word,
                source_path=hit.file_path,
                word_start=hit.start,
                word_end=hit.end,
                clip_start=clip_start,
                clip_end=clip_end,
                confidence=hit.confidence,
                context_before=hit.context_before,
                context_after=hit.context_after,
                output_path=str(output_path),
            )
        )

    if write_manifest and results:
        _write_manifest(output_dir, results)

    return results


def _write_manifest(output_dir: Path, results: list[ClipResult]) -> None:
    rows = [asdict(r) for r in results]
    (output_dir / "manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
