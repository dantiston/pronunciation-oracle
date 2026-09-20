"""Command-line interface.

    pronunciation-oracle ingest pikachu_ep01.mp4 --subtitles pikachu_ep01.srt --corpus corpus.db
    pronunciation-oracle ingest-dir ./episodes --corpus corpus.db
    pronunciation-oracle search pikachu --corpus corpus.db --out clips/
    pronunciation-oracle stats --corpus corpus.db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .align.base import Aligner
from .align.sequence_aligner import SequenceAligner
from .asr.base import ASRBackend
from .asr.fake import FakeASR
from .asr.faster_whisper_backend import FasterWhisperASR
from .clipper import extract_clips
from .corpus import Corpus
from .pipeline import transcribe_and_align
from .transcript import WordTiming

MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm",
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac",
}
SUBTITLE_EXTENSIONS = [".srt", ".vtt"]


def build_asr_backend(name: str, args: argparse.Namespace) -> ASRBackend:
    if name == "faster-whisper":
        return FasterWhisperASR(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
    if name == "fake":
        words = []
        if args.fake_words_json:
            data = json.loads(Path(args.fake_words_json).read_text(encoding="utf-8"))
            words = [WordTiming(**w) for w in data]
        return FakeASR(words=words)
    raise ValueError(f"unknown ASR backend: {name}")


def build_aligner(name: str) -> Aligner:
    if name == "sequence":
        return SequenceAligner()
    if name == "ctc":
        from .align.torchaudio_ctc import TorchaudioCTCAligner

        return TorchaudioCTCAligner()
    raise ValueError(f"unknown aligner: {name}")


def _find_subtitle_for(media_path: Path) -> Path | None:
    for ext in SUBTITLE_EXTENSIONS:
        candidate = media_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def cmd_ingest(args: argparse.Namespace) -> int:
    asr = build_asr_backend(args.asr_backend, args)
    aligner = build_aligner(args.align_backend)
    transcript = transcribe_and_align(
        args.media,
        subtitles_path=args.subtitles,
        asr=asr,
        aligner=aligner,
        language=args.language,
    )
    with Corpus(args.corpus) as corpus:
        corpus.add_transcript(transcript)
    if args.save_transcript:
        transcript.save(args.save_transcript)
    print(
        f"Ingested {args.media}: {len(transcript.words)} words, "
        f"{transcript.duration:.1f}s, source={transcript.text_origin}"
    )
    return 0


def cmd_ingest_dir(args: argparse.Namespace) -> int:
    root = Path(args.directory)
    media_files = sorted(p for p in root.rglob("*") if p.suffix.lower() in MEDIA_EXTENSIONS)
    if not media_files:
        print(f"No media files found under {root}", file=sys.stderr)
        return 1

    asr = build_asr_backend(args.asr_backend, args)
    aligner = build_aligner(args.align_backend)
    total_words = 0
    with Corpus(args.corpus) as corpus:
        for media_path in media_files:
            subtitles = _find_subtitle_for(media_path) if not args.no_subtitles else None
            try:
                transcript = transcribe_and_align(
                    media_path,
                    subtitles_path=subtitles,
                    asr=asr,
                    aligner=aligner,
                    language=args.language,
                )
            except Exception as exc:  # keep batch ingestion going past one bad file
                print(f"FAILED {media_path}: {exc}", file=sys.stderr)
                continue
            corpus.add_transcript(transcript)
            total_words += len(transcript.words)
            sub_note = f" (+ subtitles {subtitles.name})" if subtitles else ""
            print(f"Ingested {media_path.name}{sub_note}: {len(transcript.words)} words")
    print(f"Done: {len(media_files)} files scanned, {total_words} words indexed.")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    with Corpus(args.corpus) as corpus:
        hits = corpus.search(
            args.word,
            contains=args.contains,
            min_confidence=args.min_confidence,
            limit=args.limit,
        )
    if not hits:
        print(f"No matches for '{args.word}'.")
        return 0

    print(f"Found {len(hits)} instance(s) of '{args.word}'.")
    if args.out:
        results = extract_clips(hits, args.out, pad=args.pad, fmt=args.format)
        for r in results:
            print(f"  {r.output_path}  [{Path(r.source_path).name} @ {r.word_start:.2f}s]")
        print(f"Wrote {len(results)} clip(s) to {args.out} (manifest.json / manifest.csv included).")
    else:
        for hit in hits:
            ctx = f"...{hit.context_before} [{hit.word}] {hit.context_after}..."
            print(f"  {Path(hit.file_path).name} @ {hit.start:.2f}-{hit.end:.2f}s  {ctx}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    with Corpus(args.corpus) as corpus:
        stats = corpus.stats()
        files = corpus.files()
    print(f"Corpus: {args.corpus}")
    print(f"  files: {stats.num_files}")
    print(f"  words indexed: {stats.num_words}")
    print(f"  total duration: {stats.total_duration:.1f}s")
    if args.verbose:
        for f in files:
            print(f"  - {f.path} ({f.word_count} words, {f.duration:.1f}s, origin={f.text_origin})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pronunciation-oracle", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_asr_align_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--asr-backend", choices=["faster-whisper", "fake"], default="faster-whisper")
        p.add_argument("--align-backend", choices=["sequence", "ctc"], default="sequence")
        p.add_argument("--model", default="small", help="faster-whisper model size (tiny/base/small/medium/large-v3)")
        p.add_argument("--device", default="cpu")
        p.add_argument("--compute-type", default="int8")
        p.add_argument("--language", default=None, help="force a language code, e.g. en (default: auto-detect)")
        p.add_argument("--fake-words-json", default=None, help="[fake backend] JSON list of {word,start,end}")

    p_ingest = sub.add_parser("ingest", help="Transcribe/align one media file into the corpus")
    p_ingest.add_argument("media", help="video or audio file")
    p_ingest.add_argument("--subtitles", default=None, help="optional .srt/.vtt reference transcript")
    p_ingest.add_argument("--corpus", required=True, help="path to the corpus SQLite database")
    p_ingest.add_argument("--save-transcript", default=None, help="also save the raw Transcript JSON here")
    add_asr_align_args(p_ingest)
    p_ingest.set_defaults(func=cmd_ingest)

    p_dir = sub.add_parser("ingest-dir", help="Batch-ingest every media file in a directory")
    p_dir.add_argument("directory", help="directory to scan recursively")
    p_dir.add_argument("--corpus", required=True, help="path to the corpus SQLite database")
    p_dir.add_argument(
        "--no-subtitles", action="store_true",
        help="ignore same-named .srt/.vtt files and use ASR text directly",
    )
    add_asr_align_args(p_dir)
    p_dir.set_defaults(func=cmd_ingest_dir)

    p_search = sub.add_parser("search", help="Find every instance of a word or phrase and clip it out")
    p_search.add_argument("word", help="word or phrase to search for (quote multi-word phrases)")
    p_search.add_argument("--corpus", required=True, help="path to the corpus SQLite database")
    p_search.add_argument("--out", default=None, help="directory to write clips into (omit to just list hits)")
    p_search.add_argument("--pad", type=float, default=0.15, help="seconds of padding around each clip")
    p_search.add_argument("--format", choices=["wav", "mp3", "flac"], default="wav")
    p_search.add_argument(
        "--contains", action="store_true",
        help="substring match instead of exact match (applied per word for a phrase)",
    )
    p_search.add_argument("--min-confidence", type=float, default=None)
    p_search.add_argument("--limit", type=int, default=None)
    p_search.set_defaults(func=cmd_search)

    p_stats = sub.add_parser("stats", help="Show corpus summary statistics")
    p_stats.add_argument("--corpus", required=True)
    p_stats.add_argument("--verbose", action="store_true")
    p_stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
