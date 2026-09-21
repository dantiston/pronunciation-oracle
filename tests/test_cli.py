import json
import shutil
from unittest.mock import patch

import pytest

from pronunciation_oracle.asr.faster_whisper_backend import FasterWhisperASR
from pronunciation_oracle.asr.mlx_whisper_backend import MlxWhisperASR
from pronunciation_oracle.cli import build_asr_backend, build_parser, main
from pronunciation_oracle.corpus import Corpus


def _fake_words_json(tmp_path):
    words = [
        {"word": "hey", "start": 0.2, "end": 0.5, "confidence": 0.9},
        {"word": "pikachu", "start": 0.6, "end": 1.1, "confidence": 0.95},
        {"word": "use", "start": 1.3, "end": 1.5, "confidence": 0.9},
        {"word": "thunderbolt", "start": 1.6, "end": 2.2, "confidence": 0.85},
    ]
    path = tmp_path / "fake_words.json"
    path.write_text(json.dumps(words), encoding="utf-8")
    return path


def _fake_words_json_repeated_pikachu(tmp_path, count):
    words = [
        {"word": "pikachu", "start": i * 0.5, "end": i * 0.5 + 0.3, "confidence": 0.9} for i in range(count)
    ]
    path = tmp_path / "fake_words_repeated.json"
    path.write_text(json.dumps(words), encoding="utf-8")
    return path


def test_ingest_and_search_end_to_end(tmp_path, tone_wav, capsys):
    corpus_path = tmp_path / "corpus.db"
    fake_words = _fake_words_json(tmp_path)

    rc = main(
        [
            "ingest",
            str(tone_wav),
            "--corpus",
            str(corpus_path),
            "--asr-backend",
            "fake",
            "--fake-words-json",
            str(fake_words),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Ingested" in out
    assert "4 words" in out

    with Corpus(corpus_path) as corpus:
        assert corpus.stats().num_files == 1
        assert corpus.stats().num_words == 4

    out_dir = tmp_path / "clips"
    rc = main(
        [
            "search",
            "pikachu",
            "--corpus",
            str(corpus_path),
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Found 1 instance" in out

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest) == 1
    assert manifest[0]["word"] == "pikachu"
    assert manifest[0]["context_before"] == "hey"
    assert manifest[0]["context_after"] == "use thunderbolt"


def test_search_phrase_end_to_end(tmp_path, tone_wav, capsys):
    corpus_path = tmp_path / "corpus.db"
    fake_words = _fake_words_json(tmp_path)
    main(
        [
            "ingest",
            str(tone_wav),
            "--corpus",
            str(corpus_path),
            "--asr-backend",
            "fake",
            "--fake-words-json",
            str(fake_words),
        ]
    )
    capsys.readouterr()

    out_dir = tmp_path / "phrase_clips"
    rc = main(
        [
            "search",
            "use thunderbolt",
            "--corpus",
            str(corpus_path),
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Found 1 instance" in out

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest) == 1
    assert manifest[0]["word"] == "use thunderbolt"
    # Span covers from "use"'s start to "thunderbolt"'s end.
    assert manifest[0]["word_start"] == 1.3
    assert manifest[0]["word_end"] == 2.2


def _ingest_repeated_pikachu(tmp_path, tone_wav, corpus_path, count=5):
    fake_words = _fake_words_json_repeated_pikachu(tmp_path, count)
    main(
        [
            "ingest",
            str(tone_wav),
            "--corpus",
            str(corpus_path),
            "--asr-backend",
            "fake",
            "--fake-words-json",
            str(fake_words),
        ]
    )


def test_search_default_max_clips_is_3(tmp_path, tone_wav, capsys):
    corpus_path = tmp_path / "corpus.db"
    _ingest_repeated_pikachu(tmp_path, tone_wav, corpus_path, count=5)
    capsys.readouterr()

    out_dir = tmp_path / "clips"
    rc = main(["search", "pikachu", "--corpus", str(corpus_path), "--out", str(out_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Found 5 instance(s)" in out
    assert "Clipping the first 3" in out

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest) == 3


def test_search_max_clips_override(tmp_path, tone_wav, capsys):
    corpus_path = tmp_path / "corpus.db"
    _ingest_repeated_pikachu(tmp_path, tone_wav, corpus_path, count=5)
    capsys.readouterr()

    out_dir = tmp_path / "clips"
    rc = main(["search", "pikachu", "--corpus", str(corpus_path), "--out", str(out_dir), "--max-clips", "2"])
    assert rc == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest) == 2


def test_search_max_clips_zero_means_no_cap(tmp_path, tone_wav, capsys):
    corpus_path = tmp_path / "corpus.db"
    _ingest_repeated_pikachu(tmp_path, tone_wav, corpus_path, count=5)
    capsys.readouterr()

    out_dir = tmp_path / "clips"
    rc = main(["search", "pikachu", "--corpus", str(corpus_path), "--out", str(out_dir), "--max-clips", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Clipping the first" not in out
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest) == 5


def test_search_max_clips_does_not_affect_plain_listing(tmp_path, tone_wav, capsys):
    corpus_path = tmp_path / "corpus.db"
    _ingest_repeated_pikachu(tmp_path, tone_wav, corpus_path, count=5)
    capsys.readouterr()

    rc = main(["search", "pikachu", "--corpus", str(corpus_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Found 5 instance(s)" in out
    assert out.count("pikachu") >= 5  # every hit printed, not capped to 3


def test_search_no_matches(tmp_path, tone_wav, capsys):
    corpus_path = tmp_path / "corpus.db"
    fake_words = _fake_words_json(tmp_path)
    main(
        [
            "ingest",
            str(tone_wav),
            "--corpus",
            str(corpus_path),
            "--asr-backend",
            "fake",
            "--fake-words-json",
            str(fake_words),
        ]
    )
    capsys.readouterr()

    rc = main(["search", "squirtle", "--corpus", str(corpus_path)])
    assert rc == 0
    assert "No matches" in capsys.readouterr().out


def test_stats_command(tmp_path, tone_wav, capsys):
    corpus_path = tmp_path / "corpus.db"
    fake_words = _fake_words_json(tmp_path)
    main(
        [
            "ingest",
            str(tone_wav),
            "--corpus",
            str(corpus_path),
            "--asr-backend",
            "fake",
            "--fake-words-json",
            str(fake_words),
        ]
    )
    capsys.readouterr()

    rc = main(["stats", "--corpus", str(corpus_path), "--verbose"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "files: 1" in out
    assert "words indexed: 4" in out


def test_ingest_dir(tmp_path, tone_wav, capsys):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    shutil.copy(tone_wav, media_dir / "ep01.wav")
    shutil.copy(tone_wav, media_dir / "ep02.wav")
    fake_words = _fake_words_json(tmp_path)
    corpus_path = tmp_path / "corpus.db"

    rc = main(
        [
            "ingest-dir",
            str(media_dir),
            "--corpus",
            str(corpus_path),
            "--asr-backend",
            "fake",
            "--fake-words-json",
            str(fake_words),
        ]
    )
    assert rc == 0
    with Corpus(corpus_path) as corpus:
        assert corpus.stats().num_files == 2


@pytest.mark.parametrize("workers", [1, 3])
def test_ingest_dir_sequential_and_parallel_agree(tmp_path, tone_wav, workers):
    # Same batch ingested with the sequential path (--workers 1) and the
    # parallel path (--workers 3, each worker building its own FakeASR from
    # the same --fake-words-json) must produce an identical corpus.
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    for i in range(4):
        shutil.copy(tone_wav, media_dir / f"ep{i:02d}.wav")
    fake_words = _fake_words_json(tmp_path)
    corpus_path = tmp_path / "corpus.db"

    rc = main(
        [
            "ingest-dir",
            str(media_dir),
            "--corpus",
            str(corpus_path),
            "--asr-backend",
            "fake",
            "--fake-words-json",
            str(fake_words),
            "--workers",
            str(workers),
        ]
    )
    assert rc == 0
    with Corpus(corpus_path) as corpus:
        stats = corpus.stats()
        assert stats.num_files == 4
        assert stats.num_words == 16  # 4 words/file (see _fake_words_json) * 4 files


def test_ingest_dir_parallel_reports_failures_and_exit_code(tmp_path, tone_wav, capsys):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    shutil.copy(tone_wav, media_dir / "good.wav")
    (media_dir / "corrupt.wav").write_bytes(b"not actually audio")
    fake_words = _fake_words_json(tmp_path)
    corpus_path = tmp_path / "corpus.db"

    rc = main(
        [
            "ingest-dir",
            str(media_dir),
            "--corpus",
            str(corpus_path),
            "--asr-backend",
            "fake",
            "--fake-words-json",
            str(fake_words),
            "--workers",
            "2",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "FAILED" in err
    assert "corrupt.wav" in err
    assert "1 file(s) failed to ingest" in err
    with Corpus(corpus_path) as corpus:
        # The good file still got ingested despite the other one failing.
        assert corpus.stats().num_files == 1


def _ingest_dir_argv(media_dir, corpus_path, fake_words, *extra):
    return [
        "ingest-dir",
        str(media_dir),
        "--corpus",
        str(corpus_path),
        "--asr-backend",
        "fake",
        "--fake-words-json",
        str(fake_words),
        *extra,
    ]


def test_ingest_dir_resumes_by_default_skipping_completed_files(tmp_path, tone_wav, capsys):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    shutil.copy(tone_wav, media_dir / "ep01.wav")
    shutil.copy(tone_wav, media_dir / "ep02.wav")
    fake_words = _fake_words_json(tmp_path)
    corpus_path = tmp_path / "corpus.db"

    main(_ingest_dir_argv(media_dir, corpus_path, fake_words))
    capsys.readouterr()
    with Corpus(corpus_path) as corpus:
        first_run_stats = corpus.stats()

    rc = main(_ingest_dir_argv(media_dir, corpus_path, fake_words))
    assert rc == 0
    out = capsys.readouterr().out
    assert "Skipping 2 already-ingested file(s)" in out
    assert "Ingested" not in out  # nothing was actually reprocessed
    assert "Done: 2 files scanned, 0 processed, 2 skipped" in out
    with Corpus(corpus_path) as corpus:
        assert corpus.stats() == first_run_stats


def test_ingest_dir_processes_only_new_files_on_resume(tmp_path, tone_wav, capsys):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    shutil.copy(tone_wav, media_dir / "ep01.wav")
    fake_words = _fake_words_json(tmp_path)
    corpus_path = tmp_path / "corpus.db"

    main(_ingest_dir_argv(media_dir, corpus_path, fake_words))
    capsys.readouterr()

    # A new episode shows up alongside the one already ingested.
    shutil.copy(tone_wav, media_dir / "ep02.wav")
    rc = main(_ingest_dir_argv(media_dir, corpus_path, fake_words))
    assert rc == 0
    out = capsys.readouterr().out
    assert "Skipping 1 already-ingested file(s)" in out
    assert "1 to process." in out
    assert "Ingested ep02.wav" in out
    assert "Ingested ep01.wav" not in out
    with Corpus(corpus_path) as corpus:
        assert corpus.stats().num_files == 2


def test_ingest_dir_reingest_flag_forces_reprocessing(tmp_path, tone_wav, capsys):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    shutil.copy(tone_wav, media_dir / "ep01.wav")
    fake_words = _fake_words_json(tmp_path)
    corpus_path = tmp_path / "corpus.db"

    main(_ingest_dir_argv(media_dir, corpus_path, fake_words))
    capsys.readouterr()

    rc = main(_ingest_dir_argv(media_dir, corpus_path, fake_words, "--reingest"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "Skipping" not in out
    assert "Ingested ep01.wav" in out


@pytest.mark.parametrize("workers", [1, 2])
def test_ingest_dir_resume_retries_only_the_previously_failed_file(tmp_path, tone_wav, capsys, workers):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    shutil.copy(tone_wav, media_dir / "good.wav")
    (media_dir / "corrupt.wav").write_bytes(b"not actually audio")
    fake_words = _fake_words_json(tmp_path)
    corpus_path = tmp_path / "corpus.db"
    argv_extra = ("--workers", str(workers))

    rc = main(_ingest_dir_argv(media_dir, corpus_path, fake_words, *argv_extra))
    assert rc == 1
    capsys.readouterr()

    # Re-run unchanged: the good file (already in the corpus) is skipped, the
    # bad one (never made it in, since it failed) is retried -- and still fails.
    rc = main(_ingest_dir_argv(media_dir, corpus_path, fake_words, *argv_extra))
    assert rc == 1
    out_err = capsys.readouterr()
    assert "Skipping 1 already-ingested file(s)" in out_err.out
    assert "1 to process." in out_err.out
    assert "good.wav" not in out_err.out  # not reprocessed
    assert "FAILED" in out_err.err
    assert "corrupt.wav" in out_err.err
    with Corpus(corpus_path) as corpus:
        assert corpus.stats().num_files == 1


def test_vad_filter_defaults_to_on():
    args = build_parser().parse_args(["ingest", "ep01.mp4", "--corpus", "corpus.db"])
    backend = build_asr_backend("faster-whisper", args)
    assert isinstance(backend, FasterWhisperASR)
    assert backend._vad_filter is True


def test_no_vad_filter_flag_disables_it():
    args = build_parser().parse_args(["ingest", "ep01.mp4", "--corpus", "corpus.db", "--no-vad-filter"])
    backend = build_asr_backend("faster-whisper", args)
    assert isinstance(backend, FasterWhisperASR)
    assert backend._vad_filter is False


def test_asr_backend_defaults_to_auto():
    args = build_parser().parse_args(["ingest", "ep01.mp4", "--corpus", "corpus.db"])
    assert args.asr_backend == "auto"


@pytest.mark.parametrize(
    ("available", "installed", "expected"),
    [
        (True, True, MlxWhisperASR),
        (True, False, FasterWhisperASR),
        (False, True, FasterWhisperASR),
        (False, False, FasterWhisperASR),
    ],
)
def test_auto_asr_backend_resolves_by_mlx_availability(available, installed, expected):
    args = build_parser().parse_args(["ingest", "ep01.mp4", "--corpus", "corpus.db"])
    with (
        patch("pronunciation_oracle.cli.mlx_available", return_value=available),
        patch("pronunciation_oracle.cli.mlx_whisper_installed", return_value=installed),
    ):
        backend = build_asr_backend("auto", args)
    assert isinstance(backend, expected)


def test_explicit_faster_whisper_overrides_auto_even_on_apple_silicon():
    args = build_parser().parse_args(
        ["ingest", "ep01.mp4", "--corpus", "corpus.db", "--asr-backend", "faster-whisper"]
    )
    with (
        patch("pronunciation_oracle.cli.mlx_available", return_value=True),
        patch("pronunciation_oracle.cli.mlx_whisper_installed", return_value=True),
    ):
        backend = build_asr_backend(args.asr_backend, args)
    assert isinstance(backend, FasterWhisperASR)
