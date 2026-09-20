import json

from pronunciation_oracle.cli import main
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


def test_ingest_and_search_end_to_end(tmp_path, tone_wav, capsys):
    corpus_path = tmp_path / "corpus.db"
    fake_words = _fake_words_json(tmp_path)

    rc = main([
        "ingest", str(tone_wav),
        "--corpus", str(corpus_path),
        "--asr-backend", "fake",
        "--fake-words-json", str(fake_words),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Ingested" in out
    assert "4 words" in out

    with Corpus(corpus_path) as corpus:
        assert corpus.stats().num_files == 1
        assert corpus.stats().num_words == 4

    out_dir = tmp_path / "clips"
    rc = main([
        "search", "pikachu",
        "--corpus", str(corpus_path),
        "--out", str(out_dir),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Found 1 instance" in out

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest) == 1
    assert manifest[0]["word"] == "pikachu"
    assert manifest[0]["context_before"] == "hey"
    assert manifest[0]["context_after"] == "use thunderbolt"


def test_search_no_matches(tmp_path, tone_wav, capsys):
    corpus_path = tmp_path / "corpus.db"
    fake_words = _fake_words_json(tmp_path)
    main([
        "ingest", str(tone_wav),
        "--corpus", str(corpus_path),
        "--asr-backend", "fake",
        "--fake-words-json", str(fake_words),
    ])
    capsys.readouterr()

    rc = main(["search", "squirtle", "--corpus", str(corpus_path)])
    assert rc == 0
    assert "No matches" in capsys.readouterr().out


def test_stats_command(tmp_path, tone_wav, capsys):
    corpus_path = tmp_path / "corpus.db"
    fake_words = _fake_words_json(tmp_path)
    main([
        "ingest", str(tone_wav),
        "--corpus", str(corpus_path),
        "--asr-backend", "fake",
        "--fake-words-json", str(fake_words),
    ])
    capsys.readouterr()

    rc = main(["stats", "--corpus", str(corpus_path), "--verbose"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "files: 1" in out
    assert "words indexed: 4" in out


def test_ingest_dir(tmp_path, tone_wav, capsys):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    import shutil

    shutil.copy(tone_wav, media_dir / "ep01.wav")
    shutil.copy(tone_wav, media_dir / "ep02.wav")
    fake_words = _fake_words_json(tmp_path)
    corpus_path = tmp_path / "corpus.db"

    rc = main([
        "ingest-dir", str(media_dir),
        "--corpus", str(corpus_path),
        "--asr-backend", "fake",
        "--fake-words-json", str(fake_words),
    ])
    assert rc == 0
    with Corpus(corpus_path) as corpus:
        assert corpus.stats().num_files == 2
