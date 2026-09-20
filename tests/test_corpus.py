import pytest

from pronunciation_oracle.corpus import Corpus
from pronunciation_oracle.transcript import Transcript, WordTiming


def _transcript(path, words):
    return Transcript(
        source_path=path,
        duration=10.0,
        language="en",
        text_origin="asr",
        words=[WordTiming(w, i * 1.0, i * 1.0 + 0.5, 0.9) for i, w in enumerate(words)],
    )


def test_add_and_search_exact_word(tmp_path):
    corpus = Corpus(tmp_path / "corpus.db")
    corpus.add_transcript(_transcript("ep01.mp4", ["hey", "Pikachu!", "use", "thunderbolt"]))
    corpus.add_transcript(_transcript("ep02.mp4", ["Pikachu,", "I", "choose", "you"]))

    hits = corpus.search("pikachu")
    assert len(hits) == 2
    assert {h.file_path for h in hits} == {"ep01.mp4", "ep02.mp4"}
    corpus.close()


def test_search_is_case_and_punctuation_insensitive(tmp_path):
    corpus = Corpus(tmp_path / "corpus.db")
    corpus.add_transcript(_transcript("ep01.mp4", ["PIKACHU."]))
    hits = corpus.search("Pikachu")
    assert len(hits) == 1
    corpus.close()


def test_search_contains_substring(tmp_path):
    corpus = Corpus(tmp_path / "corpus.db")
    corpus.add_transcript(_transcript("ep01.mp4", ["pikachu", "pika"]))
    hits = corpus.search("pika", contains=True)
    assert {h.word.lower().strip(".,!") for h in hits} == {"pikachu", "pika"}
    exact = corpus.search("pika", contains=False)
    assert len(exact) == 1
    corpus.close()


def test_search_context_words(tmp_path):
    corpus = Corpus(tmp_path / "corpus.db")
    corpus.add_transcript(_transcript("ep01.mp4", ["hey", "pikachu", "use", "thunderbolt"]))
    hits = corpus.search("pikachu", context_words=2)
    assert hits[0].context_before == "hey"
    assert hits[0].context_after == "use thunderbolt"
    corpus.close()


def test_reingesting_same_path_replaces_words(tmp_path):
    corpus = Corpus(tmp_path / "corpus.db")
    corpus.add_transcript(_transcript("ep01.mp4", ["pikachu"]))
    corpus.add_transcript(_transcript("ep01.mp4", ["pikachu", "pikachu"]))
    assert len(corpus.search("pikachu")) == 2
    assert corpus.stats().num_files == 1
    corpus.close()


def test_reingest_without_replace_raises(tmp_path):
    corpus = Corpus(tmp_path / "corpus.db")
    corpus.add_transcript(_transcript("ep01.mp4", ["pikachu"]))
    with pytest.raises(ValueError):
        corpus.add_transcript(_transcript("ep01.mp4", ["pikachu"]), replace=False)
    corpus.close()


def test_stats_and_files(tmp_path):
    corpus = Corpus(tmp_path / "corpus.db")
    corpus.add_transcript(_transcript("ep01.mp4", ["a", "b", "c"]))
    corpus.add_transcript(_transcript("ep02.mp4", ["d", "e"]))
    stats = corpus.stats()
    assert stats.num_files == 2
    assert stats.num_words == 5
    files = corpus.files()
    assert [f.path for f in files] == ["ep01.mp4", "ep02.mp4"]
    corpus.close()


def test_context_manager_closes(tmp_path):
    with Corpus(tmp_path / "corpus.db") as corpus:
        corpus.add_transcript(_transcript("ep01.mp4", ["pikachu"]))
        assert corpus.stats().num_words == 1


def test_search_phrase_matches_contiguous_words(tmp_path):
    corpus = Corpus(tmp_path / "corpus.db")
    corpus.add_transcript(_transcript("ep01.mp4", ["Hey,", "Pikachu!", "I", "choose", "you."]))
    corpus.add_transcript(_transcript("ep02.mp4", ["I", "will", "choose", "you", "later"]))

    hits = corpus.search("I choose you")
    assert len(hits) == 1
    assert hits[0].file_path == "ep01.mp4"
    assert hits[0].word == "I choose you."
    corpus.close()


def test_search_phrase_span_covers_first_to_last_word(tmp_path):
    corpus = Corpus(tmp_path / "corpus.db")
    corpus.add_transcript(_transcript("ep01.mp4", ["hey", "pikachu", "use", "thunderbolt", "now"]))
    hits = corpus.search("use thunderbolt")
    assert len(hits) == 1
    # WordTiming for word i spans [i, i+0.5); "use" is index 2, "thunderbolt" is index 3.
    assert hits[0].start == 2.0
    assert hits[0].end == 3.5
    corpus.close()


def test_search_phrase_context_is_relative_to_whole_span(tmp_path):
    corpus = Corpus(tmp_path / "corpus.db")
    corpus.add_transcript(_transcript("ep01.mp4", ["hey", "pikachu", "use", "thunderbolt", "now", "please"]))
    hits = corpus.search("use thunderbolt", context_words=2)
    assert hits[0].context_before == "hey pikachu"
    assert hits[0].context_after == "now please"
    corpus.close()


def test_search_phrase_requires_adjacency_not_just_presence(tmp_path):
    corpus = Corpus(tmp_path / "corpus.db")
    # "use" and "thunderbolt" both appear, but not next to each other.
    corpus.add_transcript(_transcript("ep01.mp4", ["use", "the", "move", "thunderbolt"]))
    assert corpus.search("use thunderbolt") == []
    corpus.close()


def test_search_phrase_does_not_run_past_end_of_file(tmp_path):
    corpus = Corpus(tmp_path / "corpus.db")
    corpus.add_transcript(_transcript("ep01.mp4", ["hey", "pikachu"]))
    assert corpus.search("pikachu go") == []
    corpus.close()


def test_search_phrase_contains_mode_matches_per_token(tmp_path):
    corpus = Corpus(tmp_path / "corpus.db")
    corpus.add_transcript(_transcript("ep01.mp4", ["hey", "pikachu", "thunderbolt"]))
    hits = corpus.search("pika thunder", contains=True)
    assert len(hits) == 1
    assert hits[0].word == "pikachu thunderbolt"
    corpus.close()


def test_search_phrase_confidence_is_min_across_words(tmp_path):
    corpus = Corpus(tmp_path / "corpus.db")
    transcript = Transcript(
        source_path="ep01.mp4",
        duration=10.0,
        words=[
            WordTiming("use", 0.0, 0.5, 0.95),
            WordTiming("thunderbolt", 0.5, 1.0, 0.4),
        ],
    )
    corpus.add_transcript(transcript)
    hits = corpus.search("use thunderbolt")
    assert hits[0].confidence == 0.4
    assert corpus.search("use thunderbolt", min_confidence=0.5) == []
    corpus.close()
