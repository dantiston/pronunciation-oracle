import json
from pathlib import Path

from pronunciation_oracle.audio import ffprobe_duration
from pronunciation_oracle.clipper import extract_clips
from pronunciation_oracle.corpus import SearchHit


def test_extract_clips_creates_files_and_manifest(tmp_path, tone_wav):
    hits = [
        SearchHit(
            file_path=str(tone_wav), word="pikachu", start=1.0, end=1.3,
            confidence=0.9, context_before="hey", context_after="use",
        ),
        SearchHit(
            file_path=str(tone_wav), word="pikachu", start=2.0, end=2.3,
            confidence=0.8, context_before="go", context_after="now",
        ),
    ]
    out_dir = tmp_path / "clips"
    results = extract_clips(hits, out_dir, pad=0.1, fmt="wav")

    assert len(results) == 2
    for r in results:
        p = Path(r.output_path)
        assert p.exists()
        assert p.suffix == ".wav"
        duration = ffprobe_duration(p)
        # word span (0.3s) + 0.1s padding each side
        assert 0.35 <= duration <= 0.65

    manifest_json = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest_json) == 2
    assert (out_dir / "manifest.csv").exists()


def test_extract_clips_dedupes_colliding_filenames(tmp_path, tone_wav):
    hits = [
        SearchHit(str(tone_wav), "pikachu", 1.0, 1.3, 0.9, "", ""),
        SearchHit(str(tone_wav), "pikachu", 1.0, 1.3, 0.9, "", ""),
    ]
    results = extract_clips(hits, tmp_path / "clips", pad=0.1)
    paths = {r.output_path for r in results}
    assert len(paths) == 2


def test_extract_clips_clamps_to_file_duration(tmp_path, tone_wav):
    duration = ffprobe_duration(tone_wav)
    hits = [SearchHit(str(tone_wav), "end", duration - 0.05, duration - 0.01, 0.9, "", "")]
    results = extract_clips(hits, tmp_path / "clips", pad=1.0)
    assert results[0].clip_end <= duration


def test_extract_clips_groups_punctuation_variants_into_one_folder(tmp_path, tone_wav):
    # Real transcripts spell the same word differently by sentence position
    # ("Pikachu." vs "Pikachu," vs "Pikachu!"); all instances of one search
    # should land in a single output folder, not be scattered by punctuation.
    hits = [
        SearchHit(str(tone_wav), "Pikachu.", 0.5, 0.8, 0.9, "", ""),
        SearchHit(str(tone_wav), "Pikachu,", 1.0, 1.3, 0.9, "", ""),
        SearchHit(str(tone_wav), "Pikachu!", 1.5, 1.8, 0.9, "", ""),
        SearchHit(str(tone_wav), "PIKACHU", 2.0, 2.3, 0.9, "", ""),
    ]
    out_dir = tmp_path / "clips"
    results = extract_clips(hits, out_dir, pad=0.05)

    word_dirs = {Path(r.output_path).parent for r in results}
    assert word_dirs == {out_dir / "pikachu"}
    assert len(list((out_dir / "pikachu").glob("*.wav"))) == 4
