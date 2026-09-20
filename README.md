# pronunciation-oracle

Given a corpus of video/audio files (e.g. a whole anime season), find every
spoken instance of a word and cut it out as its own audio clip -- e.g. "give
me every clip of a character saying *Pikachu*".

The system has two stages:

1. **Transcribe + align** (`pronunciation_oracle.pipeline.transcribe_and_align`)
   Run ASR on a video/audio file. If a subtitle file (`.srt`/`.vtt`) is
   supplied, it's treated as the authoritative (cleaner, human-authored) text,
   and an alignment step reconciles it with the ASR's word-level timestamps so
   the output carries the subtitle's wording with audio-accurate timing. No
   subtitles -> the ASR's own word timestamps (already forced-aligned to audio
   internally) are used directly.

2. **Index + search + clip** (`pronunciation_oracle.corpus` / `clipper`)
   Every transcribed word, with its timestamp and source file, is indexed in a
   SQLite corpus database. Searching a word returns every occurrence across
   every ingested file; `extract_clips` cuts each one out (with a little
   padding) into its own audio file, plus a `manifest.json`/`.csv` recording
   source file, exact word timing, surrounding context, and confidence.

```
media/*.mp4,*.srt  --[1] transcribe_and_align-->  Transcript (per file)
                                                        |
                                                    Corpus.add_transcript
                                                        v
                                          corpus.db  --[2] search("pikachu")--> hits
                                                                                  |
                                                                          extract_clips
                                                                                  v
                                                          clips/pikachu/ep01_04200-04650.wav, ...
```

## Install

```bash
pip install -e ".[asr]"       # faster-whisper backend (default, CPU-friendly)
# or: pip install -e ".[all]" # + optional torchaudio CTC forced aligner
```

Requires `ffmpeg`/`ffprobe` on PATH (`apt-get install ffmpeg`).

## CLI

```bash
# Ingest one episode. Subtitles are auto-detected by same-basename .srt/.vtt in ingest-dir,
# or pass one explicitly:
pronunciation-oracle ingest pokemon_ep01.mp4 --subtitles pokemon_ep01.srt --corpus corpus.db

# Or batch-ingest a whole season (matches ep01.mp4 with ep01.srt automatically):
pronunciation-oracle ingest-dir ./pokemon_season1 --corpus corpus.db

# Find every "pikachu" and clip it out:
pronunciation-oracle search pikachu --corpus corpus.db --out clips/ --pad 0.15

# Phrases work too -- matched as consecutive spoken words in one file:
pronunciation-oracle search "I choose you" --corpus corpus.db --out clips/

# Just list hits without clipping:
pronunciation-oracle search pikachu --corpus corpus.db

# Corpus summary:
pronunciation-oracle stats --corpus corpus.db --verbose
```

Each clip lands at `clips/<word>/<source-file-stem>_<start_ms>-<end_ms>.wav`,
alongside `clips/manifest.json` and `clips/manifest.csv` listing every clip's
source file, word-span timing, clip timing, confidence, and surrounding
context (so you can tell which character/line each clip came from).

Useful flags:
- `--contains` on `search`: substring match (e.g. `pika` also matches `pikachu`).
- `--min-confidence`: drop low-confidence (usually interpolated) hits.
- `--format {wav,mp3,flac}`: clip output format.
- `--model {tiny,base,small,medium,large-v3}` / `--device` / `--compute-type`: faster-whisper tuning.
- `--align-backend {sequence,ctc}`: alignment engine (see below).

## Library

```python
from pronunciation_oracle import transcribe_and_align, Corpus
from pronunciation_oracle.clipper import extract_clips

transcript = transcribe_and_align("ep01.mp4", subtitles_path="ep01.srt")

with Corpus("corpus.db") as corpus:
    corpus.add_transcript(transcript)
    hits = corpus.search("pikachu")

extract_clips(hits, "clips/", pad=0.15)
```

## Architecture / extension points

Every stage is behind a small interface so pieces can be swapped independently:

- **`asr.base.ASRBackend`** -- `transcribe(audio_path) -> ASRResult` (word-level
  timestamps). Default: `FasterWhisperASR` (CTranslate2 Whisper, no torch/GPU
  needed). `FakeASR` provides canned output for tests/offline demos
  (`--asr-backend fake --fake-words-json words.json`).
- **`align.base.Aligner`** -- `align(asr_words, reference_segments) -> [WordTiming]`.
  Default: `SequenceAligner`, a dependency-light forced-alignment approximation:
  it fuzzy-matches the reference text against the ASR's word sequence
  (`difflib`), gives exactly-matched words the ASR's own timestamps, and
  interpolates unmatched words (e.g. a proper noun the ASR misheard)
  proportionally to word length between the nearest matched neighbors --
  bounded by the enclosing subtitle cue's `[start, end]` window so
  interpolation never drifts into a neighboring line. Optional:
  `align.torchaudio_ctc.TorchaudioCTCAligner` (`--align-backend ctc`) does true
  CTC forced alignment via `torchaudio.pipelines.MMS_FA` directly against the
  waveform, for higher-precision phoneme-level timing at the cost of a
  torch/torchaudio dependency and model download (`pip install .[align-ctc]`).
- **`corpus.Corpus`** -- SQLite-backed word index (`files`, `words` tables,
  indexed on normalized word). `search()` takes either a single word or a
  phrase; a phrase is matched as a contiguous run of `word_index`s in one
  file (no fuzzy/skip-word matching) and returned as one hit spanning the
  first word's start to the last word's end. Swappable for a different store
  if the corpus grows past what SQLite comfortably handles.
- **`clipper.extract_clips`** -- ffmpeg-based cutting; ffmpeg does the actual
  audio-only accurate-seek extraction (`audio.py`).

Not implemented (natural next steps, not required for the core ask):
speaker diarization/labeling (context text is included in the manifest as a
proxy for "who's probably talking"), phonetic/IPA-level pronunciation
comparison across clips, streaming ingestion for very large corpora.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests use `FakeASR` and an `ffmpeg`-generated tone WAV, so the full
ingest -> search -> clip pipeline is exercised without downloading any ASR
model.
