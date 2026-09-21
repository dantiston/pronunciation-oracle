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
pip install -e .                     # faster-whisper is a base dependency (CPU-friendly, no GPU needed)
# or: pip install -e ".[align-ctc]"  # + optional torchaudio CTC forced aligner (auto-uses cuda/mps GPU if available)
# or: pip install -e ".[mlx]"        # + optional mlx-whisper ASR backend (Apple Silicon only, Metal-accelerated)
```

Requires `ffmpeg`/`ffprobe` on PATH (`brew install ffmpeg` / `apt-get install ffmpeg`).

## CLI

```bash
# Ingest one episode. Subtitles are auto-detected by same-basename .srt/.vtt in ingest-dir,
# or pass one explicitly:
pronunciation-oracle ingest pokemon_ep01.mp4 --subtitles pokemon_ep01.srt --corpus corpus.db

# Or batch-ingest a whole season (matches ep01.mp4 with ep01.srt automatically):
pronunciation-oracle ingest-dir ./pokemon_season1 --corpus corpus.db

# Find every "pikachu" and clip it out:
pronunciation-oracle search pikachu --corpus corpus.db --out clips/ --pad 0.3

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

**If you have subtitles, use a smaller/faster `--model`.** When subtitles are
supplied, the ASR's own words are discarded in favor of the subtitle's text --
the ASR only needs to provide rough timing anchors for the aligner to snap
subtitle words onto (see [Architecture](#architecture--extension-points)).
That's a much easier job than getting the words right, so a weak model can
still do it well. Measured on real content: `--model tiny` was ~3.3x faster
than `small` and recovered the same 21/21 instances of a repeated word once
aligned against subtitle text, even though `tiny`'s own unaligned transcript
was visibly worse (hallucinated words, wrong lyrics on a song). The residual
handful of poorly-anchored words landed with `confidence=None` or very low
confidence, exactly what `search --min-confidence` is for -- so pair a fast
model with a confidence filter on search. Without subtitles (pure ASR path),
`tiny`'s raw word accuracy is the whole result, so stick with `small` or larger.

**GPU/MPS options, validated before recommending either.** `--align-backend
ctc` (the `align-ctc` extra) auto-picks cuda, then mps (Apple Silicon GPU via
Metal), then cpu -- it's the same model and math either way, so this is a
free speed win with no accuracy tradeoff (verified matching output on cpu vs
mps). Note: torchaudio's forced_align op isn't implemented for MPS in current
torch, so mps runs with a required (automatic) CPU fallback for just that one
op; measured speedup was modest (~30-40% on a short clip), not dramatic.
`--asr-backend mlx-whisper` (the `mlx` extra, Apple Silicon only) is a
different story: measured 3.9x faster than faster-whisper's cpu "small" on a
5-minute clip, but on a real recall check, it missed roughly half the actual
instances of a repeated word that faster-whisper's "small" caught correctly
(misheard as short fragments) -- and mlx-whisper's "medium", 8x slower,
did *worse*, not better, so this isn't a "just use a bigger model" fix.
**`--asr-backend` defaults to `auto`**, which picks `mlx-whisper` whenever
this is Apple Silicon with the `mlx` extra installed -- that recall cost is
accepted as the default going forward, not hidden behind an opt-in flag; pass
`--asr-backend faster-whisper` explicitly if that tradeoff isn't right for
your content, or if you haven't validated it the way this project's other
defaults were checked.

Useful flags:
- `--contains` on `search`: substring match (e.g. `pika` also matches `pikachu`).
- `--min-confidence`: drop low-confidence (usually interpolated) hits.
- `--format {wav,mp3,flac}`: clip output format.
- `--model {tiny,base,small,medium,large-v3}` / `--device` / `--compute-type` /
  `--vad-filter`/`--no-vad-filter`: faster-whisper tuning. `--vad-filter` is on
  by default (skips non-speech before decoding, ~30% faster) but on content
  with near-continuous background music under dialogue it can also drop real
  spoken words along with the music (measured ~57% recall loss on one such
  case) -- pass `--no-vad-filter` if that's a real risk for your content.
- `--asr-backend {auto,faster-whisper,mlx-whisper,fake}`: ASR engine, `auto`
  by default (see GPU/MPS above).
- `--align-backend {sequence,ctc}` / `--align-device {auto,cpu,cuda,mps}`:
  alignment engine, `sequence` by default -- `ctc` is opt-in, unlike the ASR
  backend, since it hasn't been checked for an accuracy regression the way
  mlx-whisper was (see below and GPU/MPS above).
- `ingest-dir --workers N`: transcribe N files concurrently, each in its own
  process/model instance. Defaults to 1 (sequential) -- benchmarked on Apple
  Silicon CPU, a single faster-whisper process already saturates the
  machine's usable parallel compute, so raising this made a real two-episode
  batch ~3% *slower* from process contention, not faster. May help on
  different hardware (GPU, many-core setups where one process doesn't
  saturate available compute); benchmark before relying on it.

## Library

```python
from pronunciation_oracle import transcribe_and_align, Corpus
from pronunciation_oracle.clipper import extract_clips

transcript = transcribe_and_align("ep01.mp4", subtitles_path="ep01.srt")

with Corpus("corpus.db") as corpus:
    corpus.add_transcript(transcript)
    hits = corpus.search("pikachu")

extract_clips(hits, "clips/")
```

## Architecture / extension points

Every stage is behind a small interface so pieces can be swapped independently:

- **`asr.base.ASRBackend`** -- `transcribe(audio_path) -> ASRResult` (word-level
  timestamps). `--asr-backend auto` (the CLI default) picks between the two
  real backends: `asr.mlx_whisper_backend.MlxWhisperASR` on Apple Silicon with
  the `mlx` extra installed, else `FasterWhisperASR` (CTranslate2 Whisper, no
  torch/GPU needed) -- see GPU/MPS above for the real tradeoff that choice
  accepts. `FakeASR` provides canned output for tests/offline demos
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
  Runs on cuda/mps/cpu, auto-selected (`--align-device auto`, the default).
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
