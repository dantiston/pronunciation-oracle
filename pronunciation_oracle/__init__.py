"""pronunciation-oracle: transcribe/align media and search a corpus for spoken words."""

from .corpus import Corpus, SearchHit
from .pipeline import transcribe_and_align
from .transcript import Transcript, WordTiming

__all__ = [
    "Corpus",
    "SearchHit",
    "Transcript",
    "WordTiming",
    "transcribe_and_align",
]

__version__ = "0.1.0"
