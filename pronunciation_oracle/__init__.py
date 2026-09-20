"""pronunciation-oracle: transcribe/align media and search a corpus for spoken words."""

from .transcript import Transcript, WordTiming
from .corpus import Corpus, SearchHit
from .pipeline import transcribe_and_align

__all__ = [
    "Transcript",
    "WordTiming",
    "Corpus",
    "SearchHit",
    "transcribe_and_align",
]

__version__ = "0.1.0"
