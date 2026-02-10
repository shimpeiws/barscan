"""BarScan - Lyrics word frequency analyzer using Genius API."""

from barscan.config import Settings, settings
from barscan.exceptions import (
    AnalyzerError,
    ArtistNotFoundError,
    BarScanError,
    EmptyLyricsError,
    GeniusAPIError,
    NLTKResourceError,
    NoLyricsFoundError,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Configuration
    "Settings",
    "settings",
    # Exceptions
    "AnalyzerError",
    "ArtistNotFoundError",
    "BarScanError",
    "EmptyLyricsError",
    "GeniusAPIError",
    "NLTKResourceError",
    "NoLyricsFoundError",
]
