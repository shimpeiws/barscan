"""Custom exceptions for BarScan."""


class BarScanError(Exception):
    """Base exception for BarScan."""


class GeniusAPIError(BarScanError):
    """Error from Genius API."""


class ArtistNotFoundError(BarScanError):
    """Artist not found."""


class NoLyricsFoundError(BarScanError):
    """No lyrics found for song."""


class AnalyzerError(BarScanError):
    """Base exception for analyzer errors."""


class EmptyLyricsError(AnalyzerError):
    """Lyrics text is empty or contains only whitespace."""


class NLTKResourceError(AnalyzerError):
    """NLTK resource not available."""
