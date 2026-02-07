"""Custom exceptions for BarScan."""


class BarScanError(Exception):
    """Base exception for BarScan."""


class GeniusAPIError(BarScanError):
    """Error from Genius API."""


class ArtistNotFoundError(BarScanError):
    """Artist not found."""


class NoLyricsFoundError(BarScanError):
    """No lyrics found for song."""
