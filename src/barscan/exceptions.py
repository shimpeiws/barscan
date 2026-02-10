"""Custom exceptions for BarScan."""

from typing import Any


class BarScanError(Exception):
    """Base exception for BarScan.

    Attributes:
        message: Human-readable error message
        context: Additional context information
    """

    def __init__(self, message: str, **context: Any) -> None:
        self.message = message
        self.context = context
        super().__init__(message)

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message


class GeniusAPIError(BarScanError):
    """Error from Genius API.

    Attributes:
        status_code: HTTP status code if available
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        **context: Any,
    ) -> None:
        super().__init__(message, status_code=status_code, **context)
        self.status_code = status_code


class ArtistNotFoundError(BarScanError):
    """Artist not found.

    Attributes:
        artist_name: The artist name that was searched
    """

    def __init__(self, message: str, artist_name: str | None = None) -> None:
        super().__init__(message, artist_name=artist_name)
        self.artist_name = artist_name


class NoLyricsFoundError(BarScanError):
    """No lyrics found for song.

    Attributes:
        song_id: The song ID
        song_title: The song title
    """

    def __init__(
        self,
        message: str,
        song_id: int | None = None,
        song_title: str | None = None,
    ) -> None:
        super().__init__(message, song_id=song_id, song_title=song_title)
        self.song_id = song_id
        self.song_title = song_title


class AnalyzerError(BarScanError):
    """Base exception for analyzer errors."""


class EmptyLyricsError(AnalyzerError):
    """Lyrics text is empty or contains only whitespace."""


class NLTKResourceError(AnalyzerError):
    """NLTK resource not available.

    Attributes:
        resource_name: Name of the missing NLTK resource
    """

    def __init__(self, message: str, resource_name: str | None = None) -> None:
        super().__init__(message, resource_name=resource_name)
        self.resource_name = resource_name
