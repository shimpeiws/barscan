"""Word frequency analysis for lyrics."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Protocol

from barscan.analyzer.filters import apply_filters
from barscan.analyzer.models import (
    AggregateAnalysisResult,
    AnalysisConfig,
    AnalysisResult,
    WordFrequency,
)
from barscan.analyzer.processor import preprocess
from barscan.exceptions import EmptyLyricsError


class LyricsProtocol(Protocol):
    """Protocol for lyrics objects compatible with analyze_lyrics."""

    @property
    def song_id(self) -> int: ...

    @property
    def song_title(self) -> str: ...

    @property
    def artist_name(self) -> str: ...

    @property
    def text(self) -> str: ...

    @property
    def is_empty(self) -> bool: ...


def count_frequencies(tokens: list[str]) -> Counter[str]:
    """Count word frequencies in token list.

    Args:
        tokens: List of word tokens.

    Returns:
        Counter with word frequencies.
    """
    return Counter(tokens)


def create_word_frequencies(counter: Counter[str], total_words: int) -> tuple[WordFrequency, ...]:
    """Convert Counter to sorted tuple of WordFrequency objects.

    Args:
        counter: Counter with word frequencies.
        total_words: Total word count for percentage calculation.

    Returns:
        Tuple of WordFrequency objects sorted by count descending.
    """
    if total_words == 0:
        return ()

    frequencies = tuple(
        WordFrequency(
            word=word,
            count=count,
            percentage=round((count / total_words) * 100, 2),
        )
        for word, count in counter.most_common()
    )

    return frequencies


def analyze_text(
    text: str,
    song_id: int,
    song_title: str,
    artist_name: str,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    """Perform complete word frequency analysis on lyrics text.

    Args:
        text: Raw lyrics text.
        song_id: Genius song ID.
        song_title: Title of the song.
        artist_name: Name of the artist.
        config: Analysis configuration (uses default if None).

    Returns:
        AnalysisResult with word frequencies.

    Raises:
        EmptyLyricsError: If text is empty or contains only whitespace.
    """
    if config is None:
        config = AnalysisConfig()

    # Preprocess text
    tokens = preprocess(text, config)

    # Apply filters
    filtered_tokens = apply_filters(tokens, config)

    # Handle empty result after filtering
    if not filtered_tokens:
        return AnalysisResult(
            song_id=song_id,
            song_title=song_title,
            artist_name=artist_name,
            total_words=0,
            unique_words=0,
            frequencies=(),
            analyzed_at=datetime.now(UTC),
        )

    # Count frequencies
    counter = count_frequencies(filtered_tokens)
    total_words = len(filtered_tokens)
    unique_words = len(counter)

    # Create frequency tuple
    frequencies = create_word_frequencies(counter, total_words)

    return AnalysisResult(
        song_id=song_id,
        song_title=song_title,
        artist_name=artist_name,
        total_words=total_words,
        unique_words=unique_words,
        frequencies=frequencies,
        analyzed_at=datetime.now(UTC),
    )


def analyze_lyrics(
    lyrics: LyricsProtocol,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    """Perform word frequency analysis on a Lyrics object.

    Convenience wrapper around analyze_text that extracts fields from Lyrics model.

    Args:
        lyrics: Lyrics object implementing LyricsProtocol.
        config: Analysis configuration (uses default if None).

    Returns:
        AnalysisResult with word frequencies.

    Raises:
        EmptyLyricsError: If lyrics text is empty.
    """
    if lyrics.is_empty:
        raise EmptyLyricsError(f"No lyrics available for song '{lyrics.song_title}'")

    return analyze_text(
        text=lyrics.text,
        song_id=lyrics.song_id,
        song_title=lyrics.song_title,
        artist_name=lyrics.artist_name,
        config=config,
    )


def aggregate_results(
    results: list[AnalysisResult],
    artist_name: str,
) -> AggregateAnalysisResult:
    """Aggregate multiple song analysis results into one.

    Args:
        results: List of individual song analysis results.
        artist_name: Name of the artist.

    Returns:
        AggregateAnalysisResult with combined frequencies.
    """
    if not results:
        return AggregateAnalysisResult(
            artist_name=artist_name,
            songs_analyzed=0,
            total_words=0,
            unique_words=0,
            frequencies=(),
            song_results=(),
            analyzed_at=datetime.now(UTC),
        )

    # Combine all word counts
    combined_counter: Counter[str] = Counter()
    for result in results:
        for freq in result.frequencies:
            combined_counter[freq.word] += freq.count

    total_words = sum(combined_counter.values())
    unique_words = len(combined_counter)
    frequencies = create_word_frequencies(combined_counter, total_words)

    return AggregateAnalysisResult(
        artist_name=artist_name,
        songs_analyzed=len(results),
        total_words=total_words,
        unique_words=unique_words,
        frequencies=frequencies,
        song_results=tuple(results),
        analyzed_at=datetime.now(UTC),
    )
