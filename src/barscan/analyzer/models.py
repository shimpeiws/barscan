"""Pydantic models for text analysis."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ContextsMode(StrEnum):
    """Context extraction mode for copyright handling."""

    NONE = "none"
    SHORT = "short"
    FULL = "full"


class WordFrequency(BaseModel, frozen=True):
    """A single word and its frequency count.

    Attributes:
        word: The normalized word.
        count: Number of occurrences.
        percentage: Percentage of total words (0.0-100.0).
    """

    word: str = Field(..., min_length=1, description="The normalized word")
    count: int = Field(..., ge=1, description="Number of occurrences")
    percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of total words")


class AnalysisConfig(BaseModel, frozen=True):
    """Configuration for text analysis.

    Attributes:
        min_word_length: Minimum word length to include.
        min_count: Minimum occurrence count to include.
        use_lemmatization: Whether to apply lemmatization.
        remove_stop_words: Whether to filter out stop words.
        custom_stop_words: Additional stop words to filter.
        language: Language for NLTK processing.
        compute_tfidf: Whether to compute TF-IDF scores.
        compute_pos: Whether to compute POS tags.
        compute_sentiment: Whether to compute sentiment scores.
        detect_slang: Whether to detect slang words.
        contexts_mode: Context extraction mode (none/short/full).
        max_contexts_per_word: Maximum number of contexts per word.
    """

    min_word_length: int = Field(default=2, ge=1, description="Minimum word length to include")
    min_count: int = Field(default=1, ge=1, description="Minimum occurrence count to include")
    use_lemmatization: bool = Field(default=False, description="Whether to apply lemmatization")
    remove_stop_words: bool = Field(default=True, description="Whether to filter out stop words")
    custom_stop_words: frozenset[str] = Field(
        default_factory=frozenset, description="Additional stop words to filter"
    )
    language: str = Field(
        default="auto", description="Language for tokenization: english, japanese, or auto"
    )
    use_pos_filtering: bool = Field(
        default=True,
        description="Filter by part-of-speech for Japanese (keep nouns, verbs, adjectives)",
    )

    # Enhanced NLP analysis options
    compute_tfidf: bool = Field(default=False, description="Whether to compute TF-IDF scores")
    compute_pos: bool = Field(default=False, description="Whether to compute POS tags")
    compute_sentiment: bool = Field(default=False, description="Whether to compute sentiment")
    detect_slang: bool = Field(default=False, description="Whether to detect slang words")
    contexts_mode: ContextsMode = Field(
        default=ContextsMode.NONE, description="Context extraction mode"
    )
    max_contexts_per_word: int = Field(
        default=3, ge=1, le=10, description="Maximum contexts per word"
    )


class AnalysisResult(BaseModel, frozen=True):
    """Result of word frequency analysis on lyrics.

    Attributes:
        song_id: Genius song ID for the analyzed lyrics.
        song_title: Title of the song.
        artist_name: Name of the artist.
        total_words: Total word count after filtering.
        unique_words: Number of unique words.
        frequencies: List of word frequencies, sorted by count descending.
        analyzed_at: Timestamp of analysis.
    """

    song_id: int = Field(..., description="Genius song ID")
    song_title: str = Field(..., description="Title of the song")
    artist_name: str = Field(..., description="Name of the artist")
    total_words: int = Field(..., ge=0, description="Total word count after filtering")
    unique_words: int = Field(..., ge=0, description="Number of unique words")
    frequencies: tuple[WordFrequency, ...] = Field(
        default_factory=tuple, description="Word frequencies sorted by count descending"
    )
    analyzed_at: datetime = Field(..., description="Timestamp of analysis")

    def top_words(self, n: int = 10) -> tuple[WordFrequency, ...]:
        """Return the top N most frequent words.

        Args:
            n: Number of top words to return.

        Returns:
            Tuple of top N WordFrequency items.
        """
        return self.frequencies[:n]


class AggregateAnalysisResult(BaseModel, frozen=True):
    """Aggregated result of word frequency analysis across multiple songs.

    Attributes:
        artist_name: Name of the artist.
        songs_analyzed: Number of songs included in analysis.
        total_words: Total word count across all songs.
        unique_words: Number of unique words across all songs.
        frequencies: Aggregated word frequencies sorted by count descending.
        song_results: Individual analysis results per song.
        analyzed_at: Timestamp of analysis.
    """

    artist_name: str = Field(..., description="Name of the artist")
    songs_analyzed: int = Field(..., ge=0, description="Number of songs analyzed")
    total_words: int = Field(..., ge=0, description="Total word count across all songs")
    unique_words: int = Field(..., ge=0, description="Number of unique words")
    frequencies: tuple[WordFrequency, ...] = Field(
        default_factory=tuple, description="Aggregated word frequencies"
    )
    song_results: tuple[AnalysisResult, ...] = Field(
        default_factory=tuple, description="Individual song analysis results"
    )
    analyzed_at: datetime = Field(..., description="Timestamp of analysis")

    def top_words(self, n: int = 10) -> tuple[WordFrequency, ...]:
        """Return the top N most frequent words.

        Args:
            n: Number of top words to return.

        Returns:
            Tuple of top N WordFrequency items.
        """
        return self.frequencies[:n]


class TokenWithPosition(BaseModel, frozen=True):
    """A token with its position information for context extraction.

    Attributes:
        token: The normalized word token.
        line_index: Zero-based line index in the lyrics.
        word_index: Zero-based word index within the line.
        original_line: The original line text (before normalization).
        song_id: Genius song ID.
        song_title: Title of the song.
    """

    token: str = Field(..., description="The normalized word token")
    line_index: int = Field(..., ge=0, description="Zero-based line index")
    word_index: int = Field(..., ge=0, description="Zero-based word index within line")
    original_line: str = Field(..., description="Original line text")
    song_id: int = Field(..., description="Genius song ID")
    song_title: str = Field(..., description="Title of the song")


class WordContext(BaseModel, frozen=True):
    """A context example for a word occurrence.

    Used in 'full' context mode to provide rich metadata.

    Attributes:
        line: The lyrics line containing the word.
        track: The song title.
        album: Album name (optional, may not be available).
        year: Release year (optional, may not be available).
    """

    line: str = Field(..., description="The lyrics line containing the word")
    track: str = Field(..., description="The song title")
    album: str | None = Field(default=None, description="Album name if available")
    year: int | None = Field(default=None, description="Release year if available")
