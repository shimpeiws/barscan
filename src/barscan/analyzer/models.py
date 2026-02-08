"""Pydantic models for text analysis."""

from datetime import datetime

from pydantic import BaseModel, Field


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
        use_lemmatization: Whether to apply lemmatization.
        remove_stop_words: Whether to filter out stop words.
        custom_stop_words: Additional stop words to filter.
        language: Language for NLTK processing.
    """

    min_word_length: int = Field(default=2, ge=1, description="Minimum word length to include")
    use_lemmatization: bool = Field(default=False, description="Whether to apply lemmatization")
    remove_stop_words: bool = Field(default=True, description="Whether to filter out stop words")
    custom_stop_words: frozenset[str] = Field(
        default_factory=frozenset, description="Additional stop words to filter"
    )
    language: str = Field(default="english", description="Language for NLTK processing")


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
