"""WordGrain format output for BarScan.

This module provides Pydantic models and functions to export analysis results
in the WordGrain JSON format (.wg.json), a standardized schema for vocabulary
analysis data.

Reference: https://mumbl.dev/schemas/wordgrain/v0.1.0
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from datetime import datetime
from importlib.metadata import version

from pydantic import BaseModel, Field

from barscan.analyzer.context import extract_contexts_for_word
from barscan.analyzer.models import (
    AggregateAnalysisResult,
    AnalysisConfig,
    ContextsMode,
    TokenWithPosition,
    WordContext,
)
from barscan.analyzer.pos import get_pos_tags
from barscan.analyzer.sentiment import get_sentiment_scores
from barscan.analyzer.slang import detect_slang_words
from barscan.analyzer.tfidf import calculate_corpus_tfidf

WORDGRAIN_SCHEMA_URL = "https://mumbl.dev/schemas/wordgrain/v0.1.0"


class WordGrainGrain(BaseModel, frozen=True):
    """A single word entry in WordGrain format.

    Attributes:
        word: The vocabulary word.
        frequency: Raw occurrence count.
        frequency_normalized: Occurrences per 10,000 words.
        tfidf: TF-IDF score (0.0-1.0), optional.
        pos: Part-of-speech tag, optional.
        sentiment: Sentiment category (positive/negative/neutral), optional.
        sentiment_score: VADER compound score (-1.0 to 1.0), optional.
        is_slang: Whether the word is slang, optional.
        contexts: Example usage contexts, optional.
    """

    word: str = Field(..., min_length=1, description="The vocabulary word")
    frequency: int = Field(..., ge=0, description="Raw occurrence count")
    frequency_normalized: float = Field(..., ge=0.0, description="Occurrences per 10,000 words")

    # Enhanced NLP fields (all optional for backward compatibility)
    tfidf: float | None = Field(default=None, ge=0.0, le=1.0, description="TF-IDF score")
    pos: str | None = Field(default=None, description="Part-of-speech tag")
    sentiment: str | None = Field(default=None, description="Sentiment category")
    sentiment_score: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="VADER compound score"
    )
    is_slang: bool | None = Field(default=None, description="Whether word is slang")
    contexts: tuple[str, ...] | tuple[WordContext, ...] | None = Field(
        default=None, description="Example usage contexts"
    )


class WordGrainMeta(BaseModel, frozen=True):
    """Metadata section of a WordGrain document.

    Attributes:
        source: Data source identifier.
        artist: Primary artist name.
        generated_at: ISO 8601 datetime of generation.
        corpus_size: Number of tracks analyzed.
        total_words: Total word count in corpus.
        generator: Tool identifier with version.
        language: ISO 639-1 language code.
    """

    source: str = Field(default="genius", description="Data source identifier")
    artist: str = Field(..., description="Primary artist name")
    generated_at: datetime = Field(..., description="ISO 8601 datetime of generation")
    corpus_size: int = Field(..., ge=0, description="Number of tracks analyzed")
    total_words: int = Field(..., ge=0, description="Total word count in corpus")
    generator: str = Field(..., description="Tool identifier with version")
    language: str = Field(default="en", description="ISO 639-1 language code")


class WordGrainDocument(BaseModel, frozen=True):
    """Root WordGrain document structure.

    Attributes:
        schema_: JSON Schema URL (serialized as $schema).
        meta: Document metadata.
        grains: List of word entries.
    """

    schema_: str = Field(
        default=WORDGRAIN_SCHEMA_URL,
        alias="$schema",
        description="JSON Schema URL",
    )
    meta: WordGrainMeta = Field(..., description="Document metadata")
    grains: tuple[WordGrainGrain, ...] = Field(
        default_factory=tuple, description="List of word entries"
    )


def slugify(text: str) -> str:
    """Convert text to URL-safe slug.

    Args:
        text: Input text to slugify.

    Returns:
        Lowercase ASCII slug with hyphens.
    """
    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)
    # Encode to ASCII, ignoring non-ASCII chars
    text = text.encode("ascii", "ignore").decode("ascii")
    # Convert to lowercase
    text = text.lower()
    # Replace spaces and underscores with hyphens
    text = re.sub(r"[\s_]+", "-", text)
    # Remove non-alphanumeric characters except hyphens
    text = re.sub(r"[^a-z0-9-]", "", text)
    # Remove consecutive hyphens
    text = re.sub(r"-+", "-", text)
    # Strip leading/trailing hyphens
    text = text.strip("-")
    return text


def generate_filename(artist_name: str) -> str:
    """Generate WordGrain filename from artist name.

    Args:
        artist_name: Artist name to convert.

    Returns:
        Filename in format: {artist-slug}.wg.json
    """
    slug = slugify(artist_name)
    return f"{slug}.wg.json"


def _get_generator_string() -> str:
    """Get the generator string with version."""
    try:
        ver = version("barscan")
    except Exception:
        ver = "0.1.0"
    return f"barscan/{ver}"


def to_wordgrain(
    aggregate: AggregateAnalysisResult,
    language: str = "en",
) -> WordGrainDocument:
    """Convert analysis results to WordGrain format.

    Args:
        aggregate: Aggregated analysis results.
        language: ISO 639-1 language code.

    Returns:
        WordGrainDocument ready for export.
    """
    # Convert frequencies to grains
    grains: list[WordGrainGrain] = []
    for freq in aggregate.frequencies:
        # Calculate normalized frequency (per 10,000 words)
        if aggregate.total_words > 0:
            normalized = round((freq.count / aggregate.total_words) * 10000, 2)
        else:
            normalized = 0.0
        grains.append(
            WordGrainGrain(
                word=freq.word,
                frequency=freq.count,
                frequency_normalized=normalized,
            )
        )

    # Build metadata
    meta = WordGrainMeta(
        source="genius",
        artist=aggregate.artist_name,
        generated_at=aggregate.analyzed_at,
        corpus_size=aggregate.songs_analyzed,
        total_words=aggregate.total_words,
        generator=_get_generator_string(),
        language=language,
    )

    return WordGrainDocument(
        meta=meta,
        grains=tuple(grains),
    )


def export_wordgrain(
    document: WordGrainDocument,
    indent: int = 2,
) -> str:
    """Export WordGrain document to JSON string.

    Args:
        document: WordGrain document to export.
        indent: JSON indentation level.

    Returns:
        JSON string with proper formatting.
    """
    return document.model_dump_json(
        by_alias=True,
        indent=indent,
        exclude_none=True,
    )


def to_wordgrain_enhanced(
    aggregate: AggregateAnalysisResult,
    config: AnalysisConfig,
    word_counts_per_song: list[Counter[str]] | None = None,
    tokens_with_positions: list[TokenWithPosition] | None = None,
    language: str = "en",
) -> WordGrainDocument:
    """Convert analysis results to WordGrain format with enhanced NLP fields.

    This function computes additional NLP fields based on the config:
    - TF-IDF scores (requires word_counts_per_song)
    - POS tags
    - Sentiment scores
    - Slang detection
    - Context extraction (requires tokens_with_positions)

    Args:
        aggregate: Aggregated analysis results.
        config: Analysis configuration with NLP options enabled.
        word_counts_per_song: Word counts per song for TF-IDF calculation.
        tokens_with_positions: Tokens with position info for context extraction.
        language: ISO 639-1 language code.

    Returns:
        WordGrainDocument with enhanced NLP fields.
    """
    # Collect all words for batch processing
    words = [freq.word for freq in aggregate.frequencies]

    # Compute TF-IDF if enabled
    tfidf_scores: dict[str, float] = {}
    if config.compute_tfidf and word_counts_per_song:
        aggregate_counts = {freq.word: freq.count for freq in aggregate.frequencies}
        tfidf_scores = calculate_corpus_tfidf(
            word_counts_per_song=word_counts_per_song,
            aggregate_counts=aggregate_counts,
            total_words=aggregate.total_words,
            normalize=True,
        )

    # Compute POS tags if enabled
    pos_tags: dict[str, str] = {}
    if config.compute_pos:
        pos_tags = get_pos_tags(words)

    # Compute sentiment if enabled
    sentiment_scores: dict[str, tuple[str, float]] = {}
    if config.compute_sentiment:
        sentiment_scores = get_sentiment_scores(words)

    # Detect slang if enabled
    slang_flags: dict[str, bool] = {}
    if config.detect_slang:
        slang_flags = detect_slang_words(words)

    # Build grains with enhanced fields
    grains: list[WordGrainGrain] = []
    for freq in aggregate.frequencies:
        word = freq.word
        word_lower = word.lower()

        # Calculate normalized frequency
        if aggregate.total_words > 0:
            normalized = round((freq.count / aggregate.total_words) * 10000, 2)
        else:
            normalized = 0.0

        # Get TF-IDF
        tfidf = tfidf_scores.get(word) if config.compute_tfidf else None

        # Get POS
        pos = pos_tags.get(word_lower) if config.compute_pos else None

        # Get sentiment
        sentiment = None
        sentiment_score = None
        if config.compute_sentiment and word_lower in sentiment_scores:
            sentiment, sentiment_score = sentiment_scores[word_lower]

        # Get slang flag
        is_slang = slang_flags.get(word_lower) if config.detect_slang else None

        # Get contexts
        contexts: tuple[str, ...] | tuple[WordContext, ...] | None = None
        if config.contexts_mode != ContextsMode.NONE and tokens_with_positions:
            contexts = extract_contexts_for_word(
                tokens_with_positions=tokens_with_positions,
                word=word,
                mode=config.contexts_mode,
                max_contexts=config.max_contexts_per_word,
            )

        grains.append(
            WordGrainGrain(
                word=word,
                frequency=freq.count,
                frequency_normalized=normalized,
                tfidf=tfidf,
                pos=pos,
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                is_slang=is_slang,
                contexts=contexts,
            )
        )

    # Build metadata
    meta = WordGrainMeta(
        source="genius",
        artist=aggregate.artist_name,
        generated_at=aggregate.analyzed_at,
        corpus_size=aggregate.songs_analyzed,
        total_words=aggregate.total_words,
        generator=_get_generator_string(),
        language=language,
    )

    return WordGrainDocument(
        meta=meta,
        grains=tuple(grains),
    )
