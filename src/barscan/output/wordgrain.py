"""WordGrain format output for BarScan.

This module provides Pydantic models and functions to export analysis results
in the WordGrain JSON format (.wg.json), a standardized schema for vocabulary
analysis data.

Reference: https://raw.githubusercontent.com/shimpeiws/word-grain/main/schema/v0.2.0/wordgrain.schema.json
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from typing import NamedTuple

from pydantic import BaseModel, Field, field_validator

from barscan.analyzer.context import extract_contexts_for_word
from barscan.analyzer.models import (
    AggregateAnalysisResult,
    AnalysisConfig,
    ContextsMode,
    TokenWithPosition,
    WordContext,
)
from barscan.analyzer.pos import get_pos_tags
from barscan.analyzer.processor import clean_lyrics_preserve_lines, tokenize
from barscan.analyzer.sentiment import (
    analyze_sentiment,
    get_sentiment_scores,
    map_sentiment_to_mood,
)
from barscan.analyzer.slang import detect_slang_words
from barscan.analyzer.syllable import count_line_syllables
from barscan.analyzer.techniques import detect_techniques
from barscan.analyzer.tfidf import calculate_corpus_tfidf
from barscan.analyzer.tokenizer import detect_language


class SongLyricsData(NamedTuple):
    """Data about a song's lyrics for bar generation."""

    lyrics_text: str
    song_id: int
    song_title: str
    title_with_featured: str = ""


# Pattern to extract featured artists from title_with_featured
_FEATURING_PATTERN = re.compile(
    r"\(\s*(?:ft\.?|feat\.?|Ft\.?|Feat\.?)\s+(.+?)\s*\)",
    re.IGNORECASE,
)


def parse_featuring(title_with_featured: str, song_title: str) -> tuple[str, ...] | None:
    """Extract featured artist names from title_with_featured.

    Args:
        title_with_featured: Full title including featured artists.
        song_title: Base song title without features.

    Returns:
        Tuple of featured artist names, or None if no features found.
    """
    match = _FEATURING_PATTERN.search(title_with_featured)
    if not match:
        return None

    artists_str = match.group(1)
    # Split on '&', ',' and strip whitespace
    artists = re.split(r"\s*[,&]\s*", artists_str)
    artists = [a.strip() for a in artists if a.strip()]
    return tuple(artists) if artists else None


WORDGRAIN_SCHEMA_URLS: dict[str, str] = {
    "0.1.0": "https://raw.githubusercontent.com/shimpeiws/word-grain/main/schema/v0.1.0/wordgrain.schema.json",
    "0.2.0": "https://raw.githubusercontent.com/shimpeiws/word-grain/main/schema/v0.2.0/wordgrain.schema.json",
}

DEFAULT_WORDGRAIN_SCHEMA_VERSION = "0.2.0"

WORDGRAIN_SCHEMA_URL = WORDGRAIN_SCHEMA_URLS[DEFAULT_WORDGRAIN_SCHEMA_VERSION]

# Mapping from AnalysisConfig language names to ISO 639-1 codes
_LANGUAGE_TO_ISO: dict[str, str] = {
    "english": "en",
    "japanese": "ja",
}

_ISO_TO_LANGUAGE: dict[str, str] = {v: k for k, v in _LANGUAGE_TO_ISO.items()}


def resolve_wordgrain_language(config_language: str, words: list[str] | None = None) -> str:
    """Resolve AnalysisConfig language to ISO 639-1 code for WordGrain output.

    Args:
        config_language: Language from AnalysisConfig ('english', 'japanese', or 'auto').
        words: Words from analysis results, used for auto-detection.

    Returns:
        ISO 639-1 language code (e.g., 'en', 'ja').
    """
    if config_language in _LANGUAGE_TO_ISO:
        return _LANGUAGE_TO_ISO[config_language]

    # For "auto", detect from words
    if config_language == "auto" and words:
        detected = detect_language(" ".join(words))
        return _LANGUAGE_TO_ISO.get(detected, "en")

    return "en"


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


_VALID_MOODS = frozenset(
    {
        "aggressive",
        "melancholic",
        "triumphant",
        "reflective",
        "humorous",
        "romantic",
        "defiant",
        "hopeful",
        "dark",
        "celebratory",
    }
)


class BarSource(BaseModel, frozen=True):
    """Source metadata for a bar grain entry."""

    track: str = Field(..., min_length=1)
    album: str | None = Field(default=None)
    year: int | None = Field(default=None)
    featuring: tuple[str, ...] | None = Field(default=None)
    timestamp: str | None = Field(default=None)


class BarMetrics(BaseModel, frozen=True):
    """Metrics for a bar grain entry."""

    syllable_count: int | None = Field(default=None)
    word_count: int | None = Field(default=None)
    rhyme_density: float | None = Field(default=None)


class BarSemantics(BaseModel, frozen=True):
    """Semantics for a bar grain entry."""

    mood: str | None = Field(default=None)
    themes: tuple[str, ...] | None = Field(default=None)
    techniques: tuple[str, ...] | None = Field(default=None)

    @field_validator("mood")
    @classmethod
    def validate_mood(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_MOODS:
            msg = f"Invalid mood '{v}'. Must be one of: {', '.join(sorted(_VALID_MOODS))}"
            raise ValueError(msg)
        return v


class BarGrainEntry(BaseModel, frozen=True):
    """A single bar (lyric line) entry in WordGrain bar format."""

    text: str = Field(..., min_length=1)
    source: BarSource
    metrics: BarMetrics | None = Field(default=None)
    semantics: BarSemantics | None = Field(default=None)
    language: str = Field(default="en")


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
    """Root WordGrain document structure (unified format).

    Attributes:
        schema_: JSON Schema URL (serialized as $schema).
        schema_version: Schema version string (v0.2.0+).
        meta: Document metadata.
        grains: List of word entries (vocabulary).
        bars: List of bar entries (lyric lines).
    """

    schema_: str = Field(
        default=WORDGRAIN_SCHEMA_URL,
        alias="$schema",
        description="JSON Schema URL",
    )
    schema_version: str | None = Field(
        default=None,
        description="Schema version (v0.2.0+)",
    )
    meta: WordGrainMeta = Field(..., description="Document metadata")
    grains: tuple[WordGrainGrain, ...] = Field(
        default_factory=tuple, description="List of word entries"
    )
    bars: tuple[BarGrainEntry, ...] | None = Field(default=None, description="List of bar entries")


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


def _version_fields(schema_version: str) -> dict[str, str]:
    """Return version-specific fields for WordGrainDocument constructor."""
    schema_url = WORDGRAIN_SCHEMA_URLS.get(schema_version, WORDGRAIN_SCHEMA_URL)
    fields: dict[str, str] = {"$schema": schema_url}
    if schema_version >= "0.2.0":
        fields["schema_version"] = schema_version
    return fields


def to_wordgrain(
    aggregate: AggregateAnalysisResult,
    language: str = "en",
    schema_version: str = DEFAULT_WORDGRAIN_SCHEMA_VERSION,
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
        **_version_fields(schema_version),  # type: ignore[arg-type]
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
    language: str | None = None,
    schema_version: str = DEFAULT_WORDGRAIN_SCHEMA_VERSION,
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
        language: ISO 639-1 language code. If None, derived from config.language.

    Returns:
        WordGrainDocument with enhanced NLP fields.
    """
    # Collect all words for batch processing
    words = [freq.word for freq in aggregate.frequencies]

    # Resolve language from config if not explicitly provided
    if language is None:
        language = resolve_wordgrain_language(config.language, words)

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
        sentiment_language = _ISO_TO_LANGUAGE.get(language, "english")
        sentiment_scores = get_sentiment_scores(words, language=sentiment_language)

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
        **_version_fields(schema_version),  # type: ignore[arg-type]
        meta=meta,
        grains=tuple(grains),
    )


def to_wordgrain_bar(
    lyrics_data: list[SongLyricsData] | list[tuple[str, int, str]],
    artist_name: str,
    language: str = "en",
    schema_version: str = DEFAULT_WORDGRAIN_SCHEMA_VERSION,
    config: AnalysisConfig | None = None,
) -> WordGrainDocument:
    """Convert lyrics data to WordGrain bar format (line-level).

    Args:
        lyrics_data: List of SongLyricsData or (lyrics_text, song_id, song_title) tuples.
        artist_name: Primary artist name.
        language: ISO 639-1 language code.
        schema_version: WordGrain schema version (must be >= 0.2.0).
        config: When provided, compute enrichment fields (metrics, semantics, source.featuring).

    Returns:
        WordGrainDocument with bars field containing one entry per lyric line.

    Raises:
        ValueError: If schema_version < 0.2.0 (bars require v0.2.0+).
    """
    if schema_version < "0.2.0":
        raise ValueError(f"Bar type requires WordGrain schema >= 0.2.0, got '{schema_version}'")

    bars: list[BarGrainEntry] = []
    corpus_size = 0

    for item in lyrics_data:
        if isinstance(item, SongLyricsData):
            lyrics_text = item.lyrics_text
            song_title = item.song_title
            title_with_featured = item.title_with_featured
        else:
            lyrics_text, _, song_title = item
            title_with_featured = ""

        lines = clean_lyrics_preserve_lines(lyrics_text)
        corpus_size += 1

        # Pre-compute per-song fields
        featuring = (
            parse_featuring(title_with_featured, song_title)
            if config is not None and title_with_featured
            else None
        )

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            source = BarSource(track=song_title, featuring=featuring)
            metrics: BarMetrics | None = None
            semantics: BarSemantics | None = None

            if config is not None:
                # Word count (use tokenizer without POS filtering for full count)
                wc_config = (
                    config.model_copy(update={"use_pos_filtering": False})
                    if config.use_pos_filtering
                    else config
                )
                word_count = len(tokenize(stripped, wc_config))
                # Syllable count
                syllable_count = count_line_syllables(stripped, language)
                metrics = BarMetrics(word_count=word_count, syllable_count=syllable_count)

                # Mood from sentiment
                _category, compound = analyze_sentiment(stripped)
                mood = map_sentiment_to_mood(compound, stripped)
                # Techniques
                techniques = detect_techniques(stripped, language)
                semantics = BarSemantics(mood=mood, techniques=techniques)

            bars.append(
                BarGrainEntry(
                    text=stripped,
                    source=source,
                    metrics=metrics,
                    semantics=semantics,
                    language=language,
                )
            )

    meta = WordGrainMeta(
        source="genius",
        artist=artist_name,
        generated_at=datetime.now(),
        corpus_size=corpus_size,
        total_words=len(bars),
        generator=_get_generator_string(),
        language=language,
    )

    return WordGrainDocument(
        **_version_fields(schema_version),  # type: ignore[arg-type]
        meta=meta,
        bars=tuple(bars),
    )
