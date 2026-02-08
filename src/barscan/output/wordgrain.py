"""WordGrain format output for BarScan.

This module provides Pydantic models and functions to export analysis results
in the WordGrain JSON format (.wg.json), a standardized schema for vocabulary
analysis data.

Reference: https://mumbl.dev/schemas/wordgrain/v0.1.0
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from importlib.metadata import version

from pydantic import BaseModel, Field

from barscan.analyzer.models import AggregateAnalysisResult

WORDGRAIN_SCHEMA_URL = "https://mumbl.dev/schemas/wordgrain/v0.1.0"


class WordGrainGrain(BaseModel, frozen=True):
    """A single word entry in WordGrain format.

    Attributes:
        word: The vocabulary word.
        frequency: Raw occurrence count.
        frequency_normalized: Occurrences per 10,000 words.
    """

    word: str = Field(..., min_length=1, description="The vocabulary word")
    frequency: int = Field(..., ge=0, description="Raw occurrence count")
    frequency_normalized: float = Field(..., ge=0.0, description="Occurrences per 10,000 words")


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
    )
