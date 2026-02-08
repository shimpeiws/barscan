"""Tests for WordGrain output module."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from barscan.analyzer.models import AggregateAnalysisResult, WordFrequency
from barscan.output.wordgrain import (
    WORDGRAIN_SCHEMA_URL,
    WordGrainDocument,
    WordGrainGrain,
    WordGrainMeta,
    export_wordgrain,
    generate_filename,
    slugify,
    to_wordgrain,
)


class TestWordGrainGrain:
    """Tests for WordGrainGrain model."""

    def test_create_grain(self) -> None:
        """Test creating a WordGrainGrain instance."""
        grain = WordGrainGrain(word="love", frequency=50, frequency_normalized=100.0)
        assert grain.word == "love"
        assert grain.frequency == 50
        assert grain.frequency_normalized == 100.0

    def test_grain_is_frozen(self) -> None:
        """Test that WordGrainGrain is immutable."""
        grain = WordGrainGrain(word="love", frequency=50, frequency_normalized=100.0)
        with pytest.raises(ValidationError):
            grain.word = "heart"  # type: ignore[misc]

    def test_grain_validation_empty_word(self) -> None:
        """Test that empty word is rejected."""
        with pytest.raises(ValidationError):
            WordGrainGrain(word="", frequency=50, frequency_normalized=100.0)

    def test_grain_validation_negative_frequency(self) -> None:
        """Test that negative frequency is rejected."""
        with pytest.raises(ValidationError):
            WordGrainGrain(word="love", frequency=-1, frequency_normalized=100.0)

    def test_grain_validation_negative_normalized(self) -> None:
        """Test that negative frequency_normalized is rejected."""
        with pytest.raises(ValidationError):
            WordGrainGrain(word="love", frequency=50, frequency_normalized=-1.0)


class TestWordGrainMeta:
    """Tests for WordGrainMeta model."""

    def test_create_meta_with_defaults(self) -> None:
        """Test creating meta with default values."""
        now = datetime.now(UTC)
        meta = WordGrainMeta(
            artist="Kendrick Lamar",
            generated_at=now,
            corpus_size=10,
            total_words=5000,
            generator="barscan/0.1.0",
        )
        assert meta.source == "genius"
        assert meta.artist == "Kendrick Lamar"
        assert meta.language == "en"
        assert meta.corpus_size == 10
        assert meta.total_words == 5000

    def test_create_meta_custom(self) -> None:
        """Test creating meta with custom values."""
        now = datetime.now(UTC)
        meta = WordGrainMeta(
            source="custom",
            artist="J. Cole",
            generated_at=now,
            corpus_size=5,
            total_words=2500,
            generator="barscan/1.0.0",
            language="es",
        )
        assert meta.source == "custom"
        assert meta.language == "es"

    def test_meta_is_frozen(self) -> None:
        """Test that WordGrainMeta is immutable."""
        meta = WordGrainMeta(
            artist="Test",
            generated_at=datetime.now(UTC),
            corpus_size=1,
            total_words=100,
            generator="test/0.1.0",
        )
        with pytest.raises(ValidationError):
            meta.artist = "New Artist"  # type: ignore[misc]


class TestWordGrainDocument:
    """Tests for WordGrainDocument model."""

    def test_create_document(self) -> None:
        """Test creating a WordGrainDocument."""
        now = datetime.now(UTC)
        meta = WordGrainMeta(
            artist="Test Artist",
            generated_at=now,
            corpus_size=5,
            total_words=1000,
            generator="barscan/0.1.0",
        )
        grains = (
            WordGrainGrain(word="love", frequency=50, frequency_normalized=500.0),
            WordGrainGrain(word="heart", frequency=30, frequency_normalized=300.0),
        )
        doc = WordGrainDocument(meta=meta, grains=grains)
        assert doc.schema_ == WORDGRAIN_SCHEMA_URL
        assert doc.meta == meta
        assert len(doc.grains) == 2

    def test_document_default_schema(self) -> None:
        """Test default schema URL."""
        doc = WordGrainDocument(
            meta=WordGrainMeta(
                artist="Test",
                generated_at=datetime.now(UTC),
                corpus_size=1,
                total_words=100,
                generator="test/0.1.0",
            ),
            grains=(),
        )
        assert doc.schema_ == "https://mumbl.dev/schemas/wordgrain/v0.1.0"

    def test_schema_field_alias(self) -> None:
        """Test that $schema field is serialized correctly."""
        doc = WordGrainDocument(
            meta=WordGrainMeta(
                artist="Test",
                generated_at=datetime.now(UTC),
                corpus_size=1,
                total_words=100,
                generator="test/0.1.0",
            ),
            grains=(),
        )
        json_str = doc.model_dump_json(by_alias=True)
        data = json.loads(json_str)
        assert "$schema" in data
        assert "schema_" not in data
        assert data["$schema"] == WORDGRAIN_SCHEMA_URL


class TestSlugify:
    """Tests for slugify function."""

    def test_slugify_simple(self) -> None:
        """Test slugifying a simple name."""
        assert slugify("Kendrick Lamar") == "kendrick-lamar"

    def test_slugify_with_special_chars(self) -> None:
        """Test slugifying a name with special characters."""
        assert slugify("J. Cole") == "j-cole"

    def test_slugify_with_unicode(self) -> None:
        """Test slugifying a name with unicode characters."""
        assert slugify("Björk") == "bjork"

    def test_slugify_multiple_spaces(self) -> None:
        """Test slugifying a name with multiple spaces."""
        assert slugify("The   Weeknd") == "the-weeknd"

    def test_slugify_empty(self) -> None:
        """Test slugifying an empty string."""
        assert slugify("") == ""

    def test_slugify_special_only(self) -> None:
        """Test slugifying only special characters."""
        assert slugify("!@#$%") == ""


class TestGenerateFilename:
    """Tests for generate_filename function."""

    def test_generate_filename_simple(self) -> None:
        """Test generating filename from simple name."""
        assert generate_filename("Kendrick Lamar") == "kendrick-lamar.wg.json"

    def test_generate_filename_special_chars(self) -> None:
        """Test generating filename with special characters."""
        assert generate_filename("Tyler, the Creator") == "tyler-the-creator.wg.json"


class TestToWordgrain:
    """Tests for to_wordgrain converter."""

    @pytest.fixture
    def sample_aggregate(self) -> AggregateAnalysisResult:
        """Create a sample AggregateAnalysisResult for testing."""
        frequencies = (
            WordFrequency(word="love", count=50, percentage=1.0),
            WordFrequency(word="heart", count=30, percentage=0.6),
            WordFrequency(word="soul", count=20, percentage=0.4),
        )
        return AggregateAnalysisResult(
            artist_name="Test Artist",
            songs_analyzed=5,
            total_words=5000,
            unique_words=500,
            frequencies=frequencies,
            song_results=(),
            analyzed_at=datetime(2026, 2, 9, 12, 0, 0, tzinfo=UTC),
        )

    def test_converts_frequencies(self, sample_aggregate: AggregateAnalysisResult) -> None:
        """Test that frequencies are converted correctly."""
        doc = to_wordgrain(sample_aggregate)
        assert len(doc.grains) == 3
        assert doc.grains[0].word == "love"
        assert doc.grains[0].frequency == 50
        assert doc.grains[1].word == "heart"
        assert doc.grains[1].frequency == 30

    def test_frequency_normalization(self, sample_aggregate: AggregateAnalysisResult) -> None:
        """Test that frequency_normalized is calculated correctly (per 10,000 words)."""
        doc = to_wordgrain(sample_aggregate)
        # 50 / 5000 * 10000 = 100.0
        assert doc.grains[0].frequency_normalized == 100.0
        # 30 / 5000 * 10000 = 60.0
        assert doc.grains[1].frequency_normalized == 60.0
        # 20 / 5000 * 10000 = 40.0
        assert doc.grains[2].frequency_normalized == 40.0

    def test_meta_mapping(self, sample_aggregate: AggregateAnalysisResult) -> None:
        """Test that meta fields are mapped correctly."""
        doc = to_wordgrain(sample_aggregate)
        assert doc.meta.artist == "Test Artist"
        assert doc.meta.corpus_size == 5
        assert doc.meta.total_words == 5000
        assert doc.meta.generated_at == datetime(2026, 2, 9, 12, 0, 0, tzinfo=UTC)
        assert doc.meta.source == "genius"

    def test_custom_language(self, sample_aggregate: AggregateAnalysisResult) -> None:
        """Test custom language setting."""
        doc = to_wordgrain(sample_aggregate, language="es")
        assert doc.meta.language == "es"

    def test_zero_total_words(self) -> None:
        """Test handling of zero total_words."""
        aggregate = AggregateAnalysisResult(
            artist_name="Empty Artist",
            songs_analyzed=0,
            total_words=0,
            unique_words=0,
            frequencies=(),
            song_results=(),
            analyzed_at=datetime.now(UTC),
        )
        doc = to_wordgrain(aggregate)
        assert len(doc.grains) == 0


class TestExportWordgrain:
    """Tests for export_wordgrain function."""

    def test_export_valid_json(self) -> None:
        """Test that export produces valid JSON."""
        doc = WordGrainDocument(
            meta=WordGrainMeta(
                artist="Test Artist",
                generated_at=datetime(2026, 2, 9, 12, 0, 0, tzinfo=UTC),
                corpus_size=5,
                total_words=1000,
                generator="barscan/0.1.0",
            ),
            grains=(WordGrainGrain(word="love", frequency=50, frequency_normalized=500.0),),
        )
        json_str = export_wordgrain(doc)
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_export_schema_field(self) -> None:
        """Test that $schema appears in output."""
        doc = WordGrainDocument(
            meta=WordGrainMeta(
                artist="Test",
                generated_at=datetime.now(UTC),
                corpus_size=1,
                total_words=100,
                generator="test/0.1.0",
            ),
            grains=(),
        )
        json_str = export_wordgrain(doc)
        data = json.loads(json_str)
        assert "$schema" in data
        assert data["$schema"] == WORDGRAIN_SCHEMA_URL

    def test_export_format_indentation(self) -> None:
        """Test JSON formatting with indentation."""
        doc = WordGrainDocument(
            meta=WordGrainMeta(
                artist="Test",
                generated_at=datetime.now(UTC),
                corpus_size=1,
                total_words=100,
                generator="test/0.1.0",
            ),
            grains=(),
        )
        json_str = export_wordgrain(doc, indent=2)
        assert "\n" in json_str
        assert "  " in json_str

    def test_export_no_indentation(self) -> None:
        """Test JSON formatting without indentation."""
        doc = WordGrainDocument(
            meta=WordGrainMeta(
                artist="Test",
                generated_at=datetime.now(UTC),
                corpus_size=1,
                total_words=100,
                generator="test/0.1.0",
            ),
            grains=(),
        )
        json_str = export_wordgrain(doc, indent=0)
        lines = json_str.strip().split("\n")
        assert len(lines) > 1

    def test_export_grains_structure(self) -> None:
        """Test that grains are exported with correct structure."""
        doc = WordGrainDocument(
            meta=WordGrainMeta(
                artist="Test",
                generated_at=datetime.now(UTC),
                corpus_size=1,
                total_words=100,
                generator="test/0.1.0",
            ),
            grains=(
                WordGrainGrain(word="love", frequency=50, frequency_normalized=5000.0),
                WordGrainGrain(word="heart", frequency=30, frequency_normalized=3000.0),
            ),
        )
        json_str = export_wordgrain(doc)
        data = json.loads(json_str)
        assert "grains" in data
        assert len(data["grains"]) == 2
        assert data["grains"][0]["word"] == "love"
        assert data["grains"][0]["frequency"] == 50
        assert data["grains"][0]["frequency_normalized"] == 5000.0
