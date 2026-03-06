"""Tests for WordGrain output module."""

import json
from collections import Counter
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from barscan.analyzer.models import AggregateAnalysisResult, AnalysisConfig, WordFrequency
from barscan.output.wordgrain import (
    DEFAULT_WORDGRAIN_SCHEMA_VERSION,
    WORDGRAIN_SCHEMA_URL,
    WORDGRAIN_SCHEMA_URLS,
    BarGrainEntry,
    BarMetrics,
    BarSemantics,
    BarSource,
    SongLyricsData,
    WordGrainDocument,
    WordGrainGrain,
    WordGrainMeta,
    _get_generator_string,
    _version_fields,
    export_wordgrain,
    generate_filename,
    parse_featuring,
    resolve_wordgrain_language,
    slugify,
    to_wordgrain,
    to_wordgrain_bar,
    to_wordgrain_enhanced,
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
        """Test default schema URL points to v0.2.0."""
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
        assert doc.schema_ == WORDGRAIN_SCHEMA_URLS["0.2.0"]

    def test_document_v020_fields(self) -> None:
        """Test v0.2.0 fields (schema_version, no type)."""
        doc = WordGrainDocument(
            **{"$schema": WORDGRAIN_SCHEMA_URLS["0.2.0"]},
            schema_version="0.2.0",
            meta=WordGrainMeta(
                artist="Test",
                generated_at=datetime.now(UTC),
                corpus_size=1,
                total_words=100,
                generator="test/0.1.0",
            ),
            grains=(),
        )
        assert doc.schema_version == "0.2.0"

    def test_document_v010_no_extra_fields(self) -> None:
        """Test v0.1.0 document has no schema_version."""
        doc = WordGrainDocument(
            **{"$schema": WORDGRAIN_SCHEMA_URLS["0.1.0"]},
            meta=WordGrainMeta(
                artist="Test",
                generated_at=datetime.now(UTC),
                corpus_size=1,
                total_words=100,
                generator="test/0.1.0",
            ),
            grains=(),
        )
        assert doc.schema_version is None

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

    def test_document_with_bars(self) -> None:
        """Test creating a unified document with both grains and bars."""
        meta = WordGrainMeta(
            artist="Test Artist",
            generated_at=datetime.now(UTC),
            corpus_size=2,
            total_words=100,
            generator="barscan/0.3.0",
        )
        grains = (
            WordGrainGrain(word="love", frequency=50, frequency_normalized=500.0),
        )
        bars = (
            BarGrainEntry(
                text="I got love in my heart",
                source=BarSource(track="Song 1"),
                language="en",
            ),
        )
        doc = WordGrainDocument(
            **{"$schema": WORDGRAIN_SCHEMA_URLS["0.2.0"]},
            schema_version="0.2.0",
            meta=meta,
            grains=grains,
            bars=bars,
        )
        assert len(doc.grains) == 1
        assert doc.bars is not None
        assert len(doc.bars) == 1
        assert doc.bars[0].text == "I got love in my heart"

    def test_unified_document_serialization(self) -> None:
        """Test that unified document serializes with both grains and bars."""
        meta = WordGrainMeta(
            artist="Test",
            generated_at=datetime.now(UTC),
            corpus_size=1,
            total_words=10,
            generator="barscan/0.3.0",
        )
        doc = WordGrainDocument(
            **{"$schema": WORDGRAIN_SCHEMA_URLS["0.2.0"]},
            schema_version="0.2.0",
            meta=meta,
            grains=(WordGrainGrain(word="test", frequency=1, frequency_normalized=100.0),),
            bars=(BarGrainEntry(text="test line", source=BarSource(track="Song"), language="en"),),
        )
        json_str = export_wordgrain(doc)
        data = json.loads(json_str)
        assert "grains" in data
        assert "bars" in data
        assert "type" not in data
        assert len(data["grains"]) == 1
        assert len(data["bars"]) == 1


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


class TestResolveWordgrainLanguage:
    """Tests for resolve_wordgrain_language function."""

    def test_english_maps_to_en(self) -> None:
        """Test that 'english' maps to 'en'."""
        assert resolve_wordgrain_language("english") == "en"

    def test_japanese_maps_to_ja(self) -> None:
        """Test that 'japanese' maps to 'ja'."""
        assert resolve_wordgrain_language("japanese") == "ja"

    def test_auto_detects_english(self) -> None:
        """Test that 'auto' detects English from words."""
        assert resolve_wordgrain_language("auto", ["love", "heart", "soul"]) == "en"

    def test_auto_detects_japanese(self) -> None:
        """Test that 'auto' detects Japanese from words."""
        assert resolve_wordgrain_language("auto", ["愛", "心", "魂"]) == "ja"

    def test_auto_without_words_defaults_to_en(self) -> None:
        """Test that 'auto' without words defaults to 'en'."""
        assert resolve_wordgrain_language("auto") == "en"

    def test_auto_with_empty_words_defaults_to_en(self) -> None:
        """Test that 'auto' with empty words list defaults to 'en'."""
        assert resolve_wordgrain_language("auto", []) == "en"

    def test_unknown_language_defaults_to_en(self) -> None:
        """Test that unknown language defaults to 'en'."""
        assert resolve_wordgrain_language("unknown") == "en"


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

    def test_default_schema_version(self, sample_aggregate: AggregateAnalysisResult) -> None:
        """Test default schema version is 0.2.0 with schema_version field."""
        doc = to_wordgrain(sample_aggregate)
        assert doc.schema_ == WORDGRAIN_SCHEMA_URLS["0.2.0"]
        assert doc.schema_version == "0.2.0"

    def test_schema_version_010(self, sample_aggregate: AggregateAnalysisResult) -> None:
        """Test schema version 0.1.0 has no schema_version field."""
        doc = to_wordgrain(sample_aggregate, schema_version="0.1.0")
        assert doc.schema_ == WORDGRAIN_SCHEMA_URLS["0.1.0"]
        assert doc.schema_version is None

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

    def test_export_v020_includes_schema_version(self) -> None:
        """Test that v0.2.0 export includes schema_version but no type."""
        doc = WordGrainDocument(
            **{"$schema": WORDGRAIN_SCHEMA_URLS["0.2.0"]},
            schema_version="0.2.0",
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
        assert data["schema_version"] == "0.2.0"
        assert "type" not in data

    def test_export_v010_excludes_schema_version(self) -> None:
        """Test that v0.1.0 export excludes schema_version."""
        doc = WordGrainDocument(
            **{"$schema": WORDGRAIN_SCHEMA_URLS["0.1.0"]},
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
        assert "schema_version" not in data
        assert "type" not in data

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


class TestGetGeneratorString:
    """Tests for _get_generator_string function."""

    def test_returns_version_string(self) -> None:
        """Test that generator string includes version."""
        result = _get_generator_string()
        assert result.startswith("barscan/")
        assert len(result) > len("barscan/")

    @patch("barscan.output.wordgrain.version")
    def test_fallback_on_version_error(self, mock_version: patch) -> None:
        """Test fallback when version lookup fails."""
        mock_version.side_effect = Exception("Package not found")
        result = _get_generator_string()
        assert result == "barscan/0.1.0"


class TestToWordgrainEnhanced:
    """Tests for to_wordgrain_enhanced function."""

    @pytest.fixture
    def sample_aggregate(self) -> AggregateAnalysisResult:
        """Create a sample aggregate result."""
        frequencies = (
            WordFrequency(word="love", count=50, percentage=1.0),
            WordFrequency(word="heart", count=30, percentage=0.6),
        )
        return AggregateAnalysisResult(
            artist_name="Test Artist",
            songs_analyzed=2,
            total_words=1000,
            unique_words=100,
            frequencies=frequencies,
            song_results=(),
            analyzed_at=datetime(2026, 2, 9, 12, 0, 0, tzinfo=UTC),
        )

    def test_basic_enhanced_output(self, sample_aggregate: AggregateAnalysisResult) -> None:
        """Test basic enhanced output without any NLP features."""
        config = AnalysisConfig()
        doc = to_wordgrain_enhanced(sample_aggregate, config)

        assert len(doc.grains) == 2
        assert doc.grains[0].word == "love"
        assert doc.grains[0].frequency == 50

    def test_enhanced_with_tfidf(self, sample_aggregate: AggregateAnalysisResult) -> None:
        """Test enhanced output with TF-IDF enabled."""
        config = AnalysisConfig(compute_tfidf=True)
        word_counts = [
            Counter({"love": 30, "heart": 20}),
            Counter({"love": 20, "heart": 10}),
        ]

        doc = to_wordgrain_enhanced(
            aggregate=sample_aggregate,
            config=config,
            word_counts_per_song=word_counts,
        )

        # TF-IDF should be computed
        assert doc.grains[0].tfidf is not None

    def test_enhanced_with_pos(self, sample_aggregate: AggregateAnalysisResult) -> None:
        """Test enhanced output with POS tagging enabled."""
        config = AnalysisConfig(compute_pos=True)

        doc = to_wordgrain_enhanced(aggregate=sample_aggregate, config=config)

        # POS tags should be computed
        assert doc.grains[0].pos is not None

    def test_enhanced_with_sentiment(self, sample_aggregate: AggregateAnalysisResult) -> None:
        """Test enhanced output with sentiment enabled."""
        config = AnalysisConfig(compute_sentiment=True)

        doc = to_wordgrain_enhanced(aggregate=sample_aggregate, config=config)

        # Sentiment should be computed for "love" (positive word)
        assert doc.grains[0].sentiment is not None
        assert doc.grains[0].sentiment_score is not None

    def test_enhanced_with_slang_detection(
        self, sample_aggregate: AggregateAnalysisResult
    ) -> None:
        """Test enhanced output with slang detection enabled."""
        config = AnalysisConfig(detect_slang=True)

        doc = to_wordgrain_enhanced(aggregate=sample_aggregate, config=config)

        # Slang flag should be set
        assert doc.grains[0].is_slang is not None

    def test_enhanced_without_word_counts(
        self, sample_aggregate: AggregateAnalysisResult
    ) -> None:
        """Test that TF-IDF is None when word_counts_per_song is not provided."""
        config = AnalysisConfig(compute_tfidf=True)

        doc = to_wordgrain_enhanced(aggregate=sample_aggregate, config=config)

        # TF-IDF should be None since word_counts not provided
        assert doc.grains[0].tfidf is None

    def test_enhanced_meta_mapping(self, sample_aggregate: AggregateAnalysisResult) -> None:
        """Test that meta fields are mapped correctly."""
        config = AnalysisConfig()

        doc = to_wordgrain_enhanced(aggregate=sample_aggregate, config=config)

        assert doc.meta.artist == "Test Artist"
        assert doc.meta.corpus_size == 2
        assert doc.meta.total_words == 1000

    def test_enhanced_custom_language(self, sample_aggregate: AggregateAnalysisResult) -> None:
        """Test custom language setting."""
        config = AnalysisConfig()

        doc = to_wordgrain_enhanced(
            aggregate=sample_aggregate, config=config, language="ja"
        )

        assert doc.meta.language == "ja"

    def test_enhanced_derives_language_from_config_english(
        self, sample_aggregate: AggregateAnalysisResult
    ) -> None:
        """Test that language is derived from config when not explicitly passed."""
        config = AnalysisConfig(language="english")

        doc = to_wordgrain_enhanced(aggregate=sample_aggregate, config=config)

        assert doc.meta.language == "en"

    def test_enhanced_derives_language_from_config_japanese(self) -> None:
        """Test that Japanese config language produces 'ja' output."""
        frequencies = (
            WordFrequency(word="愛", count=50, percentage=1.0),
            WordFrequency(word="心", count=30, percentage=0.6),
        )
        aggregate = AggregateAnalysisResult(
            artist_name="Test Artist",
            songs_analyzed=2,
            total_words=1000,
            unique_words=100,
            frequencies=frequencies,
            song_results=(),
            analyzed_at=datetime(2026, 2, 9, 12, 0, 0, tzinfo=UTC),
        )
        config = AnalysisConfig(language="japanese")

        doc = to_wordgrain_enhanced(aggregate=aggregate, config=config)

        assert doc.meta.language == "ja"

    def test_enhanced_derives_language_from_config_auto(self) -> None:
        """Test that 'auto' config detects language from words."""
        frequencies = (
            WordFrequency(word="愛", count=50, percentage=1.0),
            WordFrequency(word="心", count=30, percentage=0.6),
        )
        aggregate = AggregateAnalysisResult(
            artist_name="Test Artist",
            songs_analyzed=2,
            total_words=1000,
            unique_words=100,
            frequencies=frequencies,
            song_results=(),
            analyzed_at=datetime(2026, 2, 9, 12, 0, 0, tzinfo=UTC),
        )
        config = AnalysisConfig(language="auto")

        doc = to_wordgrain_enhanced(aggregate=aggregate, config=config)

        assert doc.meta.language == "ja"

    def test_enhanced_frequency_normalization(
        self, sample_aggregate: AggregateAnalysisResult
    ) -> None:
        """Test that frequency_normalized is calculated correctly."""
        config = AnalysisConfig()

        doc = to_wordgrain_enhanced(aggregate=sample_aggregate, config=config)

        # 50 / 1000 * 10000 = 500.0
        assert doc.grains[0].frequency_normalized == 500.0
        # 30 / 1000 * 10000 = 300.0
        assert doc.grains[1].frequency_normalized == 300.0

    def test_enhanced_default_schema_version(
        self, sample_aggregate: AggregateAnalysisResult
    ) -> None:
        """Test default schema version is 0.2.0."""
        config = AnalysisConfig()
        doc = to_wordgrain_enhanced(aggregate=sample_aggregate, config=config)
        assert doc.schema_ == WORDGRAIN_SCHEMA_URLS["0.2.0"]
        assert doc.schema_version == "0.2.0"

    def test_enhanced_schema_version_010(
        self, sample_aggregate: AggregateAnalysisResult
    ) -> None:
        """Test schema version 0.1.0 has no extra fields."""
        config = AnalysisConfig()
        doc = to_wordgrain_enhanced(
            aggregate=sample_aggregate, config=config, schema_version="0.1.0"
        )
        assert doc.schema_ == WORDGRAIN_SCHEMA_URLS["0.1.0"]
        assert doc.schema_version is None


class TestBarGrainModels:
    """Tests for bar grain models."""

    def test_bar_source_creation(self) -> None:
        """Test creating a BarSource."""
        source = BarSource(track="HUMBLE.")
        assert source.track == "HUMBLE."
        assert source.album is None
        assert source.year is None
        assert source.featuring is None
        assert source.timestamp is None

    def test_bar_source_with_optional_fields(self) -> None:
        """Test BarSource with all optional fields."""
        source = BarSource(
            track="HUMBLE.",
            album="DAMN.",
            year=2017,
            featuring=("Someone",),
            timestamp="01:23",
        )
        assert source.album == "DAMN."
        assert source.year == 2017
        assert source.featuring == ("Someone",)
        assert source.timestamp == "01:23"

    def test_bar_source_empty_track_rejected(self) -> None:
        """Test that empty track is rejected."""
        with pytest.raises(ValidationError):
            BarSource(track="")

    def test_bar_metrics_creation(self) -> None:
        """Test creating BarMetrics."""
        metrics = BarMetrics(syllable_count=5, word_count=3)
        assert metrics.syllable_count == 5
        assert metrics.word_count == 3
        assert metrics.rhyme_density is None

    def test_bar_metrics_all_optional(self) -> None:
        """Test that all BarMetrics fields are optional."""
        metrics = BarMetrics()
        assert metrics.syllable_count is None
        assert metrics.word_count is None
        assert metrics.rhyme_density is None

    def test_bar_semantics_valid_mood(self) -> None:
        """Test BarSemantics with valid mood."""
        sem = BarSemantics(mood="aggressive")
        assert sem.mood == "aggressive"

    def test_bar_semantics_new_moods(self) -> None:
        """Test BarSemantics with new v0.2.0 moods."""
        for mood in ("triumphant", "humorous", "hopeful", "celebratory"):
            sem = BarSemantics(mood=mood)
            assert sem.mood == mood

    def test_bar_semantics_invalid_mood_rejected(self) -> None:
        """Test BarSemantics with invalid mood."""
        with pytest.raises(ValidationError):
            BarSemantics(mood="happy")

    def test_bar_semantics_removed_moods_rejected(self) -> None:
        """Test that removed moods (euphoric, playful) are rejected."""
        with pytest.raises(ValidationError):
            BarSemantics(mood="euphoric")
        with pytest.raises(ValidationError):
            BarSemantics(mood="playful")

    def test_bar_semantics_none_mood(self) -> None:
        """Test BarSemantics with no mood."""
        sem = BarSemantics()
        assert sem.mood is None

    def test_bar_semantics_themes_and_techniques(self) -> None:
        """Test BarSemantics with themes and techniques."""
        sem = BarSemantics(
            mood="defiant",
            themes=("struggle", "resilience"),
            techniques=("metaphor", "alliteration"),
        )
        assert sem.themes == ("struggle", "resilience")
        assert sem.techniques == ("metaphor", "alliteration")

    def test_bar_grain_entry_creation(self) -> None:
        """Test creating a BarGrainEntry."""
        entry = BarGrainEntry(
            text="I got loyalty, got royalty inside my DNA",
            source=BarSource(track="DNA."),
            language="en",
        )
        assert entry.text == "I got loyalty, got royalty inside my DNA"
        assert entry.source.track == "DNA."
        assert entry.language == "en"

    def test_bar_grain_entry_empty_text_rejected(self) -> None:
        """Test that empty text is rejected."""
        with pytest.raises(ValidationError):
            BarGrainEntry(
                text="",
                source=BarSource(track="Test"),
            )

    def test_bar_grain_entry_is_frozen(self) -> None:
        """Test that BarGrainEntry is immutable."""
        entry = BarGrainEntry(
            text="test line",
            source=BarSource(track="Test"),
        )
        with pytest.raises(ValidationError):
            entry.text = "new text"  # type: ignore[misc]


class TestVersionFields:
    """Tests for _version_fields function."""

    def test_version_fields_020(self) -> None:
        """Test _version_fields for v0.2.0."""
        fields = _version_fields("0.2.0")
        assert fields["schema_version"] == "0.2.0"
        assert "type" not in fields

    def test_version_fields_010_no_schema_version(self) -> None:
        """Test _version_fields for v0.1.0 has no schema_version."""
        fields = _version_fields("0.1.0")
        assert "type" not in fields
        assert "schema_version" not in fields


class TestToWordgrainBar:
    """Tests for to_wordgrain_bar function."""

    def test_basic_conversion(self) -> None:
        """Test basic conversion from lyrics to bar entries."""
        lyrics_data = [
            ("First line\nSecond line", 1, "Song One"),
        ]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test Artist")
        assert doc.bars is not None
        assert len(doc.bars) == 2
        assert doc.bars[0].text == "First line"
        assert doc.bars[1].text == "Second line"
        assert doc.bars[0].source.track == "Song One"

    def test_multiple_songs(self) -> None:
        """Test conversion with multiple songs."""
        lyrics_data = [
            ("Line from song 1", 1, "Song One"),
            ("Line from song 2", 2, "Song Two"),
        ]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test Artist")
        assert doc.bars is not None
        assert len(doc.bars) == 2
        assert doc.bars[0].source.track == "Song One"
        assert doc.bars[1].source.track == "Song Two"
        assert doc.meta.corpus_size == 2

    def test_empty_lines_skipped(self) -> None:
        """Test that empty lines are skipped."""
        lyrics_data = [
            ("Line one\n\nLine two\n  \nLine three", 1, "Song"),
        ]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test")
        assert doc.bars is not None
        assert len(doc.bars) == 3

    def test_no_metrics_by_default(self) -> None:
        """Test that bars have no metrics by default."""
        lyrics_data = [("A line", 1, "Song")]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test")
        assert doc.bars is not None
        assert doc.bars[0].metrics is None

    def test_language_propagation(self) -> None:
        """Test that language is propagated to bars and meta."""
        lyrics_data = [("テスト", 1, "Song")]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test", language="ja")
        assert doc.meta.language == "ja"
        assert doc.bars is not None
        assert doc.bars[0].language == "ja"

    def test_total_words_is_line_count(self) -> None:
        """Test that meta.total_words equals total line count for bar type."""
        lyrics_data = [
            ("Line 1\nLine 2\nLine 3", 1, "Song"),
        ]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test")
        assert doc.meta.total_words == 3

    def test_no_type_field(self) -> None:
        """Test that document has no type field (unified format)."""
        lyrics_data = [("A line", 1, "Song")]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test")
        assert doc.schema_version == "0.2.0"
        # No type_ attribute in unified format
        assert not hasattr(doc, "type_") or not hasattr(WordGrainDocument, "type_")

    def test_schema_below_020_raises_error(self) -> None:
        """Test that schema version below 0.2.0 raises ValueError."""
        lyrics_data = [("A line", 1, "Song")]
        with pytest.raises(ValueError, match="Bar type requires WordGrain schema >= 0.2.0"):
            to_wordgrain_bar(lyrics_data, artist_name="Test", schema_version="0.1.0")

    def test_chorus_repeats_kept(self) -> None:
        """Test that repeated lines (chorus) are kept."""
        lyrics_data = [
            ("Chorus line\nVerse line\nChorus line", 1, "Song"),
        ]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test")
        assert doc.bars is not None
        assert len(doc.bars) == 3
        assert doc.bars[0].text == doc.bars[2].text

    def test_artist_in_meta_not_source(self) -> None:
        """Test that artist is in meta, not in bar source."""
        lyrics_data = [("A line", 1, "Song")]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test Artist")
        assert doc.meta.artist == "Test Artist"
        assert doc.bars is not None
        # BarSource should not have artist field
        source_data = doc.bars[0].source.model_dump()
        assert "artist" not in source_data


class TestSongLyricsData:
    """Tests for SongLyricsData NamedTuple."""

    def test_create_basic(self) -> None:
        """Test creating with required fields."""
        data = SongLyricsData("lyrics", 1, "Song")
        assert data.lyrics_text == "lyrics"
        assert data.song_id == 1
        assert data.song_title == "Song"
        assert data.title_with_featured == ""

    def test_create_with_featured(self) -> None:
        """Test creating with title_with_featured."""
        data = SongLyricsData("lyrics", 1, "Song", "Song (ft. Guest)")
        assert data.title_with_featured == "Song (ft. Guest)"

    def test_tuple_unpacking(self) -> None:
        """Test backward-compatible tuple unpacking."""
        data = SongLyricsData("lyrics", 1, "Song")
        text, sid, title, twf = data
        assert text == "lyrics"
        assert sid == 1
        assert title == "Song"
        assert twf == ""


class TestParseFeaturing:
    """Tests for parse_featuring function."""

    def test_ft_single_artist(self) -> None:
        result = parse_featuring("Song (ft. Guest)", "Song")
        assert result == ("Guest",)

    def test_feat_single_artist(self) -> None:
        result = parse_featuring("Song (feat. Guest)", "Song")
        assert result == ("Guest",)

    def test_ft_multiple_ampersand(self) -> None:
        result = parse_featuring("Song (ft. A & B)", "Song")
        assert result == ("A", "B")

    def test_feat_multiple_comma_and_ampersand(self) -> None:
        result = parse_featuring("Song (Ft. A, B & C)", "Song")
        assert result == ("A", "B", "C")

    def test_no_featuring(self) -> None:
        result = parse_featuring("Song", "Song")
        assert result is None

    def test_case_insensitive(self) -> None:
        result = parse_featuring("Song (FEAT. Guest)", "Song")
        assert result == ("Guest",)


class TestToWordgrainBarEnriched:
    """Tests for enriched bar output with config."""

    def test_word_count_english(self) -> None:
        """Test that word_count is computed for English."""
        config = AnalysisConfig()
        lyrics_data = [SongLyricsData("hello world foo bar", 1, "Song")]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test", config=config)
        assert doc.bars is not None
        assert doc.bars[0].metrics is not None
        assert doc.bars[0].metrics.word_count == 4

    def test_word_count_japanese(self) -> None:
        """Test that word_count uses tokenizer for Japanese (not whitespace split)."""
        config = AnalysisConfig(language="japanese")
        lyrics_data = [SongLyricsData("君は俺の過去より今を", 1, "Song")]
        doc = to_wordgrain_bar(
            lyrics_data, artist_name="Test", language="ja", config=config
        )
        assert doc.bars is not None
        assert doc.bars[0].metrics is not None
        # 君/は/俺/の/過去/より/今/を = 8 tokens (POS filtering disabled for word count)
        assert doc.bars[0].metrics.word_count == 8

    def test_syllable_count_english(self) -> None:
        """Test that syllable_count is computed for English."""
        config = AnalysisConfig()
        lyrics_data = [SongLyricsData("hello world", 1, "Song")]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test", config=config)
        assert doc.bars is not None
        assert doc.bars[0].metrics is not None
        assert doc.bars[0].metrics.syllable_count is not None
        assert doc.bars[0].metrics.syllable_count >= 2

    def test_syllable_count_japanese_is_mora(self) -> None:
        """Test that syllable_count returns mora count for Japanese."""
        config = AnalysisConfig()
        lyrics_data = [SongLyricsData("テスト", 1, "Song")]
        doc = to_wordgrain_bar(
            lyrics_data, artist_name="Test", language="ja", config=config
        )
        assert doc.bars is not None
        assert doc.bars[0].metrics is not None
        # テスト = テ(1) + ス(1) + ト(1) = 3 morae
        assert doc.bars[0].metrics.syllable_count == 3

    def test_mood_present(self) -> None:
        """Test that mood is computed when config is provided."""
        config = AnalysisConfig()
        lyrics_data = [SongLyricsData("I love you so much", 1, "Song")]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test", config=config)
        assert doc.bars is not None
        assert doc.bars[0].semantics is not None
        assert doc.bars[0].semantics.mood is not None

    def test_techniques_detected(self) -> None:
        """Test that techniques are detected when present."""
        config = AnalysisConfig()
        lyrics_data = [SongLyricsData("big bad boy broke barriers", 1, "Song")]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test", config=config)
        assert doc.bars is not None
        assert doc.bars[0].semantics is not None
        assert doc.bars[0].semantics.techniques is not None
        assert "alliteration" in doc.bars[0].semantics.techniques

    def test_featuring_parsed(self) -> None:
        """Test that featuring artists are parsed from title_with_featured."""
        config = AnalysisConfig()
        lyrics_data = [
            SongLyricsData("test line", 1, "Song", "Song (ft. Guest)")
        ]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test", config=config)
        assert doc.bars is not None
        assert doc.bars[0].source.featuring == ("Guest",)

    def test_no_enrichment_without_config(self) -> None:
        """Test that no enrichment happens when config is None."""
        lyrics_data = [SongLyricsData("hello world", 1, "Song")]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test")
        assert doc.bars is not None
        assert doc.bars[0].metrics is None
        assert doc.bars[0].semantics is None

    def test_backward_compatible_tuple_input(self) -> None:
        """Test that plain tuple input still works."""
        lyrics_data = [("hello world", 1, "Song")]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test")
        assert doc.bars is not None
        assert len(doc.bars) == 1
        assert doc.bars[0].text == "hello world"


class TestExportWordgrainBar:
    """Tests for exporting bar documents."""

    def test_export_bar_document(self) -> None:
        """Test that bar document exports correctly."""
        lyrics_data = [("Test line one\nTest line two", 1, "Song")]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test Artist")
        json_str = export_wordgrain(doc)
        data = json.loads(json_str)
        assert "type" not in data
        assert data["schema_version"] == "0.2.0"
        assert "bars" in data
        assert len(data["bars"]) == 2
        assert data["bars"][0]["text"] == "Test line one"
        assert data["bars"][0]["source"]["track"] == "Song"
        assert "artist" not in data["bars"][0]["source"]
        assert data["bars"][0]["language"] == "en"
        assert data["meta"]["artist"] == "Test Artist"
