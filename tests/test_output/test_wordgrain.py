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
    BarGrainDocument,
    BarGrainEntry,
    BarMetrics,
    BarSemantics,
    BarSource,
    WordGrainDocument,
    WordGrainGrain,
    WordGrainMeta,
    _get_generator_string,
    _version_fields,
    export_wordgrain,
    generate_filename,
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
        """Test v0.2.0 fields (schema_version, type)."""
        doc = WordGrainDocument(
            **{"$schema": WORDGRAIN_SCHEMA_URLS["0.2.0"]},
            schema_version="0.2.0",
            **{"type": "word"},
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
        assert doc.type_ == "word"

    def test_document_v010_no_extra_fields(self) -> None:
        """Test v0.1.0 document has no schema_version or type."""
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
        assert doc.type_ is None

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
        """Test default schema version is 0.2.0 with schema_version and type fields."""
        doc = to_wordgrain(sample_aggregate)
        assert doc.schema_ == WORDGRAIN_SCHEMA_URLS["0.2.0"]
        assert doc.schema_version == "0.2.0"
        assert doc.type_ == "word"

    def test_schema_version_010(self, sample_aggregate: AggregateAnalysisResult) -> None:
        """Test schema version 0.1.0 has no schema_version or type fields."""
        doc = to_wordgrain(sample_aggregate, schema_version="0.1.0")
        assert doc.schema_ == WORDGRAIN_SCHEMA_URLS["0.1.0"]
        assert doc.schema_version is None
        assert doc.type_ is None

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

    def test_export_v020_includes_schema_version_and_type(self) -> None:
        """Test that v0.2.0 export includes schema_version and type."""
        doc = WordGrainDocument(
            **{"$schema": WORDGRAIN_SCHEMA_URLS["0.2.0"]},
            schema_version="0.2.0",
            **{"type": "word"},
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
        assert data["type"] == "word"

    def test_export_v010_excludes_schema_version_and_type(self) -> None:
        """Test that v0.1.0 export excludes schema_version and type."""
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
        assert doc.type_ == "word"

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
        assert doc.type_ is None


class TestBarGrainModels:
    """Tests for bar grain models."""

    def test_bar_source_creation(self) -> None:
        """Test creating a BarSource."""
        source = BarSource(artist="Kendrick Lamar", track="HUMBLE.")
        assert source.artist == "Kendrick Lamar"
        assert source.track == "HUMBLE."
        assert source.album is None
        assert source.year is None
        assert source.featuring is None

    def test_bar_source_with_optional_fields(self) -> None:
        """Test BarSource with all optional fields."""
        source = BarSource(
            artist="Kendrick Lamar",
            track="HUMBLE.",
            album="DAMN.",
            year=2017,
            featuring="feat. Someone",
        )
        assert source.album == "DAMN."
        assert source.year == 2017

    def test_bar_source_empty_artist_rejected(self) -> None:
        """Test that empty artist is rejected."""
        with pytest.raises(ValidationError):
            BarSource(artist="", track="Test")

    def test_bar_source_empty_track_rejected(self) -> None:
        """Test that empty track is rejected."""
        with pytest.raises(ValidationError):
            BarSource(artist="Test", track="")

    def test_bar_metrics_creation(self) -> None:
        """Test creating BarMetrics."""
        metrics = BarMetrics(lines=1)
        assert metrics.lines == 1
        assert metrics.syllables is None
        assert metrics.mora is None

    def test_bar_metrics_zero_lines_rejected(self) -> None:
        """Test that zero lines is rejected."""
        with pytest.raises(ValidationError):
            BarMetrics(lines=0)

    def test_bar_semantics_valid_mood(self) -> None:
        """Test BarSemantics with valid mood."""
        sem = BarSemantics(mood="aggressive")
        assert sem.mood == "aggressive"

    def test_bar_semantics_invalid_mood_rejected(self) -> None:
        """Test BarSemantics with invalid mood."""
        with pytest.raises(ValidationError):
            BarSemantics(mood="happy")

    def test_bar_semantics_none_mood(self) -> None:
        """Test BarSemantics with no mood."""
        sem = BarSemantics()
        assert sem.mood is None

    def test_bar_grain_entry_creation(self) -> None:
        """Test creating a BarGrainEntry."""
        entry = BarGrainEntry(
            text="I got loyalty, got royalty inside my DNA",
            source=BarSource(artist="Kendrick Lamar", track="DNA."),
            metrics=BarMetrics(lines=1),
            language="en",
        )
        assert entry.text == "I got loyalty, got royalty inside my DNA"
        assert entry.source.artist == "Kendrick Lamar"
        assert entry.metrics.lines == 1
        assert entry.language == "en"

    def test_bar_grain_entry_empty_text_rejected(self) -> None:
        """Test that empty text is rejected."""
        with pytest.raises(ValidationError):
            BarGrainEntry(
                text="",
                source=BarSource(artist="Test", track="Test"),
            )

    def test_bar_grain_entry_is_frozen(self) -> None:
        """Test that BarGrainEntry is immutable."""
        entry = BarGrainEntry(
            text="test line",
            source=BarSource(artist="Test", track="Test"),
        )
        with pytest.raises(ValidationError):
            entry.text = "new text"  # type: ignore[misc]


class TestBarGrainDocument:
    """Tests for BarGrainDocument model."""

    def test_create_bar_document(self) -> None:
        """Test creating a BarGrainDocument."""
        meta = WordGrainMeta(
            artist="Test Artist",
            generated_at=datetime.now(UTC),
            corpus_size=1,
            total_words=2,
            generator="barscan/0.3.0",
        )
        grains = (
            BarGrainEntry(
                text="line one",
                source=BarSource(artist="Test Artist", track="Song 1"),
                metrics=BarMetrics(lines=1),
            ),
            BarGrainEntry(
                text="line two",
                source=BarSource(artist="Test Artist", track="Song 1"),
                metrics=BarMetrics(lines=1),
            ),
        )
        doc = BarGrainDocument(
            **{"$schema": WORDGRAIN_SCHEMA_URLS["0.2.0"]},
            schema_version="0.2.0",
            **{"type": "bar"},
            meta=meta,
            grains=grains,
        )
        assert doc.type_ == "bar"
        assert doc.schema_version == "0.2.0"
        assert len(doc.grains) == 2

    def test_bar_document_serialization(self) -> None:
        """Test BarGrainDocument serializes with correct type."""
        meta = WordGrainMeta(
            artist="Test",
            generated_at=datetime.now(UTC),
            corpus_size=1,
            total_words=1,
            generator="barscan/0.3.0",
        )
        doc = BarGrainDocument(
            **{"$schema": WORDGRAIN_SCHEMA_URLS["0.2.0"]},
            schema_version="0.2.0",
            **{"type": "bar"},
            meta=meta,
            grains=(
                BarGrainEntry(
                    text="test line",
                    source=BarSource(artist="Test", track="Song"),
                    metrics=BarMetrics(lines=1),
                ),
            ),
        )
        json_str = doc.model_dump_json(by_alias=True, exclude_none=True)
        data = json.loads(json_str)
        assert data["type"] == "bar"
        assert data["schema_version"] == "0.2.0"
        assert data["grains"][0]["text"] == "test line"


class TestVersionFields:
    """Tests for _version_fields function."""

    def test_version_fields_word_type(self) -> None:
        """Test _version_fields with default word type."""
        fields = _version_fields("0.2.0")
        assert fields["type"] == "word"
        assert fields["schema_version"] == "0.2.0"

    def test_version_fields_bar_type(self) -> None:
        """Test _version_fields with bar type."""
        fields = _version_fields("0.2.0", "bar")
        assert fields["type"] == "bar"
        assert fields["schema_version"] == "0.2.0"

    def test_version_fields_010_no_type(self) -> None:
        """Test _version_fields for v0.1.0 has no type or schema_version."""
        fields = _version_fields("0.1.0")
        assert "type" not in fields
        assert "schema_version" not in fields

    def test_version_fields_010_bar_type_ignored(self) -> None:
        """Test _version_fields for v0.1.0 ignores bar type."""
        fields = _version_fields("0.1.0", "bar")
        assert "type" not in fields


class TestToWordgrainBar:
    """Tests for to_wordgrain_bar function."""

    def test_basic_conversion(self) -> None:
        """Test basic conversion from lyrics to bar grains."""
        lyrics_data = [
            ("First line\nSecond line", 1, "Song One"),
        ]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test Artist")
        assert len(doc.grains) == 2
        assert doc.grains[0].text == "First line"
        assert doc.grains[1].text == "Second line"
        assert doc.grains[0].source.artist == "Test Artist"
        assert doc.grains[0].source.track == "Song One"

    def test_multiple_songs(self) -> None:
        """Test conversion with multiple songs."""
        lyrics_data = [
            ("Line from song 1", 1, "Song One"),
            ("Line from song 2", 2, "Song Two"),
        ]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test Artist")
        assert len(doc.grains) == 2
        assert doc.grains[0].source.track == "Song One"
        assert doc.grains[1].source.track == "Song Two"
        assert doc.meta.corpus_size == 2

    def test_empty_lines_skipped(self) -> None:
        """Test that empty lines are skipped."""
        lyrics_data = [
            ("Line one\n\nLine two\n  \nLine three", 1, "Song"),
        ]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test")
        assert len(doc.grains) == 3

    def test_metrics_lines_always_one(self) -> None:
        """Test that each grain has metrics.lines = 1."""
        lyrics_data = [("A line", 1, "Song")]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test")
        assert doc.grains[0].metrics is not None
        assert doc.grains[0].metrics.lines == 1

    def test_language_propagation(self) -> None:
        """Test that language is propagated to grains and meta."""
        lyrics_data = [("テスト", 1, "Song")]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test", language="ja")
        assert doc.meta.language == "ja"
        assert doc.grains[0].language == "ja"

    def test_total_words_is_line_count(self) -> None:
        """Test that meta.total_words equals total line count for bar type."""
        lyrics_data = [
            ("Line 1\nLine 2\nLine 3", 1, "Song"),
        ]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test")
        assert doc.meta.total_words == 3

    def test_type_is_bar(self) -> None:
        """Test that document type is 'bar'."""
        lyrics_data = [("A line", 1, "Song")]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test")
        assert doc.type_ == "bar"
        assert doc.schema_version == "0.2.0"

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
        assert len(doc.grains) == 3
        assert doc.grains[0].text == doc.grains[2].text


class TestExportWordgrainBar:
    """Tests for exporting bar documents."""

    def test_export_bar_document(self) -> None:
        """Test that bar document exports correctly."""
        lyrics_data = [("Test line one\nTest line two", 1, "Song")]
        doc = to_wordgrain_bar(lyrics_data, artist_name="Test Artist")
        json_str = export_wordgrain(doc)
        data = json.loads(json_str)
        assert data["type"] == "bar"
        assert data["schema_version"] == "0.2.0"
        assert len(data["grains"]) == 2
        assert data["grains"][0]["text"] == "Test line one"
        assert data["grains"][0]["source"]["artist"] == "Test Artist"
        assert data["grains"][0]["source"]["track"] == "Song"
        assert data["grains"][0]["metrics"]["lines"] == 1
        assert data["grains"][0]["language"] == "en"
