"""Tests for WordGrain output module."""

import json
from collections import Counter
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from barscan.analyzer.models import AggregateAnalysisResult, AnalysisConfig, WordFrequency
from barscan.output.wordgrain import (
    WORDGRAIN_SCHEMA_URL,
    WordGrainDocument,
    WordGrainGrain,
    WordGrainMeta,
    _get_generator_string,
    export_wordgrain,
    generate_filename,
    resolve_wordgrain_language,
    slugify,
    to_wordgrain,
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
        assert doc.schema_ == "https://raw.githubusercontent.com/shimpeiws/word-grain/main/schema/v0.1.0/wordgrain.schema.json"

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
