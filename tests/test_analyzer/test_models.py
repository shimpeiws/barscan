"""Tests for analyzer models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from barscan.analyzer.models import (
    AggregateAnalysisResult,
    AnalysisConfig,
    AnalysisResult,
    WordFrequency,
)


class TestWordFrequency:
    """Tests for WordFrequency model."""

    def test_create_word_frequency(self) -> None:
        """Test creating a WordFrequency instance."""
        wf = WordFrequency(word="hello", count=5, percentage=25.0)
        assert wf.word == "hello"
        assert wf.count == 5
        assert wf.percentage == 25.0

    def test_word_frequency_is_frozen(self) -> None:
        """Test that WordFrequency is immutable."""
        wf = WordFrequency(word="hello", count=5, percentage=25.0)
        with pytest.raises(ValidationError):
            wf.word = "world"  # type: ignore[misc]

    def test_word_frequency_validation_empty_word(self) -> None:
        """Test that empty word is rejected."""
        with pytest.raises(ValidationError):
            WordFrequency(word="", count=5, percentage=25.0)

    def test_word_frequency_validation_zero_count(self) -> None:
        """Test that zero count is rejected."""
        with pytest.raises(ValidationError):
            WordFrequency(word="hello", count=0, percentage=25.0)

    def test_word_frequency_validation_negative_percentage(self) -> None:
        """Test that negative percentage is rejected."""
        with pytest.raises(ValidationError):
            WordFrequency(word="hello", count=5, percentage=-1.0)

    def test_word_frequency_validation_over_100_percentage(self) -> None:
        """Test that percentage over 100 is rejected."""
        with pytest.raises(ValidationError):
            WordFrequency(word="hello", count=5, percentage=101.0)


class TestAnalysisConfig:
    """Tests for AnalysisConfig model."""

    def test_default_config(self) -> None:
        """Test creating config with defaults."""
        config = AnalysisConfig()
        assert config.min_word_length == 2
        assert config.use_lemmatization is False
        assert config.remove_stop_words is True
        assert config.custom_stop_words == frozenset()
        assert config.language == "auto"
        assert config.use_pos_filtering is True

    def test_custom_config(self) -> None:
        """Test creating config with custom values."""
        config = AnalysisConfig(
            min_word_length=3,
            use_lemmatization=True,
            remove_stop_words=False,
            custom_stop_words=frozenset(["yeah", "oh"]),
            language="spanish",
        )
        assert config.min_word_length == 3
        assert config.use_lemmatization is True
        assert config.remove_stop_words is False
        assert config.custom_stop_words == frozenset(["yeah", "oh"])
        assert config.language == "spanish"

    def test_config_is_frozen(self) -> None:
        """Test that AnalysisConfig is immutable."""
        config = AnalysisConfig()
        with pytest.raises(ValidationError):
            config.min_word_length = 5  # type: ignore[misc]

    def test_config_min_word_length_validation(self) -> None:
        """Test that min_word_length must be at least 1."""
        with pytest.raises(ValidationError):
            AnalysisConfig(min_word_length=0)


class TestAnalysisResult:
    """Tests for AnalysisResult model."""

    def test_create_analysis_result(self) -> None:
        """Test creating an AnalysisResult instance."""
        now = datetime.now(UTC)
        frequencies = (
            WordFrequency(word="hello", count=5, percentage=50.0),
            WordFrequency(word="world", count=5, percentage=50.0),
        )
        result = AnalysisResult(
            song_id=123,
            song_title="Test Song",
            artist_name="Test Artist",
            total_words=10,
            unique_words=2,
            frequencies=frequencies,
            analyzed_at=now,
        )
        assert result.song_id == 123
        assert result.song_title == "Test Song"
        assert result.artist_name == "Test Artist"
        assert result.total_words == 10
        assert result.unique_words == 2
        assert len(result.frequencies) == 2
        assert result.analyzed_at == now

    def test_top_words(self) -> None:
        """Test top_words method."""
        frequencies = tuple(
            WordFrequency(word=f"word{i}", count=10 - i, percentage=10.0) for i in range(5)
        )
        result = AnalysisResult(
            song_id=1,
            song_title="Test",
            artist_name="Artist",
            total_words=50,
            unique_words=5,
            frequencies=frequencies,
            analyzed_at=datetime.now(UTC),
        )
        top3 = result.top_words(3)
        assert len(top3) == 3
        assert top3[0].word == "word0"
        assert top3[1].word == "word1"
        assert top3[2].word == "word2"

    def test_top_words_default(self) -> None:
        """Test top_words with default n=10."""
        frequencies = tuple(
            WordFrequency(word=f"word{i}", count=20 - i, percentage=5.0) for i in range(15)
        )
        result = AnalysisResult(
            song_id=1,
            song_title="Test",
            artist_name="Artist",
            total_words=100,
            unique_words=15,
            frequencies=frequencies,
            analyzed_at=datetime.now(UTC),
        )
        top = result.top_words()
        assert len(top) == 10

    def test_analysis_result_is_frozen(self) -> None:
        """Test that AnalysisResult is immutable."""
        result = AnalysisResult(
            song_id=1,
            song_title="Test",
            artist_name="Artist",
            total_words=0,
            unique_words=0,
            frequencies=(),
            analyzed_at=datetime.now(UTC),
        )
        with pytest.raises(ValidationError):
            result.song_title = "New Title"  # type: ignore[misc]


class TestAggregateAnalysisResult:
    """Tests for AggregateAnalysisResult model."""

    def test_create_aggregate_result(self) -> None:
        """Test creating an AggregateAnalysisResult instance."""
        now = datetime.now(UTC)
        result = AggregateAnalysisResult(
            artist_name="Test Artist",
            songs_analyzed=5,
            total_words=1000,
            unique_words=200,
            frequencies=(),
            song_results=(),
            analyzed_at=now,
        )
        assert result.artist_name == "Test Artist"
        assert result.songs_analyzed == 5
        assert result.total_words == 1000
        assert result.unique_words == 200

    def test_aggregate_top_words(self) -> None:
        """Test top_words method on aggregate result."""
        frequencies = tuple(
            WordFrequency(word=f"word{i}", count=100 - i, percentage=1.0) for i in range(20)
        )
        result = AggregateAnalysisResult(
            artist_name="Artist",
            songs_analyzed=10,
            total_words=1000,
            unique_words=20,
            frequencies=frequencies,
            song_results=(),
            analyzed_at=datetime.now(UTC),
        )
        top5 = result.top_words(5)
        assert len(top5) == 5
        assert top5[0].word == "word0"

    def test_aggregate_result_is_frozen(self) -> None:
        """Test that AggregateAnalysisResult is immutable."""
        result = AggregateAnalysisResult(
            artist_name="Artist",
            songs_analyzed=0,
            total_words=0,
            unique_words=0,
            frequencies=(),
            song_results=(),
            analyzed_at=datetime.now(UTC),
        )
        with pytest.raises(ValidationError):
            result.artist_name = "New Artist"  # type: ignore[misc]
