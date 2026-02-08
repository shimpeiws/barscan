"""Tests for analyzer filters."""

from barscan.analyzer.filters import (
    apply_filters,
    filter_by_length,
    filter_non_alphabetic,
    filter_stop_words,
    get_stop_words,
)
from barscan.analyzer.models import AnalysisConfig


class TestGetStopWords:
    """Tests for get_stop_words function."""

    def test_english_stop_words(self, default_config: AnalysisConfig) -> None:
        """Test getting English stop words."""
        stop_words = get_stop_words(default_config)
        assert isinstance(stop_words, frozenset)
        assert "the" in stop_words
        assert "a" in stop_words
        assert "is" in stop_words
        assert "and" in stop_words

    def test_with_custom_stop_words(self) -> None:
        """Test adding custom stop words."""
        config = AnalysisConfig(custom_stop_words=frozenset(["yeah", "oh", "baby"]))
        stop_words = get_stop_words(config)
        assert "yeah" in stop_words
        assert "oh" in stop_words
        assert "baby" in stop_words
        # Should still include NLTK stop words
        assert "the" in stop_words

    def test_none_config_uses_default(self) -> None:
        """Test that None config uses default settings."""
        stop_words = get_stop_words(None)
        assert isinstance(stop_words, frozenset)
        assert "the" in stop_words


class TestFilterStopWords:
    """Tests for filter_stop_words function."""

    def test_removes_stop_words(self, default_config: AnalysisConfig) -> None:
        """Test that stop words are removed."""
        tokens = ["the", "quick", "brown", "fox", "is", "a", "animal"]
        result = filter_stop_words(tokens, default_config)
        assert "the" not in result
        assert "is" not in result
        assert "a" not in result
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result

    def test_disabled_stop_word_filtering(self, config_no_stop_words: AnalysisConfig) -> None:
        """Test that stop word filtering can be disabled."""
        tokens = ["the", "quick", "fox"]
        result = filter_stop_words(tokens, config_no_stop_words)
        assert result == tokens

    def test_empty_tokens(self, default_config: AnalysisConfig) -> None:
        """Test filtering empty token list."""
        result = filter_stop_words([], default_config)
        assert result == []


class TestFilterByLength:
    """Tests for filter_by_length function."""

    def test_default_min_length(self, default_config: AnalysisConfig) -> None:
        """Test filtering with default min length of 2."""
        tokens = ["a", "be", "cat", "door"]
        result = filter_by_length(tokens, default_config)
        assert "a" not in result
        assert "be" in result
        assert "cat" in result
        assert "door" in result

    def test_custom_min_length(self, config_min_length_3: AnalysisConfig) -> None:
        """Test filtering with custom min length of 3."""
        tokens = ["a", "be", "cat", "door"]
        result = filter_by_length(tokens, config_min_length_3)
        assert "a" not in result
        assert "be" not in result
        assert "cat" in result
        assert "door" in result

    def test_empty_tokens(self, default_config: AnalysisConfig) -> None:
        """Test filtering empty token list."""
        result = filter_by_length([], default_config)
        assert result == []


class TestFilterNonAlphabetic:
    """Tests for filter_non_alphabetic function."""

    def test_removes_numbers(self) -> None:
        """Test that tokens containing numbers are removed."""
        tokens = ["hello", "world123", "456", "test"]
        result = filter_non_alphabetic(tokens)
        assert "hello" in result
        assert "world123" not in result
        assert "456" not in result
        assert "test" in result

    def test_removes_special_characters(self) -> None:
        """Test that tokens with special characters are removed."""
        tokens = ["hello", "world!", "@test", "normal"]
        result = filter_non_alphabetic(tokens)
        assert "hello" in result
        assert "world!" not in result
        assert "@test" not in result
        assert "normal" in result

    def test_keeps_pure_alphabetic(self) -> None:
        """Test that pure alphabetic tokens are kept."""
        tokens = ["Hello", "WORLD", "test", "Word"]
        result = filter_non_alphabetic(tokens)
        assert len(result) == 4

    def test_empty_tokens(self) -> None:
        """Test filtering empty token list."""
        result = filter_non_alphabetic([])
        assert result == []


class TestApplyFilters:
    """Tests for apply_filters function."""

    def test_full_filter_pipeline(self, default_config: AnalysisConfig) -> None:
        """Test applying all filters in order."""
        tokens = ["The", "quick123", "brown", "a", "fox", "!", "is"]
        result = apply_filters(tokens, default_config)
        # Non-alphabetic removed
        assert "quick123" not in result
        assert "!" not in result
        # Stop words removed (case-insensitive check)
        result_lower = [t.lower() for t in result]
        assert "the" not in result_lower
        assert "is" not in result_lower
        # Short words removed (length < 2)
        assert "a" not in result
        # Content words kept
        assert "brown" in result
        assert "fox" in result

    def test_with_custom_filter(self, default_config: AnalysisConfig) -> None:
        """Test applying additional custom filters."""

        def remove_hello(tokens: list[str]) -> list[str]:
            return [t for t in tokens if t != "hello"]

        tokens = ["hello", "world", "goodbye"]
        result = apply_filters(tokens, default_config, additional_filters=[remove_hello])
        assert "hello" not in result
        assert "world" in result
        assert "goodbye" in result

    def test_none_config_uses_default(self) -> None:
        """Test that None config uses default settings."""
        tokens = ["the", "quick", "fox"]
        result = apply_filters(tokens, None)
        assert isinstance(result, list)
        # Stop words should be filtered with default config
        assert "the" not in result

    def test_empty_tokens(self, default_config: AnalysisConfig) -> None:
        """Test filtering empty token list."""
        result = apply_filters([], default_config)
        assert result == []
