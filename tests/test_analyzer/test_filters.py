"""Tests for analyzer filters."""

import pytest

from barscan.analyzer.filters import (
    apply_filters,
    filter_by_length,
    filter_non_alphabetic,
    filter_stop_words,
    get_stop_words,
)
from barscan.analyzer.models import AnalysisConfig

# Check if Japanese dependencies are available
try:
    from stopwordsiso import stopwords as _  # noqa: F401

    HAS_JAPANESE_DEPS = True
except ImportError:
    HAS_JAPANESE_DEPS = False

requires_japanese = pytest.mark.skipif(
    not HAS_JAPANESE_DEPS,
    reason="Japanese dependencies not installed (pip install barscan[japanese])",
)


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

    @requires_japanese
    def test_japanese_includes_english_stop_words(self) -> None:
        """Test that Japanese mode includes English stop words for mixed text."""
        config = AnalysisConfig(language="japanese")
        stop_words = get_stop_words(config)
        # Should include English stop words
        assert "the" in stop_words
        assert "a" in stop_words
        assert "is" in stop_words
        assert "me" in stop_words
        assert "it" in stop_words
        assert "on" in stop_words


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

    @requires_japanese
    def test_japanese_mode_filters_english_stop_words(self) -> None:
        """Test that Japanese mode filters English stop words in mixed text."""
        config = AnalysisConfig(language="japanese")
        # Mixed Japanese/English tokens
        tokens = ["知る", "the", "me", "仲間", "it", "on", "見る"]
        result = filter_stop_words(tokens, config)
        # English stop words should be removed
        assert "the" not in result
        assert "me" not in result
        assert "it" not in result
        assert "on" not in result
        # Japanese content words should remain
        assert "知る" in result
        assert "仲間" in result
        assert "見る" in result

    @requires_japanese
    def test_japanese_mode_filters_capitalized_english_stop_words(self) -> None:
        """Test that Japanese mode filters capitalized English stop words."""
        config = AnalysisConfig(language="japanese")
        tokens = ["Yeah", "The", "Me", "All", "仲間"]
        result = filter_stop_words(tokens, config)
        # Capitalized versions should also be filtered
        assert "The" not in result
        assert "Me" not in result
        assert "All" not in result
        # Japanese content words should remain
        assert "仲間" in result


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


class TestFiltersEdgeCases:
    """Additional edge case tests for filter functions."""

    def test_filter_stop_words_with_none_config(self) -> None:
        """Test filter_stop_words with None config."""
        tokens = ["the", "quick", "fox"]
        result = filter_stop_words(tokens, None)
        # Should use default config and filter stop words
        assert "the" not in result

    def test_filter_by_length_with_none_config(self) -> None:
        """Test filter_by_length with None config."""
        tokens = ["a", "be", "cat"]
        result = filter_by_length(tokens, None)
        # Should use default min length of 2
        assert "a" not in result
        assert "be" in result
        assert "cat" in result

    def test_get_stop_words_combines_nltk_and_custom(self) -> None:
        """Test that get_stop_words combines NLTK and custom stop words."""
        config = AnalysisConfig(custom_stop_words=frozenset(["yeah", "uh", "oh"]))
        stop_words = get_stop_words(config)
        # Should include NLTK stop words
        assert "the" in stop_words
        assert "a" in stop_words
        # Should include custom stop words
        assert "yeah" in stop_words
        assert "uh" in stop_words
        assert "oh" in stop_words

    def test_filter_stop_words_case_insensitive(self, default_config: AnalysisConfig) -> None:
        """Test that stop word filtering is case-insensitive."""
        tokens = ["THE", "Quick", "THE", "FOX"]
        result = filter_stop_words(tokens, default_config)
        # THE should be filtered regardless of case
        result_lower = [t.lower() for t in result]
        assert "the" not in result_lower

    def test_filter_non_alphabetic_with_apostrophes(self) -> None:
        """Test filtering tokens with apostrophes."""
        tokens = ["don't", "won't", "it's"]
        result = filter_non_alphabetic(tokens)
        # Tokens with apostrophes are not purely alphabetic
        assert "don't" not in result

    def test_filter_non_alphabetic_with_hyphens(self) -> None:
        """Test filtering hyphenated tokens."""
        tokens = ["well-known", "self-aware", "test"]
        result = filter_non_alphabetic(tokens)
        # Hyphenated words are not purely alphabetic
        assert "well-known" not in result
        assert "test" in result

    def test_apply_filters_with_multiple_additional_filters(
        self, default_config: AnalysisConfig
    ) -> None:
        """Test applying multiple additional filters."""

        def filter_starting_with_a(tokens: list[str]) -> list[str]:
            return [t for t in tokens if not t.lower().startswith("a")]

        def filter_ending_with_s(tokens: list[str]) -> list[str]:
            return [t for t in tokens if not t.lower().endswith("s")]

        tokens = ["apple", "banana", "cats", "dog", "elephants"]
        result = apply_filters(
            tokens,
            default_config,
            additional_filters=[filter_starting_with_a, filter_ending_with_s],
        )
        # apple filtered by starts with 'a'
        assert "apple" not in result
        # cats filtered by ends with 's'
        assert "cats" not in result
        # elephants filtered by both
        assert "elephants" not in result
        # banana and dog should remain
        assert "banana" in result
        assert "dog" in result

    def test_filter_by_length_all_filtered(self) -> None:
        """Test when all tokens are filtered by length."""
        config = AnalysisConfig(min_word_length=10)
        tokens = ["a", "be", "cat", "dog"]
        result = filter_by_length(tokens, config)
        assert result == []

    def test_filter_non_alphabetic_all_filtered(self) -> None:
        """Test when all tokens are non-alphabetic."""
        tokens = ["123", "456", "78.9", "10!"]
        result = filter_non_alphabetic(tokens)
        assert result == []

    def test_apply_filters_order_matters(self, default_config: AnalysisConfig) -> None:
        """Test that filter order is correct."""
        # Non-alphabetic filter runs first, then length, then stop words
        tokens = ["The123", "quick", "a", "fox"]
        result = apply_filters(tokens, default_config)
        # The123 should be filtered by non-alphabetic filter
        assert "The123" not in result
        # 'a' should be filtered by length
        assert "a" not in result
        # 'quick' and 'fox' should remain
        assert "quick" in result
        assert "fox" in result
