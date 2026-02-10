"""Tests for sentiment analysis module."""

import pytest

from barscan.analyzer.sentiment import (
    analyze_sentiment,
    analyze_word_sentiment,
    get_sentiment_scores,
)


class TestAnalyzeSentiment:
    """Tests for analyze_sentiment function."""

    def test_positive_word(self) -> None:
        """Test sentiment analysis of a positive word."""
        category, score = analyze_sentiment("love")
        assert category == "positive"
        assert score > 0

    def test_negative_word(self) -> None:
        """Test sentiment analysis of a negative word."""
        category, score = analyze_sentiment("hate")
        assert category == "negative"
        assert score < 0

    def test_neutral_word(self) -> None:
        """Test sentiment analysis of a neutral word."""
        category, score = analyze_sentiment("the")
        assert category == "neutral"
        assert -0.05 <= score <= 0.05

    def test_positive_sentence(self) -> None:
        """Test sentiment analysis of a positive sentence."""
        category, score = analyze_sentiment("I love this beautiful day")
        assert category == "positive"
        assert score > 0

    def test_negative_sentence(self) -> None:
        """Test sentiment analysis of a negative sentence."""
        category, score = analyze_sentiment("This is terrible and awful")
        assert category == "negative"
        assert score < 0

    def test_score_range(self) -> None:
        """Test that scores are within valid range."""
        category, score = analyze_sentiment("amazing wonderful great")
        assert -1.0 <= score <= 1.0

    def test_score_rounding(self) -> None:
        """Test that scores are properly rounded."""
        category, score = analyze_sentiment("love")
        # Score should be rounded to 4 decimal places
        score_str = str(score)
        if "." in score_str:
            decimals = len(score_str.split(".")[1])
            assert decimals <= 4


class TestAnalyzeWordSentiment:
    """Tests for analyze_word_sentiment function."""

    def test_single_word(self) -> None:
        """Test analyzing a single word."""
        category, score = analyze_word_sentiment("happy")
        assert isinstance(category, str)
        assert isinstance(score, float)

    def test_returns_tuple(self) -> None:
        """Test that function returns a tuple."""
        result = analyze_word_sentiment("love")
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestGetSentimentScores:
    """Tests for get_sentiment_scores function."""

    def test_empty_list(self) -> None:
        """Test with empty word list."""
        result = get_sentiment_scores([])
        assert result == {}

    def test_single_word(self) -> None:
        """Test with single word."""
        result = get_sentiment_scores(["love"])
        assert "love" in result
        category, score = result["love"]
        assert category == "positive"

    def test_multiple_words(self) -> None:
        """Test with multiple words."""
        result = get_sentiment_scores(["love", "hate", "the"])
        assert len(result) == 3
        assert "love" in result
        assert "hate" in result
        assert "the" in result

    def test_deduplication(self) -> None:
        """Test that duplicate words are deduplicated."""
        result = get_sentiment_scores(["love", "love", "love"])
        assert len(result) == 1

    def test_case_insensitive(self) -> None:
        """Test that lookup is case-insensitive."""
        result = get_sentiment_scores(["Love", "LOVE", "love"])
        assert len(result) == 1
        assert "love" in result

    def test_result_structure(self) -> None:
        """Test result dictionary structure."""
        result = get_sentiment_scores(["happy"])
        assert "happy" in result
        value = result["happy"]
        assert isinstance(value, tuple)
        assert len(value) == 2
        category, score = value
        assert category in ("positive", "negative", "neutral")
        assert isinstance(score, float)
