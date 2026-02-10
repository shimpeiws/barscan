"""Tests for slang detection module."""

import pytest

from barscan.analyzer.slang import (
    SLANG_WORDS,
    detect_slang_words,
    get_slang_count,
    is_slang,
)


class TestIsSlang:
    """Tests for is_slang function."""

    def test_known_slang_word(self) -> None:
        """Test detection of a known slang word."""
        assert is_slang("gonna") is True
        assert is_slang("wanna") is True
        assert is_slang("lit") is True
        assert is_slang("dope") is True

    def test_non_slang_word(self) -> None:
        """Test non-slang word returns False."""
        assert is_slang("love") is False
        assert is_slang("music") is False
        assert is_slang("the") is False

    def test_case_insensitive(self) -> None:
        """Test case-insensitive detection."""
        assert is_slang("Gonna") is True
        assert is_slang("GONNA") is True
        assert is_slang("gOnNa") is True

    def test_with_additional_slang(self) -> None:
        """Test with additional slang words."""
        additional = frozenset({"customslang", "myword"})
        assert is_slang("customslang", additional) is True
        assert is_slang("myword", additional) is True
        # Original slang should still work
        assert is_slang("gonna", additional) is True

    def test_none_additional_slang(self) -> None:
        """Test with None as additional slang."""
        assert is_slang("gonna", None) is True


class TestDetectSlangWords:
    """Tests for detect_slang_words function."""

    def test_empty_list(self) -> None:
        """Test with empty word list."""
        result = detect_slang_words([])
        assert result == {}

    def test_mixed_words(self) -> None:
        """Test with mix of slang and non-slang words."""
        words = ["gonna", "love", "lit", "music", "dope"]
        result = detect_slang_words(words)
        assert result["gonna"] is True
        assert result["love"] is False
        assert result["lit"] is True
        assert result["music"] is False
        assert result["dope"] is True

    def test_deduplication(self) -> None:
        """Test that duplicate words are deduplicated."""
        words = ["gonna", "gonna", "gonna"]
        result = detect_slang_words(words)
        assert len(result) == 1
        assert result["gonna"] is True

    def test_case_normalization(self) -> None:
        """Test that words are normalized to lowercase."""
        words = ["Gonna", "GONNA", "gonna"]
        result = detect_slang_words(words)
        assert len(result) == 1
        assert "gonna" in result

    def test_with_additional_slang(self) -> None:
        """Test with additional slang words."""
        additional = frozenset({"customword"})
        words = ["customword", "gonna", "music"]
        result = detect_slang_words(words, additional)
        assert result["customword"] is True
        assert result["gonna"] is True
        assert result["music"] is False


class TestGetSlangCount:
    """Tests for get_slang_count function."""

    def test_empty_list(self) -> None:
        """Test with empty word list."""
        result = get_slang_count([])
        assert result == 0

    def test_no_slang(self) -> None:
        """Test with no slang words."""
        words = ["love", "music", "peace"]
        result = get_slang_count(words)
        assert result == 0

    def test_all_slang(self) -> None:
        """Test with all slang words."""
        words = ["gonna", "wanna", "lit"]
        result = get_slang_count(words)
        assert result == 3

    def test_mixed_words(self) -> None:
        """Test with mixed slang and non-slang."""
        words = ["gonna", "love", "lit", "music"]
        result = get_slang_count(words)
        assert result == 2

    def test_counts_duplicates(self) -> None:
        """Test that duplicate slang words are counted."""
        words = ["gonna", "gonna", "gonna"]
        result = get_slang_count(words)
        assert result == 3


class TestSlangWordsConstant:
    """Tests for SLANG_WORDS constant."""

    def test_is_frozenset(self) -> None:
        """Test that SLANG_WORDS is a frozenset."""
        assert isinstance(SLANG_WORDS, frozenset)

    def test_contains_common_slang(self) -> None:
        """Test that common slang words are included."""
        common_slang = ["gonna", "wanna", "ain't", "yo", "bruh", "lit", "dope"]
        for word in common_slang:
            assert word in SLANG_WORDS

    def test_all_lowercase(self) -> None:
        """Test that all words are lowercase."""
        for word in SLANG_WORDS:
            assert word == word.lower()
