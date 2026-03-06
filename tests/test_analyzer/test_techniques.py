"""Tests for lyrical technique detection module."""

from barscan.analyzer.techniques import (
    _has_alliteration,
    _has_repetition,
    detect_techniques,
)


class TestAlliteration:
    """Tests for alliteration detection."""

    def test_three_consecutive_same_consonant(self) -> None:
        assert _has_alliteration("big bad boy") is True

    def test_no_alliteration(self) -> None:
        assert _has_alliteration("the quick fox") is False

    def test_vowel_start_not_counted(self) -> None:
        assert _has_alliteration("an apple arrived") is False

    def test_too_few_words(self) -> None:
        assert _has_alliteration("big bad") is False

    def test_mixed_with_alliteration(self) -> None:
        assert _has_alliteration("I saw sweet silver stars shine") is True

    def test_case_insensitive(self) -> None:
        assert _has_alliteration("Big Bad Boy") is True


class TestRepetition:
    """Tests for repetition detection."""

    def test_word_repeated(self) -> None:
        assert _has_repetition("go go go") is True

    def test_no_repetition(self) -> None:
        assert _has_repetition("one two three") is False

    def test_two_occurrences(self) -> None:
        assert _has_repetition("love is love") is True

    def test_single_word(self) -> None:
        assert _has_repetition("word") is False

    def test_case_insensitive(self) -> None:
        assert _has_repetition("Love is LOVE") is True


class TestDetectTechniques:
    """Tests for detect_techniques function."""

    def test_alliteration_detected(self) -> None:
        result = detect_techniques("big bad boy went home", "en")
        assert result is not None
        assert "alliteration" in result

    def test_repetition_detected(self) -> None:
        result = detect_techniques("go go go let it go", "en")
        assert result is not None
        assert "repetition" in result

    def test_both_techniques(self) -> None:
        result = detect_techniques("big bad boy big bad boy", "en")
        assert result is not None
        assert "alliteration" in result
        assert "repetition" in result

    def test_no_techniques(self) -> None:
        result = detect_techniques("the quick fox", "en")
        assert result is None

    def test_non_english_returns_none(self) -> None:
        result = detect_techniques("大きい悪い男", "ja")
        assert result is None
