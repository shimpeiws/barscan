"""Tests for syllable counting module."""

from barscan.analyzer.syllable import (
    _count_syllables_heuristic,
    count_line_syllables,
    count_syllables_english,
)


class TestCountSyllablesHeuristic:
    """Tests for vowel-group heuristic fallback."""

    def test_one_syllable(self) -> None:
        assert _count_syllables_heuristic("cat") == 1

    def test_two_syllables(self) -> None:
        assert _count_syllables_heuristic("hello") == 2

    def test_silent_e(self) -> None:
        assert _count_syllables_heuristic("cake") == 1

    def test_empty_string(self) -> None:
        assert _count_syllables_heuristic("") == 0

    def test_minimum_one(self) -> None:
        assert _count_syllables_heuristic("rhythm") >= 1


class TestCountSyllablesEnglish:
    """Tests for English syllable counting with cmudict."""

    def test_known_word(self) -> None:
        count = count_syllables_english("hello")
        assert count == 2

    def test_one_syllable_word(self) -> None:
        count = count_syllables_english("cat")
        assert count == 1

    def test_multisyllable_word(self) -> None:
        count = count_syllables_english("beautiful")
        assert count == 3

    def test_empty_string(self) -> None:
        assert count_syllables_english("") == 0

    def test_unknown_word_uses_heuristic(self) -> None:
        # Made-up word not in cmudict
        count = count_syllables_english("xyloquartz")
        assert count >= 1

    def test_case_insensitive(self) -> None:
        assert count_syllables_english("Hello") == count_syllables_english("hello")


class TestCountLineSyllables:
    """Tests for line-level syllable counting."""

    def test_english_line(self) -> None:
        count = count_line_syllables("hello world", "en")
        assert count is not None
        assert count >= 2

    def test_non_english_returns_none(self) -> None:
        assert count_line_syllables("こんにちは", "ja") is None

    def test_empty_line(self) -> None:
        assert count_line_syllables("", "en") == 0

    def test_multiple_words(self) -> None:
        count = count_line_syllables("I love you", "en")
        assert count is not None
        assert count >= 3
