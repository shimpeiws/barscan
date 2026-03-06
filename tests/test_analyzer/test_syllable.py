"""Tests for syllable counting module."""

import pytest

from barscan.analyzer.syllable import (
    _count_syllables_heuristic,
    count_line_syllables,
    count_mora,
    count_syllables_english,
)

janome_available = True
try:
    from janome.tokenizer import Tokenizer as _JanomeTokenizer
except ImportError:
    janome_available = False

skip_without_janome = pytest.mark.skipif(
    not janome_available, reason="janome is not installed"
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
        assert count_line_syllables("bonjour", "fr") is None

    def test_empty_line(self) -> None:
        assert count_line_syllables("", "en") == 0

    def test_multiple_words(self) -> None:
        count = count_line_syllables("I love you", "en")
        assert count is not None
        assert count >= 3

    @skip_without_janome
    def test_count_line_syllables_japanese(self) -> None:
        count = count_line_syllables("こんにちは", "ja")
        assert count is not None
        assert isinstance(count, int)
        assert count > 0


@skip_without_janome
class TestCountMora:
    """Tests for Japanese mora counting."""

    def test_count_mora_hiragana(self) -> None:
        assert count_mora("あいうえお") == 5

    def test_count_mora_katakana(self) -> None:
        assert count_mora("アイウエオ") == 5

    def test_count_mora_youon(self) -> None:
        # きゃりー = き+ゃ(1) + り(1) + ー(1) = 3 morae
        assert count_mora("きゃりー") == 3

    def test_count_mora_sokuon(self) -> None:
        # がっこう = が(1) + っ(1) + こ(1) + う(1) = 4 morae
        assert count_mora("がっこう") == 4

    def test_count_mora_hatsuon(self) -> None:
        # にほん = に(1) + ほ(1) + ん(1) = 3 morae
        assert count_mora("にほん") == 3

    def test_count_mora_kanji(self) -> None:
        # 東京 -> トウキョウ = ト(1) + ウ(1) + キ(1) + ョ combines(0) + ウ(1) = 4 morae
        count = count_mora("東京")
        assert count == 4

    def test_count_mora_mixed(self) -> None:
        # Mixed Japanese/ASCII text
        count = count_mora("KOHHは最高")
        assert count is not None
        assert count > 0

    def test_count_mora_empty(self) -> None:
        assert count_mora("") == 0

    def test_count_mora_long_vowel(self) -> None:
        # ラーメン = ラ(1) + ー(1) + メ(1) + ン(1) = 4 morae
        assert count_mora("ラーメン") == 4

    def test_count_mora_complex_youon(self) -> None:
        # しゅっぱつ = シュ(1, ュ is small) + ッ(1) + パ(1) + ツ(1) = 4 morae
        assert count_mora("しゅっぱつ") == 4
