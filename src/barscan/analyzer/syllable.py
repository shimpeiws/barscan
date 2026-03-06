"""Syllable/mora counting for English and Japanese text."""

from __future__ import annotations

import re
from typing import Any

from barscan.analyzer.nltk_resources import SYLLABLE_RESOURCES, ensure_resources

_cmudict: dict[str, list[list[str]]] | None = None

# Small kana that form compound morae with the preceding character (youon)
_SMALL_KANA = frozenset("ァィゥェォャュョぁぃぅぇぉゃゅょ")


def _get_cmudict() -> dict[str, list[list[str]]]:
    """Get or load the CMU Pronouncing Dictionary."""
    global _cmudict
    if _cmudict is None:
        ensure_resources(SYLLABLE_RESOURCES)
        from nltk.corpus import cmudict

        _cmudict = cmudict.dict()
    return _cmudict


def _count_syllables_heuristic(word: str) -> int:
    """Count syllables using vowel-group heuristic as fallback.

    Args:
        word: Lowercase English word.

    Returns:
        Estimated syllable count (minimum 1).
    """
    word = word.lower().strip()
    if not word:
        return 0

    # Remove trailing 'e' (silent e)
    if word.endswith("e") and len(word) > 2:
        word = word[:-1]

    # Count vowel groups
    count = len(re.findall(r"[aeiouy]+", word))
    return max(count, 1)


def count_syllables_english(word: str) -> int:
    """Count syllables in an English word using cmudict with fallback heuristic.

    Args:
        word: English word to count syllables for.

    Returns:
        Syllable count.
    """
    word_lower = word.lower().strip()
    if not word_lower:
        return 0

    cmu = _get_cmudict()
    if word_lower in cmu:
        # Count vowel phonemes (digits in ARPAbet indicate stress on vowels)
        pronunciation = cmu[word_lower][0]
        return sum(1 for phoneme in pronunciation if phoneme[-1].isdigit())

    return _count_syllables_heuristic(word_lower)


def _count_mora_from_kana(reading: str) -> int:
    """Count morae from a katakana/hiragana reading string.

    Rules:
    - Each kana character = 1 mora
    - Small kana (ァィゥェォャュョ and hiragana equivalents) combine with
      the preceding character to form 1 mora (youon)
    - Sokuon (っ/ッ) = 1 mora
    - Hatsuon (ん/ン) = 1 mora
    - Chouon (ー) = 1 mora

    Args:
        reading: Katakana or hiragana string.

    Returns:
        Mora count.
    """
    count = 0
    for char in reading:
        if char in _SMALL_KANA:
            # Small kana combines with previous char; don't add a mora
            continue
        count += 1
    return count


_janome_tokenizer: Any = None


def _get_janome_tokenizer() -> Any:
    """Get or create a Janome tokenizer instance."""
    global _janome_tokenizer
    if _janome_tokenizer is None:
        try:
            from janome.tokenizer import Tokenizer as JanomeTokenizer

            _janome_tokenizer = JanomeTokenizer()
        except ImportError as e:
            raise ImportError(
                "Janome is required for Japanese mora counting. "
                "Install it with: pip install barscan[japanese]"
            ) from e
    return _janome_tokenizer


def count_mora(text: str) -> int:
    """Count morae in Japanese text using Janome for morphological analysis.

    Converts text to katakana readings via Janome, then counts morae.
    For tokens without readings (e.g. ASCII words), falls back to character count.

    Args:
        text: Japanese text to count morae for.

    Returns:
        Total mora count.
    """
    text = text.strip()
    if not text:
        return 0

    tokenizer = _get_janome_tokenizer()
    total = 0

    for token in tokenizer.tokenize(text):
        reading = token.reading
        if reading and reading != "*":
            total += _count_mora_from_kana(reading)
        else:
            # No reading available (e.g. ASCII, symbols) - use character count
            surface = token.surface.strip()
            if surface:
                total += len(surface)

    return total


def count_line_syllables(text: str, language: str) -> int | None:
    """Count total syllables/morae in a line of text.

    Args:
        text: Text line to count syllables for.
        language: Language code ('en' for English, 'ja' for Japanese).

    Returns:
        Syllable count for English, mora count for Japanese, None for other languages.
    """
    if language == "ja":
        return count_mora(text)

    if language == "en":
        words = text.split()
        if not words:
            return 0
        return sum(count_syllables_english(w) for w in words)

    return None
