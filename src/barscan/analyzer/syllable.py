"""Syllable counting for English text using CMU Pronouncing Dictionary."""

from __future__ import annotations

import re

from barscan.analyzer.nltk_resources import SYLLABLE_RESOURCES, ensure_resources

_cmudict: dict[str, list[list[str]]] | None = None


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


def count_line_syllables(text: str, language: str) -> int | None:
    """Count total syllables in a line of text.

    Args:
        text: Text line to count syllables for.
        language: Language code ('en' for English).

    Returns:
        Total syllable count for English, None for other languages.
    """
    if language != "en":
        return None

    words = text.split()
    if not words:
        return 0

    return sum(count_syllables_english(w) for w in words)
