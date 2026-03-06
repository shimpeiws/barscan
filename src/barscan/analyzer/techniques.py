"""Lyrical technique detection for bar analysis."""

from __future__ import annotations

import re
from collections import Counter


def _has_alliteration(text: str) -> bool:
    """Detect alliteration: 3+ consecutive words starting with same consonant.

    Args:
        text: Text line to check.

    Returns:
        True if alliteration is detected.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 3:
        return False

    vowels = set("aeiou")
    for i in range(len(words) - 2):
        first_letters = [w[0] for w in words[i : i + 3]]
        if all(c == first_letters[0] and c not in vowels for c in first_letters):
            return True
    return False


def _has_repetition(text: str) -> bool:
    """Detect repetition: same word appears 2+ times in the line.

    Args:
        text: Text line to check.

    Returns:
        True if repetition is detected.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 2:
        return False

    counts = Counter(words)
    return any(c >= 2 for c in counts.values())


def detect_techniques(text: str, language: str) -> tuple[str, ...] | None:
    """Detect lyrical techniques in a text line.

    Args:
        text: Text line to analyze.
        language: Language code (e.g., 'en', 'ja').

    Returns:
        Tuple of detected technique names, or None if none detected.
    """
    if language != "en":
        return None

    techniques: list[str] = []

    if _has_alliteration(text):
        techniques.append("alliteration")

    if _has_repetition(text):
        techniques.append("repetition")

    return tuple(techniques) if techniques else None
