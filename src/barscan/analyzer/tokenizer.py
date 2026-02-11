"""Tokenizer module for language-aware text tokenization."""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Any

from nltk.tokenize import word_tokenize

# Unicode ranges for Japanese character detection
HIRAGANA_RANGE = (0x3040, 0x309F)
KATAKANA_RANGE = (0x30A0, 0x30FF)
KANJI_RANGE = (0x4E00, 0x9FFF)
KATAKANA_EXTENDED_RANGE = (0x31F0, 0x31FF)
HALFWIDTH_KATAKANA_RANGE = (0xFF65, 0xFF9F)

# Part-of-speech categories to keep for Japanese content word analysis
# Keeps: nouns, verbs, adjectives, adverbs, interjections
# Removes: particles (が, を, に), auxiliaries (た, て, ない)
MEANINGFUL_POS = frozenset({"名詞", "動詞", "形容詞", "副詞", "感動詞"})

# POS subcategories to exclude (pos2 level)
# 非自立: non-independent verbs like てる, いる, ある
# 接尾: suffixes like さん, 様
EXCLUDED_POS2 = frozenset({"非自立", "接尾"})


def is_japanese_char(char: str) -> bool:
    """Check if a character is Japanese (Hiragana, Katakana, or Kanji).

    Args:
        char: A single character to check.

    Returns:
        True if the character is Japanese, False otherwise.
    """
    if len(char) != 1:
        return False

    code = ord(char)
    return (
        HIRAGANA_RANGE[0] <= code <= HIRAGANA_RANGE[1]
        or KATAKANA_RANGE[0] <= code <= KATAKANA_RANGE[1]
        or KANJI_RANGE[0] <= code <= KANJI_RANGE[1]
        or KATAKANA_EXTENDED_RANGE[0] <= code <= KATAKANA_EXTENDED_RANGE[1]
        or HALFWIDTH_KATAKANA_RANGE[0] <= code <= HALFWIDTH_KATAKANA_RANGE[1]
    )


def is_japanese_text(text: str) -> bool:
    """Check if the text contains Japanese characters.

    Args:
        text: The text to check.

    Returns:
        True if the text contains Japanese characters, False otherwise.
    """
    return any(is_japanese_char(char) for char in text)


def detect_language(text: str) -> str:
    """Detect the language of the text based on character composition.

    Args:
        text: The text to analyze.

    Returns:
        'japanese' if text contains significant Japanese characters,
        'english' otherwise.
    """
    if not text:
        return "english"

    # If there are any Japanese characters, treat as Japanese
    # (Japanese text often mixes with English words)
    if any(is_japanese_char(char) for char in text):
        return "japanese"

    return "english"


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into a list of tokens.

        Args:
            text: The text to tokenize.

        Returns:
            A list of tokens.
        """
        pass

    @abstractmethod
    def get_base_forms(self, text: str) -> list[str]:
        """Tokenize text and return base forms (lemmas).

        Args:
            text: The text to tokenize.

        Returns:
            A list of base form tokens.
        """
        pass


class EnglishTokenizer(Tokenizer):
    """Tokenizer for English text using NLTK."""

    def tokenize(self, text: str) -> list[str]:
        """Tokenize English text using NLTK word_tokenize.

        Args:
            text: The text to tokenize.

        Returns:
            A list of tokens.
        """
        tokens: list[str] = word_tokenize(text)
        return tokens

    def get_base_forms(self, text: str) -> list[str]:
        """Tokenize and return tokens (lemmatization handled separately).

        Args:
            text: The text to tokenize.

        Returns:
            A list of tokens.
        """
        return self.tokenize(text)


class JapaneseTokenizer(Tokenizer):
    """Tokenizer for Japanese text using Janome."""

    def __init__(self) -> None:
        """Initialize the Japanese tokenizer."""
        self._tokenizer: Any = None

    def _get_tokenizer(self) -> Any:
        """Lazy initialization of Janome tokenizer."""
        if self._tokenizer is None:
            try:
                from janome.tokenizer import Tokenizer as JanomeTokenizer

                self._tokenizer = JanomeTokenizer()
            except ImportError as e:
                raise ImportError(
                    "Janome is required for Japanese tokenization. "
                    "Install it with: pip install barscan[japanese]"
                ) from e
        return self._tokenizer

    def tokenize(self, text: str) -> list[str]:
        """Tokenize Japanese text using Janome.

        Returns base forms for better frequency analysis.
        E.g., "食べた", "食べる", "食べている" -> all become "食べる"

        Args:
            text: The text to tokenize.

        Returns:
            A list of tokens (base forms).
        """
        return self.get_base_forms(text)

    def get_base_forms(self, text: str) -> list[str]:
        """Tokenize Japanese text and return base forms.

        Args:
            text: The text to tokenize.

        Returns:
            A list of base form tokens.
        """
        tokenizer = self._get_tokenizer()
        tokens = []
        for token in tokenizer.tokenize(text):
            # Use base form if available, otherwise use surface form
            base_form = token.base_form
            if base_form == "*":
                tokens.append(token.surface)
            else:
                tokens.append(base_form)
        return tokens

    def tokenize_with_pos_filter(self, text: str) -> list[str]:
        """Tokenize Japanese text and filter by part-of-speech.

        Only keeps content words (nouns, verbs, adjectives, adverbs, interjections).
        Removes particles (が, を, に), auxiliaries (た, て, ない),
        non-independent verbs (てる, いる), and suffixes (さん, 様).

        Args:
            text: The text to tokenize.

        Returns:
            A list of base form tokens for content words only.
        """
        tokenizer = self._get_tokenizer()
        tokens = []
        for token in tokenizer.tokenize(text):
            # Parse POS tag (e.g., "動詞,自立,*,*" or "動詞,非自立,*,*")
            parts = token.part_of_speech.split(",")
            pos1 = parts[0]  # Primary POS: 動詞, 名詞, etc.
            pos2 = parts[1] if len(parts) > 1 else "*"  # Secondary: 自立, 非自立, etc.

            # Skip if not a meaningful primary POS
            if pos1 not in MEANINGFUL_POS:
                continue

            # Skip non-independent forms (てる, いる) and suffixes (さん, 様)
            if pos2 in EXCLUDED_POS2:
                continue

            base_form = token.base_form
            if base_form == "*":
                tokens.append(token.surface)
            else:
                tokens.append(base_form)
        return tokens


def normalize_text_for_language(text: str, language: str) -> str:
    """Normalize text based on language.

    Args:
        text: The text to normalize.
        language: The language ('english', 'japanese', or 'auto').

    Returns:
        Normalized text.
    """
    if language == "auto":
        language = detect_language(text)

    if language == "japanese":
        # NFKC normalization for Japanese
        # Converts full-width characters to half-width, normalizes unicode
        text = unicodedata.normalize("NFKC", text)
        # Remove common punctuation but keep Japanese-specific punctuation
        text = re.sub(r"[!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]", " ", text)
        # Normalize whitespace
        text = " ".join(text.split())
    else:
        # English normalization
        text = text.lower()
        # Remove punctuation except apostrophes in contractions
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
        text = " ".join(text.split())

    return text


def get_tokenizer(language: str, text: str | None = None) -> Tokenizer:
    """Factory function to get the appropriate tokenizer.

    Args:
        language: The language ('english', 'japanese', or 'auto').
        text: Optional text for language detection when language is 'auto'.

    Returns:
        An appropriate Tokenizer instance.
    """
    if language == "auto":
        if text is None:
            return EnglishTokenizer()
        language = detect_language(text)

    if language == "japanese":
        return JapaneseTokenizer()

    return EnglishTokenizer()
