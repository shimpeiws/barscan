"""Word filtering for lyrics analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from nltk.corpus import stopwords

from barscan.analyzer.stopwords_ja import get_japanese_stop_words
from barscan.analyzer.tokenizer import detect_language, is_japanese_char
from barscan.exceptions import NLTKResourceError

if TYPE_CHECKING:
    from barscan.analyzer.models import AnalysisConfig


def get_stop_words(config: AnalysisConfig | None = None, text: str | None = None) -> frozenset[str]:
    """Get the set of stop words for filtering.

    For English: Combines NLTK stop words with any custom stop words from config.
    For Japanese: Uses Japanese stop words + English stop words for mixed text.

    Args:
        config: Analysis configuration (uses default if None).
        text: Optional text for language detection when language is 'auto'.

    Returns:
        Frozen set of stop words.

    Raises:
        NLTKResourceError: If NLTK stopwords corpus is not available.
    """
    if config is None:
        from barscan.analyzer.models import AnalysisConfig

        config = AnalysisConfig()

    language = config.language
    if language == "auto":
        if text:
            language = detect_language(text)
        else:
            language = "english"

    if language == "japanese":
        # Import here to avoid circular import
        from barscan.analyzer.processor import ensure_nltk_resources

        base_stop_words = get_japanese_stop_words()
        # Also include English stop words for mixed Japanese/English lyrics
        try:
            ensure_nltk_resources()
            english_stop_words = frozenset(stopwords.words("english"))
            base_stop_words = base_stop_words | english_stop_words
        except LookupError:
            pass  # English stopwords not available, use Japanese only
    else:
        # Import here to avoid circular import
        from barscan.analyzer.processor import ensure_nltk_resources

        ensure_nltk_resources()
        try:
            base_stop_words = frozenset(stopwords.words(language))
        except LookupError as e:
            raise NLTKResourceError(f"Failed to load NLTK stop words for '{language}': {e}") from e

    # Combine with custom stop words
    return frozenset(base_stop_words | set(config.custom_stop_words))


def filter_stop_words(
    tokens: list[str], config: AnalysisConfig | None = None, text: str | None = None
) -> list[str]:
    """Remove stop words from token list.

    Args:
        tokens: List of word tokens.
        config: Analysis configuration (uses default if None).
        text: Optional text for language detection when language is 'auto'.

    Returns:
        Filtered list of tokens with stop words removed.
    """
    if config is None:
        from barscan.analyzer.models import AnalysisConfig

        config = AnalysisConfig()

    if not config.remove_stop_words:
        return tokens

    stop_words = get_stop_words(config, text)

    # Determine language for case handling
    language = config.language
    if language == "auto" and text:
        language = detect_language(text)

    if language == "japanese":
        # For mixed Japanese/English text, check both original and lowercase
        # Japanese characters are case-insensitive, but English words need lowercase check
        return [
            token for token in tokens if token not in stop_words and token.lower() not in stop_words
        ]
    else:
        return [token for token in tokens if token.lower() not in stop_words]


def filter_by_length(tokens: list[str], config: AnalysisConfig | None = None) -> list[str]:
    """Filter tokens by minimum length.

    Args:
        tokens: List of word tokens.
        config: Analysis configuration (uses default if None).

    Returns:
        Filtered list of tokens meeting minimum length requirement.
    """
    if config is None:
        from barscan.analyzer.models import AnalysisConfig

        config = AnalysisConfig()

    return [token for token in tokens if len(token) >= config.min_word_length]


def is_valid_word(token: str, language: str) -> bool:
    """Check if a token is a valid word for the given language.

    Args:
        token: The token to check.
        language: The language ('english', 'japanese', or 'auto').

    Returns:
        True if the token is a valid word, False otherwise.
    """
    if not token:
        return False

    if language == "japanese":
        # For Japanese: accept tokens containing Japanese characters or alphabetic
        return any(is_japanese_char(char) or char.isalpha() for char in token)
    else:
        # For English: accept purely alphabetic tokens
        return token.isalpha()


def filter_non_alphabetic(
    tokens: list[str], config: AnalysisConfig | None = None, text: str | None = None
) -> list[str]:
    """Remove tokens that are not valid words.

    For English: Removes tokens that are not purely alphabetic.
    For Japanese: Accepts tokens containing Japanese characters (hiragana, katakana, kanji).

    Args:
        tokens: List of word tokens.
        config: Analysis configuration (uses default if None).
        text: Optional text for language detection when language is 'auto'.

    Returns:
        Filtered list of valid word tokens.
    """
    if config is None:
        from barscan.analyzer.models import AnalysisConfig

        config = AnalysisConfig()

    language = config.language
    if language == "auto":
        if text:
            language = detect_language(text)
        else:
            language = "english"

    return [token for token in tokens if is_valid_word(token, language)]


def apply_filters(
    tokens: list[str],
    config: AnalysisConfig | None = None,
    additional_filters: list[Callable[[list[str]], list[str]]] | None = None,
    text: str | None = None,
) -> list[str]:
    """Apply all configured filters to token list.

    Default filter order:
    1. Remove non-alphabetic tokens
    2. Filter by minimum length
    3. Remove stop words
    4. Apply any additional custom filters

    Args:
        tokens: List of word tokens.
        config: Analysis configuration (uses default if None).
        additional_filters: Optional list of additional filter functions.
        text: Optional text for language detection when language is 'auto'.

    Returns:
        Filtered list of tokens.
    """
    if config is None:
        from barscan.analyzer.models import AnalysisConfig

        config = AnalysisConfig()

    # Apply default filters in order
    filtered = filter_non_alphabetic(tokens, config, text)
    filtered = filter_by_length(filtered, config)
    filtered = filter_stop_words(filtered, config, text)

    # Apply additional custom filters if provided
    if additional_filters:
        for filter_func in additional_filters:
            filtered = filter_func(filtered)

    return filtered
