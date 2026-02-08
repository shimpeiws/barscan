"""Word filtering for lyrics analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from nltk.corpus import stopwords

from barscan.analyzer.processor import ensure_nltk_resources
from barscan.exceptions import NLTKResourceError

if TYPE_CHECKING:
    from barscan.analyzer.models import AnalysisConfig


def get_stop_words(config: AnalysisConfig | None = None) -> frozenset[str]:
    """Get the set of stop words for filtering.

    Combines NLTK stop words with any custom stop words from config.

    Args:
        config: Analysis configuration (uses default if None).

    Returns:
        Frozen set of stop words.

    Raises:
        NLTKResourceError: If NLTK stopwords corpus is not available.
    """
    if config is None:
        from barscan.analyzer.models import AnalysisConfig

        config = AnalysisConfig()

    ensure_nltk_resources()

    try:
        nltk_stop_words = set(stopwords.words(config.language))
    except LookupError as e:
        raise NLTKResourceError(
            f"Failed to load NLTK stop words for '{config.language}': {e}"
        ) from e

    # Combine with custom stop words
    return frozenset(nltk_stop_words | set(config.custom_stop_words))


def filter_stop_words(tokens: list[str], config: AnalysisConfig | None = None) -> list[str]:
    """Remove stop words from token list.

    Args:
        tokens: List of word tokens.
        config: Analysis configuration (uses default if None).

    Returns:
        Filtered list of tokens with stop words removed.
    """
    if config is None:
        from barscan.analyzer.models import AnalysisConfig

        config = AnalysisConfig()

    if not config.remove_stop_words:
        return tokens

    stop_words = get_stop_words(config)
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


def filter_non_alphabetic(tokens: list[str]) -> list[str]:
    """Remove tokens that are not purely alphabetic.

    Args:
        tokens: List of word tokens.

    Returns:
        Filtered list of alphabetic-only tokens.
    """
    return [token for token in tokens if token.isalpha()]


def apply_filters(
    tokens: list[str],
    config: AnalysisConfig | None = None,
    additional_filters: list[Callable[[list[str]], list[str]]] | None = None,
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

    Returns:
        Filtered list of tokens.
    """
    if config is None:
        from barscan.analyzer.models import AnalysisConfig

        config = AnalysisConfig()

    # Apply default filters in order
    filtered = filter_non_alphabetic(tokens)
    filtered = filter_by_length(filtered, config)
    filtered = filter_stop_words(filtered, config)

    # Apply additional custom filters if provided
    if additional_filters:
        for filter_func in additional_filters:
            filtered = filter_func(filtered)

    return filtered
