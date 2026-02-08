"""Text preprocessing for lyrics analysis."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from barscan.exceptions import EmptyLyricsError, NLTKResourceError

if TYPE_CHECKING:
    from barscan.analyzer.models import AnalysisConfig

# Pattern to match section headers like [Verse 1], [Chorus], [Bridge], etc.
SECTION_HEADER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\[([A-Za-z0-9\s\-:]+)\]",
    re.IGNORECASE,
)

# Pattern to match non-word characters (except apostrophes for contractions)
PUNCTUATION_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^\w\s']")

# Pattern to match standalone apostrophes
STANDALONE_APOSTROPHE_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?<!\w)'|'(?!\w)")


def ensure_nltk_resources() -> None:
    """Ensure required NLTK resources are downloaded.

    Downloads punkt tokenizer, stopwords, and wordnet if not present.

    Raises:
        NLTKResourceError: If resources cannot be downloaded.
    """
    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception as e:
                raise NLTKResourceError(f"Failed to download NLTK resource '{name}': {e}") from e


def clean_lyrics(text: str) -> str:
    """Remove section headers and clean raw lyrics text.

    Removes patterns like [Verse 1], [Chorus], [Bridge], [Intro], etc.

    Args:
        text: Raw lyrics text.

    Returns:
        Cleaned lyrics text with section headers removed.

    Raises:
        EmptyLyricsError: If text is empty or whitespace only.
    """
    if not text or not text.strip():
        raise EmptyLyricsError("Lyrics text is empty or contains only whitespace")

    # Remove section headers
    cleaned = SECTION_HEADER_PATTERN.sub("", text)

    # Normalize whitespace
    cleaned = " ".join(cleaned.split())

    return cleaned


def normalize_text(text: str) -> str:
    """Normalize text for analysis.

    Converts to lowercase, removes punctuation (except apostrophes in contractions),
    and normalizes whitespace.

    Args:
        text: Text to normalize.

    Returns:
        Normalized text.
    """
    # Lowercase
    normalized = text.lower()

    # Remove punctuation except apostrophes in contractions
    normalized = PUNCTUATION_PATTERN.sub(" ", normalized)

    # Remove standalone apostrophes
    normalized = STANDALONE_APOSTROPHE_PATTERN.sub("", normalized)

    # Normalize whitespace
    normalized = " ".join(normalized.split())

    return normalized


def tokenize(text: str) -> list[str]:
    """Tokenize text into words using NLTK.

    Args:
        text: Text to tokenize.

    Returns:
        List of word tokens.

    Raises:
        NLTKResourceError: If NLTK punkt tokenizer is not available.
    """
    ensure_nltk_resources()
    try:
        tokens: list[str] = word_tokenize(text)
        return tokens
    except LookupError as e:
        raise NLTKResourceError(f"NLTK tokenization failed: {e}") from e


def lemmatize(tokens: list[str], config: AnalysisConfig | None = None) -> list[str]:
    """Lemmatize tokens to their base form.

    Args:
        tokens: List of word tokens.
        config: Analysis configuration (uses default if None).

    Returns:
        List of lemmatized tokens.

    Raises:
        NLTKResourceError: If NLTK WordNet is not available.
    """
    if config is None:
        from barscan.analyzer.models import AnalysisConfig

        config = AnalysisConfig()

    if not config.use_lemmatization:
        return tokens

    ensure_nltk_resources()
    try:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]
    except LookupError as e:
        raise NLTKResourceError(f"NLTK lemmatization failed: {e}") from e


def preprocess(text: str, config: AnalysisConfig | None = None) -> list[str]:
    """Full preprocessing pipeline: clean, normalize, tokenize, and optionally lemmatize.

    Args:
        text: Raw lyrics text.
        config: Analysis configuration (uses default if None).

    Returns:
        List of preprocessed word tokens.

    Raises:
        EmptyLyricsError: If text is empty or whitespace only.
        NLTKResourceError: If NLTK resources are not available.
    """
    if config is None:
        from barscan.analyzer.models import AnalysisConfig

        config = AnalysisConfig()

    cleaned = clean_lyrics(text)
    normalized = normalize_text(cleaned)
    tokens = tokenize(normalized)
    tokens = lemmatize(tokens, config)

    return tokens
