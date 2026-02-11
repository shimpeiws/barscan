"""Text preprocessing for lyrics analysis."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from nltk.stem import WordNetLemmatizer

from barscan.analyzer.nltk_resources import PROCESSOR_RESOURCES, ensure_resources
from barscan.analyzer.tokenizer import (
    JapaneseTokenizer,
    detect_language,
    get_tokenizer,
    normalize_text_for_language,
)
from barscan.exceptions import EmptyLyricsError, NLTKResourceError

if TYPE_CHECKING:
    from barscan.analyzer.models import AnalysisConfig

from barscan.analyzer.models import TokenWithPosition

# Pattern to match section headers like [Verse 1], [Chorus], [Bridge], etc.
SECTION_HEADER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\[([A-Za-z0-9\s\-:]+)\]",
    re.IGNORECASE,
)

# Pattern to match non-word characters (except apostrophes for contractions)
PUNCTUATION_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^\w\s']")

# Pattern to match standalone apostrophes
STANDALONE_APOSTROPHE_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?<!\w)'|'(?!\w)")

# Module-level cached lemmatizer for performance
_lemmatizer: WordNetLemmatizer | None = None


def _get_lemmatizer() -> WordNetLemmatizer:
    """Get or create the WordNet lemmatizer.

    Returns:
        WordNetLemmatizer instance.

    Raises:
        NLTKResourceError: If WordNet is not available.
    """
    global _lemmatizer
    if _lemmatizer is None:
        ensure_nltk_resources()
        try:
            _lemmatizer = WordNetLemmatizer()
        except LookupError as e:
            raise NLTKResourceError(f"NLTK WordNet initialization failed: {e}") from e
    return _lemmatizer


def ensure_nltk_resources() -> None:
    """Ensure required NLTK resources are downloaded.

    Downloads punkt tokenizer, stopwords, and wordnet if not present.

    Raises:
        NLTKResourceError: If resources cannot be downloaded.
    """
    ensure_resources(PROCESSOR_RESOURCES)


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


def normalize_text(text: str, config: AnalysisConfig | None = None) -> str:
    """Normalize text for analysis.

    For English: Converts to lowercase, removes punctuation (except apostrophes
    in contractions), and normalizes whitespace.
    For Japanese: Applies NFKC normalization and removes common punctuation.

    Args:
        text: Text to normalize.
        config: Analysis configuration (uses default if None).

    Returns:
        Normalized text.
    """
    if config is None:
        from barscan.analyzer.models import AnalysisConfig

        config = AnalysisConfig()

    language = config.language
    if language == "auto":
        language = detect_language(text)

    return normalize_text_for_language(text, language)


def tokenize(text: str, config: AnalysisConfig | None = None) -> list[str]:
    """Tokenize text into words using appropriate tokenizer.

    For English: Uses NLTK word_tokenize.
    For Japanese: Uses Janome morphological analyzer with optional POS filtering.

    Args:
        text: Text to tokenize.
        config: Analysis configuration (uses default if None).

    Returns:
        List of word tokens.

    Raises:
        NLTKResourceError: If NLTK resources are not available (for English).
        ImportError: If Janome is not installed (for Japanese).
    """
    if config is None:
        from barscan.analyzer.models import AnalysisConfig

        config = AnalysisConfig()

    language = config.language
    if language == "auto":
        language = detect_language(text)

    if language == "english":
        ensure_nltk_resources()

    tokenizer = get_tokenizer(language, text)
    try:
        # Use POS filtering for Japanese if enabled
        if language == "japanese" and config.use_pos_filtering:
            if isinstance(tokenizer, JapaneseTokenizer):
                return tokenizer.tokenize_with_pos_filter(text)
        return tokenizer.tokenize(text)
    except LookupError as e:
        raise NLTKResourceError(f"Tokenization failed: {e}") from e


def lemmatize(
    tokens: list[str], config: AnalysisConfig | None = None, text: str | None = None
) -> list[str]:
    """Lemmatize tokens to their base form.

    For English: Uses WordNet lemmatizer.
    For Japanese: Skipped (Janome already returns base forms when using get_base_forms).

    Args:
        tokens: List of word tokens.
        config: Analysis configuration (uses default if None).
        text: Original text for language detection (optional).

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

    # Skip lemmatization for Japanese (Janome handles base forms)
    language = config.language
    if language == "auto" and text:
        language = detect_language(text)

    if language == "japanese":
        return tokens

    lemmatizer = _get_lemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


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
    normalized = normalize_text(cleaned, config)
    tokens = tokenize(normalized, config)
    # Ensure all tokens are lowercase (safety measure for mixed-language text)
    tokens = [token.lower() for token in tokens]
    tokens = lemmatize(tokens, config, text=cleaned)

    return tokens


def clean_lyrics_preserve_lines(text: str) -> list[str]:
    """Clean lyrics while preserving line structure.

    Removes section headers but keeps line breaks for context extraction.

    Args:
        text: Raw lyrics text.

    Returns:
        List of cleaned lyrics lines.

    Raises:
        EmptyLyricsError: If text is empty or whitespace only.
    """
    if not text or not text.strip():
        raise EmptyLyricsError("Lyrics text is empty or contains only whitespace")

    # Split into lines
    lines = text.split("\n")

    # Clean each line (remove section headers)
    cleaned_lines: list[str] = []
    for line in lines:
        cleaned = SECTION_HEADER_PATTERN.sub("", line)
        cleaned = cleaned.strip()
        if cleaned:  # Only keep non-empty lines
            cleaned_lines.append(cleaned)

    return cleaned_lines


def tokenize_with_positions(
    text: str,
    song_id: int,
    song_title: str,
    config: AnalysisConfig | None = None,
) -> list[TokenWithPosition]:
    """Tokenize text while preserving line and position information.

    This is used for context extraction, where we need to track where each
    token appears in the original lyrics.

    Args:
        text: Raw lyrics text.
        song_id: Genius song ID.
        song_title: Title of the song.
        config: Analysis configuration (uses default if None).

    Returns:
        List of tokens with their position information.

    Raises:
        EmptyLyricsError: If text is empty or whitespace only.
        NLTKResourceError: If NLTK resources are not available.
    """
    if config is None:
        from barscan.analyzer.models import AnalysisConfig

        config = AnalysisConfig()

    # Determine language once for consistency
    language = config.language
    if language == "auto":
        language = detect_language(text)

    if language == "english":
        ensure_nltk_resources()

    # Get lines while preserving structure
    lines = clean_lyrics_preserve_lines(text)

    tokens_with_positions: list[TokenWithPosition] = []
    tokenizer = get_tokenizer(language)

    for line_index, line in enumerate(lines):
        # Normalize the line for tokenization
        normalized_line = normalize_text_for_language(line, language)

        # Tokenize the normalized line (use POS filtering for Japanese if enabled)
        try:
            if language == "japanese" and config.use_pos_filtering:
                if isinstance(tokenizer, JapaneseTokenizer):
                    line_tokens = tokenizer.tokenize_with_pos_filter(normalized_line)
                else:
                    line_tokens = tokenizer.tokenize(normalized_line)
            else:
                line_tokens = tokenizer.tokenize(normalized_line)
        except LookupError as e:
            raise NLTKResourceError(f"Tokenization failed: {e}") from e

        # Ensure all tokens are lowercase (safety measure for mixed-language text)
        line_tokens = [token.lower() for token in line_tokens]

        # Optionally lemmatize (only for English)
        if config.use_lemmatization and language == "english":
            lemmatizer = _get_lemmatizer()
            line_tokens = [lemmatizer.lemmatize(token) for token in line_tokens]

        # Create TokenWithPosition for each token
        for word_index, token in enumerate(line_tokens):
            tokens_with_positions.append(
                TokenWithPosition(
                    token=token,
                    line_index=line_index,
                    word_index=word_index,
                    original_line=line,  # Keep original line for context
                    song_id=song_id,
                    song_title=song_title,
                )
            )

    return tokens_with_positions
