"""Sentiment analysis using NLTK VADER."""

from __future__ import annotations

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from barscan.exceptions import NLTKResourceError

# Singleton instance for performance
_sia: SentimentIntensityAnalyzer | None = None


def ensure_sentiment_resources() -> None:
    """Ensure required NLTK resources for sentiment analysis are downloaded.

    Raises:
        NLTKResourceError: If resources cannot be downloaded.
    """
    resources = [
        ("sentiment/vader_lexicon.zip", "vader_lexicon"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception as e:
                raise NLTKResourceError(f"Failed to download NLTK resource '{name}': {e}") from e


def _get_analyzer() -> SentimentIntensityAnalyzer:
    """Get or create the VADER sentiment analyzer.

    Returns:
        SentimentIntensityAnalyzer instance.

    Raises:
        NLTKResourceError: If VADER lexicon is not available.
    """
    global _sia
    if _sia is None:
        ensure_sentiment_resources()
        try:
            _sia = SentimentIntensityAnalyzer()
        except LookupError as e:
            raise NLTKResourceError(f"NLTK VADER initialization failed: {e}") from e
    return _sia


def analyze_sentiment(text: str) -> tuple[str, float]:
    """Analyze sentiment of text.

    Args:
        text: Text to analyze (word, phrase, or sentence).

    Returns:
        Tuple of (category, compound_score) where:
        - category: 'positive', 'negative', or 'neutral'
        - compound_score: VADER compound score from -1.0 to 1.0

    Raises:
        NLTKResourceError: If VADER is not available.
    """
    sia = _get_analyzer()
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    # Classify based on compound score thresholds
    if compound >= 0.05:
        category = "positive"
    elif compound <= -0.05:
        category = "negative"
    else:
        category = "neutral"

    return (category, round(compound, 4))


def analyze_word_sentiment(word: str) -> tuple[str, float]:
    """Analyze sentiment of a single word.

    Note: Single words often have neutral sentiment as VADER is designed
    for sentences. Context matters for accurate sentiment analysis.

    Args:
        word: Word to analyze.

    Returns:
        Tuple of (category, compound_score).

    Raises:
        NLTKResourceError: If VADER is not available.
    """
    return analyze_sentiment(word)


def get_sentiment_scores(words: list[str]) -> dict[str, tuple[str, float]]:
    """Get sentiment scores for a list of words.

    Args:
        words: List of words to analyze.

    Returns:
        Dictionary mapping words to (category, compound_score) tuples.

    Raises:
        NLTKResourceError: If VADER is not available.
    """
    if not words:
        return {}

    result: dict[str, tuple[str, float]] = {}
    for word in set(words):  # Deduplicate
        word_lower = word.lower()
        if word_lower not in result:
            result[word_lower] = analyze_word_sentiment(word_lower)

    return result
