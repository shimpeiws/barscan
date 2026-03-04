"""Sentiment analysis using NLTK VADER and oseti dictionaries for Japanese."""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from barscan.analyzer.nltk_resources import SENTIMENT_RESOURCES, ensure_resources
from barscan.exceptions import NLTKResourceError

# Singleton instance for performance
_sia: SentimentIntensityAnalyzer | None = None

# Singleton for Japanese sentiment dictionaries
_ja_dict: dict[str, float] | None = None


def ensure_sentiment_resources() -> None:
    """Ensure required NLTK resources for sentiment analysis are downloaded.

    Raises:
        NLTKResourceError: If resources cannot be downloaded.
    """
    ensure_resources(SENTIMENT_RESOURCES)


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


def _load_japanese_dict() -> dict[str, float]:
    """Load oseti sentiment dictionaries for Japanese.

    Returns:
        Dictionary mapping words to polarity scores (+1.0 or -1.0).

    Raises:
        NLTKResourceError: If oseti is not installed.
    """
    global _ja_dict
    if _ja_dict is not None:
        return _ja_dict

    spec = find_spec("oseti")
    if spec is None or spec.origin is None:
        raise NLTKResourceError(
            "oseti package is required for Japanese sentiment analysis. "
            "Install it with: pip install barscan[japanese]",
            resource_name="oseti",
        )

    dic_dir = Path(spec.origin).parent / "dic"
    result: dict[str, float] = {}

    # Load pn_noun.json: "p" → +1.0, "n" → -1.0, others → skip
    noun_path = dic_dir / "pn_noun.json"
    if noun_path.exists():
        with noun_path.open(encoding="utf-8") as f:
            pn_noun: dict[str, str] = json.load(f)
        for word, label in pn_noun.items():
            if label == "p":
                result[word] = 1.0
            elif label == "n":
                result[word] = -1.0

    # Load pn_wago.json: "ポジ" in value → +1.0, "ネガ" in value → -1.0
    wago_path = dic_dir / "pn_wago.json"
    if wago_path.exists():
        with wago_path.open(encoding="utf-8") as f:
            pn_wago: dict[str, str] = json.load(f)
        for word, label in pn_wago.items():
            if "ポジ" in label:
                result[word] = 1.0
            elif "ネガ" in label:
                result[word] = -1.0

    _ja_dict = result
    return _ja_dict


def _analyze_japanese_word(word: str) -> tuple[str, float]:
    """Analyze sentiment of a Japanese word using oseti dictionaries.

    Args:
        word: Japanese word to analyze.

    Returns:
        Tuple of (category, score).

    Raises:
        NLTKResourceError: If oseti is not installed.
    """
    ja_dict = _load_japanese_dict()
    score = ja_dict.get(word, 0.0)

    if score >= 0.05:
        category = "positive"
    elif score <= -0.05:
        category = "negative"
    else:
        category = "neutral"

    return (category, score)


def analyze_sentiment(text: str, language: str = "english") -> tuple[str, float]:
    """Analyze sentiment of text.

    Args:
        text: Text to analyze (word, phrase, or sentence).
        language: Language for analysis ("english" or "japanese").

    Returns:
        Tuple of (category, compound_score) where:
        - category: 'positive', 'negative', or 'neutral'
        - compound_score: Score from -1.0 to 1.0

    Raises:
        NLTKResourceError: If required resources are not available.
    """
    if language == "japanese":
        return _analyze_japanese_word(text)

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


def analyze_word_sentiment(word: str, language: str = "english") -> tuple[str, float]:
    """Analyze sentiment of a single word.

    Note: Single words often have neutral sentiment as VADER is designed
    for sentences. Context matters for accurate sentiment analysis.

    Args:
        word: Word to analyze.
        language: Language for analysis ("english" or "japanese").

    Returns:
        Tuple of (category, compound_score).

    Raises:
        NLTKResourceError: If required resources are not available.
    """
    return analyze_sentiment(word, language=language)


def get_sentiment_scores(
    words: list[str], language: str = "english"
) -> dict[str, tuple[str, float]]:
    """Get sentiment scores for a list of words.

    Args:
        words: List of words to analyze.
        language: Language for analysis ("english" or "japanese").

    Returns:
        Dictionary mapping words to (category, compound_score) tuples.

    Raises:
        NLTKResourceError: If required resources are not available.
    """
    if not words:
        return {}

    result: dict[str, tuple[str, float]] = {}
    for word in set(words):  # Deduplicate
        word_lower = word.lower()
        if word_lower not in result:
            result[word_lower] = analyze_word_sentiment(word_lower, language=language)

    return result
