"""Text analysis module for lyrics word frequency analysis."""

from barscan.analyzer.filters import (
    apply_filters,
    filter_by_length,
    filter_non_alphabetic,
    filter_stop_words,
    get_stop_words,
)
from barscan.analyzer.frequency import (
    LyricsProtocol,
    aggregate_results,
    analyze_lyrics,
    analyze_text,
    count_frequencies,
    create_word_frequencies,
)
from barscan.analyzer.models import (
    AggregateAnalysisResult,
    AnalysisConfig,
    AnalysisResult,
    WordFrequency,
)
from barscan.analyzer.processor import (
    clean_lyrics,
    ensure_nltk_resources,
    lemmatize,
    normalize_text,
    preprocess,
    tokenize,
)

__all__ = [
    # Models
    "AnalysisConfig",
    "AnalysisResult",
    "AggregateAnalysisResult",
    "WordFrequency",
    # Processor
    "clean_lyrics",
    "normalize_text",
    "tokenize",
    "lemmatize",
    "preprocess",
    "ensure_nltk_resources",
    # Filters
    "filter_stop_words",
    "filter_by_length",
    "filter_non_alphabetic",
    "apply_filters",
    "get_stop_words",
    # Frequency
    "LyricsProtocol",
    "analyze_text",
    "analyze_lyrics",
    "aggregate_results",
    "count_frequencies",
    "create_word_frequencies",
]
