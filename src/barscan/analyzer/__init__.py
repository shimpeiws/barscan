"""Text analysis module for lyrics word frequency analysis."""

from barscan.analyzer.filters import (
    apply_filters,
    filter_by_length,
    filter_non_alphabetic,
    filter_stop_words,
    get_stop_words,
)
from barscan.analyzer.frequency import (
    aggregate_results,
    analyze_lyrics,
    analyze_text,
    collect_tokens_with_positions,
    count_frequencies,
    create_word_frequencies,
    get_word_counts_per_song,
)
from barscan.analyzer.models import (
    AggregateAnalysisResult,
    AnalysisConfig,
    AnalysisResult,
    ContextsMode,
    TokenWithPosition,
    WordContext,
    WordFrequency,
)
from barscan.analyzer.processor import (
    clean_lyrics,
    clean_lyrics_preserve_lines,
    ensure_nltk_resources,
    lemmatize,
    normalize_text,
    preprocess,
    tokenize,
    tokenize_with_positions,
)

__all__ = [
    # Models
    "AnalysisConfig",
    "AnalysisResult",
    "AggregateAnalysisResult",
    "WordFrequency",
    "ContextsMode",
    "TokenWithPosition",
    "WordContext",
    # Processor
    "clean_lyrics",
    "clean_lyrics_preserve_lines",
    "normalize_text",
    "tokenize",
    "tokenize_with_positions",
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
    "analyze_text",
    "analyze_lyrics",
    "aggregate_results",
    "count_frequencies",
    "create_word_frequencies",
    "get_word_counts_per_song",
    "collect_tokens_with_positions",
]
