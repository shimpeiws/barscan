"""TF-IDF calculation for lyrics corpus."""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def calculate_document_frequencies(
    word_counts_per_song: list[Counter[str]],
) -> dict[str, int]:
    """Calculate document frequency for each word.

    Document frequency is the number of songs that contain each word.

    Args:
        word_counts_per_song: List of Counter objects, one per song.

    Returns:
        Dictionary mapping words to their document frequency.
    """
    doc_freq: Counter[str] = Counter()
    for word_counts in word_counts_per_song:
        # Each word counts once per document, regardless of frequency
        for word in word_counts:
            doc_freq[word] += 1
    return dict(doc_freq)


def calculate_idf(
    document_frequency: int,
    total_documents: int,
) -> float:
    """Calculate IDF (Inverse Document Frequency) score.

    Uses log(N/df) formula where N is total documents and df is document frequency.
    Returns 0 if the word appears in no documents.

    Args:
        document_frequency: Number of documents containing the word.
        total_documents: Total number of documents in corpus.

    Returns:
        IDF score (always >= 0).
    """
    if document_frequency == 0 or total_documents == 0:
        return 0.0
    return math.log(total_documents / document_frequency)


def calculate_tf(word_count: int, total_words: int) -> float:
    """Calculate TF (Term Frequency) score.

    Uses raw term frequency divided by total words in document.

    Args:
        word_count: Number of times the word appears.
        total_words: Total words in the document/corpus.

    Returns:
        TF score (0.0 to 1.0).
    """
    if total_words == 0:
        return 0.0
    return word_count / total_words


def calculate_tfidf_scores(
    word_counts: dict[str, int],
    total_words: int,
    document_frequencies: dict[str, int],
    total_documents: int,
    normalize: bool = True,
) -> dict[str, float]:
    """Calculate TF-IDF scores for all words.

    Args:
        word_counts: Dictionary mapping words to their counts.
        total_words: Total words in the corpus.
        document_frequencies: Dictionary mapping words to their document frequency.
        total_documents: Total number of documents in corpus.
        normalize: Whether to normalize scores to 0.0-1.0 range.

    Returns:
        Dictionary mapping words to their TF-IDF scores.
    """
    if not word_counts:
        return {}

    tfidf_scores: dict[str, float] = {}

    for word, count in word_counts.items():
        tf = calculate_tf(count, total_words)
        doc_freq = document_frequencies.get(word, 0)
        idf = calculate_idf(doc_freq, total_documents)
        tfidf_scores[word] = tf * idf

    if normalize and tfidf_scores:
        max_score = max(tfidf_scores.values())
        if max_score > 0:
            tfidf_scores = {
                word: round(score / max_score, 4) for word, score in tfidf_scores.items()
            }

    return tfidf_scores


def calculate_corpus_tfidf(
    word_counts_per_song: list[Counter[str]],
    aggregate_counts: dict[str, int],
    total_words: int,
    normalize: bool = True,
) -> dict[str, float]:
    """Calculate TF-IDF scores for an aggregated corpus.

    This is the main function to use when computing TF-IDF for WordGrain output.
    It treats each song as a document and calculates TF-IDF based on:
    - TF: word frequency in the entire corpus
    - IDF: inverse of how many songs contain the word

    Args:
        word_counts_per_song: List of Counter objects, one per song.
        aggregate_counts: Aggregated word counts across all songs.
        total_words: Total words in the entire corpus.
        normalize: Whether to normalize scores to 0.0-1.0 range.

    Returns:
        Dictionary mapping words to their TF-IDF scores.
    """
    total_documents = len(word_counts_per_song)
    if total_documents == 0:
        return {}

    doc_frequencies = calculate_document_frequencies(word_counts_per_song)

    return calculate_tfidf_scores(
        word_counts=aggregate_counts,
        total_words=total_words,
        document_frequencies=doc_frequencies,
        total_documents=total_documents,
        normalize=normalize,
    )
