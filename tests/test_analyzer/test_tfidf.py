"""Tests for TF-IDF calculation module."""

import math
from collections import Counter

import pytest

from barscan.analyzer.tfidf import (
    calculate_corpus_tfidf,
    calculate_document_frequencies,
    calculate_idf,
    calculate_tf,
    calculate_tfidf_scores,
)


class TestCalculateDocumentFrequencies:
    """Tests for calculate_document_frequencies function."""

    def test_empty_list(self) -> None:
        """Test with empty document list."""
        result = calculate_document_frequencies([])
        assert result == {}

    def test_single_document(self) -> None:
        """Test with single document."""
        docs = [Counter({"love": 3, "hate": 1})]
        result = calculate_document_frequencies(docs)
        assert result == {"love": 1, "hate": 1}

    def test_multiple_documents(self) -> None:
        """Test with multiple documents."""
        docs = [
            Counter({"love": 3, "hate": 1}),
            Counter({"love": 2, "peace": 1}),
            Counter({"peace": 2, "joy": 1}),
        ]
        result = calculate_document_frequencies(docs)
        assert result["love"] == 2  # appears in 2 docs
        assert result["hate"] == 1  # appears in 1 doc
        assert result["peace"] == 2  # appears in 2 docs
        assert result["joy"] == 1  # appears in 1 doc

    def test_word_in_all_documents(self) -> None:
        """Test word appearing in all documents."""
        docs = [
            Counter({"common": 1}),
            Counter({"common": 2}),
            Counter({"common": 3}),
        ]
        result = calculate_document_frequencies(docs)
        assert result["common"] == 3


class TestCalculateIdf:
    """Tests for calculate_idf function."""

    def test_zero_document_frequency(self) -> None:
        """Test IDF when word appears in no documents."""
        result = calculate_idf(0, 10)
        assert result == 0.0

    def test_zero_total_documents(self) -> None:
        """Test IDF when there are no documents."""
        result = calculate_idf(5, 0)
        assert result == 0.0

    def test_word_in_all_documents(self) -> None:
        """Test IDF when word appears in all documents."""
        result = calculate_idf(10, 10)
        assert result == math.log(1)  # = 0

    def test_word_in_one_document(self) -> None:
        """Test IDF when word appears in one document."""
        result = calculate_idf(1, 10)
        assert result == math.log(10)

    def test_idf_is_non_negative(self) -> None:
        """Test that IDF is always non-negative."""
        for df in range(1, 11):
            for n in range(df, 11):
                result = calculate_idf(df, n)
                assert result >= 0


class TestCalculateTf:
    """Tests for calculate_tf function."""

    def test_zero_total_words(self) -> None:
        """Test TF when there are no words."""
        result = calculate_tf(5, 0)
        assert result == 0.0

    def test_basic_tf(self) -> None:
        """Test basic TF calculation."""
        result = calculate_tf(5, 100)
        assert result == 0.05

    def test_tf_range(self) -> None:
        """Test that TF is between 0 and 1."""
        result = calculate_tf(50, 100)
        assert 0 <= result <= 1


class TestCalculateTfidfScores:
    """Tests for calculate_tfidf_scores function."""

    def test_empty_word_counts(self) -> None:
        """Test with empty word counts."""
        result = calculate_tfidf_scores({}, 100, {"love": 1}, 10)
        assert result == {}

    def test_basic_calculation(self) -> None:
        """Test basic TF-IDF calculation."""
        word_counts = {"love": 10}
        doc_freqs = {"love": 1}
        result = calculate_tfidf_scores(word_counts, 100, doc_freqs, 10, normalize=False)
        assert "love" in result

    def test_normalization(self) -> None:
        """Test normalized scores are in [0, 1] range."""
        word_counts = {"love": 10, "hate": 5, "peace": 2}
        doc_freqs = {"love": 1, "hate": 2, "peace": 5}
        result = calculate_tfidf_scores(word_counts, 100, doc_freqs, 10, normalize=True)

        for score in result.values():
            assert 0 <= score <= 1

        # Highest TF-IDF should be normalized to 1.0
        assert max(result.values()) == 1.0

    def test_missing_doc_frequency(self) -> None:
        """Test handling of words missing from doc_freqs."""
        word_counts = {"unknown": 5}
        doc_freqs = {}
        result = calculate_tfidf_scores(word_counts, 100, doc_freqs, 10)
        assert result["unknown"] == 0.0


class TestCalculateCorpusTfidf:
    """Tests for calculate_corpus_tfidf function."""

    def test_empty_corpus(self) -> None:
        """Test with empty corpus."""
        result = calculate_corpus_tfidf([], {}, 0)
        assert result == {}

    def test_single_song(self) -> None:
        """Test with single song."""
        word_counts_per_song = [Counter({"love": 5, "hate": 2})]
        aggregate_counts = {"love": 5, "hate": 2}
        result = calculate_corpus_tfidf(word_counts_per_song, aggregate_counts, 7)
        assert "love" in result
        assert "hate" in result

    def test_multiple_songs(self) -> None:
        """Test with multiple songs."""
        word_counts_per_song = [
            Counter({"love": 5, "music": 3}),
            Counter({"love": 2, "dance": 4}),
            Counter({"music": 2, "dance": 1}),
        ]
        aggregate_counts = {"love": 7, "music": 5, "dance": 5}
        result = calculate_corpus_tfidf(word_counts_per_song, aggregate_counts, 17)

        assert "love" in result
        assert "music" in result
        assert "dance" in result

    def test_normalization_max_is_one(self) -> None:
        """Test that maximum normalized score is 1.0."""
        word_counts_per_song = [
            Counter({"rare": 10}),  # High TF, appears in one doc
            Counter({"common": 1}),
        ]
        aggregate_counts = {"rare": 10, "common": 1}
        result = calculate_corpus_tfidf(word_counts_per_song, aggregate_counts, 11)
        assert max(result.values()) == 1.0
