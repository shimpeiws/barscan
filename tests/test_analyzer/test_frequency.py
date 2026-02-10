"""Tests for analyzer frequency."""

from collections import Counter
from datetime import UTC, datetime

import pytest

from barscan.analyzer.frequency import (
    aggregate_results,
    analyze_lyrics,
    analyze_text,
    collect_tokens_with_positions,
    count_frequencies,
    create_word_frequencies,
    get_word_counts_per_song,
)
from barscan.analyzer.models import AnalysisConfig, AnalysisResult, WordFrequency
from barscan.exceptions import EmptyLyricsError
from barscan.genius.models import Lyrics


class TestCountFrequencies:
    """Tests for count_frequencies function."""

    def test_basic_counting(self) -> None:
        """Test basic word frequency counting."""
        tokens = ["hello", "world", "hello", "hello", "world"]
        result = count_frequencies(tokens)
        assert isinstance(result, Counter)
        assert result["hello"] == 3
        assert result["world"] == 2

    def test_empty_tokens(self) -> None:
        """Test counting empty token list."""
        result = count_frequencies([])
        assert len(result) == 0

    def test_single_token(self) -> None:
        """Test counting single token."""
        result = count_frequencies(["hello"])
        assert result["hello"] == 1

    def test_all_unique(self) -> None:
        """Test counting all unique tokens."""
        tokens = ["one", "two", "three", "four"]
        result = count_frequencies(tokens)
        assert all(count == 1 for count in result.values())


class TestCreateWordFrequencies:
    """Tests for create_word_frequencies function."""

    def test_creates_sorted_frequencies(self) -> None:
        """Test that frequencies are sorted by count descending."""
        counter: Counter[str] = Counter({"hello": 5, "world": 3, "test": 1})
        result = create_word_frequencies(counter, 9)
        assert result[0].word == "hello"
        assert result[1].word == "world"
        assert result[2].word == "test"

    def test_calculates_percentages(self) -> None:
        """Test that percentages are calculated correctly."""
        counter: Counter[str] = Counter({"hello": 50, "world": 50})
        result = create_word_frequencies(counter, 100)
        assert result[0].percentage == 50.0
        assert result[1].percentage == 50.0

    def test_empty_counter(self) -> None:
        """Test with empty counter."""
        result = create_word_frequencies(Counter(), 0)
        assert result == ()

    def test_zero_total_words(self) -> None:
        """Test with zero total words."""
        counter: Counter[str] = Counter({"hello": 5})
        result = create_word_frequencies(counter, 0)
        assert result == ()

    def test_returns_tuple(self) -> None:
        """Test that result is a tuple."""
        counter: Counter[str] = Counter({"hello": 1})
        result = create_word_frequencies(counter, 1)
        assert isinstance(result, tuple)

    def test_filters_by_min_count(self) -> None:
        """Test that words below min_count are filtered out."""
        counter: Counter[str] = Counter({"hello": 5, "world": 3, "rare": 1})
        result = create_word_frequencies(counter, 9, min_count=3)
        words = {f.word for f in result}
        assert "hello" in words
        assert "world" in words
        assert "rare" not in words

    def test_min_count_default_includes_all(self) -> None:
        """Test that default min_count (1) includes all words."""
        counter: Counter[str] = Counter({"hello": 5, "world": 3, "rare": 1})
        result = create_word_frequencies(counter, 9)
        assert len(result) == 3


class TestAnalyzeText:
    """Tests for analyze_text function."""

    def test_complete_analysis(self) -> None:
        """Test complete word frequency analysis."""
        text = "[Verse 1]\nHello world, hello universe\nThe world is beautiful"
        result = analyze_text(
            text=text,
            song_id=123,
            song_title="Test Song",
            artist_name="Test Artist",
        )
        assert isinstance(result, AnalysisResult)
        assert result.song_id == 123
        assert result.song_title == "Test Song"
        assert result.artist_name == "Test Artist"
        assert result.total_words > 0
        assert result.unique_words > 0
        assert len(result.frequencies) > 0

    def test_analysis_with_config(self) -> None:
        """Test analysis with custom config."""
        text = "Hello world goodbye world"
        config = AnalysisConfig(min_word_length=5, remove_stop_words=False)
        result = analyze_text(
            text=text,
            song_id=1,
            song_title="Test",
            artist_name="Artist",
            config=config,
        )
        # "hello", "world", "goodbye" all have 5+ chars, "world" has 5
        assert result.total_words > 0

    def test_empty_after_filtering(self) -> None:
        """Test when all words are filtered out."""
        text = "the a an is"  # All stop words
        result = analyze_text(
            text=text,
            song_id=1,
            song_title="Test",
            artist_name="Artist",
        )
        assert result.total_words == 0
        assert result.unique_words == 0
        assert len(result.frequencies) == 0

    def test_analyzed_at_is_set(self) -> None:
        """Test that analyzed_at timestamp is set."""
        text = "Hello world"
        before = datetime.now(UTC)
        result = analyze_text(
            text=text,
            song_id=1,
            song_title="Test",
            artist_name="Artist",
        )
        after = datetime.now(UTC)
        assert before <= result.analyzed_at <= after

    def test_frequency_order(self) -> None:
        """Test that frequencies are ordered by count descending."""
        text = "hello hello hello world world test"
        result = analyze_text(
            text=text,
            song_id=1,
            song_title="Test",
            artist_name="Artist",
            config=AnalysisConfig(remove_stop_words=False, min_word_length=1),
        )
        counts = [f.count for f in result.frequencies]
        assert counts == sorted(counts, reverse=True)


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def test_aggregates_multiple_results(self) -> None:
        """Test aggregating multiple song results."""
        now = datetime.now(UTC)
        result1 = AnalysisResult(
            song_id=1,
            song_title="Song 1",
            artist_name="Artist",
            total_words=10,
            unique_words=5,
            frequencies=(
                WordFrequency(word="hello", count=5, percentage=50.0),
                WordFrequency(word="world", count=5, percentage=50.0),
            ),
            analyzed_at=now,
        )
        result2 = AnalysisResult(
            song_id=2,
            song_title="Song 2",
            artist_name="Artist",
            total_words=10,
            unique_words=5,
            frequencies=(
                WordFrequency(word="hello", count=3, percentage=30.0),
                WordFrequency(word="goodbye", count=7, percentage=70.0),
            ),
            analyzed_at=now,
        )
        aggregate = aggregate_results([result1, result2], "Artist")
        assert aggregate.artist_name == "Artist"
        assert aggregate.songs_analyzed == 2
        assert aggregate.total_words == 20  # 10 + 10
        # Check combined frequencies
        freq_dict = {f.word: f.count for f in aggregate.frequencies}
        assert freq_dict["hello"] == 8  # 5 + 3
        assert freq_dict["world"] == 5
        assert freq_dict["goodbye"] == 7

    def test_empty_results(self) -> None:
        """Test aggregating empty results list."""
        aggregate = aggregate_results([], "Artist")
        assert aggregate.songs_analyzed == 0
        assert aggregate.total_words == 0
        assert aggregate.unique_words == 0
        assert len(aggregate.frequencies) == 0

    def test_single_result(self) -> None:
        """Test aggregating single result."""
        now = datetime.now(UTC)
        result = AnalysisResult(
            song_id=1,
            song_title="Song 1",
            artist_name="Artist",
            total_words=10,
            unique_words=2,
            frequencies=(
                WordFrequency(word="hello", count=6, percentage=60.0),
                WordFrequency(word="world", count=4, percentage=40.0),
            ),
            analyzed_at=now,
        )
        aggregate = aggregate_results([result], "Artist")
        assert aggregate.songs_analyzed == 1
        assert aggregate.total_words == 10
        assert len(aggregate.song_results) == 1

    def test_includes_song_results(self) -> None:
        """Test that individual song results are included."""
        now = datetime.now(UTC)
        results = [
            AnalysisResult(
                song_id=i,
                song_title=f"Song {i}",
                artist_name="Artist",
                total_words=10,
                unique_words=2,
                frequencies=(),
                analyzed_at=now,
            )
            for i in range(3)
        ]
        aggregate = aggregate_results(results, "Artist")
        assert len(aggregate.song_results) == 3

    def test_applies_min_count_filter(self) -> None:
        """Test that min_count from config filters aggregate frequencies."""
        now = datetime.now(UTC)
        result1 = AnalysisResult(
            song_id=1,
            song_title="Song 1",
            artist_name="Artist",
            total_words=10,
            unique_words=3,
            frequencies=(
                WordFrequency(word="hello", count=5, percentage=50.0),
                WordFrequency(word="world", count=4, percentage=40.0),
                WordFrequency(word="rare", count=1, percentage=10.0),
            ),
            analyzed_at=now,
        )
        result2 = AnalysisResult(
            song_id=2,
            song_title="Song 2",
            artist_name="Artist",
            total_words=10,
            unique_words=3,
            frequencies=(
                WordFrequency(word="hello", count=3, percentage=30.0),
                WordFrequency(word="goodbye", count=6, percentage=60.0),
                WordFrequency(word="unique", count=1, percentage=10.0),
            ),
            analyzed_at=now,
        )
        config = AnalysisConfig(min_count=3)
        aggregate = aggregate_results([result1, result2], "Artist", config)

        freq_dict = {f.word: f.count for f in aggregate.frequencies}
        # hello: 5+3=8, world: 4, goodbye: 6 should be included
        assert "hello" in freq_dict
        assert "world" in freq_dict
        assert "goodbye" in freq_dict
        # rare: 1, unique: 1 should be filtered out
        assert "rare" not in freq_dict
        assert "unique" not in freq_dict


class TestAnalyzeLyrics:
    """Tests for analyze_lyrics function."""

    def test_analyze_lyrics_with_valid_lyrics(self) -> None:
        """Test analyzing a Lyrics object."""
        lyrics = Lyrics(
            song_id=123,
            song_title="Test Song",
            artist_name="Test Artist",
            lyrics_text="Hello world hello universe",
        )
        result = analyze_lyrics(lyrics)

        assert result.song_id == 123
        assert result.song_title == "Test Song"
        assert result.artist_name == "Test Artist"
        assert result.total_words > 0

    def test_analyze_lyrics_empty_raises_error(self) -> None:
        """Test that empty lyrics raises EmptyLyricsError."""
        lyrics = Lyrics(
            song_id=123,
            song_title="Test Song",
            artist_name="Test Artist",
            lyrics_text="",
        )
        with pytest.raises(EmptyLyricsError, match="No lyrics available"):
            analyze_lyrics(lyrics)

    def test_analyze_lyrics_with_config(self) -> None:
        """Test analyze_lyrics with custom config."""
        lyrics = Lyrics(
            song_id=123,
            song_title="Test Song",
            artist_name="Test Artist",
            lyrics_text="Hello world goodbye world",
        )
        config = AnalysisConfig(use_lemmatization=True)
        result = analyze_lyrics(lyrics, config=config)

        assert result.song_id == 123
        assert result.total_words > 0


class TestGetWordCountsPerSong:
    """Tests for get_word_counts_per_song function."""

    def test_extracts_counters_from_results(self) -> None:
        """Test extracting word counters from analysis results."""
        now = datetime.now(UTC)
        results = [
            AnalysisResult(
                song_id=1,
                song_title="Song 1",
                artist_name="Artist",
                total_words=10,
                unique_words=2,
                frequencies=(
                    WordFrequency(word="hello", count=6, percentage=60.0),
                    WordFrequency(word="world", count=4, percentage=40.0),
                ),
                analyzed_at=now,
            ),
        ]

        counters = get_word_counts_per_song(results)

        assert len(counters) == 1
        assert counters[0]["hello"] == 6
        assert counters[0]["world"] == 4

    def test_multiple_songs(self) -> None:
        """Test extracting counters from multiple songs."""
        now = datetime.now(UTC)
        results = [
            AnalysisResult(
                song_id=1,
                song_title="Song 1",
                artist_name="Artist",
                total_words=10,
                unique_words=2,
                frequencies=(
                    WordFrequency(word="hello", count=6, percentage=60.0),
                    WordFrequency(word="world", count=4, percentage=40.0),
                ),
                analyzed_at=now,
            ),
            AnalysisResult(
                song_id=2,
                song_title="Song 2",
                artist_name="Artist",
                total_words=10,
                unique_words=2,
                frequencies=(
                    WordFrequency(word="goodbye", count=7, percentage=70.0),
                    WordFrequency(word="universe", count=3, percentage=30.0),
                ),
                analyzed_at=now,
            ),
        ]

        counters = get_word_counts_per_song(results)

        assert len(counters) == 2
        assert counters[0]["hello"] == 6
        assert counters[1]["goodbye"] == 7

    def test_empty_results(self) -> None:
        """Test with empty results list."""
        counters = get_word_counts_per_song([])
        assert counters == []


class TestCollectTokensWithPositions:
    """Tests for collect_tokens_with_positions function."""

    def test_collects_from_multiple_songs(self) -> None:
        """Test collecting tokens from multiple songs."""
        lyrics_data = [
            ("[Verse 1]\nHello world", 1, "Song 1"),
            ("[Verse 1]\nGoodbye world", 2, "Song 2"),
        ]

        tokens = collect_tokens_with_positions(lyrics_data)

        assert len(tokens) > 0
        song_ids = {t.song_id for t in tokens}
        assert 1 in song_ids
        assert 2 in song_ids

    def test_skips_empty_lyrics(self) -> None:
        """Test that empty lyrics are skipped."""
        lyrics_data = [
            ("[Verse 1]\nHello world", 1, "Song 1"),
            ("", 2, "Song 2"),  # Empty
            ("[Verse 1]\nGoodbye", 3, "Song 3"),
        ]

        tokens = collect_tokens_with_positions(lyrics_data)

        song_ids = {t.song_id for t in tokens}
        assert 1 in song_ids
        assert 2 not in song_ids  # Empty lyrics skipped
        assert 3 in song_ids

    def test_with_config(self) -> None:
        """Test collecting tokens with custom config."""
        lyrics_data = [
            ("[Verse 1]\nRunning cats dogs", 1, "Song 1"),
        ]
        config = AnalysisConfig(use_lemmatization=True)

        tokens = collect_tokens_with_positions(lyrics_data, config=config)

        assert len(tokens) > 0

    def test_empty_lyrics_data(self) -> None:
        """Test with empty lyrics data list."""
        tokens = collect_tokens_with_positions([])
        assert tokens == []

    def test_tokens_have_position_info(self) -> None:
        """Test that tokens have correct position information."""
        lyrics_data = [
            ("[Verse 1]\nHello world", 123, "Test Song"),
        ]

        tokens = collect_tokens_with_positions(lyrics_data)

        assert all(t.song_id == 123 for t in tokens)
        assert all(t.song_title == "Test Song" for t in tokens)
        assert all(isinstance(t.line_index, int) for t in tokens)
        assert all(isinstance(t.word_index, int) for t in tokens)
