"""Tests for context extraction module."""

import pytest

from barscan.analyzer.context import (
    extract_contexts_for_word,
    extract_full_context,
    extract_short_context,
    group_tokens_by_word,
)
from barscan.analyzer.models import ContextsMode, TokenWithPosition, WordContext


class TestExtractShortContext:
    """Tests for extract_short_context function."""

    def test_word_in_middle(self) -> None:
        """Test extracting context when word is in the middle."""
        line = "I love you baby"
        result = extract_short_context(line, "love", window_size=1)
        assert "I" in result
        assert "love" in result
        assert "you" in result

    def test_word_at_start(self) -> None:
        """Test extracting context when word is at the start."""
        line = "Love is all you need"
        result = extract_short_context(line, "love", window_size=2)
        assert "Love" in result
        assert "is" in result
        # Should not have leading ellipsis
        assert not result.startswith("...")

    def test_word_at_end(self) -> None:
        """Test extracting context when word is at the end."""
        line = "All you need is love"
        result = extract_short_context(line, "love", window_size=2)
        assert "love" in result
        assert "is" in result
        # Should not have trailing ellipsis
        assert not result.endswith("...")

    def test_ellipsis_markers(self) -> None:
        """Test that ellipsis markers are added when needed."""
        line = "one two three four five six seven eight"
        result = extract_short_context(line, "five", window_size=1)
        assert result.startswith("...")
        assert result.endswith("...")

    def test_word_not_found(self) -> None:
        """Test when word is not found in line."""
        line = "I love music"
        result = extract_short_context(line, "xyz", window_size=2)
        # Should return truncated line
        assert result is not None

    def test_case_insensitive(self) -> None:
        """Test case-insensitive word matching."""
        line = "I LOVE you"
        result = extract_short_context(line, "love", window_size=1)
        assert "LOVE" in result


class TestExtractFullContext:
    """Tests for extract_full_context function."""

    def test_basic_context(self) -> None:
        """Test basic full context extraction."""
        result = extract_full_context(
            line="Every day I'm hustlin'",
            song_title="Hustlin'",
        )
        assert isinstance(result, WordContext)
        assert result.line == "Every day I'm hustlin'"
        assert result.track == "Hustlin'"
        assert result.album is None
        assert result.year is None

    def test_with_album_and_year(self) -> None:
        """Test full context with album and year."""
        result = extract_full_context(
            line="Every day I'm hustlin'",
            song_title="Hustlin'",
            album="Port of Miami",
            year=2006,
        )
        assert result.album == "Port of Miami"
        assert result.year == 2006

    def test_strips_whitespace(self) -> None:
        """Test that line is stripped of whitespace."""
        result = extract_full_context(
            line="  Some lyrics here  ",
            song_title="Test Song",
        )
        assert result.line == "Some lyrics here"


class TestExtractContextsForWord:
    """Tests for extract_contexts_for_word function."""

    @pytest.fixture
    def sample_tokens(self) -> list[TokenWithPosition]:
        """Create sample tokens with positions."""
        return [
            TokenWithPosition(
                token="love",
                line_index=0,
                word_index=1,
                original_line="I love you",
                song_id=1,
                song_title="Song 1",
            ),
            TokenWithPosition(
                token="you",
                line_index=0,
                word_index=2,
                original_line="I love you",
                song_id=1,
                song_title="Song 1",
            ),
            TokenWithPosition(
                token="love",
                line_index=1,
                word_index=0,
                original_line="Love is all",
                song_id=1,
                song_title="Song 1",
            ),
            TokenWithPosition(
                token="love",
                line_index=0,
                word_index=2,
                original_line="My endless love",
                song_id=2,
                song_title="Song 2",
            ),
        ]

    def test_none_mode(self, sample_tokens: list[TokenWithPosition]) -> None:
        """Test that NONE mode returns None."""
        result = extract_contexts_for_word(
            tokens_with_positions=sample_tokens,
            word="love",
            mode=ContextsMode.NONE,
        )
        assert result is None

    def test_short_mode(self, sample_tokens: list[TokenWithPosition]) -> None:
        """Test SHORT mode returns string contexts."""
        result = extract_contexts_for_word(
            tokens_with_positions=sample_tokens,
            word="love",
            mode=ContextsMode.SHORT,
            max_contexts=3,
        )
        assert result is not None
        assert isinstance(result, tuple)
        assert all(isinstance(ctx, str) for ctx in result)

    def test_full_mode(self, sample_tokens: list[TokenWithPosition]) -> None:
        """Test FULL mode returns WordContext objects."""
        result = extract_contexts_for_word(
            tokens_with_positions=sample_tokens,
            word="love",
            mode=ContextsMode.FULL,
            max_contexts=3,
        )
        assert result is not None
        assert isinstance(result, tuple)
        assert all(isinstance(ctx, WordContext) for ctx in result)

    def test_max_contexts_limit(self, sample_tokens: list[TokenWithPosition]) -> None:
        """Test that max_contexts limits the number of results."""
        result = extract_contexts_for_word(
            tokens_with_positions=sample_tokens,
            word="love",
            mode=ContextsMode.SHORT,
            max_contexts=2,
        )
        assert result is not None
        assert len(result) <= 2

    def test_word_not_found(self, sample_tokens: list[TokenWithPosition]) -> None:
        """Test when word is not found in tokens."""
        result = extract_contexts_for_word(
            tokens_with_positions=sample_tokens,
            word="xyz",
            mode=ContextsMode.SHORT,
        )
        assert result is None

    def test_case_insensitive(self, sample_tokens: list[TokenWithPosition]) -> None:
        """Test case-insensitive word matching."""
        result = extract_contexts_for_word(
            tokens_with_positions=sample_tokens,
            word="LOVE",
            mode=ContextsMode.SHORT,
        )
        assert result is not None

    def test_deduplicates_same_line(self, sample_tokens: list[TokenWithPosition]) -> None:
        """Test that duplicate contexts from same line are deduplicated."""
        # Add duplicate token on same line
        tokens = sample_tokens + [
            TokenWithPosition(
                token="love",
                line_index=0,
                word_index=3,
                original_line="I love you",
                song_id=1,
                song_title="Song 1",
            ),
        ]
        result = extract_contexts_for_word(
            tokens_with_positions=tokens,
            word="love",
            mode=ContextsMode.SHORT,
            max_contexts=10,
        )
        # Should deduplicate by (song_id, line_index)
        assert result is not None
        # Count unique (song_id, line_index) pairs
        assert len(result) == 3


class TestGroupTokensByWord:
    """Tests for group_tokens_by_word function."""

    def test_empty_list(self) -> None:
        """Test with empty token list."""
        result = group_tokens_by_word([])
        assert result == {}

    def test_grouping(self) -> None:
        """Test basic grouping."""
        tokens = [
            TokenWithPosition(
                token="love", line_index=0, word_index=0,
                original_line="love", song_id=1, song_title="Song"
            ),
            TokenWithPosition(
                token="hate", line_index=0, word_index=1,
                original_line="hate", song_id=1, song_title="Song"
            ),
            TokenWithPosition(
                token="love", line_index=1, word_index=0,
                original_line="love", song_id=1, song_title="Song"
            ),
        ]
        result = group_tokens_by_word(tokens)
        assert len(result) == 2
        assert len(result["love"]) == 2
        assert len(result["hate"]) == 1

    def test_case_normalization(self) -> None:
        """Test that words are normalized to lowercase."""
        tokens = [
            TokenWithPosition(
                token="Love", line_index=0, word_index=0,
                original_line="Love", song_id=1, song_title="Song"
            ),
            TokenWithPosition(
                token="LOVE", line_index=1, word_index=0,
                original_line="LOVE", song_id=1, song_title="Song"
            ),
        ]
        result = group_tokens_by_word(tokens)
        assert len(result) == 1
        assert "love" in result
        assert len(result["love"]) == 2
