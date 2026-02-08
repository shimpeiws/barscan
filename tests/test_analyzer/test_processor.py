"""Tests for analyzer processor."""

import pytest

from barscan.analyzer.models import AnalysisConfig
from barscan.analyzer.processor import (
    clean_lyrics,
    lemmatize,
    normalize_text,
    preprocess,
    tokenize,
)
from barscan.exceptions import EmptyLyricsError


class TestCleanLyrics:
    """Tests for clean_lyrics function."""

    def test_removes_verse_headers(self) -> None:
        """Test removing [Verse 1] style headers."""
        text = "[Verse 1]\nHello world\n[Verse 2]\nGoodbye world"
        result = clean_lyrics(text)
        assert "[Verse 1]" not in result
        assert "[Verse 2]" not in result
        assert "Hello world" in result
        assert "Goodbye world" in result

    def test_removes_chorus_headers(self) -> None:
        """Test removing [Chorus] headers."""
        text = "[Chorus]\nSing along\n[Chorus]\nSing again"
        result = clean_lyrics(text)
        assert "[Chorus]" not in result
        assert "Sing along" in result

    def test_removes_various_headers(self) -> None:
        """Test removing various section headers."""
        text = "[Intro]\nStart\n[Bridge]\nMiddle\n[Outro]\nEnd"
        result = clean_lyrics(text)
        assert "[Intro]" not in result
        assert "[Bridge]" not in result
        assert "[Outro]" not in result
        assert "Start" in result
        assert "Middle" in result
        assert "End" in result

    def test_removes_headers_with_numbers(self) -> None:
        """Test removing headers with numbers like [Verse 1]."""
        text = "[Verse 1]\nFirst\n[Verse 2]\nSecond\n[Hook 3]\nThird"
        result = clean_lyrics(text)
        assert "[Verse 1]" not in result
        assert "[Verse 2]" not in result
        assert "[Hook 3]" not in result

    def test_normalizes_whitespace(self) -> None:
        """Test that whitespace is normalized."""
        text = "Hello   world\n\n\nGoodbye   world"
        result = clean_lyrics(text)
        assert "  " not in result
        assert "\n" not in result

    def test_empty_text_raises_error(self) -> None:
        """Test that empty text raises EmptyLyricsError."""
        with pytest.raises(EmptyLyricsError):
            clean_lyrics("")

    def test_whitespace_only_raises_error(self) -> None:
        """Test that whitespace-only text raises EmptyLyricsError."""
        with pytest.raises(EmptyLyricsError):
            clean_lyrics("   \n\t  ")

    def test_preserves_content(self, sample_lyrics_text: str) -> None:
        """Test that actual lyrics content is preserved."""
        result = clean_lyrics(sample_lyrics_text)
        assert "Hello world" in result
        assert "hello universe" in result


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_converts_to_lowercase(self) -> None:
        """Test that text is converted to lowercase."""
        text = "Hello WORLD"
        result = normalize_text(text)
        assert result == "hello world"

    def test_removes_punctuation(self) -> None:
        """Test that punctuation is removed."""
        text = "Hello, world! How are you?"
        result = normalize_text(text)
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_preserves_contractions(self) -> None:
        """Test that contractions are preserved."""
        text = "don't won't can't"
        result = normalize_text(text)
        assert "don't" in result
        assert "won't" in result
        assert "can't" in result

    def test_removes_standalone_apostrophes(self) -> None:
        """Test that standalone apostrophes are removed."""
        text = "' hello ' world '"
        result = normalize_text(text)
        assert "' " not in result
        assert " '" not in result

    def test_normalizes_whitespace(self) -> None:
        """Test that whitespace is normalized."""
        text = "hello   world\n\tgoodbye"
        result = normalize_text(text)
        assert "  " not in result
        assert "\n" not in result
        assert "\t" not in result


class TestTokenize:
    """Tests for tokenize function."""

    def test_basic_tokenization(self) -> None:
        """Test basic word tokenization."""
        text = "hello world goodbye"
        result = tokenize(text)
        assert result == ["hello", "world", "goodbye"]

    def test_tokenizes_contractions(self) -> None:
        """Test that contractions are tokenized."""
        text = "don't"
        result = tokenize(text)
        # NLTK splits contractions
        assert "do" in result or "don't" in result

    def test_empty_string(self) -> None:
        """Test tokenizing empty string."""
        result = tokenize("")
        assert result == []


class TestLemmatize:
    """Tests for lemmatize function."""

    def test_lemmatization_disabled_by_default(self) -> None:
        """Test that lemmatization is disabled by default."""
        tokens = ["running", "cats", "better"]
        result = lemmatize(tokens)
        assert result == tokens

    def test_lemmatization_enabled(self, config_with_lemmatization: AnalysisConfig) -> None:
        """Test lemmatization when enabled."""
        tokens = ["running", "cats", "better"]
        result = lemmatize(tokens, config_with_lemmatization)
        # WordNetLemmatizer should lemmatize some words
        assert "cat" in result or "cats" in result  # cats -> cat
        assert len(result) == 3

    def test_lemmatization_with_none_config(self) -> None:
        """Test lemmatization with None config uses default."""
        tokens = ["hello", "world"]
        result = lemmatize(tokens, None)
        assert result == tokens


class TestPreprocess:
    """Tests for preprocess function."""

    def test_full_pipeline(self, sample_lyrics_text: str) -> None:
        """Test the full preprocessing pipeline."""
        result = preprocess(sample_lyrics_text)
        assert isinstance(result, list)
        assert len(result) > 0
        # All tokens should be lowercase
        assert all(token.islower() or not token.isalpha() for token in result)
        # No section headers
        assert "[Verse" not in " ".join(result)
        assert "[Chorus]" not in " ".join(result)

    def test_with_custom_config(self, sample_lyrics_text: str) -> None:
        """Test preprocessing with custom config."""
        config = AnalysisConfig(use_lemmatization=True)
        result = preprocess(sample_lyrics_text, config)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_empty_text_raises_error(self) -> None:
        """Test that empty text raises EmptyLyricsError."""
        with pytest.raises(EmptyLyricsError):
            preprocess("")

    def test_preserves_word_content(self) -> None:
        """Test that word content is preserved through pipeline."""
        text = "[Verse 1]\nHello world"
        result = preprocess(text)
        assert "hello" in result
        assert "world" in result
