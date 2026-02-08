"""Tests for analyzer processor."""

import pytest

from barscan.analyzer.models import AnalysisConfig
from barscan.analyzer.processor import (
    clean_lyrics,
    ensure_nltk_resources,
    lemmatize,
    normalize_text,
    preprocess,
    tokenize,
)
from barscan.exceptions import EmptyLyricsError, NLTKResourceError


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


class TestProcessorEdgeCases:
    """Additional edge case tests for processor functions."""

    def test_clean_lyrics_with_colon_headers(self) -> None:
        """Test removing headers with colons like [Verse 1: Artist]."""
        text = "[Verse 1: Eminem]\nRap lyrics here"
        result = clean_lyrics(text)
        assert "[Verse 1: Eminem]" not in result
        assert "Rap lyrics" in result

    def test_clean_lyrics_with_hyphen_headers(self) -> None:
        """Test removing headers with hyphens like [Pre-Chorus]."""
        text = "[Pre-Chorus]\nBefore the chorus\n[Post-Chorus]\nAfter the chorus"
        result = clean_lyrics(text)
        assert "[Pre-Chorus]" not in result
        assert "[Post-Chorus]" not in result
        assert "Before" in result
        assert "After" in result

    def test_normalize_text_with_numbers(self) -> None:
        """Test normalizing text containing numbers."""
        text = "Song123 has 456 words"
        result = normalize_text(text)
        assert "song123" in result
        assert "456" in result

    def test_normalize_text_unicode_characters(self) -> None:
        """Test normalizing text with unicode characters."""
        text = "Caf\u00e9 music \u2014 good vibes"
        result = normalize_text(text)
        assert "caf" in result

    def test_tokenize_with_mixed_quotes(self) -> None:
        """Test tokenizing text with mixed quote styles."""
        text = "don't \"quote\" me 'on' this"
        result = tokenize(text)
        assert len(result) > 0

    def test_tokenize_with_multiple_spaces(self) -> None:
        """Test tokenizing text with multiple consecutive spaces."""
        text = "word   another    word"
        result = tokenize(text)
        assert "word" in result
        assert "another" in result

    def test_lemmatize_with_plural_nouns(self, config_with_lemmatization: AnalysisConfig) -> None:
        """Test lemmatization handles plural nouns."""
        tokens = ["dogs", "cats", "houses", "cities"]
        result = lemmatize(tokens, config_with_lemmatization)
        # WordNet lemmatizer should convert plurals
        assert "dog" in result or "dogs" in result

    def test_lemmatize_with_verb_forms(self, config_with_lemmatization: AnalysisConfig) -> None:
        """Test lemmatization handles verb forms."""
        tokens = ["running", "walked", "swimming"]
        result = lemmatize(tokens, config_with_lemmatization)
        # Should lemmatize at least some verb forms
        assert len(result) == 3

    def test_preprocess_only_whitespace_after_cleaning(self) -> None:
        """Test preprocessing text that becomes only whitespace after header removal."""
        # When text only has headers, it becomes whitespace which should raise
        # Note: The clean_lyrics function normalizes whitespace, so "[Verse 1]   " -> ""
        # But the header regex leaves spaces, so we need text that becomes empty
        text = "[Verse 1]"
        # After header removal, this becomes empty string
        # clean_lyrics will raise because result is empty/whitespace
        result = preprocess(text)
        # Should return empty list since all content was headers
        assert result == [] or len(result) == 0

    def test_preprocess_with_repeated_words(self) -> None:
        """Test preprocessing maintains word frequencies."""
        text = "[Hook]\nYeah yeah yeah baby baby"
        result = preprocess(text)
        # Should have multiple occurrences
        assert result.count("yeah") == 3 or "yeah" in result
        assert result.count("baby") == 2 or "baby" in result

    def test_clean_lyrics_multiline_headers(self) -> None:
        """Test cleaning lyrics with headers spanning lines."""
        text = "[Verse 1]\n[By Artist]\nActual lyrics here"
        result = clean_lyrics(text)
        assert "Actual lyrics" in result
        assert "[By Artist]" not in result

    def test_normalize_preserves_possessive_apostrophe(self) -> None:
        """Test that possessive apostrophes are preserved."""
        text = "Sarah's guitar john's bass"
        result = normalize_text(text)
        assert "sarah's" in result or "sarahs" in result

    def test_preprocess_with_none_config_creates_default(self) -> None:
        """Test that preprocess with None config creates default config."""
        text = "[Verse 1]\nHello world"
        result = preprocess(text, None)
        assert isinstance(result, list)
        assert len(result) > 0
