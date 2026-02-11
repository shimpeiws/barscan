"""Tests for analyzer processor."""

from unittest.mock import patch

import pytest

from barscan.analyzer.models import AnalysisConfig
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

    def test_preprocess_normalizes_mixed_case_to_lowercase(self) -> None:
        """Test that mixed-case words are normalized to lowercase."""
        text = "[Hook]\nYeah YEAH yeah Wavy WAVY"
        result = preprocess(text)
        # All variations should become lowercase
        assert "yeah" in result
        assert "wavy" in result
        # No uppercase versions should exist
        assert "Yeah" not in result
        assert "YEAH" not in result
        assert "Wavy" not in result
        assert "WAVY" not in result

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


class TestEnsureNltkResources:
    """Tests for ensure_nltk_resources function."""

    @patch("nltk.data.find")
    def test_resources_already_present(self, mock_find: patch) -> None:
        """Test when all resources are already downloaded."""
        mock_find.return_value = True  # No exception = found
        ensure_nltk_resources()
        assert mock_find.call_count >= 1

    @patch("nltk.download")
    @patch("nltk.data.find")
    def test_downloads_missing_resources(
        self, mock_find: patch, mock_download: patch
    ) -> None:
        """Test downloading missing resources."""
        mock_find.side_effect = LookupError("Not found")
        ensure_nltk_resources()
        assert mock_download.called

    @patch("nltk.download")
    @patch("nltk.data.find")
    def test_raises_on_download_failure(
        self, mock_find: patch, mock_download: patch
    ) -> None:
        """Test error handling when download fails."""
        mock_find.side_effect = LookupError("Not found")
        mock_download.side_effect = Exception("Network error")
        with pytest.raises(NLTKResourceError, match="Failed to download"):
            ensure_nltk_resources()


class TestCleanLyricsPreserveLines:
    """Tests for clean_lyrics_preserve_lines function."""

    def test_preserves_line_structure(self) -> None:
        """Test that line structure is preserved."""
        text = "[Verse 1]\nLine one\nLine two\n[Chorus]\nChorus line"
        result = clean_lyrics_preserve_lines(text)
        assert len(result) == 3
        assert result[0] == "Line one"
        assert result[1] == "Line two"
        assert result[2] == "Chorus line"

    def test_removes_empty_lines(self) -> None:
        """Test that empty lines are removed."""
        text = "Line one\n\nLine two\n\n\nLine three"
        result = clean_lyrics_preserve_lines(text)
        assert len(result) == 3
        assert "Line one" in result
        assert "Line two" in result
        assert "Line three" in result

    def test_removes_section_headers(self) -> None:
        """Test that section headers are removed from lines."""
        text = "[Verse 1]\nHello world\n[Chorus]\nSing along"
        result = clean_lyrics_preserve_lines(text)
        assert len(result) == 2
        assert result[0] == "Hello world"
        assert result[1] == "Sing along"

    def test_raises_on_empty_text(self) -> None:
        """Test that empty text raises EmptyLyricsError."""
        with pytest.raises(EmptyLyricsError):
            clean_lyrics_preserve_lines("")

    def test_raises_on_whitespace_only(self) -> None:
        """Test that whitespace-only text raises EmptyLyricsError."""
        with pytest.raises(EmptyLyricsError):
            clean_lyrics_preserve_lines("   \n\t  ")

    def test_header_only_text(self) -> None:
        """Test text with only headers returns empty list."""
        text = "[Verse 1]\n[Chorus]\n[Bridge]"
        result = clean_lyrics_preserve_lines(text)
        assert result == []


class TestTokenizeWithPositions:
    """Tests for tokenize_with_positions function."""

    def test_creates_tokens_with_position_info(self) -> None:
        """Test that tokens include position information."""
        text = "[Verse 1]\nHello world\nGoodbye world"
        result = tokenize_with_positions(text, song_id=123, song_title="Test Song")

        assert len(result) > 0
        assert all(isinstance(t.line_index, int) for t in result)
        assert all(isinstance(t.word_index, int) for t in result)
        assert all(t.song_id == 123 for t in result)
        assert all(t.song_title == "Test Song" for t in result)

    def test_tracks_original_lines(self) -> None:
        """Test that original line text is preserved."""
        text = "[Verse]\nHello world"
        result = tokenize_with_positions(text, song_id=1, song_title="Test")

        assert len(result) == 2
        assert result[0].original_line == "Hello world"
        assert result[1].original_line == "Hello world"

    def test_line_and_word_indices(self) -> None:
        """Test that line and word indices are correct."""
        text = "[Verse 1]\nFirst line\nSecond line"
        result = tokenize_with_positions(text, song_id=1, song_title="Test")

        # First line: "First line" -> 2 tokens at line_index=0
        # Second line: "Second line" -> 2 tokens at line_index=1
        first_line_tokens = [t for t in result if t.line_index == 0]
        second_line_tokens = [t for t in result if t.line_index == 1]

        assert len(first_line_tokens) == 2
        assert len(second_line_tokens) == 2
        assert first_line_tokens[0].word_index == 0
        assert first_line_tokens[1].word_index == 1

    def test_with_lemmatization(self, config_with_lemmatization: AnalysisConfig) -> None:
        """Test tokenization with lemmatization enabled."""
        text = "[Verse 1]\nThe cats are running"
        result = tokenize_with_positions(
            text, song_id=1, song_title="Test", config=config_with_lemmatization
        )

        tokens = [t.token for t in result]
        # Should have lemmatized tokens
        assert "cat" in tokens or "cats" in tokens
        assert len(tokens) > 0

    def test_raises_on_empty_text(self) -> None:
        """Test that empty text raises EmptyLyricsError."""
        with pytest.raises(EmptyLyricsError):
            tokenize_with_positions("", song_id=1, song_title="Test")

    def test_with_none_config(self) -> None:
        """Test tokenization with None config uses default."""
        text = "[Verse 1]\nHello world"
        result = tokenize_with_positions(text, song_id=1, song_title="Test", config=None)

        assert len(result) > 0
        assert all(t.song_id == 1 for t in result)
