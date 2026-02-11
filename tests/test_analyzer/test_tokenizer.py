"""Tests for the tokenizer module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from barscan.analyzer.tokenizer import (
    EXCLUDED_POS2,
    MEANINGFUL_POS,
    EnglishTokenizer,
    JapaneseTokenizer,
    detect_language,
    get_tokenizer,
    is_japanese_char,
    is_japanese_text,
    normalize_text_for_language,
)


class TestMeaningfulPOS:
    """Tests for MEANINGFUL_POS constant."""

    def test_meaningful_pos_contains_content_words(self) -> None:
        """Test MEANINGFUL_POS contains expected content word categories."""
        assert "名詞" in MEANINGFUL_POS  # Noun
        assert "動詞" in MEANINGFUL_POS  # Verb
        assert "形容詞" in MEANINGFUL_POS  # Adjective
        assert "副詞" in MEANINGFUL_POS  # Adverb
        assert "感動詞" in MEANINGFUL_POS  # Interjection

    def test_meaningful_pos_excludes_function_words(self) -> None:
        """Test MEANINGFUL_POS excludes function word categories."""
        assert "助詞" not in MEANINGFUL_POS  # Particle
        assert "助動詞" not in MEANINGFUL_POS  # Auxiliary verb
        assert "接続詞" not in MEANINGFUL_POS  # Conjunction
        assert "記号" not in MEANINGFUL_POS  # Symbol


class TestExcludedPOS2:
    """Tests for EXCLUDED_POS2 constant."""

    def test_excluded_pos2_contains_non_independent(self) -> None:
        """Test EXCLUDED_POS2 contains non-independent forms."""
        assert "非自立" in EXCLUDED_POS2  # Non-independent (てる, いる)
        assert "接尾" in EXCLUDED_POS2  # Suffix (さん, 様)

    def test_excluded_pos2_does_not_contain_independent(self) -> None:
        """Test EXCLUDED_POS2 does not contain independent forms."""
        assert "自立" not in EXCLUDED_POS2  # Independent verbs
        assert "一般" not in EXCLUDED_POS2  # General nouns


class TestIsJapaneseChar:
    """Tests for is_japanese_char function."""

    def test_hiragana(self) -> None:
        """Test detection of hiragana characters."""
        assert is_japanese_char("あ") is True
        assert is_japanese_char("ん") is True

    def test_katakana(self) -> None:
        """Test detection of katakana characters."""
        assert is_japanese_char("ア") is True
        assert is_japanese_char("ン") is True

    def test_kanji(self) -> None:
        """Test detection of kanji characters."""
        assert is_japanese_char("漢") is True
        assert is_japanese_char("字") is True

    def test_ascii_characters(self) -> None:
        """Test that ASCII characters are not Japanese."""
        assert is_japanese_char("a") is False
        assert is_japanese_char("Z") is False
        assert is_japanese_char("1") is False

    def test_empty_string(self) -> None:
        """Test empty string returns False."""
        assert is_japanese_char("") is False

    def test_multi_char_string(self) -> None:
        """Test multi-character string returns False."""
        assert is_japanese_char("あい") is False


class TestIsJapaneseText:
    """Tests for is_japanese_text function."""

    def test_japanese_text(self) -> None:
        """Test detection of Japanese text."""
        assert is_japanese_text("こんにちは") is True
        assert is_japanese_text("カタカナ") is True
        assert is_japanese_text("漢字") is True

    def test_english_text(self) -> None:
        """Test that English text is not Japanese."""
        assert is_japanese_text("hello world") is False
        assert is_japanese_text("Hello World") is False

    def test_mixed_text(self) -> None:
        """Test mixed Japanese and English text."""
        assert is_japanese_text("Hello こんにちは") is True
        assert is_japanese_text("Testing テスト") is True

    def test_empty_text(self) -> None:
        """Test empty text returns False."""
        assert is_japanese_text("") is False

    def test_numbers_only(self) -> None:
        """Test numbers-only text returns False."""
        assert is_japanese_text("12345") is False


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_japanese_text(self) -> None:
        """Test detection of Japanese text."""
        assert detect_language("こんにちは世界") == "japanese"
        assert detect_language("テスト") == "japanese"

    def test_english_text(self) -> None:
        """Test detection of English text."""
        assert detect_language("hello world") == "english"
        assert detect_language("This is a test") == "english"

    def test_mixed_text_with_japanese(self) -> None:
        """Test mixed text returns japanese if any Japanese chars present."""
        assert detect_language("Hello こんにちは") == "japanese"
        assert detect_language("Testing テスト 123") == "japanese"

    def test_empty_text(self) -> None:
        """Test empty text returns english."""
        assert detect_language("") == "english"


class TestNormalizeTextForLanguage:
    """Tests for normalize_text_for_language function."""

    def test_english_lowercase(self) -> None:
        """Test English text is lowercased."""
        result = normalize_text_for_language("Hello World", "english")
        assert result == "hello world"

    def test_english_punctuation_removed(self) -> None:
        """Test punctuation is removed for English."""
        result = normalize_text_for_language("Hello, World!", "english")
        assert "," not in result
        assert "!" not in result

    def test_english_preserves_apostrophe_in_contractions(self) -> None:
        """Test apostrophes in contractions are preserved."""
        result = normalize_text_for_language("don't can't", "english")
        assert "'" in result

    def test_japanese_nfkc_normalization(self) -> None:
        """Test Japanese uses NFKC normalization."""
        # Full-width numbers to half-width
        result = normalize_text_for_language("１２３", "japanese")
        assert "123" in result

    def test_japanese_punctuation_removed(self) -> None:
        """Test common punctuation is removed for Japanese."""
        result = normalize_text_for_language("こんにちは！世界？", "japanese")
        assert "！" not in result
        assert "？" not in result


class TestEnglishTokenizer:
    """Tests for EnglishTokenizer class."""

    def test_tokenize_simple_sentence(self) -> None:
        """Test tokenization of simple English sentence."""
        tokenizer = EnglishTokenizer()
        tokens = tokenizer.tokenize("hello world")
        assert tokens == ["hello", "world"]

    def test_tokenize_with_punctuation(self) -> None:
        """Test tokenization handles punctuation."""
        tokenizer = EnglishTokenizer()
        tokens = tokenizer.tokenize("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_get_base_forms(self) -> None:
        """Test get_base_forms returns same as tokenize."""
        tokenizer = EnglishTokenizer()
        tokens = tokenizer.get_base_forms("hello world")
        assert tokens == ["hello", "world"]


class TestJapaneseTokenizer:
    """Tests for JapaneseTokenizer class."""

    def test_init_lazy_loading(self) -> None:
        """Test Janome is not loaded until first use."""
        tokenizer = JapaneseTokenizer()
        assert tokenizer._tokenizer is None

    def test_tokenize_raises_import_error_when_janome_not_installed(self) -> None:
        """Test ImportError is raised when Janome is not installed."""
        tokenizer = JapaneseTokenizer()
        with patch.dict("sys.modules", {"janome": None, "janome.tokenizer": None}):
            # Force re-import by clearing cached tokenizer
            tokenizer._tokenizer = None
            with patch(
                "barscan.analyzer.tokenizer.JapaneseTokenizer._get_tokenizer",
                side_effect=ImportError("Janome is required"),
            ):
                with pytest.raises(ImportError, match="Janome is required"):
                    tokenizer.tokenize("テスト")

    def test_tokenize_with_mock_janome(self) -> None:
        """Test tokenization with mocked Janome."""
        tokenizer = JapaneseTokenizer()

        # Create mock token objects
        mock_token1 = MagicMock()
        mock_token1.surface = "こんにちは"
        mock_token1.base_form = "こんにちは"

        mock_token2 = MagicMock()
        mock_token2.surface = "世界"
        mock_token2.base_form = "世界"

        # Create mock Janome tokenizer
        mock_janome_tokenizer = MagicMock()
        mock_janome_tokenizer.tokenize.return_value = [mock_token1, mock_token2]

        tokenizer._tokenizer = mock_janome_tokenizer

        tokens = tokenizer.tokenize("こんにちは世界")
        assert tokens == ["こんにちは", "世界"]

    def test_get_base_forms_with_mock_janome(self) -> None:
        """Test get_base_forms returns base forms from Janome."""
        tokenizer = JapaneseTokenizer()

        # Create mock token with different surface and base forms
        mock_token = MagicMock()
        mock_token.surface = "食べた"
        mock_token.base_form = "食べる"

        mock_janome_tokenizer = MagicMock()
        mock_janome_tokenizer.tokenize.return_value = [mock_token]

        tokenizer._tokenizer = mock_janome_tokenizer

        tokens = tokenizer.get_base_forms("食べた")
        assert tokens == ["食べる"]

    def test_get_base_forms_uses_surface_when_base_is_asterisk(self) -> None:
        """Test surface form is used when base_form is '*'."""
        tokenizer = JapaneseTokenizer()

        mock_token = MagicMock()
        mock_token.surface = "テスト"
        mock_token.base_form = "*"

        mock_janome_tokenizer = MagicMock()
        mock_janome_tokenizer.tokenize.return_value = [mock_token]

        tokenizer._tokenizer = mock_janome_tokenizer

        tokens = tokenizer.get_base_forms("テスト")
        assert tokens == ["テスト"]

    def test_tokenize_returns_base_forms(self) -> None:
        """Test tokenize() returns base forms, not surface forms."""
        tokenizer = JapaneseTokenizer()

        # Mock token with different surface and base forms
        mock_token = MagicMock()
        mock_token.surface = "食べた"  # Surface form (past tense)
        mock_token.base_form = "食べる"  # Base form (dictionary form)

        mock_janome_tokenizer = MagicMock()
        mock_janome_tokenizer.tokenize.return_value = [mock_token]

        tokenizer._tokenizer = mock_janome_tokenizer

        tokens = tokenizer.tokenize("食べた")
        assert tokens == ["食べる"]

    def test_tokenize_with_pos_filter_keeps_content_words(self) -> None:
        """Test POS filtering keeps nouns, verbs, adjectives."""
        tokenizer = JapaneseTokenizer()

        # Create mock tokens for different parts of speech
        mock_noun = MagicMock()
        mock_noun.surface = "猫"
        mock_noun.base_form = "猫"
        mock_noun.part_of_speech = "名詞,一般,*,*"

        mock_verb = MagicMock()
        mock_verb.surface = "走る"
        mock_verb.base_form = "走る"
        mock_verb.part_of_speech = "動詞,自立,*,*"

        mock_adjective = MagicMock()
        mock_adjective.surface = "高い"
        mock_adjective.base_form = "高い"
        mock_adjective.part_of_speech = "形容詞,自立,*,*"

        mock_janome_tokenizer = MagicMock()
        mock_janome_tokenizer.tokenize.return_value = [mock_noun, mock_verb, mock_adjective]

        tokenizer._tokenizer = mock_janome_tokenizer

        tokens = tokenizer.tokenize_with_pos_filter("猫が走る高い")
        assert "猫" in tokens
        assert "走る" in tokens
        assert "高い" in tokens

    def test_tokenize_with_pos_filter_removes_particles(self) -> None:
        """Test POS filtering removes particles (助詞)."""
        tokenizer = JapaneseTokenizer()

        # Create mock tokens: noun + particle
        mock_noun = MagicMock()
        mock_noun.surface = "猫"
        mock_noun.base_form = "猫"
        mock_noun.part_of_speech = "名詞,一般,*,*"

        mock_particle = MagicMock()
        mock_particle.surface = "が"
        mock_particle.base_form = "が"
        mock_particle.part_of_speech = "助詞,格助詞,一般,*"

        mock_janome_tokenizer = MagicMock()
        mock_janome_tokenizer.tokenize.return_value = [mock_noun, mock_particle]

        tokenizer._tokenizer = mock_janome_tokenizer

        tokens = tokenizer.tokenize_with_pos_filter("猫が")
        assert tokens == ["猫"]
        assert "が" not in tokens

    def test_tokenize_with_pos_filter_removes_auxiliaries(self) -> None:
        """Test POS filtering removes auxiliary verbs (助動詞)."""
        tokenizer = JapaneseTokenizer()

        # Create mock tokens: verb stem + auxiliary (たい)
        mock_verb = MagicMock()
        mock_verb.surface = "食べ"
        mock_verb.base_form = "食べる"
        mock_verb.part_of_speech = "動詞,自立,*,*"

        mock_auxiliary = MagicMock()
        mock_auxiliary.surface = "たい"
        mock_auxiliary.base_form = "たい"
        mock_auxiliary.part_of_speech = "助動詞,*,*,*"

        mock_janome_tokenizer = MagicMock()
        mock_janome_tokenizer.tokenize.return_value = [mock_verb, mock_auxiliary]

        tokenizer._tokenizer = mock_janome_tokenizer

        tokens = tokenizer.tokenize_with_pos_filter("食べたい")
        assert tokens == ["食べる"]
        assert "たい" not in tokens

    def test_tokenize_with_pos_filter_uses_base_forms(self) -> None:
        """Test POS filtering uses base forms, not surface forms."""
        tokenizer = JapaneseTokenizer()

        mock_verb = MagicMock()
        mock_verb.surface = "走った"  # Surface form (past tense)
        mock_verb.base_form = "走る"  # Base form
        mock_verb.part_of_speech = "動詞,自立,*,*"

        mock_janome_tokenizer = MagicMock()
        mock_janome_tokenizer.tokenize.return_value = [mock_verb]

        tokenizer._tokenizer = mock_janome_tokenizer

        tokens = tokenizer.tokenize_with_pos_filter("走った")
        assert tokens == ["走る"]

    def test_tokenize_with_pos_filter_removes_non_independent_verbs(self) -> None:
        """Test POS filtering removes non-independent verbs (非自立) like てる."""
        tokenizer = JapaneseTokenizer()

        # Create mock tokens: verb + non-independent verb (てる)
        mock_verb = MagicMock()
        mock_verb.surface = "見"
        mock_verb.base_form = "見る"
        mock_verb.part_of_speech = "動詞,自立,*,*"

        mock_teru = MagicMock()
        mock_teru.surface = "てる"
        mock_teru.base_form = "てる"
        mock_teru.part_of_speech = "動詞,非自立,*,*"  # Non-independent verb

        mock_janome_tokenizer = MagicMock()
        mock_janome_tokenizer.tokenize.return_value = [mock_verb, mock_teru]

        tokenizer._tokenizer = mock_janome_tokenizer

        tokens = tokenizer.tokenize_with_pos_filter("見てる")
        assert tokens == ["見る"]
        assert "てる" not in tokens

    def test_tokenize_with_pos_filter_removes_suffixes(self) -> None:
        """Test POS filtering removes suffixes (接尾) like さん."""
        tokenizer = JapaneseTokenizer()

        # Create mock tokens: noun + suffix
        mock_noun = MagicMock()
        mock_noun.surface = "田中"
        mock_noun.base_form = "田中"
        mock_noun.part_of_speech = "名詞,固有名詞,人名,姓"

        mock_suffix = MagicMock()
        mock_suffix.surface = "さん"
        mock_suffix.base_form = "さん"
        mock_suffix.part_of_speech = "名詞,接尾,人名,*"

        mock_janome_tokenizer = MagicMock()
        mock_janome_tokenizer.tokenize.return_value = [mock_noun, mock_suffix]

        tokenizer._tokenizer = mock_janome_tokenizer

        tokens = tokenizer.tokenize_with_pos_filter("田中さん")
        assert tokens == ["田中"]
        assert "さん" not in tokens


class TestGetTokenizer:
    """Tests for get_tokenizer factory function."""

    def test_get_english_tokenizer(self) -> None:
        """Test getting English tokenizer."""
        tokenizer = get_tokenizer("english")
        assert isinstance(tokenizer, EnglishTokenizer)

    def test_get_japanese_tokenizer(self) -> None:
        """Test getting Japanese tokenizer."""
        tokenizer = get_tokenizer("japanese")
        assert isinstance(tokenizer, JapaneseTokenizer)

    def test_auto_detect_english(self) -> None:
        """Test auto-detection returns English tokenizer for English text."""
        tokenizer = get_tokenizer("auto", "hello world")
        assert isinstance(tokenizer, EnglishTokenizer)

    def test_auto_detect_japanese(self) -> None:
        """Test auto-detection returns Japanese tokenizer for Japanese text."""
        tokenizer = get_tokenizer("auto", "こんにちは")
        assert isinstance(tokenizer, JapaneseTokenizer)

    def test_auto_without_text_returns_english(self) -> None:
        """Test auto without text returns English tokenizer."""
        tokenizer = get_tokenizer("auto")
        assert isinstance(tokenizer, EnglishTokenizer)
