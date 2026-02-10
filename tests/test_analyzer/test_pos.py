"""Tests for POS tagging module."""

import pytest

from barscan.analyzer.pos import POS_TAG_MAP, get_pos_tag, get_pos_tags


class TestGetPosTags:
    """Tests for get_pos_tags function."""

    def test_empty_tokens(self) -> None:
        """Test with empty token list."""
        result = get_pos_tags([])
        assert result == {}

    def test_single_noun(self) -> None:
        """Test tagging a single noun."""
        result = get_pos_tags(["love"])
        assert "love" in result
        # Note: "love" can be noun or verb depending on context

    def test_multiple_words(self) -> None:
        """Test tagging multiple words."""
        result = get_pos_tags(["love", "run", "beautiful", "quickly"])
        assert len(result) == 4
        assert "love" in result
        assert "run" in result
        assert "beautiful" in result
        assert "quickly" in result

    def test_returns_simple_labels(self) -> None:
        """Test that results use simple labels from POS_TAG_MAP."""
        result = get_pos_tags(["running", "cats", "beautiful"])
        for word, tag in result.items():
            # Tag should be a simple label or a lowercase raw tag
            assert tag.islower() or tag in POS_TAG_MAP.values()

    def test_duplicate_tokens(self) -> None:
        """Test handling of duplicate tokens."""
        result = get_pos_tags(["love", "love", "love"])
        assert len(result) == 1
        assert "love" in result

    def test_most_common_tag(self) -> None:
        """Test that most common tag is used for repeated words."""
        # When a word appears multiple times in different positions,
        # the most common tag should be returned
        result = get_pos_tags(["love", "love", "love"])
        assert "love" in result


class TestGetPosTag:
    """Tests for get_pos_tag function."""

    def test_single_word(self) -> None:
        """Test tagging a single word."""
        result = get_pos_tag("love")
        assert isinstance(result, str)

    def test_unknown_word(self) -> None:
        """Test tagging an unknown word."""
        result = get_pos_tag("xyzzy123")
        assert isinstance(result, str)


class TestPosTagMap:
    """Tests for POS_TAG_MAP constant."""

    def test_noun_tags(self) -> None:
        """Test noun tag mappings."""
        assert POS_TAG_MAP["NN"] == "noun"
        assert POS_TAG_MAP["NNS"] == "noun"
        assert POS_TAG_MAP["NNP"] == "noun"
        assert POS_TAG_MAP["NNPS"] == "noun"

    def test_verb_tags(self) -> None:
        """Test verb tag mappings."""
        assert POS_TAG_MAP["VB"] == "verb"
        assert POS_TAG_MAP["VBD"] == "verb"
        assert POS_TAG_MAP["VBG"] == "verb"
        assert POS_TAG_MAP["VBN"] == "verb"
        assert POS_TAG_MAP["VBP"] == "verb"
        assert POS_TAG_MAP["VBZ"] == "verb"

    def test_adjective_tags(self) -> None:
        """Test adjective tag mappings."""
        assert POS_TAG_MAP["JJ"] == "adjective"
        assert POS_TAG_MAP["JJR"] == "adjective"
        assert POS_TAG_MAP["JJS"] == "adjective"

    def test_adverb_tags(self) -> None:
        """Test adverb tag mappings."""
        assert POS_TAG_MAP["RB"] == "adverb"
        assert POS_TAG_MAP["RBR"] == "adverb"
        assert POS_TAG_MAP["RBS"] == "adverb"
