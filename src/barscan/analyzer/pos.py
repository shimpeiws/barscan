"""Part-of-speech tagging for lyrics analysis."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import nltk

from barscan.analyzer.nltk_resources import POS_RESOURCES, ensure_resources
from barscan.exceptions import NLTKResourceError

if TYPE_CHECKING:
    pass

# Penn Treebank POS tag to simple label mapping
POS_TAG_MAP: dict[str, str] = {
    # Nouns
    "NN": "noun",
    "NNS": "noun",
    "NNP": "noun",
    "NNPS": "noun",
    # Verbs
    "VB": "verb",
    "VBD": "verb",
    "VBG": "verb",
    "VBN": "verb",
    "VBP": "verb",
    "VBZ": "verb",
    # Adjectives
    "JJ": "adjective",
    "JJR": "adjective",
    "JJS": "adjective",
    # Adverbs
    "RB": "adverb",
    "RBR": "adverb",
    "RBS": "adverb",
    # Pronouns
    "PRP": "pronoun",
    "PRP$": "pronoun",
    "WP": "pronoun",
    "WP$": "pronoun",
    # Prepositions
    "IN": "preposition",
    "TO": "preposition",
    # Conjunctions
    "CC": "conjunction",
    # Determiners
    "DT": "determiner",
    "PDT": "determiner",
    "WDT": "determiner",
    # Interjections
    "UH": "interjection",
    # Modals
    "MD": "modal",
    # Other
    "CD": "number",
    "EX": "existential",
    "FW": "foreign",
    "LS": "list",
    "POS": "possessive",
    "RP": "particle",
    "SYM": "symbol",
    "WRB": "wh-adverb",
}


def ensure_pos_resources() -> None:
    """Ensure required NLTK resources for POS tagging are downloaded.

    Raises:
        NLTKResourceError: If resources cannot be downloaded.
    """
    ensure_resources(POS_RESOURCES)


def get_pos_tags(tokens: list[str]) -> dict[str, str]:
    """Get POS tags for a list of tokens.

    Returns the most common POS tag for each unique word. Tags are mapped to
    simple labels (noun, verb, adjective, etc.) for readability.

    Args:
        tokens: List of word tokens.

    Returns:
        Dictionary mapping words to their POS tags (simple labels).

    Raises:
        NLTKResourceError: If POS tagger is not available.
    """
    if not tokens:
        return {}

    ensure_pos_resources()

    try:
        # Get POS tags for all tokens
        tagged = nltk.pos_tag(tokens)
    except LookupError as e:
        raise NLTKResourceError(f"NLTK POS tagging failed: {e}") from e

    # Aggregate: find most common tag for each unique word
    word_tags: dict[str, Counter[str]] = {}
    for word, tag in tagged:
        word_lower = word.lower()
        if word_lower not in word_tags:
            word_tags[word_lower] = Counter()
        word_tags[word_lower][tag] += 1

    # Get most common tag for each word and map to simple label
    result: dict[str, str] = {}
    for word, tag_counts in word_tags.items():
        most_common_tag = tag_counts.most_common(1)[0][0]
        # Map to simple label, default to the raw tag if not in map
        result[word] = POS_TAG_MAP.get(most_common_tag, most_common_tag.lower())

    return result


def get_pos_tag(word: str) -> str:
    """Get POS tag for a single word.

    Args:
        word: The word to tag.

    Returns:
        Simple POS label (noun, verb, etc.).

    Raises:
        NLTKResourceError: If POS tagger is not available.
    """
    tags = get_pos_tags([word])
    return tags.get(word.lower(), "unknown")
