"""Slang word detection for lyrics analysis."""

from __future__ import annotations

# Curated slang dictionary for music/hip-hop lyrics
# This is a representative set - can be extended via configuration
SLANG_WORDS: frozenset[str] = frozenset(
    {
        # Common contractions/informal
        "ain't",
        "gonna",
        "wanna",
        "gotta",
        "kinda",
        "sorta",
        "lemme",
        "gimme",
        "dunno",
        "tryna",
        "finna",
        "boutta",
        "shoulda",
        "coulda",
        "woulda",
        "ima",
        "imma",
        "aint",
        # Pronouns/addressing
        "ya",
        "yo",
        "yall",
        "y'all",
        "em",
        "'em",
        "da",
        "tha",
        "dat",
        "dis",
        "dem",
        "dey",
        "wit",
        "wid",
        # People
        "bruh",
        "bro",
        "homie",
        "homies",
        "dawg",
        "fam",
        "cuz",
        "playa",
        "pimp",
        "shorty",
        "shawty",
        "mama",
        "mami",
        "papi",
        "bae",
        "boo",
        # Actions/states
        "chillin",
        "trippin",
        "flexin",
        "stuntin",
        "ballin",
        "poppin",
        "rockin",
        "vibin",
        "slidin",
        "drippin",
        "sippin",
        "hittin",
        "whippin",
        # Adjectives/interjections
        "lit",
        "dope",
        "sick",
        "fire",
        "tight",
        "wack",
        "whack",
        "sus",
        "bougie",
        "boujee",
        "fly",
        "fresh",
        "cold",
        "icy",
        "wavy",
        "lowkey",
        "highkey",
        "deadass",
        "hype",
        "hyped",
        # Money/success
        "bands",
        "racks",
        "stacks",
        "guap",
        "bread",
        "cheddar",
        "dough",
        "paper",
        "moolah",
        "cake",
        "gwap",
        "benjis",
        "bucks",
        "hunnids",
        # Lifestyle
        "flex",
        "drip",
        "swag",
        "clout",
        "hustle",
        "grind",
        "trap",
        "hood",
        "block",
        "turf",
        "whip",
        "ride",
        "crib",
        "pad",
        # Expressions
        "bet",
        "cap",
        "nocap",
        "facts",
        "word",
        "aight",
        "ight",
        "iight",
        "yeet",
        "slay",
        "bop",
        "slap",
        "slaps",
        "bussin",
        "goat",
        "goated",
        "goats",
        "fye",
        # Negatives/criticism
        "hater",
        "haters",
        "opps",
        "ops",
        "snitch",
        "fake",
        "lame",
        "corny",
        "bogus",
        # Misc slang
        "vibe",
        "vibes",
        "mood",
        "wave",
        "sauce",
        "drako",
        "choppa",
        "glizzy",
        "thang",
        "thangs",
        "nah",
        "yuh",
        "yeah",
        "ayy",
        "aye",
        "huh",
        "uh",
        "uhh",
        "mmm",
        "ooh",
        "woah",
        "skrrt",
        "skrt",
        "brr",
        "grr",
    }
)


def is_slang(word: str, additional_slang: frozenset[str] | None = None) -> bool:
    """Check if a word is slang.

    Args:
        word: Word to check.
        additional_slang: Optional additional slang words to include.

    Returns:
        True if the word is in the slang dictionary.
    """
    word_lower = word.lower()
    slang_set = SLANG_WORDS | (additional_slang or frozenset())
    return word_lower in slang_set


def detect_slang_words(
    words: list[str],
    additional_slang: frozenset[str] | None = None,
) -> dict[str, bool]:
    """Detect slang words in a list.

    Args:
        words: List of words to check.
        additional_slang: Optional additional slang words to include.

    Returns:
        Dictionary mapping words to their slang status.
    """
    if not words:
        return {}

    slang_set = SLANG_WORDS | (additional_slang or frozenset())
    result: dict[str, bool] = {}

    for word in set(words):  # Deduplicate
        word_lower = word.lower()
        if word_lower not in result:
            result[word_lower] = word_lower in slang_set

    return result


def get_slang_count(
    words: list[str],
    additional_slang: frozenset[str] | None = None,
) -> int:
    """Count slang words in a list.

    Args:
        words: List of words to check.
        additional_slang: Optional additional slang words to include.

    Returns:
        Number of slang word occurrences.
    """
    slang_set = SLANG_WORDS | (additional_slang or frozenset())
    return sum(1 for word in words if word.lower() in slang_set)
