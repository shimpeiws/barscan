"""Context extraction for word occurrences in lyrics."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from barscan.analyzer.models import ContextsMode, TokenWithPosition, WordContext

if TYPE_CHECKING:
    pass


def extract_short_context(
    line: str,
    word: str,
    window_size: int = 2,
) -> str:
    """Extract a short context window around a word in a line.

    Creates a snippet like "...my love for you..." with the target word
    surrounded by window_size words on each side.

    Args:
        line: The full line of lyrics.
        word: The target word to find context for.
        window_size: Number of words to include on each side.

    Returns:
        Context snippet with ellipsis markers.
    """
    # Split line into words (preserving punctuation attached to words)
    words = line.split()
    word_lower = word.lower()

    # Find the word position (case-insensitive)
    word_index = -1
    for i, w in enumerate(words):
        # Strip punctuation for comparison
        clean_w = re.sub(r"[^\w']", "", w.lower())
        if clean_w == word_lower:
            word_index = i
            break

    if word_index == -1:
        # Word not found, return truncated line
        if len(words) <= window_size * 2 + 1:
            return line.strip()
        return "..." + " ".join(words[: window_size * 2 + 1]) + "..."

    # Extract window
    start = max(0, word_index - window_size)
    end = min(len(words), word_index + window_size + 1)
    context_words = words[start:end]

    # Add ellipsis markers
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(words) else ""

    return f"{prefix}{' '.join(context_words)}{suffix}"


def extract_full_context(
    line: str,
    song_title: str,
    album: str | None = None,
    year: int | None = None,
) -> WordContext:
    """Extract full context with metadata.

    Args:
        line: The full line of lyrics.
        song_title: The song title.
        album: Optional album name.
        year: Optional release year.

    Returns:
        WordContext object with full metadata.
    """
    return WordContext(
        line=line.strip(),
        track=song_title,
        album=album,
        year=year,
    )


def extract_contexts_for_word(
    tokens_with_positions: list[TokenWithPosition],
    word: str,
    mode: ContextsMode,
    max_contexts: int = 3,
    window_size: int = 2,
) -> tuple[str, ...] | tuple[WordContext, ...] | None:
    """Extract contexts for a specific word from tokens with positions.

    Args:
        tokens_with_positions: List of tokens with their position information.
        word: The target word to find contexts for.
        mode: Context extraction mode (none/short/full).
        max_contexts: Maximum number of contexts to extract.
        window_size: Window size for short context mode.

    Returns:
        Tuple of context strings (short mode) or WordContext objects (full mode),
        or None if mode is 'none'.
    """
    if mode == ContextsMode.NONE:
        return None

    word_lower = word.lower()

    # Find all occurrences of the word
    occurrences: list[TokenWithPosition] = [
        token for token in tokens_with_positions if token.token.lower() == word_lower
    ]

    if not occurrences:
        return None

    # Deduplicate by (song_id, line_index) to avoid duplicate contexts from same line
    seen_lines: set[tuple[int, int]] = set()
    unique_occurrences: list[TokenWithPosition] = []
    for occ in occurrences:
        key = (occ.song_id, occ.line_index)
        if key not in seen_lines:
            seen_lines.add(key)
            unique_occurrences.append(occ)

    # Limit to max_contexts
    selected = unique_occurrences[:max_contexts]

    if mode == ContextsMode.SHORT:
        contexts: list[str] = []
        for occ in selected:
            context = extract_short_context(occ.original_line, word, window_size)
            contexts.append(context)
        return tuple(contexts)

    elif mode == ContextsMode.FULL:
        full_contexts: list[WordContext] = []
        for occ in selected:
            full_ctx = extract_full_context(
                line=occ.original_line,
                song_title=occ.song_title,
                album=None,  # Not available in current model
                year=None,  # Not available in current model
            )
            full_contexts.append(full_ctx)
        return tuple(full_contexts)

    return None


def group_tokens_by_word(
    tokens_with_positions: list[TokenWithPosition],
) -> dict[str, list[TokenWithPosition]]:
    """Group tokens by their normalized word.

    Args:
        tokens_with_positions: List of tokens with position information.

    Returns:
        Dictionary mapping lowercase words to their token occurrences.
    """
    groups: dict[str, list[TokenWithPosition]] = {}
    for token in tokens_with_positions:
        word_lower = token.token.lower()
        if word_lower not in groups:
            groups[word_lower] = []
        groups[word_lower].append(token)
    return groups
