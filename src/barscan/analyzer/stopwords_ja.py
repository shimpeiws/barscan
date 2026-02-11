"""Japanese stop words for lyrics analysis."""

from __future__ import annotations


def get_japanese_stop_words() -> frozenset[str]:
    """Get the set of Japanese stop words using stopwordsiso.

    Returns:
        A frozenset of Japanese stop words.

    Raises:
        ImportError: If stopwordsiso is not installed.
    """
    try:
        from stopwordsiso import stopwords

        return frozenset(stopwords("ja"))
    except ImportError as e:
        raise ImportError(
            "stopwordsiso is required for Japanese stop words. "
            "Install it with: pip install barscan[japanese]"
        ) from e
