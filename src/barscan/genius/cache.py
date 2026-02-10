"""Local file-based cache for lyrics data."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from barscan.logging import get_logger

if TYPE_CHECKING:
    from .models import Lyrics

logger = get_logger("genius.cache")


def _ensure_timezone_aware(dt_str: str) -> datetime:
    """Parse ISO datetime string and ensure it's timezone-aware.

    Args:
        dt_str: ISO format datetime string.

    Returns:
        Timezone-aware datetime (UTC if no timezone info in string).
    """
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


class LyricsCache:
    """File-based cache for lyrics with TTL support."""

    CACHE_VERSION = "1"

    def __init__(
        self,
        cache_dir: Path,
        ttl_hours: int = 168,
    ) -> None:
        """
        Initialize lyrics cache.

        Args:
            cache_dir: Directory to store cache files.
            ttl_hours: Time-to-live in hours for cached entries.
        """
        self.cache_dir = cache_dir / "lyrics" / f"v{self.CACHE_VERSION}"
        self.ttl = timedelta(hours=ttl_hours)
        self._ensure_cache_dir()

    def get_lyrics(self, song_id: int) -> Lyrics | None:
        """
        Retrieve lyrics from cache.

        Args:
            song_id: Genius song ID.

        Returns:
            Lyrics model if cached and valid, None otherwise.
        """
        cache_file = self._get_cache_path(song_id)

        if not cache_file.exists():
            logger.debug("Cache miss for song_id=%d (file not found)", song_id)
            return None

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))

            fetched_at = _ensure_timezone_aware(data["fetched_at"])
            if datetime.now(UTC) - fetched_at > self.ttl:
                logger.debug("Cache miss for song_id=%d (expired)", song_id)
                cache_file.unlink()
                return None

            from .models import Lyrics

            logger.debug("Cache hit for song_id=%d", song_id)
            return Lyrics(
                song_id=data["song_id"],
                song_title=data["song_title"],
                artist_name=data["artist_name"],
                lyrics_text=data["lyrics_text"],
                fetched_at=fetched_at,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Invalid cache entry for song_id=%d: %s", song_id, e)
            cache_file.unlink(missing_ok=True)
            return None

    def store_lyrics(self, lyrics: Lyrics) -> None:
        """
        Store lyrics in cache.

        Args:
            lyrics: Lyrics model to cache.
        """
        cache_file = self._get_cache_path(lyrics.song_id)

        data = {
            "song_id": lyrics.song_id,
            "song_title": lyrics.song_title,
            "artist_name": lyrics.artist_name,
            "lyrics_text": lyrics.lyrics_text,
            "fetched_at": lyrics.fetched_at.isoformat(),
        }

        cache_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("Cached lyrics for song_id=%d", lyrics.song_id)

    def clear(self) -> int:
        """
        Clear all cached lyrics.

        Returns:
            Number of cache entries removed.
        """
        count = 0
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.rglob("*.json"):
                cache_file.unlink()
                count += 1
        logger.info("Cleared %d cache entries", count)
        return count

    def clear_expired(self) -> int:
        """
        Remove only expired cache entries.

        Returns:
            Number of expired entries removed.
        """
        count = 0
        if not self.cache_dir.exists():
            return count

        now = datetime.now(UTC)
        for cache_file in self.cache_dir.rglob("*.json"):
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                fetched_at = _ensure_timezone_aware(data["fetched_at"])
                if now - fetched_at > self.ttl:
                    cache_file.unlink()
                    count += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                cache_file.unlink()
                count += 1

        logger.info("Cleared %d expired cache entries", count)
        return count

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        if not self.cache_dir.exists():
            return {"total_entries": 0, "size_bytes": 0, "expired": 0}

        total = 0
        size = 0
        expired = 0
        now = datetime.now(UTC)

        for cache_file in self.cache_dir.rglob("*.json"):
            total += 1
            size += cache_file.stat().st_size
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                fetched_at = _ensure_timezone_aware(data["fetched_at"])
                if now - fetched_at > self.ttl:
                    expired += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                expired += 1

        return {
            "total_entries": total,
            "size_bytes": size,
            "expired": expired,
        }

    def _get_cache_path(self, song_id: int) -> Path:
        """Generate cache file path for a song ID."""
        hash_prefix = hashlib.md5(str(song_id).encode()).hexdigest()[:2]
        subdir = self.cache_dir / hash_prefix
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{song_id}.json"

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
