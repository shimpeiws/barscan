"""Local file-based cache for lyrics data."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Lyrics


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
            return None

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))

            fetched_at = datetime.fromisoformat(data["fetched_at"])
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=UTC)
            if datetime.now(UTC) - fetched_at > self.ttl:
                cache_file.unlink()
                return None

            from .models import Lyrics

            return Lyrics(
                song_id=data["song_id"],
                song_title=data["song_title"],
                artist_name=data["artist_name"],
                lyrics_text=data["lyrics_text"],
                fetched_at=fetched_at,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
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
                fetched_at = datetime.fromisoformat(data["fetched_at"])
                if fetched_at.tzinfo is None:
                    fetched_at = fetched_at.replace(tzinfo=UTC)
                if now - fetched_at > self.ttl:
                    cache_file.unlink()
                    count += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                cache_file.unlink()
                count += 1

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
                fetched_at = datetime.fromisoformat(data["fetched_at"])
                if fetched_at.tzinfo is None:
                    fetched_at = fetched_at.replace(tzinfo=UTC)
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
