"""Tests for LyricsCache."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from barscan.genius.cache import LyricsCache
from barscan.genius.models import Lyrics


class TestLyricsCache:
    def test_store_and_retrieve(self, temp_cache_dir: Path):
        cache = LyricsCache(cache_dir=temp_cache_dir, ttl_hours=24)

        lyrics = Lyrics(
            song_id=123,
            song_title="Test Song",
            artist_name="Test Artist",
            lyrics_text="These are test lyrics",
        )

        cache.store_lyrics(lyrics)
        retrieved = cache.get_lyrics(123)

        assert retrieved is not None
        assert retrieved.song_id == 123
        assert retrieved.lyrics_text == "These are test lyrics"

    def test_returns_none_for_missing(self, temp_cache_dir: Path):
        cache = LyricsCache(cache_dir=temp_cache_dir)
        assert cache.get_lyrics(999) is None

    def test_ttl_expiration(self, temp_cache_dir: Path):
        cache = LyricsCache(cache_dir=temp_cache_dir, ttl_hours=1)

        lyrics = Lyrics(
            song_id=123,
            song_title="Test",
            artist_name="Test",
            lyrics_text="Test",
            fetched_at=datetime.now(UTC) - timedelta(hours=2),
        )

        cache.store_lyrics(lyrics)

        # Should return None because TTL expired
        assert cache.get_lyrics(123) is None

    def test_clear_all(self, temp_cache_dir: Path):
        cache = LyricsCache(cache_dir=temp_cache_dir)

        for i in range(5):
            cache.store_lyrics(
                Lyrics(
                    song_id=i,
                    song_title=f"Song {i}",
                    artist_name="Artist",
                    lyrics_text="Text",
                )
            )

        removed = cache.clear()
        assert removed == 5

        for i in range(5):
            assert cache.get_lyrics(i) is None

    def test_clear_expired(self, temp_cache_dir: Path):
        cache = LyricsCache(cache_dir=temp_cache_dir, ttl_hours=1)

        # Add expired lyrics
        cache.store_lyrics(
            Lyrics(
                song_id=1,
                song_title="Expired",
                artist_name="Artist",
                lyrics_text="Text",
                fetched_at=datetime.now(UTC) - timedelta(hours=2),
            )
        )

        # Add fresh lyrics
        cache.store_lyrics(
            Lyrics(
                song_id=2,
                song_title="Fresh",
                artist_name="Artist",
                lyrics_text="Text",
            )
        )

        removed = cache.clear_expired()
        assert removed == 1

        # Expired should be gone
        assert cache.get_lyrics(1) is None
        # Fresh should still exist
        assert cache.get_lyrics(2) is not None

    def test_cache_stats(self, temp_cache_dir: Path):
        cache = LyricsCache(cache_dir=temp_cache_dir)

        cache.store_lyrics(
            Lyrics(
                song_id=1,
                song_title="Song",
                artist_name="Artist",
                lyrics_text="Some lyrics text here",
            )
        )

        stats = cache.get_stats()
        assert stats["total_entries"] == 1
        assert stats["size_bytes"] > 0
        assert stats["expired"] == 0

    def test_cache_stats_with_expired(self, temp_cache_dir: Path):
        cache = LyricsCache(cache_dir=temp_cache_dir, ttl_hours=1)

        cache.store_lyrics(
            Lyrics(
                song_id=1,
                song_title="Expired",
                artist_name="Artist",
                lyrics_text="Text",
                fetched_at=datetime.now(UTC) - timedelta(hours=2),
            )
        )

        stats = cache.get_stats()
        assert stats["total_entries"] == 1
        assert stats["expired"] == 1

    def test_unicode_lyrics(self, temp_cache_dir: Path):
        cache = LyricsCache(cache_dir=temp_cache_dir)

        lyrics = Lyrics(
            song_id=123,
            song_title="日本語の歌",
            artist_name="アーティスト",
            lyrics_text="これは日本語の歌詞です\n한국어도 있어요\nEmoji: 🎵🎶",
        )

        cache.store_lyrics(lyrics)
        retrieved = cache.get_lyrics(123)

        assert retrieved is not None
        assert retrieved.song_title == "日本語の歌"
        assert "日本語" in retrieved.lyrics_text
        assert "🎵" in retrieved.lyrics_text

    def test_handles_corrupted_cache_file(self, temp_cache_dir: Path):
        cache = LyricsCache(cache_dir=temp_cache_dir)

        # Store valid lyrics
        cache.store_lyrics(
            Lyrics(
                song_id=123,
                song_title="Test",
                artist_name="Test",
                lyrics_text="Test",
            )
        )

        # Corrupt the cache file
        cache_path = cache._get_cache_path(123)
        cache_path.write_text("not valid json", encoding="utf-8")

        # Should return None and not raise
        result = cache.get_lyrics(123)
        assert result is None

        # Corrupted file should be removed
        assert not cache_path.exists()

    def test_empty_cache_stats(self, temp_cache_dir: Path):
        cache = LyricsCache(cache_dir=temp_cache_dir)
        stats = cache.get_stats()

        assert stats["total_entries"] == 0
        assert stats["size_bytes"] == 0
        assert stats["expired"] == 0


class TestCacheVersioning:
    def test_different_versions_use_different_dirs(self, temp_cache_dir: Path):
        cache1 = LyricsCache(cache_dir=temp_cache_dir, ttl_hours=24)

        cache1.store_lyrics(
            Lyrics(
                song_id=123,
                song_title="Test",
                artist_name="Test",
                lyrics_text="Test",
            )
        )

        # Verify the version is in the path
        assert "v1" in str(cache1.cache_dir)
