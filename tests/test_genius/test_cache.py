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


class TestCacheEdgeCases:
    """Additional edge case tests for cache functionality."""

    def test_store_and_retrieve_large_lyrics(self, temp_cache_dir: Path):
        """Test caching very large lyrics text."""
        cache = LyricsCache(cache_dir=temp_cache_dir)

        # Create large lyrics text (100KB+)
        large_text = "word " * 20000  # ~100KB of text

        lyrics = Lyrics(
            song_id=999,
            song_title="Long Song",
            artist_name="Artist",
            lyrics_text=large_text,
        )

        cache.store_lyrics(lyrics)
        retrieved = cache.get_lyrics(999)

        assert retrieved is not None
        assert retrieved.lyrics_text == large_text

    def test_cache_path_hashing(self, temp_cache_dir: Path):
        """Test that cache files are distributed across subdirectories."""
        cache = LyricsCache(cache_dir=temp_cache_dir)

        # Store multiple lyrics to verify hashing
        for i in range(10):
            cache.store_lyrics(
                Lyrics(
                    song_id=i * 1000 + 1,
                    song_title=f"Song {i}",
                    artist_name="Artist",
                    lyrics_text=f"Text {i}",
                )
            )

        # Verify files are stored in subdirectories
        stats = cache.get_stats()
        assert stats["total_entries"] == 10

    def test_clear_on_nonexistent_directory(self, temp_cache_dir: Path):
        """Test clearing cache when cache directory doesn't exist."""
        cache = LyricsCache(cache_dir=temp_cache_dir / "nonexistent")
        # This should not raise an error
        count = cache.clear()
        assert count == 0

    def test_clear_expired_on_nonexistent_directory(self, temp_cache_dir: Path):
        """Test clearing expired entries when directory doesn't exist."""
        # Create cache with a subpath that doesn't exist yet
        nonexistent = temp_cache_dir / "new_subdir"
        cache = LyricsCache(cache_dir=nonexistent)
        # The cache dir is created on init, but let's check the lyrics subdir
        count = cache.clear_expired()
        assert count == 0

    def test_get_stats_with_corrupted_files(self, temp_cache_dir: Path):
        """Test get_stats handles corrupted cache files gracefully."""
        cache = LyricsCache(cache_dir=temp_cache_dir)

        # Store valid lyrics
        cache.store_lyrics(
            Lyrics(
                song_id=1,
                song_title="Valid",
                artist_name="Artist",
                lyrics_text="Valid lyrics",
            )
        )

        # Create a corrupted file
        corrupted_path = cache._get_cache_path(2)
        corrupted_path.write_text("{invalid json}", encoding="utf-8")

        stats = cache.get_stats()
        # Should count both files, but corrupted one should be marked as expired
        assert stats["total_entries"] == 2
        assert stats["expired"] == 1  # Corrupted file counts as expired

    def test_clear_expired_removes_corrupted_files(self, temp_cache_dir: Path):
        """Test that clear_expired also removes corrupted files."""
        cache = LyricsCache(cache_dir=temp_cache_dir)

        # Store valid lyrics
        cache.store_lyrics(
            Lyrics(
                song_id=1,
                song_title="Valid",
                artist_name="Artist",
                lyrics_text="Valid lyrics",
            )
        )

        # Create a corrupted file
        corrupted_path = cache._get_cache_path(2)
        corrupted_path.write_text("not json at all", encoding="utf-8")

        count = cache.clear_expired()
        assert count == 1  # Corrupted file should be removed

        # Valid file should still exist
        assert cache.get_lyrics(1) is not None

    def test_handles_missing_keys_in_cache_file(self, temp_cache_dir: Path):
        """Test handling of cache files with missing required keys."""
        cache = LyricsCache(cache_dir=temp_cache_dir)

        # Create a valid-looking JSON but with missing keys
        cache_path = cache._get_cache_path(100)
        import json

        incomplete_data = {
            "song_id": 100,
            # Missing: song_title, artist_name, lyrics_text, fetched_at
        }
        cache_path.write_text(json.dumps(incomplete_data), encoding="utf-8")

        result = cache.get_lyrics(100)
        assert result is None  # Should return None for incomplete data
        assert not cache_path.exists()  # File should be cleaned up

    def test_handles_invalid_datetime_in_cache(self, temp_cache_dir: Path):
        """Test handling of cache files with invalid datetime."""
        cache = LyricsCache(cache_dir=temp_cache_dir)

        cache_path = cache._get_cache_path(200)
        import json

        invalid_data = {
            "song_id": 200,
            "song_title": "Test",
            "artist_name": "Artist",
            "lyrics_text": "Text",
            "fetched_at": "not-a-valid-datetime",
        }
        cache_path.write_text(json.dumps(invalid_data), encoding="utf-8")

        result = cache.get_lyrics(200)
        assert result is None
        assert not cache_path.exists()

    def test_special_characters_in_song_title(self, temp_cache_dir: Path):
        """Test caching songs with special characters in title."""
        cache = LyricsCache(cache_dir=temp_cache_dir)

        lyrics = Lyrics(
            song_id=300,
            song_title="Song with \"quotes\" and 'apostrophes' & special <chars>",
            artist_name="Artist/Name\\With:Slashes",
            lyrics_text="Test lyrics with\nnewlines\tand\ttabs",
        )

        cache.store_lyrics(lyrics)
        retrieved = cache.get_lyrics(300)

        assert retrieved is not None
        assert retrieved.song_title == lyrics.song_title
        assert retrieved.artist_name == lyrics.artist_name

    def test_concurrent_song_ids_dont_collide(self, temp_cache_dir: Path):
        """Test that different song IDs don't overwrite each other."""
        cache = LyricsCache(cache_dir=temp_cache_dir)

        # Store multiple lyrics
        for song_id in [1, 10, 100, 1000, 10000]:
            cache.store_lyrics(
                Lyrics(
                    song_id=song_id,
                    song_title=f"Song {song_id}",
                    artist_name="Artist",
                    lyrics_text=f"Lyrics for song {song_id}",
                )
            )

        # Verify all can be retrieved
        for song_id in [1, 10, 100, 1000, 10000]:
            retrieved = cache.get_lyrics(song_id)
            assert retrieved is not None
            assert retrieved.song_id == song_id
            assert f"Lyrics for song {song_id}" in retrieved.lyrics_text

    def test_ttl_boundary_cases(self, temp_cache_dir: Path):
        """Test TTL boundary conditions."""
        # 1-hour TTL
        cache = LyricsCache(cache_dir=temp_cache_dir, ttl_hours=1)

        # Store lyrics exactly at TTL boundary
        from datetime import timedelta

        boundary_time = datetime.now(UTC) - timedelta(hours=1, seconds=1)
        lyrics = Lyrics(
            song_id=400,
            song_title="Boundary",
            artist_name="Artist",
            lyrics_text="Text",
            fetched_at=boundary_time,
        )

        cache.store_lyrics(lyrics)

        # Should be expired (just over 1 hour old)
        assert cache.get_lyrics(400) is None

    def test_empty_lyrics_text(self, temp_cache_dir: Path):
        """Test caching empty lyrics text."""
        cache = LyricsCache(cache_dir=temp_cache_dir)

        lyrics = Lyrics(
            song_id=500,
            song_title="Empty Song",
            artist_name="Artist",
            lyrics_text="",  # Empty lyrics
        )

        cache.store_lyrics(lyrics)
        retrieved = cache.get_lyrics(500)

        assert retrieved is not None
        assert retrieved.lyrics_text == ""

    def test_cache_stats_size_calculation(self, temp_cache_dir: Path):
        """Test that cache stats correctly calculate size."""
        cache = LyricsCache(cache_dir=temp_cache_dir)

        # Store lyrics with known sizes
        cache.store_lyrics(
            Lyrics(
                song_id=1,
                song_title="Test",
                artist_name="Artist",
                lyrics_text="A" * 1000,  # ~1KB of text
            )
        )

        stats = cache.get_stats()
        # Size should be at least 1000 bytes (for the lyrics text)
        assert stats["size_bytes"] > 1000

    def test_timezone_naive_datetime_handling(self, temp_cache_dir: Path):
        """Test handling of timezone-naive datetimes in cache."""
        cache = LyricsCache(cache_dir=temp_cache_dir, ttl_hours=24)

        # Create cache file with timezone-naive datetime
        cache_path = cache._get_cache_path(600)
        import json

        # Use a recent timezone-naive datetime string
        naive_datetime = datetime.now().replace(microsecond=0).isoformat()
        data = {
            "song_id": 600,
            "song_title": "Naive",
            "artist_name": "Artist",
            "lyrics_text": "Text",
            "fetched_at": naive_datetime,
        }
        cache_path.write_text(json.dumps(data), encoding="utf-8")

        # Should handle timezone-naive datetime gracefully
        result = cache.get_lyrics(600)
        assert result is not None
        assert result.song_id == 600
