"""Shared fixtures for CLI tests."""

from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Temporary directory for cache testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def mock_settings(temp_cache_dir: Path):
    """Mock Settings object for testing."""
    from barscan.config import Settings

    return Settings(
        genius_access_token="test_token_123",
        cache_dir=temp_cache_dir,
        cache_ttl_hours=24,
        default_max_songs=5,
        default_top_words=10,
    )


@pytest.fixture
def mock_artist():
    """Create a mock Artist model."""
    from barscan.genius.models import Artist

    return Artist(
        id=123,
        name="Test Artist",
        url="https://genius.com/artists/Test-Artist",
        image_url="https://images.genius.com/test.jpg",
        is_verified=True,
    )


@pytest.fixture
def mock_song():
    """Create a mock Song model."""
    from barscan.genius.models import Song

    return Song(
        id=456,
        title="Test Song",
        title_with_featured="Test Song (ft. Other Artist)",
        artist="Test Artist",
        artist_id=123,
        url="https://genius.com/Test-artist-test-song-lyrics",
        lyrics_state="complete",
        header_image_url="https://images.genius.com/song.jpg",
    )


@pytest.fixture
def mock_lyrics():
    """Create a mock Lyrics model."""
    from barscan.genius.models import Lyrics

    return Lyrics(
        song_id=456,
        song_title="Test Song",
        artist_name="Test Artist",
        lyrics_text="hello world hello test song word word word",
    )


@pytest.fixture
def mock_artist_with_songs(mock_artist, mock_song):
    """Create a mock ArtistWithSongs model."""
    from barscan.genius.models import ArtistWithSongs

    return ArtistWithSongs(
        artist=mock_artist,
        songs=[mock_song],
        total_songs_fetched=1,
    )
