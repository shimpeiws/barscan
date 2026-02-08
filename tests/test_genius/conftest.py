"""Shared fixtures for genius module tests."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_genius_artist():
    """Create a mock lyricsgenius Artist."""
    artist = MagicMock()
    artist.id = 123
    artist.name = "Test Artist"
    artist.url = "https://genius.com/artists/Test-Artist"
    artist.image_url = "https://images.genius.com/test.jpg"
    artist.is_verified = True
    artist.songs = []
    return artist


@pytest.fixture
def mock_genius_song():
    """Create a mock lyricsgenius Song."""
    song = MagicMock()
    song.id = 456
    song.title = "Test Song"
    song.title_with_featured = "Test Song (ft. Other Artist)"
    song.artist = "Test Artist"
    song.url = "https://genius.com/Test-artist-test-song-lyrics"
    song.lyrics_state = "complete"
    song.header_image_url = "https://images.genius.com/song.jpg"

    song.primary_artist = {"id": 123, "name": "Test Artist"}

    return song


@pytest.fixture
def sample_lyrics_text():
    """Sample lyrics text for testing."""
    return """[Verse 1]
This is the first verse
With some test lyrics
Word word word

[Chorus]
Repeated chorus line
Repeated chorus line
"""


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
    )
