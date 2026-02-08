"""Tests for GeniusClient."""

from unittest.mock import MagicMock, patch

import pytest

from barscan.exceptions import ArtistNotFoundError, GeniusAPIError, NoLyricsFoundError
from barscan.genius.client import GeniusClient
from barscan.genius.models import Song


class TestGeniusClientInit:
    def test_requires_access_token(self, mock_settings):
        mock_settings.genius_access_token = ""
        with pytest.raises(GeniusAPIError, match="access token is required"):
            GeniusClient(settings_obj=mock_settings)

    @patch("barscan.genius.client.Genius")
    def test_initializes_with_settings(self, mock_genius_class, mock_settings):
        client = GeniusClient(settings_obj=mock_settings)
        mock_genius_class.assert_called_once()
        assert client._cache is not None

    @patch("barscan.genius.client.Genius")
    def test_initializes_without_cache(self, mock_genius_class, mock_settings):
        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)
        assert client._cache is None

    @patch("barscan.genius.client.Genius")
    def test_accepts_token_directly(self, mock_genius_class, mock_settings):
        mock_settings.genius_access_token = ""
        client = GeniusClient(
            access_token="direct_token",
            settings_obj=mock_settings,
            enable_cache=False,
        )
        assert client._token == "direct_token"


class TestSearchArtist:
    @patch("barscan.genius.client.Genius")
    def test_search_artist_success(self, mock_genius_class, mock_settings, mock_genius_artist):
        mock_genius = MagicMock()
        mock_genius.search_artist.return_value = mock_genius_artist
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)
        result = client.search_artist("Test Artist")

        assert result.id == 123
        assert result.name == "Test Artist"

    @patch("barscan.genius.client.Genius")
    def test_search_artist_not_found(self, mock_genius_class, mock_settings):
        mock_genius = MagicMock()
        mock_genius.search_artist.return_value = None
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)

        with pytest.raises(ArtistNotFoundError):
            client.search_artist("Unknown Artist")


class TestGetArtistSongs:
    @patch("barscan.genius.client.Genius")
    def test_get_artist_songs_success(
        self, mock_genius_class, mock_settings, mock_genius_artist, mock_genius_song
    ):
        mock_genius_artist.songs = [mock_genius_song]
        mock_genius = MagicMock()
        mock_genius.search_artist.return_value = mock_genius_artist
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)
        result = client.get_artist_songs("Test Artist", max_songs=5)

        assert result.artist.id == 123
        assert len(result.songs) == 1
        assert result.songs[0].title == "Test Song"

    @patch("barscan.genius.client.Genius")
    def test_get_artist_songs_not_found(self, mock_genius_class, mock_settings):
        mock_genius = MagicMock()
        mock_genius.search_artist.return_value = None
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)

        with pytest.raises(ArtistNotFoundError):
            client.get_artist_songs("Unknown Artist")


class TestGetLyrics:
    @patch("barscan.genius.client.Genius")
    def test_get_lyrics_success(self, mock_genius_class, mock_settings, sample_lyrics_text):
        mock_genius = MagicMock()
        mock_genius.lyrics.return_value = sample_lyrics_text
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)

        song = Song(
            id=456,
            title="Test",
            title_with_featured="Test",
            artist="Artist",
            artist_id=123,
            url="https://genius.com/test",
        )

        result = client.get_lyrics(song)
        assert result.lyrics_text == sample_lyrics_text

    @patch("barscan.genius.client.Genius")
    def test_get_lyrics_not_found(self, mock_genius_class, mock_settings):
        mock_genius = MagicMock()
        mock_genius.lyrics.return_value = None
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)

        song = Song(
            id=456,
            title="Test",
            title_with_featured="Test",
            artist="Artist",
            artist_id=123,
            url="https://genius.com/test",
        )

        with pytest.raises(NoLyricsFoundError):
            client.get_lyrics(song)

    @patch("barscan.genius.client.Genius")
    def test_get_lyrics_empty(self, mock_genius_class, mock_settings):
        mock_genius = MagicMock()
        mock_genius.lyrics.return_value = "   "
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)

        song = Song(
            id=456,
            title="Test",
            title_with_featured="Test",
            artist="Artist",
            artist_id=123,
            url="https://genius.com/test",
        )

        with pytest.raises(NoLyricsFoundError):
            client.get_lyrics(song)


class TestGetLyricsWithCache:
    @patch("barscan.genius.client.Genius")
    def test_returns_cached_lyrics(self, mock_genius_class, mock_settings, sample_lyrics_text):
        mock_genius = MagicMock()
        mock_genius.lyrics.return_value = sample_lyrics_text
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=True)

        song = Song(
            id=456,
            title="Test",
            title_with_featured="Test",
            artist="Artist",
            artist_id=123,
            url="https://genius.com/test",
        )

        # First call - should hit API
        result1 = client.get_lyrics(song)
        assert result1.lyrics_text == sample_lyrics_text
        assert mock_genius.lyrics.call_count == 1

        # Second call - should return cached
        result2 = client.get_lyrics(song)
        assert result2.lyrics_text == sample_lyrics_text
        assert mock_genius.lyrics.call_count == 1  # No additional API call


class TestRetryLogic:
    @patch("barscan.genius.client.Genius")
    @patch("barscan.genius.client.time.sleep")
    def test_retry_on_failure(
        self, mock_sleep, mock_genius_class, mock_settings, mock_genius_artist
    ):
        mock_genius = MagicMock()
        mock_genius.search_artist.side_effect = [
            Exception("Network error"),
            Exception("Timeout"),
            mock_genius_artist,
        ]
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(
            settings_obj=mock_settings,
            enable_cache=False,
            max_retries=3,
            retry_delay=0.1,
        )

        result = client.search_artist("Test")
        assert result.id == 123
        assert mock_sleep.call_count == 2

    @patch("barscan.genius.client.Genius")
    @patch("barscan.genius.client.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep, mock_genius_class, mock_settings):
        mock_genius = MagicMock()
        mock_genius.search_artist.side_effect = Exception("Persistent error")
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(
            settings_obj=mock_settings,
            enable_cache=False,
            max_retries=3,
            retry_delay=0.1,
        )

        with pytest.raises(GeniusAPIError, match="Request failed after 3 attempts"):
            client.search_artist("Test")


class TestGetAllLyrics:
    @patch("barscan.genius.client.Genius")
    def test_get_all_lyrics(
        self,
        mock_genius_class,
        mock_settings,
        mock_genius_artist,
        mock_genius_song,
        sample_lyrics_text,
    ):
        mock_genius_artist.songs = [mock_genius_song]
        mock_genius = MagicMock()
        mock_genius.search_artist.return_value = mock_genius_artist
        mock_genius.lyrics.return_value = sample_lyrics_text
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)
        result = client.get_all_lyrics("Test Artist", max_songs=5)

        assert len(result) == 1
        assert result[0].lyrics_text == sample_lyrics_text

    @patch("barscan.genius.client.Genius")
    def test_skips_songs_without_lyrics(
        self,
        mock_genius_class,
        mock_settings,
        mock_genius_artist,
        mock_genius_song,
    ):
        mock_genius_artist.songs = [mock_genius_song]
        mock_genius = MagicMock()
        mock_genius.search_artist.return_value = mock_genius_artist
        mock_genius.lyrics.return_value = None
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)
        result = client.get_all_lyrics("Test Artist", max_songs=5)

        assert len(result) == 0
