"""Tests for GeniusClient."""

from unittest.mock import MagicMock, patch

import pytest

from pydantic import SecretStr

from barscan.exceptions import ArtistNotFoundError, GeniusAPIError, NoLyricsFoundError
from barscan.genius.client import GeniusClient
from barscan.genius.models import Song


class TestGeniusClientInit:
    def test_requires_access_token(self, mock_settings):
        mock_settings.genius_access_token = SecretStr("")
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


class TestGetSongsPaginated:
    """Tests for get_songs_paginated method."""

    @patch("barscan.genius.client.Genius")
    def test_get_songs_paginated_success(self, mock_genius_class, mock_settings):
        """Test getting paginated songs."""
        mock_genius = MagicMock()
        mock_genius.artist_songs.return_value = {
            "songs": [
                {
                    "id": 1,
                    "title": "Song 1",
                    "url": "https://genius.com/song-1",
                    "primary_artist": {"id": 123, "name": "Artist"},
                },
                {
                    "id": 2,
                    "title": "Song 2",
                    "url": "https://genius.com/song-2",
                    "primary_artist": {"id": 123, "name": "Artist"},
                },
            ],
            "next_page": 2,
        }
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)
        result = client.get_songs_paginated(artist_id=123, page=1, per_page=2)

        assert len(result.songs) == 2
        assert result.page == 1
        assert result.has_next is True

    @patch("barscan.genius.client.Genius")
    def test_get_songs_paginated_last_page(self, mock_genius_class, mock_settings):
        """Test getting last page of songs."""
        mock_genius = MagicMock()
        mock_genius.artist_songs.return_value = {
            "songs": [
                {
                    "id": 1,
                    "title": "Song 1",
                    "url": "https://genius.com/song-1",
                    "primary_artist": {"id": 123, "name": "Artist"},
                },
            ],
            "next_page": None,
        }
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)
        result = client.get_songs_paginated(artist_id=123, page=5)

        assert result.has_next is False

    @patch("barscan.genius.client.Genius")
    def test_get_songs_paginated_empty(self, mock_genius_class, mock_settings):
        """Test getting empty page of songs."""
        mock_genius = MagicMock()
        mock_genius.artist_songs.return_value = None
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)
        result = client.get_songs_paginated(artist_id=123, page=100)

        assert len(result.songs) == 0
        assert result.has_next is False


class TestGetLyricsById:
    """Tests for get_lyrics_by_id method."""

    @patch("barscan.genius.client.Genius")
    def test_get_lyrics_by_id_success(
        self, mock_genius_class, mock_settings, mock_genius_song, sample_lyrics_text
    ):
        """Test getting lyrics by song ID."""
        mock_genius = MagicMock()
        mock_genius.search_song.return_value = mock_genius_song
        mock_genius.lyrics.return_value = sample_lyrics_text
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)
        result = client.get_lyrics_by_id(456)

        assert result.lyrics_text == sample_lyrics_text

    @patch("barscan.genius.client.Genius")
    def test_get_lyrics_by_id_not_found(self, mock_genius_class, mock_settings):
        """Test getting lyrics by ID when song not found."""
        mock_genius = MagicMock()
        mock_genius.search_song.return_value = None
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)

        with pytest.raises(NoLyricsFoundError, match="Song not found"):
            client.get_lyrics_by_id(999)

    @patch("barscan.genius.client.Genius")
    def test_get_lyrics_by_id_uses_cache(
        self, mock_genius_class, mock_settings, sample_lyrics_text
    ):
        """Test that get_lyrics_by_id uses cache."""
        mock_genius = MagicMock()
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=True)

        # Pre-populate cache
        from barscan.genius.models import Lyrics

        cached_lyrics = Lyrics(
            song_id=789,
            song_title="Cached Song",
            artist_name="Artist",
            lyrics_text="Cached lyrics text",
        )
        client._cache.store_lyrics(cached_lyrics)

        result = client.get_lyrics_by_id(789)

        # Should use cache, not call API
        mock_genius.search_song.assert_not_called()
        assert result.lyrics_text == "Cached lyrics text"


class TestConvertSongFromDict:
    """Tests for _convert_song_from_dict method."""

    @patch("barscan.genius.client.Genius")
    def test_convert_song_from_dict_full(self, mock_genius_class, mock_settings):
        """Test converting full song dict to Song model."""
        mock_genius_class.return_value = MagicMock()
        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)

        song_dict = {
            "id": 123,
            "title": "Test Song",
            "title_with_featured": "Test Song (ft. Guest)",
            "url": "https://genius.com/test",
            "lyrics_state": "complete",
            "header_image_url": "https://images.genius.com/test.jpg",
            "primary_artist": {"id": 456, "name": "Artist Name"},
        }

        result = client._convert_song_from_dict(song_dict)

        assert result.id == 123
        assert result.title == "Test Song"
        assert result.title_with_featured == "Test Song (ft. Guest)"
        assert result.artist == "Artist Name"
        assert result.artist_id == 456

    @patch("barscan.genius.client.Genius")
    def test_convert_song_from_dict_minimal(self, mock_genius_class, mock_settings):
        """Test converting minimal song dict to Song model."""
        mock_genius_class.return_value = MagicMock()
        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)

        song_dict = {
            "id": 123,
            "title": "Test Song",
            "url": "https://genius.com/test",
        }

        result = client._convert_song_from_dict(song_dict)

        assert result.id == 123
        assert result.title == "Test Song"
        assert result.artist == "Unknown"
        assert result.artist_id == 0


class TestExponentialBackoff:
    """Tests for exponential backoff behavior."""

    @patch("barscan.genius.client.Genius")
    @patch("barscan.genius.client.time.sleep")
    def test_exponential_backoff_delays(
        self, mock_sleep, mock_genius_class, mock_settings, mock_genius_artist
    ):
        """Test that retry delays follow exponential pattern."""
        mock_genius = MagicMock()
        mock_genius.search_artist.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            mock_genius_artist,
        ]
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(
            settings_obj=mock_settings,
            enable_cache=False,
            max_retries=3,
            retry_delay=1.0,
        )

        client.search_artist("Test")

        # Check exponential backoff: 1.0 * 2^0 = 1.0, 1.0 * 2^1 = 2.0
        calls = mock_sleep.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == 1.0  # First retry delay
        assert calls[1][0][0] == 2.0  # Second retry delay (doubled)


class TestClientEdgeCases:
    """Additional edge case tests for GeniusClient."""

    @patch("barscan.genius.client.Genius")
    def test_get_artist_songs_with_empty_songs_list(
        self, mock_genius_class, mock_settings, mock_genius_artist
    ):
        """Test getting artist with empty songs list."""
        mock_genius_artist.songs = []
        mock_genius = MagicMock()
        mock_genius.search_artist.return_value = mock_genius_artist
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)
        result = client.get_artist_songs("Test Artist")

        assert len(result.songs) == 0
        assert result.total_songs_fetched == 0

    @patch("barscan.genius.client.Genius")
    def test_get_artist_songs_with_none_songs(
        self, mock_genius_class, mock_settings, mock_genius_artist
    ):
        """Test getting artist when songs attribute is None."""
        mock_genius_artist.songs = None
        mock_genius = MagicMock()
        mock_genius.search_artist.return_value = mock_genius_artist
        mock_genius_class.return_value = mock_genius

        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)
        result = client.get_artist_songs("Test Artist")

        assert len(result.songs) == 0

    @patch("barscan.genius.client.Genius")
    def test_convert_artist_with_missing_optional_fields(
        self, mock_genius_class, mock_settings
    ):
        """Test converting artist with missing optional fields."""
        mock_genius_class.return_value = MagicMock()
        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)

        # Create minimal artist mock without optional attributes
        artist = MagicMock(spec=["id", "name", "url"])
        artist.id = 123
        artist.name = "Test Artist"
        artist.url = "https://genius.com/artists/test"
        # image_url and is_verified attributes don't exist

        result = client._convert_artist(artist)

        assert result.id == 123
        assert result.name == "Test Artist"
        assert result.image_url is None
        assert result.is_verified is False

    @patch("barscan.genius.client.Genius")
    def test_convert_song_with_missing_primary_artist(
        self, mock_genius_class, mock_settings
    ):
        """Test converting song when primary_artist is missing."""
        mock_genius_class.return_value = MagicMock()
        client = GeniusClient(settings_obj=mock_settings, enable_cache=False)

        # Create song mock without primary_artist having an id
        song = MagicMock(spec=["id", "title", "artist", "url", "primary_artist"])
        song.id = 456
        song.title = "Test Song"
        song.artist = "Artist Name"
        song.url = "https://genius.com/test"
        song.primary_artist = None  # No primary artist

        result = client._convert_song(song)

        assert result.id == 456
        assert result.artist_id == 0  # Default when no primary artist
