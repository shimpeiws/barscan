"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from barscan.genius.models import Artist, ArtistWithSongs, Lyrics, PaginatedSongs, Song


class TestArtist:
    def test_create_artist(self):
        artist = Artist(
            id=123,
            name="Test Artist",
            url="https://genius.com/artists/Test-Artist",
        )
        assert artist.id == 123
        assert artist.name == "Test Artist"
        assert artist.image_url is None
        assert artist.is_verified is False

    def test_artist_with_all_fields(self):
        artist = Artist(
            id=123,
            name="Test Artist",
            url="https://genius.com/artists/Test-Artist",
            image_url="https://images.genius.com/test.jpg",
            is_verified=True,
        )
        assert artist.is_verified is True
        assert str(artist.image_url) == "https://images.genius.com/test.jpg"

    def test_artist_immutable(self):
        artist = Artist(
            id=123,
            name="Test",
            url="https://genius.com/test",
        )
        with pytest.raises(ValidationError):
            artist.name = "Changed"

    def test_artist_invalid_url(self):
        with pytest.raises(ValidationError):
            Artist(
                id=123,
                name="Test",
                url="not-a-url",
            )


class TestSong:
    def test_create_song(self):
        song = Song(
            id=456,
            title="Test Song",
            title_with_featured="Test Song",
            artist="Test Artist",
            artist_id=123,
            url="https://genius.com/test-song",
        )
        assert song.id == 456
        assert song.lyrics_state == "complete"
        assert song.header_image_url is None

    def test_song_with_all_fields(self):
        song = Song(
            id=456,
            title="Test Song",
            title_with_featured="Test Song (ft. Guest)",
            artist="Test Artist",
            artist_id=123,
            url="https://genius.com/test-song",
            lyrics_state="complete",
            header_image_url="https://images.genius.com/header.jpg",
        )
        assert song.title_with_featured == "Test Song (ft. Guest)"
        assert str(song.header_image_url) == "https://images.genius.com/header.jpg"


class TestLyrics:
    def test_create_lyrics(self, sample_lyrics_text: str):
        lyrics = Lyrics(
            song_id=456,
            song_title="Test Song",
            artist_name="Test Artist",
            lyrics_text=sample_lyrics_text,
        )
        assert lyrics.song_id == 456
        assert lyrics.word_count > 0
        assert not lyrics.is_empty

    def test_empty_lyrics(self):
        lyrics = Lyrics(
            song_id=456,
            song_title="Instrumental",
            artist_name="Test",
            lyrics_text="",
        )
        assert lyrics.is_empty
        assert lyrics.word_count == 0

    def test_whitespace_only_lyrics(self):
        lyrics = Lyrics(
            song_id=456,
            song_title="Test",
            artist_name="Test",
            lyrics_text="   \n\t  ",
        )
        assert lyrics.is_empty

    def test_word_count(self):
        lyrics = Lyrics(
            song_id=456,
            song_title="Test",
            artist_name="Test",
            lyrics_text="one two three four five",
        )
        assert lyrics.word_count == 5


class TestArtistWithSongs:
    def test_create_artist_with_songs(self):
        artist = Artist(
            id=123,
            name="Test Artist",
            url="https://genius.com/artists/Test-Artist",
        )
        song = Song(
            id=456,
            title="Test Song",
            title_with_featured="Test Song",
            artist="Test Artist",
            artist_id=123,
            url="https://genius.com/test-song",
        )
        artist_with_songs = ArtistWithSongs(
            artist=artist,
            songs=[song],
            total_songs_fetched=1,
        )
        assert artist_with_songs.artist.id == 123
        assert len(artist_with_songs.songs) == 1
        assert artist_with_songs.total_songs_fetched == 1

    def test_default_empty_songs(self):
        artist = Artist(
            id=123,
            name="Test Artist",
            url="https://genius.com/artists/Test-Artist",
        )
        artist_with_songs = ArtistWithSongs(artist=artist)
        assert artist_with_songs.songs == []
        assert artist_with_songs.total_songs_fetched == 0


class TestPaginatedSongs:
    def test_create_paginated_songs(self):
        song = Song(
            id=456,
            title="Test Song",
            title_with_featured="Test Song",
            artist="Test Artist",
            artist_id=123,
            url="https://genius.com/test-song",
        )
        paginated = PaginatedSongs(
            songs=[song],
            page=1,
            per_page=20,
            has_next=True,
            total_fetched=1,
        )
        assert len(paginated.songs) == 1
        assert paginated.page == 1
        assert paginated.has_next is True

    def test_default_values(self):
        paginated = PaginatedSongs(songs=[])
        assert paginated.page == 1
        assert paginated.per_page == 20
        assert paginated.has_next is False
        assert paginated.total_fetched == 0
