"""Genius API client wrapper with retry logic."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import requests.exceptions
from lyricsgenius import Genius

from barscan.config import Settings, settings
from barscan.exceptions import (
    ArtistNotFoundError,
    GeniusAPIError,
    NoLyricsFoundError,
)
from barscan.logging import get_logger

from .cache import LyricsCache
from .models import Artist, ArtistWithSongs, Lyrics, PaginatedSongs, Song

if TYPE_CHECKING:
    from lyricsgenius.artist import Artist as GeniusArtist
    from lyricsgenius.song import Song as GeniusSong

logger = get_logger("genius.client")


class GeniusClient:
    """High-level wrapper for Genius API with caching and retry logic."""

    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0
    DEFAULT_PER_PAGE = 20

    def __init__(
        self,
        access_token: str | None = None,
        settings_obj: Settings | None = None,
        enable_cache: bool = True,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> None:
        """
        Initialize GeniusClient.

        Args:
            access_token: Genius API access token. Falls back to settings.
            settings_obj: Settings instance. Uses global settings if None.
            enable_cache: Whether to enable local caching.
            max_retries: Maximum retry attempts for failed requests.
            retry_delay: Base delay between retries (exponential backoff).
        """
        self._settings = settings_obj or settings
        self._token = access_token or self._settings.get_access_token()

        if not self._token:
            raise GeniusAPIError("Genius access token is required")

        self._client = Genius(
            access_token=self._token,
            timeout=15,
            retries=max_retries,
            remove_section_headers=True,
            skip_non_songs=True,
            verbose=False,
        )

        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._cache: LyricsCache | None = None

        if enable_cache:
            self._cache = LyricsCache(
                cache_dir=self._settings.cache_dir,
                ttl_hours=self._settings.cache_ttl_hours,
            )

    def search_artist(self, artist_name: str) -> Artist:
        """
        Search for an artist by name.

        Args:
            artist_name: Name of the artist to search for.

        Returns:
            Artist model with basic artist information.

        Raises:
            ArtistNotFoundError: If no artist matches the search.
            GeniusAPIError: If API request fails.
        """
        logger.debug("Searching for artist: %s", artist_name)
        result = self._retry_request(
            lambda: self._client.search_artist(
                artist_name,
                max_songs=0,
                get_full_info=False,
            )
        )

        if result is None:
            raise ArtistNotFoundError(f"Artist not found: {artist_name}", artist_name=artist_name)

        artist = self._convert_artist(result)
        logger.info("Found artist: %s (id=%d)", artist.name, artist.id)
        return artist

    def get_artist_songs(
        self,
        artist_name: str,
        max_songs: int | None = None,
        sort: str = "popularity",
    ) -> ArtistWithSongs:
        """
        Get an artist and their songs.

        Args:
            artist_name: Name of the artist.
            max_songs: Maximum number of songs to fetch. Uses settings default if None.
            sort: Sort order - "popularity", "title", or "release_date".

        Returns:
            ArtistWithSongs containing artist info and song list.

        Raises:
            ArtistNotFoundError: If no artist matches the search.
            GeniusAPIError: If API request fails.
        """
        max_songs = max_songs or self._settings.default_max_songs

        logger.debug("Fetching artist songs: %s (max_songs=%d)", artist_name, max_songs)
        result = self._retry_request(
            lambda: self._client.search_artist(
                artist_name,
                max_songs=max_songs,
                sort=sort,
                get_full_info=True,
            )
        )

        if result is None:
            raise ArtistNotFoundError(f"Artist not found: {artist_name}", artist_name=artist_name)

        artist = self._convert_artist(result)
        songs = [self._convert_song(s) for s in (result.songs or [])]

        return ArtistWithSongs(
            artist=artist,
            songs=songs,
            total_songs_fetched=len(songs),
        )

    def get_songs_paginated(
        self,
        artist_id: int,
        page: int = 1,
        per_page: int = DEFAULT_PER_PAGE,
        sort: str = "popularity",
    ) -> PaginatedSongs:
        """
        Get songs for an artist with pagination.

        Args:
            artist_id: Genius artist ID.
            page: Page number (1-indexed).
            per_page: Number of songs per page.
            sort: Sort order - "popularity", "title", or "release_date".

        Returns:
            PaginatedSongs with songs for the requested page.

        Raises:
            GeniusAPIError: If API request fails.
        """
        result = self._retry_request(
            lambda: self._client.artist_songs(
                artist_id,
                per_page=per_page,
                page=page,
                sort=sort,
            )
        )

        if result is None:
            return PaginatedSongs(
                songs=[], page=page, per_page=per_page, has_next=False, total_fetched=0
            )

        songs_data = result.get("songs", [])
        next_page = result.get("next_page")

        songs = [self._convert_song_from_dict(s) for s in songs_data]

        return PaginatedSongs(
            songs=songs,
            page=page,
            per_page=per_page,
            has_next=next_page is not None,
            total_fetched=len(songs),
        )

    def get_lyrics(self, song: Song) -> Lyrics:
        """
        Get lyrics for a song.

        Args:
            song: Song model to fetch lyrics for.

        Returns:
            Lyrics model with the song's lyrics.

        Raises:
            NoLyricsFoundError: If lyrics are not available.
            GeniusAPIError: If API request fails.
        """
        logger.debug("Fetching lyrics for: %s (id=%d)", song.title, song.id)
        if self._cache:
            cached = self._cache.get_lyrics(song.id)
            if cached is not None:
                logger.debug("Cache hit for song: %s", song.title)
                return cached

        lyrics_text = self._retry_request(lambda: self._client.lyrics(song_url=str(song.url)))

        if lyrics_text is None or lyrics_text.strip() == "":
            raise NoLyricsFoundError(
                f"No lyrics found for: {song.title} by {song.artist}",
                song_id=song.id,
                song_title=song.title,
            )

        lyrics = Lyrics(
            song_id=song.id,
            song_title=song.title,
            artist_name=song.artist,
            lyrics_text=lyrics_text,
        )

        if self._cache:
            self._cache.store_lyrics(lyrics)

        return lyrics

    def get_lyrics_by_id(self, song_id: int) -> Lyrics:
        """
        Get lyrics by song ID (requires additional API call for song info).

        Args:
            song_id: Genius song ID.

        Returns:
            Lyrics model.

        Raises:
            NoLyricsFoundError: If song or lyrics not found.
            GeniusAPIError: If API request fails.
        """
        if self._cache:
            cached = self._cache.get_lyrics(song_id)
            if cached is not None:
                return cached

        result = self._retry_request(lambda: self._client.search_song(song_id=song_id))

        if result is None:
            raise NoLyricsFoundError(f"Song not found with ID: {song_id}")

        song = self._convert_song(result)
        return self.get_lyrics(song)

    def get_all_lyrics(
        self,
        artist_name: str,
        max_songs: int | None = None,
        sort: str = "popularity",
    ) -> list[Lyrics]:
        """
        Convenience method to get all lyrics for an artist.

        Args:
            artist_name: Name of the artist.
            max_songs: Maximum songs to fetch.
            sort: Sort order.

        Returns:
            List of Lyrics models (songs without lyrics are skipped).
        """
        artist_data = self.get_artist_songs(artist_name, max_songs, sort)

        lyrics_list: list[Lyrics] = []
        for song in artist_data.songs:
            try:
                lyrics = self.get_lyrics(song)
                lyrics_list.append(lyrics)
            except NoLyricsFoundError:
                continue

        return lyrics_list

    def _retry_request(self, request_fn: Any, retries: int | None = None) -> Any:
        """Execute a request with retry logic and exponential backoff.

        Catches network-related exceptions (ConnectionError, Timeout, HTTPError, etc.)
        and retries with exponential backoff. Programming errors and system signals
        are not caught and will propagate immediately.
        """
        retries = retries or self._max_retries
        last_error: Exception | None = None

        for attempt in range(retries):
            try:
                return request_fn()
            except requests.exceptions.RequestException as e:
                # Network errors: ConnectionError, Timeout, HTTPError, etc.
                last_error = e
                if attempt < retries - 1:
                    delay = self._retry_delay * (2**attempt)
                    time.sleep(delay)

        raise GeniusAPIError(
            f"Request failed after {retries} attempts: {last_error}"
        ) from last_error

    def _convert_artist(self, genius_artist: GeniusArtist) -> Artist:
        """Convert lyricsgenius Artist to Pydantic model."""
        return Artist(
            id=genius_artist._body["id"],
            name=genius_artist.name,
            url=genius_artist.url,
            image_url=getattr(genius_artist, "image_url", None),
            is_verified=getattr(genius_artist, "is_verified", False),
        )

    def _convert_song(self, genius_song: GeniusSong) -> Song:
        """Convert lyricsgenius Song to Pydantic model."""
        artist_id = 0
        if hasattr(genius_song, "primary_artist") and genius_song.primary_artist:
            artist_id = genius_song.primary_artist.get("id", 0)

        return Song(
            id=genius_song._body["id"],
            title=genius_song.title,
            title_with_featured=getattr(genius_song, "title_with_featured", genius_song.title),
            artist=genius_song.artist,
            artist_id=artist_id,
            url=genius_song.url,
            lyrics_state=getattr(genius_song, "lyrics_state", "complete"),
            header_image_url=getattr(genius_song, "header_image_url", None),
        )

    def _convert_song_from_dict(self, song_dict: dict[str, Any]) -> Song:
        """Convert API response dict to Song model."""
        primary_artist = song_dict.get("primary_artist", {})
        return Song(
            id=song_dict["id"],
            title=song_dict["title"],
            title_with_featured=song_dict.get("title_with_featured", song_dict["title"]),
            artist=primary_artist.get("name", "Unknown"),
            artist_id=primary_artist.get("id", 0),
            url=song_dict["url"],
            lyrics_state=song_dict.get("lyrics_state", "complete"),
            header_image_url=song_dict.get("header_image_url"),
        )
