"""Pydantic data models for Genius API data."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field, HttpUrl


class Artist(BaseModel):
    """Represents a Genius artist."""

    id: int = Field(..., description="Genius artist ID")
    name: str = Field(..., description="Artist name")
    url: HttpUrl = Field(..., description="Genius profile URL")
    image_url: HttpUrl | None = Field(None, description="Artist image URL")
    is_verified: bool = Field(False, description="Whether artist is verified")

    model_config = {"frozen": True}


class Song(BaseModel):
    """Represents a Genius song."""

    id: int = Field(..., description="Genius song ID")
    title: str = Field(..., description="Song title")
    title_with_featured: str = Field(..., description="Title with featured artists")
    artist: str = Field(..., description="Primary artist name")
    artist_id: int = Field(..., description="Primary artist Genius ID")
    url: HttpUrl = Field(..., description="Genius song URL")
    lyrics_state: str = Field("complete", description="Lyrics availability state")
    header_image_url: HttpUrl | None = Field(None, description="Header image URL")

    model_config = {"frozen": True}


class Lyrics(BaseModel):
    """Represents song lyrics with metadata."""

    song_id: int = Field(..., description="Genius song ID")
    song_title: str = Field(..., description="Song title")
    artist_name: str = Field(..., description="Artist name")
    lyrics_text: str = Field(..., description="Raw lyrics text")
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when lyrics were fetched",
    )

    @property
    def word_count(self) -> int:
        """Return approximate word count of lyrics."""
        return len(self.lyrics_text.split())

    @property
    def is_empty(self) -> bool:
        """Check if lyrics are empty or instrumental."""
        return len(self.lyrics_text.strip()) == 0


class ArtistWithSongs(BaseModel):
    """Artist with their songs list."""

    artist: Artist
    songs: list[Song] = Field(default_factory=list)
    total_songs_fetched: int = Field(0, description="Number of songs fetched")


class PaginatedSongs(BaseModel):
    """Paginated list of songs from artist."""

    songs: list[Song]
    page: int = Field(1, description="Current page number")
    per_page: int = Field(20, description="Songs per page")
    has_next: bool = Field(False, description="Whether more pages exist")
    total_fetched: int = Field(0, description="Total songs fetched so far")
