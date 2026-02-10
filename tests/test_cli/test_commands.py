"""Tests for CLI commands."""

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from barscan.cli import app
from barscan.genius.cache import LyricsCache
from barscan.genius.client import GeniusClient


class TestConfigCommand:
    """Tests for the config command."""

    def test_config_shows_settings(self, cli_runner: CliRunner, mock_settings, temp_cache_dir):
        """Test config command displays settings."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.LyricsCache") as mock_cache_class:
                mock_cache = MagicMock(spec=LyricsCache)
                mock_cache.get_stats.return_value = {
                    "total_entries": 5,
                    "size_bytes": 1024,
                    "expired": 1,
                }
                mock_cache_class.return_value = mock_cache

                result = cli_runner.invoke(app, ["config"])

                assert result.exit_code == 0
                assert "BARSCAN_GENIUS_ACCESS_TOKEN" in result.output
                assert "BARSCAN_CACHE_DIR" in result.output
                assert "BARSCAN_CACHE_TTL_HOURS" in result.output
                assert "Cache Statistics" in result.output

    def test_config_masks_token(self, cli_runner: CliRunner, mock_settings):
        """Test config command masks the API token."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.LyricsCache") as mock_cache_class:
                mock_cache = MagicMock(spec=LyricsCache)
                mock_cache.get_stats.return_value = {
                    "total_entries": 0,
                    "size_bytes": 0,
                    "expired": 0,
                }
                mock_cache_class.return_value = mock_cache

                result = cli_runner.invoke(app, ["config"])

                assert result.exit_code == 0
                # Token should be masked (showing only first and last 4 chars)
                assert "test_token_123" not in result.output


class TestClearCacheCommand:
    """Tests for the clear-cache command."""

    def test_clear_cache_empty(self, cli_runner: CliRunner, mock_settings):
        """Test clear-cache when cache is already empty."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.LyricsCache") as mock_cache_class:
                mock_cache = MagicMock(spec=LyricsCache)
                mock_cache.get_stats.return_value = {
                    "total_entries": 0,
                    "size_bytes": 0,
                    "expired": 0,
                }
                mock_cache_class.return_value = mock_cache

                result = cli_runner.invoke(app, ["clear-cache"])

                assert result.exit_code == 0
                assert "already empty" in result.output.lower()

    def test_clear_cache_with_force(self, cli_runner: CliRunner, mock_settings):
        """Test clear-cache with --force flag."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.LyricsCache") as mock_cache_class:
                mock_cache = MagicMock(spec=LyricsCache)
                mock_cache.get_stats.return_value = {
                    "total_entries": 5,
                    "size_bytes": 1024,
                    "expired": 1,
                }
                mock_cache.clear.return_value = 5
                mock_cache_class.return_value = mock_cache

                result = cli_runner.invoke(app, ["clear-cache", "--force"])

                assert result.exit_code == 0
                assert "Cleared 5 cache entries" in result.output
                mock_cache.clear.assert_called_once()

    def test_clear_cache_expired_only(self, cli_runner: CliRunner, mock_settings):
        """Test clear-cache with --expired-only flag."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.LyricsCache") as mock_cache_class:
                mock_cache = MagicMock(spec=LyricsCache)
                mock_cache.get_stats.return_value = {
                    "total_entries": 5,
                    "size_bytes": 1024,
                    "expired": 2,
                }
                mock_cache.clear_expired.return_value = 2
                mock_cache_class.return_value = mock_cache

                result = cli_runner.invoke(app, ["clear-cache", "--expired-only", "--force"])

                assert result.exit_code == 0
                assert "Cleared 2 expired cache entries" in result.output
                mock_cache.clear_expired.assert_called_once()

    def test_clear_cache_no_expired(self, cli_runner: CliRunner, mock_settings):
        """Test clear-cache --expired-only when no expired entries."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.LyricsCache") as mock_cache_class:
                mock_cache = MagicMock(spec=LyricsCache)
                mock_cache.get_stats.return_value = {
                    "total_entries": 5,
                    "size_bytes": 1024,
                    "expired": 0,
                }
                mock_cache_class.return_value = mock_cache

                result = cli_runner.invoke(app, ["clear-cache", "--expired-only"])

                assert result.exit_code == 0
                assert "No expired cache entries" in result.output

    def test_clear_cache_cancelled(self, cli_runner: CliRunner, mock_settings):
        """Test clear-cache cancelled by user."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.LyricsCache") as mock_cache_class:
                mock_cache = MagicMock(spec=LyricsCache)
                mock_cache.get_stats.return_value = {
                    "total_entries": 5,
                    "size_bytes": 1024,
                    "expired": 0,
                }
                mock_cache_class.return_value = mock_cache

                result = cli_runner.invoke(app, ["clear-cache"], input="n\n")

                assert result.exit_code == 0
                assert "Cancelled" in result.output


class TestAnalyzeCommand:
    """Tests for the analyze command."""

    def test_analyze_requires_api_token(self, cli_runner: CliRunner):
        """Test analyze fails without API token configured."""
        from barscan.config import Settings

        mock_settings = Settings(genius_access_token="")

        with patch("barscan.cli.settings", mock_settings):
            result = cli_runner.invoke(app, ["analyze", "Test Artist"])

            assert result.exit_code == 1
            assert "Genius API token not configured" in result.output

    def test_analyze_artist_not_found(self, cli_runner: CliRunner, mock_settings):
        """Test analyze handles artist not found."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                from barscan.exceptions import ArtistNotFoundError

                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.side_effect = ArtistNotFoundError("Not found")
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(app, ["analyze", "Unknown Artist"])

                assert result.exit_code == 1
                assert "Artist not found" in result.output

    def test_analyze_success_table_format(
        self, cli_runner: CliRunner, mock_settings, mock_artist_with_songs, mock_lyrics
    ):
        """Test analyze with table output format."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.return_value = mock_artist_with_songs
                mock_client.get_lyrics.return_value = mock_lyrics
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(app, ["analyze", "Test Artist", "-t", "5"])

                assert result.exit_code == 0
                assert "Test Artist" in result.output
                assert "Word Frequencies" in result.output

    def test_analyze_json_format(
        self, cli_runner: CliRunner, mock_settings, mock_artist_with_songs, mock_lyrics
    ):
        """Test analyze with JSON output format."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.return_value = mock_artist_with_songs
                mock_client.get_lyrics.return_value = mock_lyrics
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(app, ["analyze", "Test Artist", "-f", "json", "-t", "5"])

                assert result.exit_code == 0
                # Find the JSON block in the output (starts with '{')
                output_lines = result.output.strip().split("\n")
                json_start = None
                for i, line in enumerate(output_lines):
                    if line.strip().startswith("{"):
                        json_start = i
                        break
                assert json_start is not None, "JSON output not found"
                json_text = "\n".join(output_lines[json_start:])
                output_data = json.loads(json_text)
                assert "artist" in output_data
                assert "frequencies" in output_data

    def test_analyze_csv_format(
        self, cli_runner: CliRunner, mock_settings, mock_artist_with_songs, mock_lyrics
    ):
        """Test analyze with CSV output format."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.return_value = mock_artist_with_songs
                mock_client.get_lyrics.return_value = mock_lyrics
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(app, ["analyze", "Test Artist", "-f", "csv", "-t", "5"])

                assert result.exit_code == 0
                assert "word,count,percentage" in result.output

    def test_analyze_output_file(
        self, cli_runner: CliRunner, mock_settings, mock_artist_with_songs, mock_lyrics, tmp_path
    ):
        """Test analyze with output file."""
        output_file = tmp_path / "results.json"

        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.return_value = mock_artist_with_songs
                mock_client.get_lyrics.return_value = mock_lyrics
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(
                    app, ["analyze", "Test Artist", "-f", "json", "-o", str(output_file), "-t", "5"]
                )

                assert result.exit_code == 0
                assert output_file.exists()
                data = json.loads(output_file.read_text())
                assert "artist" in data

    def test_analyze_no_songs_found(self, cli_runner: CliRunner, mock_settings, mock_artist):
        """Test analyze when no songs are found."""
        from barscan.genius.models import ArtistWithSongs

        empty_result = ArtistWithSongs(artist=mock_artist, songs=[], total_songs_fetched=0)

        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.return_value = empty_result
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(app, ["analyze", "Test Artist"])

                assert result.exit_code == 0
                assert "No songs found" in result.output

    def test_analyze_empty_artist_name(self, cli_runner: CliRunner):
        """Test analyze with empty artist name."""
        result = cli_runner.invoke(app, ["analyze", "   "])

        assert result.exit_code != 0

    def test_analyze_with_exclude_words(
        self, cli_runner: CliRunner, mock_settings, mock_artist_with_songs, mock_lyrics
    ):
        """Test analyze with --exclude option."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.return_value = mock_artist_with_songs
                mock_client.get_lyrics.return_value = mock_lyrics
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(
                    app,
                    ["analyze", "Test Artist", "-e", "hello", "-e", "world", "-t", "5"],
                )

                assert result.exit_code == 0

    def test_analyze_no_stop_words(
        self, cli_runner: CliRunner, mock_settings, mock_artist_with_songs, mock_lyrics
    ):
        """Test analyze with --no-stop-words option."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.return_value = mock_artist_with_songs
                mock_client.get_lyrics.return_value = mock_lyrics
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(
                    app, ["analyze", "Test Artist", "--no-stop-words", "-t", "5"]
                )

                assert result.exit_code == 0

    def test_analyze_genius_api_error(self, cli_runner: CliRunner, mock_settings):
        """Test analyze handles generic API errors."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                from barscan.exceptions import GeniusAPIError

                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.side_effect = GeniusAPIError("API rate limit")
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(app, ["analyze", "Test Artist"])

                assert result.exit_code == 1
                assert "API error" in result.output

    def test_analyze_barscan_error(self, cli_runner: CliRunner, mock_settings):
        """Test analyze handles generic BarScan errors."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                from barscan.exceptions import BarScanError

                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.side_effect = BarScanError("Generic error")
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(app, ["analyze", "Test Artist"])

                assert result.exit_code == 1

    def test_analyze_no_lyrics_found_all_songs(
        self, cli_runner: CliRunner, mock_settings, mock_artist_with_songs
    ):
        """Test analyze when no lyrics are found for any song."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                from barscan.exceptions import NoLyricsFoundError

                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.return_value = mock_artist_with_songs
                mock_client.get_lyrics.side_effect = NoLyricsFoundError("No lyrics")
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(app, ["analyze", "Test Artist"])

                assert result.exit_code == 0
                # Should report no lyrics found to analyze
                assert "No lyrics found" in result.output or "Skipped" in result.output

    def test_analyze_skips_empty_lyrics(
        self, cli_runner: CliRunner, mock_settings, mock_artist_with_songs
    ):
        """Test analyze skips songs with empty lyrics."""
        from barscan.genius.models import Lyrics

        empty_lyrics = Lyrics(
            song_id=456,
            song_title="Empty Song",
            artist_name="Test Artist",
            lyrics_text="",  # Empty lyrics
        )

        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.return_value = mock_artist_with_songs
                mock_client.get_lyrics.return_value = empty_lyrics
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(app, ["analyze", "Test Artist"])

                # Should handle empty lyrics gracefully
                assert result.exit_code == 0

    def test_analyze_client_init_failure(self, cli_runner: CliRunner, mock_settings):
        """Test analyze handles client initialization failure."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                from barscan.exceptions import GeniusAPIError

                mock_client_class.side_effect = GeniusAPIError("Invalid token")

                result = cli_runner.invoke(app, ["analyze", "Test Artist"])

                assert result.exit_code == 1
                assert "Error" in result.output

    def test_analyze_wordgrain_format(
        self, cli_runner: CliRunner, mock_settings, mock_artist_with_songs, mock_lyrics
    ):
        """Test analyze with wordgrain output format."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.return_value = mock_artist_with_songs
                mock_client.get_lyrics.return_value = mock_lyrics
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(
                    app, ["analyze", "Test Artist", "-f", "wordgrain", "-t", "5"]
                )

                assert result.exit_code == 0
                assert "grains" in result.output or "$schema" in result.output

    def test_analyze_wordgrain_to_file(
        self, cli_runner: CliRunner, mock_settings, mock_artist_with_songs, mock_lyrics, tmp_path
    ):
        """Test analyze with wordgrain format to file."""
        output_file = tmp_path / "output.wg.json"

        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.return_value = mock_artist_with_songs
                mock_client.get_lyrics.return_value = mock_lyrics
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(
                    app,
                    ["analyze", "Test Artist", "-f", "wordgrain", "-o", str(output_file), "-t", "5"],
                )

                assert result.exit_code == 0
                assert output_file.exists()
                content = output_file.read_text()
                assert "$schema" in content

    def test_analyze_table_to_file(
        self, cli_runner: CliRunner, mock_settings, mock_artist_with_songs, mock_lyrics, tmp_path
    ):
        """Test analyze with table format written to file."""
        output_file = tmp_path / "results.txt"

        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.return_value = mock_artist_with_songs
                mock_client.get_lyrics.return_value = mock_lyrics
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(
                    app,
                    ["analyze", "Test Artist", "-f", "table", "-o", str(output_file), "-t", "5"],
                )

                assert result.exit_code == 0
                assert output_file.exists()
                content = output_file.read_text()
                assert "Artist:" in content

    def test_analyze_with_multiple_songs(
        self, cli_runner: CliRunner, mock_settings, mock_artist
    ):
        """Test analyze with multiple songs."""
        from barscan.genius.models import ArtistWithSongs, Lyrics, Song

        songs = [
            Song(
                id=i,
                title=f"Song {i}",
                title_with_featured=f"Song {i}",
                artist="Test Artist",
                artist_id=123,
                url=f"https://genius.com/song-{i}",
            )
            for i in range(3)
        ]

        artist_with_songs = ArtistWithSongs(
            artist=mock_artist,
            songs=songs,
            total_songs_fetched=3,
        )

        def get_lyrics_for_song(song):
            return Lyrics(
                song_id=song.id,
                song_title=song.title,
                artist_name="Test Artist",
                lyrics_text=f"lyrics for song {song.id} word word test",
            )

        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.return_value = artist_with_songs
                mock_client.get_lyrics.side_effect = get_lyrics_for_song
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(app, ["analyze", "Test Artist", "-t", "5"])

                assert result.exit_code == 0
                assert "3 songs" in result.output

    def test_analyze_with_max_songs_option(
        self, cli_runner: CliRunner, mock_settings, mock_artist_with_songs, mock_lyrics
    ):
        """Test analyze with --max-songs option."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.return_value = mock_artist_with_songs
                mock_client.get_lyrics.return_value = mock_lyrics
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(
                    app, ["analyze", "Test Artist", "--max-songs", "5", "-t", "5"]
                )

                assert result.exit_code == 0
                mock_client.get_artist_songs.assert_called_once()
                call_kwargs = mock_client.get_artist_songs.call_args
                assert call_kwargs[1]["max_songs"] == 5


class TestClearCacheEdgeCases:
    """Additional edge case tests for clear-cache command."""

    def test_clear_cache_expired_cancelled(self, cli_runner: CliRunner, mock_settings):
        """Test clear-cache --expired-only cancelled by user."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.LyricsCache") as mock_cache_class:
                mock_cache = MagicMock(spec=LyricsCache)
                mock_cache.get_stats.return_value = {
                    "total_entries": 5,
                    "size_bytes": 1024,
                    "expired": 2,
                }
                mock_cache_class.return_value = mock_cache

                result = cli_runner.invoke(
                    app, ["clear-cache", "--expired-only"], input="n\n"
                )

                assert result.exit_code == 0
                assert "Cancelled" in result.output


class TestConfigCommand:
    """Additional tests for config command."""

    def test_config_with_short_token(self, cli_runner: CliRunner, temp_cache_dir):
        """Test config with a short token that can't be properly masked."""
        from barscan.config import Settings

        short_token_settings = Settings(
            genius_access_token="abc",  # Very short token
            cache_dir=temp_cache_dir,
        )

        with patch("barscan.cli.settings", short_token_settings):
            with patch("barscan.cli.LyricsCache") as mock_cache_class:
                mock_cache = MagicMock(spec=LyricsCache)
                mock_cache.get_stats.return_value = {
                    "total_entries": 0,
                    "size_bytes": 0,
                    "expired": 0,
                }
                mock_cache_class.return_value = mock_cache

                result = cli_runner.invoke(app, ["config"])

                assert result.exit_code == 0
                # Short tokens should be masked with ****
                assert "abc" not in result.output

    def test_config_with_no_token(self, cli_runner: CliRunner, temp_cache_dir):
        """Test config when no token is configured."""
        from barscan.config import Settings

        no_token_settings = Settings(
            genius_access_token="",
            cache_dir=temp_cache_dir,
        )

        with patch("barscan.cli.settings", no_token_settings):
            with patch("barscan.cli.LyricsCache") as mock_cache_class:
                mock_cache = MagicMock(spec=LyricsCache)
                mock_cache.get_stats.return_value = {
                    "total_entries": 0,
                    "size_bytes": 0,
                    "expired": 0,
                }
                mock_cache_class.return_value = mock_cache

                result = cli_runner.invoke(app, ["config"])

                assert result.exit_code == 0
                assert "Not set" in result.output


class TestFormatOutputFunction:
    """Tests for format_output function."""

    def test_format_output_wordgrain_requires_aggregate(self, cli_runner: CliRunner):
        """Test that wordgrain format requires aggregate parameter."""
        from barscan.cli import OutputFormat, format_output

        import pytest

        with pytest.raises(ValueError, match="aggregate is required"):
            format_output(
                artist_name="Test",
                songs_analyzed=1,
                total_words=100,
                unique_words=50,
                frequencies=[],
                output_format=OutputFormat.WORDGRAIN,
                aggregate=None,  # Missing aggregate
            )


class TestValidation:
    """Tests for validation functions."""

    def test_validate_artist_name_whitespace_only(self, cli_runner: CliRunner):
        """Test that whitespace-only artist name is rejected."""
        result = cli_runner.invoke(app, ["analyze", "   "])
        assert result.exit_code != 0

    def test_validate_artist_name_strips_whitespace(
        self, cli_runner: CliRunner, mock_settings, mock_artist_with_songs, mock_lyrics
    ):
        """Test that artist name whitespace is stripped."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.GeniusClient") as mock_client_class:
                mock_client = MagicMock(spec=GeniusClient)
                mock_client.get_artist_songs.return_value = mock_artist_with_songs
                mock_client.get_lyrics.return_value = mock_lyrics
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(
                    app, ["analyze", "  Test Artist  ", "-t", "5"]
                )

                assert result.exit_code == 0
