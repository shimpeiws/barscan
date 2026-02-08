"""Tests for CLI commands."""

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from barscan.cli import app


class TestConfigCommand:
    """Tests for the config command."""

    def test_config_shows_settings(self, cli_runner: CliRunner, mock_settings, temp_cache_dir):
        """Test config command displays settings."""
        with patch("barscan.cli.settings", mock_settings):
            with patch("barscan.cli.LyricsCache") as mock_cache_class:
                mock_cache = MagicMock()
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
                mock_cache = MagicMock()
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
                mock_cache = MagicMock()
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
                mock_cache = MagicMock()
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
                mock_cache = MagicMock()
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
                mock_cache = MagicMock()
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
                mock_cache = MagicMock()
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

                mock_client = MagicMock()
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
                mock_client = MagicMock()
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
                mock_client = MagicMock()
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
                mock_client = MagicMock()
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
                mock_client = MagicMock()
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
                mock_client = MagicMock()
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
                mock_client = MagicMock()
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
                mock_client = MagicMock()
                mock_client.get_artist_songs.return_value = mock_artist_with_songs
                mock_client.get_lyrics.return_value = mock_lyrics
                mock_client_class.return_value = mock_client

                result = cli_runner.invoke(
                    app, ["analyze", "Test Artist", "--no-stop-words", "-t", "5"]
                )

                assert result.exit_code == 0
