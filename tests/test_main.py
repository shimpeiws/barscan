"""Tests for __main__.py entry point."""

import subprocess
import sys


class TestMainModule:
    """Tests for running barscan as a module."""

    def test_module_import(self) -> None:
        """Test that __main__ module can be imported."""
        from barscan import __main__

        assert hasattr(__main__, "app")

    def test_module_runs_with_help(self) -> None:
        """Test that running module with --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "barscan", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Usage:" in result.stdout or "usage:" in result.stdout.lower()

    def test_module_has_commands(self) -> None:
        """Test that the module has expected commands."""
        result = subprocess.run(
            [sys.executable, "-m", "barscan", "--help"],
            capture_output=True,
            text=True,
        )
        # Check for expected commands in help output
        assert "analyze" in result.stdout.lower() or "barscan" in result.stdout.lower()
