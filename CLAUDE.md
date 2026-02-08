# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BarScan is a Python CLI tool that analyzes word frequency in song lyrics using the Genius API. It fetches lyrics for artists via the lyricsgenius library, processes them with NLTK, and outputs frequency analysis.

## Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest

# Run single test file
pytest tests/test_genius/test_client.py -v

# Run single test
pytest tests/test_genius/test_client.py::TestSearchArtist::test_search_artist_success -v

# Lint
ruff check src/

# Format
ruff format src/

# Type check (use --ignore-missing-imports for external libs without stubs)
mypy src/barscan/ --ignore-missing-imports
```

## Architecture

```
src/barscan/
├── cli.py          # Typer CLI entry point (barscan command)
├── config.py       # Pydantic Settings (env: BARSCAN_*)
├── exceptions.py   # Exception hierarchy (BarScanError base)
├── genius/         # Genius API integration
│   ├── models.py   # Pydantic models (Artist, Song, Lyrics)
│   ├── client.py   # GeniusClient wrapper with retry logic
│   └── cache.py    # File-based lyrics cache with TTL
├── analyzer/       # Word frequency analysis (not yet implemented)
└── output/         # Result formatting (not yet implemented)
```

## Key Patterns

- **Configuration**: All settings use `BARSCAN_` prefix (e.g., `BARSCAN_GENIUS_ACCESS_TOKEN`)
- **Models**: Pydantic models with `frozen=True` for immutability where appropriate
- **Exceptions**: Domain-specific exceptions inherit from `BarScanError`
- **Cache**: JSON-based file cache at `~/.cache/barscan/` with configurable TTL
- **Testing**: Mock the `lyricsgenius.Genius` class in client tests

## Git Conventions

- Use English only for Issues, PRs, and commit messages (no Japanese)
- Do not include `Co-Authored-By` in commit messages
- Do not include `Co-Authored-By` or AI attribution in PR descriptions
