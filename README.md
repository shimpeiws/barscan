# BarScan

A Python CLI tool that analyzes word frequency in song lyrics using the Genius API.

## Features

- Fetch lyrics for any artist from the Genius API
- Analyze word frequency across multiple songs
- Natural language processing with NLTK for accurate tokenization
- Customizable stop word filtering and exclusions
- Multiple output formats: table, JSON, CSV, and WordGrain
- Local caching to reduce API calls and improve performance
- Retry logic with exponential backoff for robust API communication

## Installation

### Prerequisites

- Python 3.11 or higher
- pip (latest version recommended)

### From PyPI

```bash
pip install barscan
```

#### With Japanese Support

To analyze Japanese lyrics, install with the `japanese` extra:

```bash
pip install barscan[japanese]
```

This includes [Janome](https://mocobeta.github.io/janome/) for Japanese tokenization and additional stop words.

### From Source

```bash
git clone https://github.com/shimpeiws/barscan.git
cd barscan
pip install -e ".[dev]"
```

## Setup

### Getting a Genius API Token

1. Go to [Genius API Clients](https://genius.com/api-clients)
2. Sign in with your Genius account (or create one)
3. Click "Create an API Client"
4. Fill in the app details:
   - App Name: Any name (e.g., "BarScan CLI")
   - App Website URL: Any URL (e.g., your GitHub profile)
   - Redirect URI: Leave default or use `http://localhost`
5. Click "Save"
6. Copy the "Client Access Token" (not the Client ID or Secret)

### Configuring the Token

Set the token as an environment variable:

```bash
export BARSCAN_GENIUS_ACCESS_TOKEN=your_token_here
```

Or create a `.env` file in your project directory:

```bash
BARSCAN_GENIUS_ACCESS_TOKEN=your_token_here
```

## Usage

### Basic Analysis

Analyze the most common words in an artist's lyrics:

```bash
barscan analyze "Kendrick Lamar"
```

### Command Options

```bash
# Analyze more songs
barscan analyze "Drake" --max-songs 20

# Show more words in results
barscan analyze "J. Cole" --top 100

# Combine options
barscan analyze "Tyler, The Creator" -n 15 -t 50
```

### Output Formats

```bash
# Default table format (console)
barscan analyze "Beyonce"

# JSON format
barscan analyze "Beyonce" --format json

# CSV format
barscan analyze "Beyonce" --format csv

# WordGrain format (structured JSON schema)
barscan analyze "Beyonce" --format wordgrain

# Save to file
barscan analyze "Beyonce" --format json --output results.json
```

### Filtering Options

```bash
# Disable stop word filtering (include "the", "a", "is", etc.)
barscan analyze "Eminem" --no-stop-words

# Exclude specific words
barscan analyze "Eminem" --exclude "yeah" --exclude "oh"

# Combine exclusions
barscan analyze "Eminem" -e "uh" -e "like" -e "yo"
```

### Cache Management

BarScan caches lyrics locally to reduce API calls:

```bash
# Clear all cached lyrics
barscan clear-cache --force

# Clear only expired cache entries
barscan clear-cache --expired-only --force

# Interactive confirmation (without --force)
barscan clear-cache
```

### View Configuration

```bash
# Show current configuration and cache statistics
barscan config
```

## Configuration Options

All settings can be configured via environment variables with the `BARSCAN_` prefix:

| Variable | Description | Default |
|----------|-------------|---------|
| `BARSCAN_GENIUS_ACCESS_TOKEN` | Genius API access token | (required) |
| `BARSCAN_CACHE_DIR` | Directory for caching lyrics | `~/.cache/barscan` |
| `BARSCAN_CACHE_TTL_HOURS` | Cache time-to-live in hours | `168` (7 days) |
| `BARSCAN_DEFAULT_MAX_SONGS` | Default number of songs to analyze | `10` |
| `BARSCAN_DEFAULT_TOP_WORDS` | Default number of top words to show | `50` |

## Output Formats

### Table Format (default)

Human-readable table with word rankings:

```
Artist: Kendrick Lamar
Songs analyzed: 10
Total words: 5,432
Unique words: 1,203

Word Frequencies
┌──────┬─────────┬───────┬────────────┐
│ Rank │ Word    │ Count │ Percentage │
├──────┼─────────┼───────┼────────────┤
│ 1    │ love    │ 87    │ 1.60%      │
│ 2    │ know    │ 65    │ 1.20%      │
│ ...  │ ...     │ ...   │ ...        │
└──────┴─────────┴───────┴────────────┘
```

### JSON Format

Structured JSON for programmatic use:

```json
{
  "artist": "Kendrick Lamar",
  "songs_analyzed": 10,
  "total_words": 5432,
  "unique_words": 1203,
  "frequencies": [
    {"word": "love", "count": 87, "percentage": 1.60},
    {"word": "know", "count": 65, "percentage": 1.20}
  ]
}
```

### CSV Format

Comma-separated values for spreadsheet import:

```csv
word,count,percentage
love,87,1.60
know,65,1.20
```

### WordGrain Format

[WordGrain](https://github.com/shimpeiws/word-grain) is a standardized JSON schema for vocabulary analysis data. It enables interoperability between different word frequency analysis tools. See the [documentation](https://shimpeiws.github.io/word-grain/) for details.

Output example:

```json
{
  "$schema": "https://raw.githubusercontent.com/shimpeiws/word-grain/main/schema/v0.1.0/wordgrain.schema.json",
  "meta": {
    "source": "genius",
    "artist": "Kendrick Lamar",
    "generated_at": "2024-01-15T10:30:00Z",
    "corpus_size": 10,
    "total_words": 5432,
    "generator": "barscan/0.1.0",
    "language": "en"
  },
  "grains": [
    {"word": "love", "frequency": 87, "frequency_normalized": 160.18}
  ]
}
```

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/shimpeiws/barscan.git
cd barscan

# Install with development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_genius/test_client.py -v

# Run specific test
pytest tests/test_genius/test_client.py::TestSearchArtist::test_search_artist_success -v
```

### Code Quality

```bash
# Lint code
ruff check src/

# Format code
ruff format src/

# Type check
mypy src/barscan/ --ignore-missing-imports
```

## Architecture

```
src/barscan/
├── cli.py          # Typer CLI entry point (barscan command)
├── config.py       # Pydantic Settings configuration
├── exceptions.py   # Exception hierarchy (BarScanError base)
├── genius/         # Genius API integration
│   ├── models.py   # Pydantic models (Artist, Song, Lyrics)
│   ├── client.py   # GeniusClient with retry logic
│   └── cache.py    # File-based lyrics cache with TTL
├── analyzer/       # Word frequency analysis
│   ├── models.py   # Analysis result models
│   ├── processor.py # Text preprocessing with NLTK
│   ├── filters.py  # Stop word and length filtering
│   └── frequency.py # Word counting and aggregation
└── output/         # Result formatting
    └── wordgrain.py # WordGrain schema export
```

## Troubleshooting

### "Genius API token not configured"

Make sure you've set the `BARSCAN_GENIUS_ACCESS_TOKEN` environment variable or created a `.env` file with the token.

### "Artist not found"

- Check the spelling of the artist name
- Try using the artist's name exactly as it appears on Genius
- Some artists may have limited or no presence on Genius

### Rate Limiting

BarScan includes automatic retry logic with exponential backoff. If you encounter rate limiting:

- The tool will automatically retry failed requests
- Consider reducing `--max-songs` for large analyses
- Cached lyrics won't trigger new API calls

### Empty Results

If no words appear in results after filtering:

- Try `--no-stop-words` to include common words
- Check if the artist has lyrics available on Genius
- Some songs may be instrumental or have no lyrics

## License

MIT
