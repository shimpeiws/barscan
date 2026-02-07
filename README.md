# BarScan

Lyrics word frequency analyzer using Genius API.

## Features

- Fetch lyrics from Genius API for any artist
- Analyze word frequency across songs
- Filter stop words and customize exclusions
- Export results to console, JSON, or CSV

## Installation

```bash
pip install barscan
```

Or install from source:

```bash
git clone https://github.com/shimpeiws/barscan.git
cd barscan
pip install -e ".[dev]"
```

## Setup

1. Get your Genius API access token at https://genius.com/api-clients
2. Create a `.env` file or set the environment variable:

```bash
export BARSCAN_GENIUS_ACCESS_TOKEN=your_token_here
```

## Usage

```bash
# Basic usage
barscan analyze "Kendrick Lamar"

# With options
barscan analyze "Drake" --max-songs 20 --top 100

# Export to JSON
barscan analyze "J. Cole" --format json -o results.json

# Clear cache
barscan clear-cache

# Show configuration
barscan config
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check src/

# Run type checker
mypy src/
```

## License

MIT
