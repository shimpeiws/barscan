"""CLI entry point for BarScan."""

from __future__ import annotations

import json
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from barscan.analyzer.models import TokenWithPosition

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from barscan.analyzer import (
    AggregateAnalysisResult,
    AnalysisConfig,
    ContextsMode,
    WordFrequency,
    aggregate_results,
    analyze_text,
    collect_tokens_with_positions,
    get_word_counts_per_song,
)
from barscan.config import settings
from barscan.exceptions import (
    ArtistNotFoundError,
    BarScanError,
    GeniusAPIError,
    NoLyricsFoundError,
)
from barscan.genius import GeniusClient, LyricsCache
from barscan.logging import setup_logging
from barscan.output import (
    export_wordgrain,
    generate_filename,
    to_wordgrain,
    to_wordgrain_enhanced,
)

app = typer.Typer(
    name="barscan",
    help="Lyrics word frequency analyzer using Genius API.",
    add_completion=False,
)

console = Console()
error_console = Console(stderr=True)


class OutputFormat(StrEnum):
    """Output format options."""

    TABLE = "table"
    JSON = "json"
    CSV = "csv"
    WORDGRAIN = "wordgrain"


def validate_artist_name(value: str) -> str:
    """Validate artist name is not empty."""
    stripped = value.strip()
    if not stripped:
        raise typer.BadParameter("Artist name cannot be empty")
    return stripped


def validate_positive_int(value: int, name: str) -> int:
    """Validate integer is positive."""
    if value < 1:
        raise typer.BadParameter(f"{name} must be at least 1")
    return value


@app.command()
def analyze(
    artist: Annotated[
        str,
        typer.Argument(
            ...,
            help="Artist name to search",
            callback=lambda v: validate_artist_name(v),
        ),
    ],
    max_songs: Annotated[
        int,
        typer.Option(
            "--max-songs",
            "-n",
            help="Maximum songs to analyze",
            min=1,
        ),
    ] = 10,
    top_words: Annotated[
        int,
        typer.Option(
            "--top",
            "-t",
            help="Number of top words to show",
            min=1,
        ),
    ] = 50,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            "-f",
            help="Output format",
            case_sensitive=False,
        ),
    ] = OutputFormat.TABLE,
    output_file: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path (default: stdout)",
        ),
    ] = None,
    no_stop_words: Annotated[
        bool,
        typer.Option(
            "--no-stop-words",
            help="Disable stop word filtering",
        ),
    ] = False,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            "-e",
            help="Additional words to exclude (can be used multiple times)",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose debug output to stderr",
        ),
    ] = False,
    enhanced: Annotated[
        bool,
        typer.Option(
            "--enhanced",
            help="Enable enhanced NLP analysis (TF-IDF, POS, sentiment) for WordGrain output",
        ),
    ] = False,
    contexts_mode: Annotated[
        str,
        typer.Option(
            "--contexts-mode",
            "-c",
            help="Context mode: none (default), short (3-4 words), full (with metadata)",
        ),
    ] = "none",
    detect_slang: Annotated[
        bool,
        typer.Option(
            "--detect-slang",
            help="Enable slang word detection for WordGrain output",
        ),
    ] = False,
    min_count: Annotated[
        int,
        typer.Option(
            "--min-count",
            help="Minimum occurrence count to include",
            min=1,
        ),
    ] = 1,
) -> None:
    """Analyze word frequency in an artist's lyrics."""
    setup_logging(verbose=verbose)
    if not settings.is_configured():
        error_console.print(
            "[red]Error:[/red] Genius API token not configured.\n"
            "Set BARSCAN_GENIUS_ACCESS_TOKEN environment variable or add to .env file."
        )
        raise typer.Exit(1)

    try:
        client = GeniusClient()
    except GeniusAPIError as e:
        error_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    # Validate contexts_mode
    try:
        contexts_mode_enum = ContextsMode(contexts_mode)
    except ValueError:
        error_console.print(
            f"[red]Error:[/red] Invalid contexts mode '{contexts_mode}'. "
            "Valid options: none, short, full"
        )
        raise typer.Exit(1) from None

    # Build analysis config
    custom_stop_words = frozenset(exclude) if exclude else frozenset()
    config = AnalysisConfig(
        remove_stop_words=not no_stop_words,
        custom_stop_words=custom_stop_words,
        compute_tfidf=enhanced,
        compute_pos=enhanced,
        compute_sentiment=enhanced,
        detect_slang=detect_slang,
        contexts_mode=contexts_mode_enum,
        min_count=min_count,
    )

    # Track if we need enhanced data
    needs_enhanced = enhanced or detect_slang or contexts_mode_enum != ContextsMode.NONE

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            # Search for artist
            progress.add_task(f"Searching for {artist}...", total=None)
            artist_data = client.get_artist_songs(artist, max_songs=max_songs)

            if not artist_data.songs:
                error_console.print(f"[yellow]Warning:[/yellow] No songs found for {artist}")
                raise typer.Exit(0)

            console.print(
                f"Found [bold]{artist_data.artist.name}[/bold] with {len(artist_data.songs)} songs"
            )

            # Fetch lyrics and analyze
            task = progress.add_task(
                "Fetching lyrics...",
                total=len(artist_data.songs),
            )
            results = []
            lyrics_data: list[tuple[str, int, str]] = []  # For context extraction
            skipped = 0

            for song in artist_data.songs:
                progress.update(task, description=f"Analyzing: {song.title[:40]}...")
                try:
                    lyrics = client.get_lyrics(song)
                    if not lyrics.is_empty:
                        result = analyze_text(
                            text=lyrics.lyrics_text,
                            song_id=lyrics.song_id,
                            song_title=lyrics.song_title,
                            artist_name=lyrics.artist_name,
                            config=config,
                        )
                        results.append(result)
                        # Store lyrics data for enhanced analysis
                        if needs_enhanced:
                            lyrics_data.append(
                                (lyrics.lyrics_text, lyrics.song_id, lyrics.song_title)
                            )
                except NoLyricsFoundError:
                    skipped += 1
                progress.advance(task)

        if skipped > 0:
            console.print(f"[dim]Skipped {skipped} songs without lyrics[/dim]")

        if not results:
            error_console.print("[yellow]Warning:[/yellow] No lyrics found to analyze")
            raise typer.Exit(0)

        # Aggregate results
        aggregate = aggregate_results(results, artist_data.artist.name, config)
        top_frequencies = aggregate.top_words(top_words)

        # Prepare enhanced data if needed
        word_counts_per_song = None
        tokens_with_positions = None
        if needs_enhanced and output_format == OutputFormat.WORDGRAIN:
            word_counts_per_song = get_word_counts_per_song(results)
            if contexts_mode_enum != ContextsMode.NONE:
                tokens_with_positions = collect_tokens_with_positions(lyrics_data, config)

        # Output results
        output_content = format_output(
            artist_name=aggregate.artist_name,
            songs_analyzed=aggregate.songs_analyzed,
            total_words=aggregate.total_words,
            unique_words=aggregate.unique_words,
            frequencies=list(top_frequencies),
            output_format=output_format,
            aggregate=aggregate,
            config=config if needs_enhanced else None,
            word_counts_per_song=word_counts_per_song,
            tokens_with_positions=tokens_with_positions,
        )

        if output_file:
            output_file.write_text(output_content, encoding="utf-8")
            console.print(f"Results written to [bold]{output_file}[/bold]")
        else:
            if output_format == OutputFormat.TABLE:
                display_table(
                    artist_name=aggregate.artist_name,
                    songs_analyzed=aggregate.songs_analyzed,
                    total_words=aggregate.total_words,
                    unique_words=aggregate.unique_words,
                    frequencies=list(top_frequencies),
                )
            elif output_format == OutputFormat.WORDGRAIN:
                suggested_filename = generate_filename(aggregate.artist_name)
                console.print(f"[dim]Suggested filename: {suggested_filename}[/dim]")
                console.print(output_content)
            else:
                console.print(output_content)

    except ArtistNotFoundError:
        error_console.print(f"[red]Error:[/red] Artist not found: {artist}")
        raise typer.Exit(1) from None
    except GeniusAPIError as e:
        error_console.print(f"[red]Error:[/red] API error: {e}")
        raise typer.Exit(1) from None
    except BarScanError as e:
        error_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


def format_output(
    artist_name: str,
    songs_analyzed: int,
    total_words: int,
    unique_words: int,
    frequencies: list[WordFrequency],
    output_format: OutputFormat,
    aggregate: AggregateAnalysisResult | None = None,
    config: AnalysisConfig | None = None,
    word_counts_per_song: list[Counter[str]] | None = None,
    tokens_with_positions: list[TokenWithPosition] | None = None,
) -> str:
    """Format analysis results for output."""
    if output_format == OutputFormat.WORDGRAIN:
        if aggregate is None:
            raise ValueError("aggregate is required for WORDGRAIN format")

        # Use enhanced output if config is provided
        if config is not None:
            document = to_wordgrain_enhanced(
                aggregate=aggregate,
                config=config,
                word_counts_per_song=word_counts_per_song,
                tokens_with_positions=tokens_with_positions,
            )
        else:
            document = to_wordgrain(aggregate)
        return export_wordgrain(document)

    if output_format == OutputFormat.JSON:
        data = {
            "artist": artist_name,
            "songs_analyzed": songs_analyzed,
            "total_words": total_words,
            "unique_words": unique_words,
            "frequencies": [
                {"word": f.word, "count": f.count, "percentage": f.percentage} for f in frequencies
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    elif output_format == OutputFormat.CSV:
        lines = ["word,count,percentage"]
        for f in frequencies:
            lines.append(f"{f.word},{f.count},{f.percentage}")
        return "\n".join(lines)

    else:  # TABLE format for file output
        lines = [
            f"Artist: {artist_name}",
            f"Songs analyzed: {songs_analyzed}",
            f"Total words: {total_words}",
            f"Unique words: {unique_words}",
            "",
            "Rank\tWord\tCount\tPercentage",
        ]
        for i, f in enumerate(frequencies, 1):
            lines.append(f"{i}\t{f.word}\t{f.count}\t{f.percentage:.2f}%")
        return "\n".join(lines)


def display_table(
    artist_name: str,
    songs_analyzed: int,
    total_words: int,
    unique_words: int,
    frequencies: list[WordFrequency],
) -> None:
    """Display results as a Rich table."""
    console.print()
    console.print(f"[bold]Artist:[/bold] {artist_name}")
    console.print(f"[bold]Songs analyzed:[/bold] {songs_analyzed}")
    console.print(f"[bold]Total words:[/bold] {total_words:,}")
    console.print(f"[bold]Unique words:[/bold] {unique_words:,}")
    console.print()

    table = Table(title="Word Frequencies")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Word", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right")

    for i, f in enumerate(frequencies, 1):
        table.add_row(str(i), f.word, str(f.count), f"{f.percentage:.2f}%")

    console.print(table)


@app.command()
def clear_cache(
    expired_only: Annotated[
        bool,
        typer.Option(
            "--expired-only",
            help="Only clear expired cache entries",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Skip confirmation prompt",
        ),
    ] = False,
) -> None:
    """Clear the local lyrics cache."""
    cache = LyricsCache(
        cache_dir=settings.cache_dir,
        ttl_hours=settings.cache_ttl_hours,
    )

    stats = cache.get_stats()
    if stats["total_entries"] == 0:
        console.print("Cache is already empty.")
        return

    if expired_only:
        if stats["expired"] == 0:
            console.print("No expired cache entries found.")
            return
        if not force:
            confirm = typer.confirm(f"Clear {stats['expired']} expired cache entries?")
            if not confirm:
                console.print("Cancelled.")
                raise typer.Exit(0)
        count = cache.clear_expired()
        console.print(f"Cleared {count} expired cache entries.")
    else:
        if not force:
            confirm = typer.confirm(
                f"Clear all {stats['total_entries']} cache entries ({stats['size_bytes']:,} bytes)?"
            )
            if not confirm:
                console.print("Cancelled.")
                raise typer.Exit(0)
        count = cache.clear()
        console.print(f"Cleared {count} cache entries.")


@app.command()
def config() -> None:
    """Show current configuration."""
    table = Table(title="BarScan Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    # API Token (masked)
    token = settings.get_access_token()
    if token:
        masked = token[:4] + "*" * (len(token) - 8) + token[-4:] if len(token) > 8 else "****"
        token_display = f"[green]{masked}[/green]"
    else:
        token_display = "[red]Not set[/red]"
    table.add_row("BARSCAN_GENIUS_ACCESS_TOKEN", token_display)

    table.add_row("BARSCAN_CACHE_DIR", str(settings.cache_dir))
    table.add_row("BARSCAN_CACHE_TTL_HOURS", str(settings.cache_ttl_hours))
    table.add_row("BARSCAN_DEFAULT_MAX_SONGS", str(settings.default_max_songs))
    table.add_row("BARSCAN_DEFAULT_TOP_WORDS", str(settings.default_top_words))

    console.print(table)

    # Show cache stats
    cache = LyricsCache(
        cache_dir=settings.cache_dir,
        ttl_hours=settings.cache_ttl_hours,
    )
    stats = cache.get_stats()

    console.print()
    console.print("[bold]Cache Statistics:[/bold]")
    console.print(f"  Entries: {stats['total_entries']}")
    console.print(f"  Size: {stats['size_bytes']:,} bytes")
    console.print(f"  Expired: {stats['expired']}")


if __name__ == "__main__":
    app()
