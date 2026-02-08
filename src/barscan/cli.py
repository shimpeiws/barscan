"""CLI entry point for BarScan."""

import typer

app = typer.Typer(
    name="barscan",
    help="Lyrics word frequency analyzer using Genius API.",
    add_completion=False,
)


@app.command()
def analyze(
    artist: str = typer.Argument(..., help="Artist name to search"),
    max_songs: int = typer.Option(10, "--max-songs", "-n", help="Maximum songs to analyze"),
    top_words: int = typer.Option(50, "--top", "-t", help="Number of top words to show"),
) -> None:
    """Analyze word frequency in an artist's lyrics."""
    typer.echo(f"Analyzing lyrics for: {artist}")
    typer.echo(f"Max songs: {max_songs}, Top words: {top_words}")
    typer.echo("Not implemented yet.")


@app.command()
def clear_cache() -> None:
    """Clear the local lyrics cache."""
    typer.echo("Cache cleared.")


@app.command()
def config() -> None:
    """Show current configuration."""
    typer.echo("Configuration:")
    typer.echo("Not implemented yet.")


if __name__ == "__main__":
    app()
