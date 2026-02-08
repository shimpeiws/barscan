"""Configuration management using Pydantic Settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="BARSCAN_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    genius_access_token: str = ""
    cache_dir: Path = Path.home() / ".cache" / "barscan"
    cache_ttl_hours: int = 168  # 1 week
    default_max_songs: int = 10
    default_top_words: int = 50

    def is_configured(self) -> bool:
        """Check if the required configuration is set."""
        return bool(self.genius_access_token)


settings = Settings()
