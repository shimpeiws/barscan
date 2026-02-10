"""Configuration management using Pydantic Settings."""

from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="BARSCAN_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    genius_access_token: SecretStr = SecretStr("")
    cache_dir: Path = Path.home() / ".cache" / "barscan"
    cache_ttl_hours: int = 168  # 1 week
    default_max_songs: int = 10
    default_top_words: int = 50

    @field_validator("cache_ttl_hours")
    @classmethod
    def validate_cache_ttl_hours(cls, v: int) -> int:
        """Validate cache TTL is at least 1 hour."""
        if v < 1:
            raise ValueError("cache_ttl_hours must be at least 1")
        return v

    @field_validator("default_max_songs")
    @classmethod
    def validate_default_max_songs(cls, v: int) -> int:
        """Validate default_max_songs is at least 1."""
        if v < 1:
            raise ValueError("default_max_songs must be at least 1")
        return v

    @field_validator("default_top_words")
    @classmethod
    def validate_default_top_words(cls, v: int) -> int:
        """Validate default_top_words is at least 1."""
        if v < 1:
            raise ValueError("default_top_words must be at least 1")
        return v

    def is_configured(self) -> bool:
        """Check if the required configuration is set."""
        return bool(self.genius_access_token.get_secret_value())

    def get_access_token(self) -> str:
        """Get the actual token value for API use."""
        return self.genius_access_token.get_secret_value()


settings = Settings()
