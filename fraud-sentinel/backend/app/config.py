"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings shared by the API and its services."""

    frontend_url: str = "http://localhost:3000"
    anthropic_api_key: str = ""
    synthetic_seed: int = 42
    synthetic_days: int = 14
    synthetic_baseline_transactions: int = 20_000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()