"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings shared by the API and its services."""

    frontend_url: str = "http://localhost:3000"
    anthropic_api_key: str = ""
    synthetic_seed: int = 42
    synthetic_days: int = 14
    synthetic_baseline_transactions: int = 20_000
    rule_large_collect_amount_threshold: float = Field(default=25_000, gt=0)
    rule_rapid_transfer_window_seconds: int = Field(default=300, gt=0)
    rule_rapid_transfer_minimum_count: int = Field(default=3, ge=2)
    rule_circular_flow_window_seconds: int = Field(default=300, gt=0)
    rule_circular_flow_max_transfers: int = Field(default=4, ge=3, le=8)
    isolation_forest_contamination: float = Field(default=0.01, gt=0, le=0.5)
    isolation_forest_n_estimators: int = Field(default=200, ge=10)
    isolation_forest_random_state: int = 42
    lightgbm_test_size: float = Field(default=0.2, gt=0, lt=1)
    lightgbm_random_state: int = 42
    lightgbm_threshold: float = Field(default=0.5, ge=0, le=1)
    lightgbm_n_estimators: int = Field(default=100, ge=10)
    lightgbm_learning_rate: float = Field(default=0.05, gt=0, le=1)
    lightgbm_num_leaves: int = Field(default=7, ge=2)
    lightgbm_max_depth: int = Field(default=4, ge=1)
    lightgbm_min_child_samples: int = Field(default=5, ge=1)
    network_rapid_path_window_seconds: int = Field(default=300, gt=0)
    network_rapid_path_minimum_hops: int = Field(default=3, ge=2)
    network_rapid_path_maximum_hops: int = Field(default=6, ge=2, le=10)
    network_cycle_window_seconds: int = Field(default=300, gt=0)
    network_cycle_maximum_length: int = Field(default=4, ge=3, le=6)
    network_hub_degree_threshold: int = Field(default=31, ge=1)
    network_hub_minimum_transactions: int = Field(default=12, ge=1)
    risk_medium_threshold: float = Field(default=30, ge=0, le=100)
    risk_high_threshold: float = Field(default=70, ge=0, le=100)
    risk_alert_threshold: float = Field(default=30, ge=0, le=100)
    llm_enabled: bool = False
    llm_provider: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_seconds: float = Field(default=15, gt=0, le=120)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()