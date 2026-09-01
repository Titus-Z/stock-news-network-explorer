"""Configuration helpers for the Stock and News Network Explorer."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

DEFAULT_ALPHA_VANTAGE_API_KEY = None


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration shared across the project."""

    alpha_vantage_api_key: str | None
    openai_api_key: str | None
    alpha_vantage_base_url: str
    openai_base_url: str
    default_openai_model: str
    request_timeout: int
    default_news_limit: int
    default_price_period: str
    data_dir: Path
    raw_data_dir: Path
    backup_data_dir: Path


def load_config() -> AppConfig:
    """Load application configuration from environment variables."""

    # Keep environment parsing here so the rest of the project can depend on
    # one stable config object instead of repeatedly reading os.environ.
    timeout_value = os.getenv("MARKET_EXPLORER_TIMEOUT", "30")

    try:
        request_timeout = int(timeout_value)
    except ValueError:
        request_timeout = 30

    # The whole project reads and writes relative to one configurable data root.
    data_dir = Path(
        os.getenv("MARKET_EXPLORER_DATA_DIR", "data/demo")
    ).expanduser()

    return AppConfig(
        alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        alpha_vantage_base_url="https://www.alphavantage.co/query",
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        default_openai_model=os.getenv(
            "MARKET_EXPLORER_OPENAI_MODEL",
            "gpt-4o-mini",
        ),
        request_timeout=request_timeout,
        default_news_limit=50,
        default_price_period=os.getenv("MARKET_EXPLORER_PRICE_PERIOD", "1y"),
        data_dir=data_dir,
        raw_data_dir=data_dir / "raw",
        backup_data_dir=data_dir / "backups",
    )
