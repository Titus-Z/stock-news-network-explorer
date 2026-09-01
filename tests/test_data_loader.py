"""Tests for Phase 1 market data loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from config import AppConfig
from data_loader import DataLoaderError, MarketDataLoader


class DummyResponse:
    """Small fake response object for unit tests."""

    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """Raise an HTTP error when the fake status code is failing."""

        if self.status_code >= 400:
            raise requests.HTTPError(f"Status code: {self.status_code}")

    def json(self) -> Any:
        """Return the prepared JSON payload."""

        return self.payload


class DummySession:
    """Small fake session object for tests that need request control."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def get(self, url: str, params: dict[str, Any], timeout: int) -> DummyResponse:
        """Return a fake response regardless of input."""

        return DummyResponse(self.payload)


def build_loader(tmp_path: Path) -> MarketDataLoader:
    """Create a loader with a safe test config."""

    data_dir = tmp_path / "data"
    config = AppConfig(
        alpha_vantage_api_key="demo",
        openai_api_key="demo-openai",
        alpha_vantage_base_url="https://www.alphavantage.co/query",
        openai_base_url="https://api.openai.com/v1",
        default_openai_model="gpt-4o-mini",
        request_timeout=30,
        default_news_limit=50,
        default_price_period="1y",
        data_dir=data_dir,
        raw_data_dir=data_dir / "raw",
        backup_data_dir=data_dir / "backups",
    )
    return MarketDataLoader(config=config)


def test_fetch_stock_price_data_parses_yfinance_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Price loading should parse yfinance rows into a DataFrame."""

    def fake_history(
        self: MarketDataLoader,
        ticker: str,
        period: str,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": [99.0, 101.0],
                "High": [102.0, 104.0],
                "Low": [98.5, 100.5],
                "Close": [100.5, 103.5],
                "Adj Close": [100.2, 103.1],
                "Volume": [1500, 2000],
                "Dividends": [0.0, 0.0],
                "Stock Splits": [0.0, 0.0],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )

    monkeypatch.setattr(
        MarketDataLoader,
        "_download_yfinance_history",
        fake_history,
    )

    loader = build_loader(tmp_path)
    prices = loader.fetch_stock_price_data("aapl", period="1y", use_cache=False)

    assert list(prices["date"].dt.strftime("%Y-%m-%d")) == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert prices.loc[1, "adjusted_close"] == pytest.approx(103.1)
    assert prices.loc[0, "ticker"] == "AAPL"
    assert (tmp_path / "data" / "raw" / "prices" / "AAPL_prices.csv").exists()


def test_fetch_stock_price_data_uses_cached_file(tmp_path: Path) -> None:
    """Cached price files should be reused when available."""

    loader = build_loader(tmp_path)
    cache_path = tmp_path / "data" / "raw" / "prices" / "AAPL_prices.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "open": [99.0],
            "high": [102.0],
            "low": [98.5],
            "close": [100.5],
            "adjusted_close": [100.2],
            "volume": [1500],
            "dividend_amount": [0.0],
            "split_coefficient": [0.0],
            "ticker": ["AAPL"],
        }
    ).to_csv(cache_path, index=False)

    prices = loader.fetch_stock_price_data("AAPL")

    assert len(prices) == 1
    assert prices.loc[0, "ticker"] == "AAPL"


def test_fetch_stock_price_data_rejects_bad_period(tmp_path: Path) -> None:
    """Invalid price periods should fail early."""

    loader = build_loader(tmp_path)

    with pytest.raises(DataLoaderError, match="Unsupported period"):
        loader.fetch_stock_price_data("AAPL", period="12months")


def test_fetch_news_json_accepts_empty_feed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An empty news feed should be allowed and returned to the caller."""

    def fake_alpha_request(
        self: MarketDataLoader,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "items": "0",
            "feed": [],
        }

    monkeypatch.setattr(
        MarketDataLoader,
        "_get_alpha_vantage_json",
        fake_alpha_request,
    )

    loader = build_loader(tmp_path)
    payload = loader.fetch_news_json(["AAPL", "MSFT"], limit=10, use_cache=False)

    assert payload["feed"] == []
    assert (
        tmp_path
        / "data"
        / "raw"
        / "news"
        / "AAPL_MSFT_all_topics_latest_10.json"
    ).exists()


def test_fetch_news_json_rejects_non_positive_limit(tmp_path: Path) -> None:
    """The loader should reject impossible limit values."""

    loader = build_loader(tmp_path)

    with pytest.raises(DataLoaderError, match="positive integer"):
        loader.fetch_news_json("AAPL", limit=0)


def test_load_sector_info_uses_yfinance_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Sector loading should use yfinance metadata and save it locally."""

    def fake_metadata(
        self: MarketDataLoader,
        ticker: str,
    ) -> dict[str, Any]:
        if ticker == "AAPL":
            return {
                "company_name": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "source": "yfinance",
            }

        return {
            "company_name": "Tesla, Inc.",
            "sector": "Consumer Cyclical",
            "industry": "Auto Manufacturers",
            "source": "yfinance",
        }

    monkeypatch.setattr(
        MarketDataLoader,
        "_fetch_yfinance_metadata",
        fake_metadata,
    )

    loader = build_loader(tmp_path)
    sector_info = loader.load_sector_info(["AAPL", "TSLA"], use_cache=False)

    assert sector_info.loc[0, "source"] == "yfinance"
    assert sector_info.loc[1, "sector"] == "Consumer Cyclical"
    assert (tmp_path / "data" / "raw" / "sectors" / "AAPL_sector.json").exists()


def test_load_sector_info_handles_missing_sector_info(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing sector data should not crash the batch load."""

    monkeypatch.setattr(
        MarketDataLoader,
        "_fetch_yfinance_metadata",
        lambda self, ticker: {},
    )

    loader = build_loader(tmp_path)
    sector_info = loader.load_sector_info(["XYZ"], use_cache=False)

    assert sector_info.loc[0, "ticker"] == "XYZ"
    assert pd.isna(sector_info.loc[0, "sector"])
    assert sector_info.loc[0, "source"] == "missing"


def test_get_alpha_vantage_json_raises_on_empty_response(tmp_path: Path) -> None:
    """Completely empty Alpha Vantage payloads should raise a clear error."""

    session = DummySession(payload={})
    data_dir = tmp_path / "data"
    config = AppConfig(
        alpha_vantage_api_key="demo",
        openai_api_key="demo-openai",
        alpha_vantage_base_url="https://www.alphavantage.co/query",
        openai_base_url="https://api.openai.com/v1",
        default_openai_model="gpt-4o-mini",
        request_timeout=30,
        default_news_limit=50,
        default_price_period="1y",
        data_dir=data_dir,
        raw_data_dir=data_dir / "raw",
        backup_data_dir=data_dir / "backups",
    )
    loader = MarketDataLoader(config=config, session=session)

    with pytest.raises(DataLoaderError, match="empty response"):
        loader._get_alpha_vantage_json({"function": "OVERVIEW", "symbol": "AAPL"})
