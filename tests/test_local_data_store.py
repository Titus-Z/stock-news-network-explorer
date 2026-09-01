"""Tests for local cached-data loading."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from config import AppConfig
from local_data_store import LocalDataStore, LocalDataStoreError


def build_store(tmp_path: Path) -> LocalDataStore:
    """Create a local data store with a temporary data directory."""

    data_dir = tmp_path / "data"
    config = AppConfig(
        alpha_vantage_api_key=None,
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
    return LocalDataStore(config=config)


def test_load_price_tables_reads_cached_csvs(tmp_path: Path) -> None:
    """The store should load local price CSV files into a ticker dictionary."""

    store = build_store(tmp_path)
    price_dir = tmp_path / "data" / "raw" / "prices"
    price_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "adjusted_close": [100.0],
        }
    ).to_csv(price_dir / "AAPL_prices.csv", index=False)

    prices = store.load_price_tables()

    assert "AAPL" in prices
    assert len(prices["AAPL"]) == 1


def test_load_sector_info_reads_cached_json(tmp_path: Path) -> None:
    """The store should load local sector JSON files into one DataFrame."""

    store = build_store(tmp_path)
    sector_dir = tmp_path / "data" / "raw" / "sectors"
    sector_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
    }
    (sector_dir / "AAPL_sector.json").write_text(json.dumps(payload), encoding="utf-8")

    sector_info = store.load_sector_info()

    assert len(sector_info) == 1
    assert sector_info.loc[0, "sector"] == "Technology"


def test_load_news_payload_reads_merged_news_file(tmp_path: Path) -> None:
    """The store should load one local news JSON payload."""

    store = build_store(tmp_path)
    news_dir = tmp_path / "data" / "raw" / "news"
    news_dir.mkdir(parents=True, exist_ok=True)

    payload = {"items": "0", "feed": []}
    (news_dir / "merged_seed_news.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    loaded_payload = store.load_news_payload()

    assert loaded_payload["feed"] == []


def test_load_processed_news_tables_builds_analysis_frames(tmp_path: Path) -> None:
    """The store should expose processed news tables for downstream analysis."""

    store = build_store(tmp_path)
    news_dir = tmp_path / "data" / "raw" / "news"
    news_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "items": "1",
        "feed": [
            {
                "title": "Apple launches a new product",
                "url": "https://example.com/a1",
                "time_published": "20260102T103000",
                "source": "Reuters",
                "summary": "Apple announced a new product.",
                "overall_sentiment_score": "0.5",
                "overall_sentiment_label": "Bullish",
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": "0.9",
                        "ticker_sentiment_score": "0.5",
                        "ticker_sentiment_label": "Bullish",
                    }
                ],
                "topics": [
                    {
                        "topic": "Technology",
                        "relevance_score": "0.8",
                    }
                ],
            }
        ],
    }
    (news_dir / "merged_seed_news.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    tables = store.load_processed_news_tables()

    assert set(tables) == {"articles", "article_tickers", "topic_stock"}
    assert len(tables["articles"]) == 1
    assert tables["article_tickers"].loc[0, "ticker"] == "AAPL"
    assert tables["topic_stock"].loc[0, "topic"] == "Technology"


def test_missing_price_directory_raises_clear_error(tmp_path: Path) -> None:
    """Missing local data should raise a clear analysis-time error."""

    store = build_store(tmp_path)

    with pytest.raises(LocalDataStoreError, match="Price directory not found"):
        store.load_price_tables()
