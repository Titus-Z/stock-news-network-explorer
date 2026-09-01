"""Local file loading helpers for analysis-time workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from config import AppConfig, load_config
from news_processor import NewsProcessor


class LocalDataStoreError(Exception):
    """Raised when required local analysis data is missing or malformed."""


class LocalDataStore:
    """Load cached project data from the local filesystem."""

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialize the store from application config."""

        self.config = config or load_config()

    def load_price_tables(
        self,
        tickers: Sequence[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load cached price tables from the local raw-data directory."""

        selected_tickers = self._normalize_ticker_list(tickers)
        price_dir = self.config.raw_data_dir / "prices"
        if not price_dir.exists():
            raise LocalDataStoreError(
                f"Price directory not found: {price_dir}. "
                "Run seed_data.py first."
            )

        price_tables: dict[str, pd.DataFrame] = {}
        for file_path in sorted(price_dir.glob("*_prices.csv")):
            ticker = file_path.name.replace("_prices.csv", "").upper()
            if selected_tickers and ticker not in selected_tickers:
                continue

            # Analysis code expects parsed dates so this happens once here,
            # instead of repeating date parsing in multiple downstream modules.
            prices = pd.read_csv(file_path, parse_dates=["date"])
            price_tables[ticker] = prices

        if not price_tables:
            raise LocalDataStoreError(
                "No local price tables were found for the requested tickers."
            )

        return price_tables

    def load_sector_info(
        self,
        tickers: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Load cached sector metadata from the local raw-data directory."""

        selected_tickers = self._normalize_ticker_list(tickers)
        sector_dir = self.config.raw_data_dir / "sectors"
        if not sector_dir.exists():
            raise LocalDataStoreError(
                f"Sector directory not found: {sector_dir}. "
                "Run seed_data.py first."
            )

        rows: list[dict[str, Any]] = []
        for file_path in sorted(sector_dir.glob("*_sector.json")):
            ticker = file_path.name.replace("_sector.json", "").upper()
            if selected_tickers and ticker not in selected_tickers:
                continue

            with file_path.open("r", encoding="utf-8") as file_handle:
                rows.append(json.load(file_handle))

        sector_info = pd.DataFrame(rows)
        if sector_info.empty:
            raise LocalDataStoreError(
                "No local sector files were found for the requested tickers."
            )

        return sector_info

    def load_news_payload(self, file_name: str = "merged_seed_news.json") -> dict[str, Any]:
        """Load one cached news payload from the local raw-data directory."""

        file_path = self.config.raw_data_dir / "news" / file_name
        if not file_path.exists():
            raise LocalDataStoreError(
                f"News file not found: {file_path}. Run seed_data.py first."
            )

        with file_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        # The rest of the processing pipeline is written against one JSON object
        # that contains a NEWS_SENTIMENT-style "feed" list.
        if not isinstance(payload, dict):
            raise LocalDataStoreError(
                f"Expected a JSON object in local news file: {file_path}"
            )

        return payload

    def load_processed_news_tables(
        self,
        file_name: str = "merged_seed_news.json",
    ) -> dict[str, pd.DataFrame]:
        """Load and process one cached news payload into analysis tables."""

        payload = self.load_news_payload(file_name=file_name)
        return NewsProcessor().process_news_payload(payload)

    def list_llm_impact_runs(self) -> list[str]:
        """List available LLM impact run stems from the local raw-data directory."""

        llm_dir = self.config.raw_data_dir / "llm_enriched"
        if not llm_dir.exists():
            return []

        run_stems: list[str] = []
        for file_path in sorted(llm_dir.glob("*_article_llm_summary.csv")):
            run_stems.append(file_path.name.replace("_article_llm_summary.csv", ""))

        return sorted(run_stems, reverse=True)

    def load_llm_impact_tables(self, file_stem: str) -> dict[str, pd.DataFrame]:
        """Load one set of cached LLM impact CSV tables."""

        llm_dir = self.config.raw_data_dir / "llm_enriched"
        if not llm_dir.exists():
            raise LocalDataStoreError(
                f"LLM enrichment directory not found: {llm_dir}. "
                "Run enrich_news_with_llm.py first."
            )

        table_map = {
            "article_llm_summary": llm_dir / f"{file_stem}_article_llm_summary.csv",
            "article_sector_impacts": llm_dir / f"{file_stem}_article_sector_impacts.csv",
            "article_stock_impacts": llm_dir / f"{file_stem}_article_stock_impacts.csv",
        }

        tables: dict[str, pd.DataFrame] = {}
        for table_name, file_path in table_map.items():
            if not file_path.exists():
                raise LocalDataStoreError(
                    f"LLM impact file not found: {file_path}. "
                    "Choose an existing run stem from the configured data root."
                )
            tables[table_name] = pd.read_csv(file_path)

        return tables

    def _normalize_ticker_list(
        self,
        tickers: Sequence[str] | None,
    ) -> set[str]:
        """Normalize an optional ticker list into a lookup set."""

        if tickers is None:
            return set()

        normalized: set[str] = set()
        for ticker in tickers:
            if not isinstance(ticker, str):
                raise LocalDataStoreError("Ticker filters must be strings.")

            cleaned = ticker.strip().upper()
            if cleaned:
                normalized.add(cleaned)

        return normalized
