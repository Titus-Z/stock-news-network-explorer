"""Data loading utilities for stock prices, news, and sector metadata."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests

from config import AppConfig, load_config

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - covered through runtime validation
    yf = None


class DataLoaderError(Exception):
    """Raised when external market data cannot be loaded or validated."""


class MarketDataLoader:
    """Load market data needed for the project."""

    VALID_NEWS_SORTS = {"LATEST", "EARLIEST", "RELEVANCE"}
    VALID_PRICE_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"}

    def __init__(
        self,
        config: AppConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the loader with config and an optional HTTP session."""

        self.config = config or load_config()
        # Accept an injected session so tests can replace real HTTP requests
        # with deterministic fake responses.
        self.session = session or requests.Session()
        self._ensure_storage_directories()

    def fetch_stock_price_data(
        self,
        ticker: str,
        period: str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch daily stock price data from yfinance and back it up locally."""

        symbol = self._normalize_ticker(ticker)
        selected_period = period or self.config.default_price_period

        if selected_period not in self.VALID_PRICE_PERIODS:
            raise DataLoaderError(
                f"Unsupported period '{selected_period}'. "
                f"Choose from {sorted(self.VALID_PRICE_PERIODS)}."
            )

        cache_path = self._raw_file_path("prices", f"{symbol}_prices.csv")
        if use_cache and cache_path.exists():
            return self._read_prices_csv(cache_path, symbol)

        # Download raw history first, then convert it into the stable project
        # schema that later graph code depends on.
        history = self._download_yfinance_history(symbol, selected_period)
        prices = self._prepare_price_dataframe(history, symbol)

        if prices.empty:
            raise DataLoaderError(f"Price table for ticker '{symbol}' is empty.")

        self._write_dataframe_snapshot(prices, "prices", f"{symbol}_prices.csv")
        return prices

    def fetch_news_json(
        self,
        tickers: str | Sequence[str],
        topics: str | Sequence[str] | None = None,
        limit: int | None = None,
        sort: str = "LATEST",
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Fetch raw news JSON from Alpha Vantage NEWS_SENTIMENT."""

        symbols = self._normalize_ticker_list(tickers)
        selected_limit = self.config.default_news_limit if limit is None else limit
        selected_sort = sort.upper()

        if not isinstance(selected_limit, int) or selected_limit <= 0:
            raise DataLoaderError("News limit must be a positive integer.")

        if selected_sort not in self.VALID_NEWS_SORTS:
            raise DataLoaderError(
                f"Unsupported sort '{selected_sort}'. "
                f"Choose from {sorted(self.VALID_NEWS_SORTS)}."
            )

        topic_list: list[str] = []
        if topics is not None:
            topic_list = self._normalize_text_list(topics, field_name="topics")

        cache_name = self._build_news_cache_name(
            symbols=symbols,
            topics=topic_list,
            limit=selected_limit,
            sort=selected_sort,
        )
        cache_path = self._raw_file_path("news", cache_name)

        if use_cache and cache_path.exists():
            return self._read_json_file(cache_path)

        params: dict[str, Any] = {
            "function": "NEWS_SENTIMENT",
            "tickers": ",".join(symbols),
            "limit": selected_limit,
            "sort": selected_sort,
        }

        if topic_list:
            params["topics"] = ",".join(topic_list)

        payload = self._get_alpha_vantage_json(params)
        feed = payload.get("feed")

        if feed is None:
            raise DataLoaderError("News response is missing the 'feed' field.")

        if not isinstance(feed, list):
            raise DataLoaderError("News response field 'feed' must be a list.")

        self._write_json_snapshot(payload, "news", cache_name)
        return payload

    def load_sector_info(
        self,
        tickers: Sequence[str],
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load sector metadata for one or more tickers and back it up locally."""

        symbols = self._normalize_ticker_list(tickers)
        rows: list[dict[str, Any]] = []

        for symbol in symbols:
            cache_path = self._raw_file_path("sectors", f"{symbol}_sector.json")

            if use_cache and cache_path.exists():
                rows.append(self._read_json_file(cache_path))
                continue

            record = self._load_single_sector_record(symbol)
            rows.append(record)
            self._write_json_snapshot(record, "sectors", f"{symbol}_sector.json")

        return pd.DataFrame(rows)

    def _load_single_sector_record(self, ticker: str) -> dict[str, Any]:
        """Load sector metadata for a single ticker from yfinance."""

        metadata = self._fetch_yfinance_metadata(ticker)

        return {
            "ticker": ticker,
            "company_name": metadata.get("company_name"),
            "sector": metadata.get("sector"),
            "industry": metadata.get("industry"),
            "source": metadata.get("source", "missing"),
        }

    def _download_yfinance_history(
        self,
        ticker: str,
        period: str,
    ) -> pd.DataFrame:
        """Download price history using yfinance."""

        if yf is None:
            raise DataLoaderError(
                "yfinance is not installed. Run 'pip install yfinance'."
            )

        try:
            history = yf.Ticker(ticker).history(
                period=period,
                interval="1d",
                auto_adjust=False,
            )
        except Exception as exc:  # pragma: no cover - third-party runtime safeguard
            raise DataLoaderError(
                f"yfinance failed to download price history for '{ticker}': {exc}"
            ) from exc

        if history is None or history.empty:
            raise DataLoaderError(
                f"No price history was returned for ticker '{ticker}'."
            )

        return history

    def _prepare_price_dataframe(
        self,
        history: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """Convert yfinance history output into a stable project schema."""

        prices = history.reset_index().rename(
            columns={
                "Date": "date",
                "index": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "Volume": "volume",
                "Dividends": "dividend_amount",
                "Stock Splits": "split_coefficient",
            }
        )

        expected_columns = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        }
        missing_columns = expected_columns - set(prices.columns)
        if missing_columns:
            raise DataLoaderError(
                f"Price history for '{ticker}' is missing columns: "
                f"{sorted(missing_columns)}."
            )

        if "dividend_amount" not in prices.columns:
            prices["dividend_amount"] = 0.0

        if "split_coefficient" not in prices.columns:
            prices["split_coefficient"] = 0.0

        # Strip timezone information to avoid mixed timezone issues when price
        # tables from different sources are aligned into one returns matrix.
        prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
            "dividend_amount",
            "split_coefficient",
        ]
        prices[numeric_columns] = prices[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        prices["ticker"] = ticker

        ordered_columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
            "dividend_amount",
            "split_coefficient",
            "ticker",
        ]

        return prices[ordered_columns].sort_values("date").reset_index(drop=True)

    def _fetch_yfinance_metadata(self, ticker: str) -> dict[str, Any]:
        """Fetch company metadata from yfinance."""

        if yf is None:
            raise DataLoaderError(
                "yfinance is not installed. Run 'pip install yfinance'."
            )

        try:
            info = yf.Ticker(ticker).info
        except Exception as exc:  # pragma: no cover - third-party runtime safeguard
            raise DataLoaderError(
                f"yfinance failed to load metadata for '{ticker}': {exc}"
            ) from exc

        if not isinstance(info, dict):
            return {}

        return {
            "company_name": self._clean_text(
                info.get("longName") or info.get("shortName")
            ),
            "sector": self._clean_text(info.get("sector")),
            "industry": self._clean_text(info.get("industry")),
            "source": "yfinance",
        }

    def _get_alpha_vantage_json(self, params: dict[str, Any]) -> dict[str, Any]:
        """Send a validated request to Alpha Vantage."""

        api_key = self.config.alpha_vantage_api_key
        if not api_key:
            raise DataLoaderError(
                "Missing Alpha Vantage API key. "
                "Set the ALPHA_VANTAGE_API_KEY environment variable."
            )

        payload = self._request_json(
            self.config.alpha_vantage_base_url,
            {**params, "apikey": api_key},
        )

        if not payload:
            raise DataLoaderError("Alpha Vantage returned an empty response.")

        if "Error Message" in payload:
            raise DataLoaderError(str(payload["Error Message"]))

        # Alpha Vantage reports throttling and plan-level issues through normal
        # JSON bodies, not only through HTTP status codes.
        if "Note" in payload:
            raise DataLoaderError(str(payload["Note"]))

        if "Information" in payload:
            raise DataLoaderError(str(payload["Information"]))

        return payload

    def _request_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """Make an HTTP GET request and parse the JSON response."""

        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.config.request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise DataLoaderError(f"Request failed for '{url}': {exc}") from exc
        except ValueError as exc:
            raise DataLoaderError(f"Invalid JSON response from '{url}'.") from exc

        if not isinstance(payload, dict):
            raise DataLoaderError("Expected a JSON object response from the API.")

        return payload

    def _ensure_storage_directories(self) -> None:
        """Create local storage directories for raw data and backups."""

        directories = [
            self.config.data_dir,
            self.config.raw_data_dir,
            self.config.raw_data_dir / "prices",
            self.config.raw_data_dir / "news",
            self.config.raw_data_dir / "sectors",
            self.config.backup_data_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _write_dataframe_snapshot(
        self,
        data_frame: pd.DataFrame,
        category: str,
        file_name: str,
    ) -> None:
        """Write a DataFrame to the latest file and a timestamped backup."""

        current_path = self._raw_file_path(category, file_name)
        backup_path = self._backup_file_path(category, file_name)

        current_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # The current file supports normal app reads; the backup file preserves
        # one timestamped copy for reproducibility and debugging.
        data_frame.to_csv(current_path, index=False)
        data_frame.to_csv(backup_path, index=False)

    def _write_json_snapshot(
        self,
        payload: dict[str, Any],
        category: str,
        file_name: str,
    ) -> None:
        """Write JSON to the latest file and a timestamped backup."""

        current_path = self._raw_file_path(category, file_name)
        backup_path = self._backup_file_path(category, file_name)

        current_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        with current_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)

        with backup_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)

    def _read_prices_csv(self, path: Path, ticker: str) -> pd.DataFrame:
        """Read cached price data from CSV."""

        prices = pd.read_csv(path, parse_dates=["date"])
        if prices.empty:
            raise DataLoaderError(f"Cached price file for '{ticker}' is empty.")

        prices["ticker"] = ticker
        return prices

    def _read_json_file(self, path: Path) -> dict[str, Any]:
        """Read a JSON object from disk."""

        with path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        if not isinstance(payload, dict):
            raise DataLoaderError(f"Expected a JSON object in '{path}'.")

        return payload

    def _raw_file_path(self, category: str, file_name: str) -> Path:
        """Build the current raw-data path for a file."""

        return self.config.raw_data_dir / category / file_name

    def _backup_file_path(self, category: str, file_name: str) -> Path:
        """Build the timestamped backup path for a file."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.config.backup_data_dir / timestamp / category / file_name

    def _build_news_cache_name(
        self,
        symbols: list[str],
        topics: list[str],
        limit: int,
        sort: str,
    ) -> str:
        """Build a stable file name for a news query."""

        symbol_part = "_".join(symbols)
        topic_part = "all_topics" if not topics else "_".join(topics)
        return f"{symbol_part}_{topic_part}_{sort.lower()}_{limit}.json"

    def _normalize_ticker(self, ticker: str) -> str:
        """Validate and normalize a single ticker symbol."""

        if not isinstance(ticker, str):
            raise DataLoaderError("Ticker must be a string.")

        cleaned = ticker.strip().upper()
        if not cleaned:
            raise DataLoaderError("Ticker cannot be empty.")

        return cleaned

    def _normalize_ticker_list(
        self,
        tickers: str | Sequence[str],
    ) -> list[str]:
        """Normalize one or many ticker symbols into a list."""

        return self._normalize_text_list(tickers, field_name="tickers", uppercase=True)

    def _normalize_text_list(
        self,
        values: str | Sequence[str],
        field_name: str,
        uppercase: bool = False,
    ) -> list[str]:
        """Normalize a string or string list into a clean list of values."""

        # Accept either one comma-separated string or a sequence so both CLI
        # arguments and internal callers can reuse the same validation path.
        if isinstance(values, str):
            raw_items = values.split(",")
        else:
            raw_items = list(values)

        cleaned_items: list[str] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, str):
                raise DataLoaderError(
                    f"Each value in '{field_name}' must be a string."
                )

            cleaned = raw_item.strip()
            if cleaned:
                cleaned_items.append(cleaned.upper() if uppercase else cleaned)

        if not cleaned_items:
            raise DataLoaderError(f"'{field_name}' cannot be empty.")

        return cleaned_items

    def _clean_text(self, value: Any) -> str | None:
        """Convert empty or placeholder strings into None."""

        if value is None:
            return None

        if not isinstance(value, str):
            value = str(value)

        cleaned = value.strip()
        if cleaned in {"", "None", "N/A", "nan"}:
            return None

        return cleaned
