"""Batch download seed market data for the project."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from data_loader import DataLoaderError, MarketDataLoader


DEFAULT_STOCK_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "META",
    "ORCL",
    "AMZN",
    "TSLA",
    "HD",
    "NKE",
    "JPM",
    "BAC",
    "GS",
    "V",
    "UNH",
    "JNJ",
    "PFE",
    "ABBV",
    "XOM",
    "CVX",
    "COP",
    "CAT",
    "HON",
    "WMT",
    "COST",
    "PG",
    "KO",
    "DIS",
    "LIN",
    "DUK",
]

DEFAULT_NEWS_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "META",
    "ORCL",
    "AMZN",
    "TSLA",
    "JPM",
    "BAC",
    "UNH",
    "JNJ",
    "XOM",
    "WMT",
    "DIS",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for batch seed downloads."""

    parser = argparse.ArgumentParser(
        description="Download seed stock, sector, and news data."
    )
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_STOCK_TICKERS),
        help="Comma-separated ticker list for price and sector downloads.",
    )
    parser.add_argument(
        "--news-tickers",
        default=",".join(DEFAULT_NEWS_TICKERS),
        help="Comma-separated ticker list for news downloads.",
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="Price period for yfinance downloads. Example: 1y, 2y, 5y, max.",
    )
    parser.add_argument(
        "--news-limit",
        type=int,
        default=1000,
        help="Maximum number of news records per ticker request.",
    )
    parser.add_argument(
        "--news-delay-seconds",
        type=float,
        default=1.2,
        help="Delay between Alpha Vantage news requests to avoid rate limiting.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore local cache files and download fresh data.",
    )
    parser.add_argument(
        "--skip-news",
        action="store_true",
        help="Skip news downloads and only pull prices and sectors.",
    )
    parser.add_argument(
        "--only-news",
        action="store_true",
        help="Skip prices and sectors, and only download news data.",
    )
    return parser


def parse_ticker_argument(raw_value: str) -> list[str]:
    """Parse a comma-separated ticker string into a clean list."""

    items = [item.strip().upper() for item in raw_value.split(",")]
    return [item for item in items if item]


def download_seed_data(
    loader: MarketDataLoader,
    stock_tickers: list[str],
    news_tickers: list[str],
    period: str,
    news_limit: int,
    news_delay_seconds: float,
    refresh: bool,
    skip_news: bool,
    only_news: bool,
) -> int:
    """Download a seed dataset and print a simple progress summary."""

    use_cache = not refresh
    failure_count = 0
    total_news_records = 0
    merged_news_payloads: list[dict[str, Any]] = []

    print(f"Local data directory: {loader.config.data_dir.resolve()}")

    if not only_news:
        print()
        print("Downloading price and sector data...")

        for ticker in stock_tickers:
            try:
                # Prices and sector metadata are stored separately so later
                # analysis can rebuild the graph from local files only.
                prices = loader.fetch_stock_price_data(
                    ticker,
                    period=period,
                    use_cache=use_cache,
                )
                sector_info = loader.load_sector_info([ticker], use_cache=use_cache)
                sector_name = get_sector_name(sector_info)
                print(
                    f"[OK] {ticker}: {len(prices)} price rows saved, "
                    f"sector={sector_name or 'missing'}"
                )
            except DataLoaderError as exc:
                failure_count += 1
                print(f"[FAILED] {ticker}: {exc}")

    if not skip_news:
        if news_delay_seconds < 0:
            raise DataLoaderError("News delay must be zero or a positive number.")

        print()
        print("Downloading news data...")

        for index, ticker in enumerate(news_tickers):
            try:
                # News is downloaded per ticker instead of multi-ticker batches
                # so we get a much larger union of articles before deduping.
                payload = loader.fetch_news_json(
                    ticker,
                    limit=news_limit,
                    use_cache=use_cache,
                )
                article_count = len(payload.get("feed", []))
                total_news_records += article_count
                merged_news_payloads.append(payload)
                print(f"[OK] {ticker}: {article_count} news records saved")
            except DataLoaderError as exc:
                failure_count += 1
                print(f"[FAILED] {ticker}: {exc}")

            if not use_cache and index < len(news_tickers) - 1:
                time.sleep(news_delay_seconds)

        if merged_news_payloads:
            # The merged file becomes the default news input for both the CLI
            # and the web apps.
            merged_payload = build_merged_news_payload(
                merged_news_payloads,
                news_tickers,
            )
            write_merged_news_snapshot(loader, merged_payload)
            merged_count = len(merged_payload.get("feed", []))
            duplicate_count = total_news_records - merged_count
            print(
                f"[OK] merged_seed_news.json: {merged_count} unique records saved "
                f"({duplicate_count} duplicates removed)"
            )

    print()
    print("Download summary")
    print(f"- Stock/sector tickers requested: {0 if only_news else len(stock_tickers)}")
    print(f"- News tickers requested: {0 if skip_news else len(news_tickers)}")
    print(f"- Raw news records downloaded: {total_news_records}")
    print(f"- Failures: {failure_count}")

    return 0 if failure_count == 0 else 1


def build_merged_news_payload(
    payloads: list[dict[str, Any]],
    news_tickers: list[str],
) -> dict[str, Any]:
    """Merge and deduplicate article payloads from multiple ticker queries."""

    merged_articles: dict[str, dict[str, Any]] = {}

    for payload in payloads:
        for article in payload.get("feed", []):
            dedupe_key = get_article_dedupe_key(article)

            if dedupe_key in merged_articles:
                # The same article often appears under multiple ticker queries.
                # Merge ticker/topic lists instead of dropping later metadata.
                merged_articles[dedupe_key] = merge_article_records(
                    merged_articles[dedupe_key],
                    article,
                )
            else:
                merged_articles[dedupe_key] = article

    merged_feed = sorted(
        merged_articles.values(),
        key=lambda item: item.get("time_published", ""),
        reverse=True,
    )

    return {
        "items": str(len(merged_feed)),
        "feed": merged_feed,
        "downloaded_tickers": news_tickers,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def get_article_dedupe_key(article: dict[str, Any]) -> str:
    """Build a stable deduplication key for a news article."""

    url = article.get("url")
    if isinstance(url, str) and url.strip():
        return url.strip()

    title = str(article.get("title", "")).strip()
    published_at = str(article.get("time_published", "")).strip()
    return f"{title}|{published_at}"


def merge_article_records(
    existing_article: dict[str, Any],
    new_article: dict[str, Any],
) -> dict[str, Any]:
    """Merge duplicate article records across ticker downloads."""

    merged_article = dict(existing_article)
    merged_article["ticker_sentiment"] = merge_unique_dict_list(
        existing_article.get("ticker_sentiment", []),
        new_article.get("ticker_sentiment", []),
        key_name="ticker",
    )
    merged_article["topics"] = merge_unique_dict_list(
        existing_article.get("topics", []),
        new_article.get("topics", []),
        key_name="topic",
    )

    return merged_article


def merge_unique_dict_list(
    first_items: list[dict[str, Any]],
    second_items: list[dict[str, Any]],
    key_name: str,
) -> list[dict[str, Any]]:
    """Merge two lists of dictionaries while keeping unique entries by key."""

    merged: dict[str, dict[str, Any]] = {}

    for item in first_items + second_items:
        key_value = item.get(key_name)
        if key_value is None:
            continue
        merged[str(key_value)] = item

    return list(merged.values())


def write_merged_news_snapshot(
    loader: MarketDataLoader,
    payload: dict[str, Any],
) -> None:
    """Write the merged news payload to the raw and backup directories."""

    raw_path = loader.config.raw_data_dir / "news" / "merged_seed_news.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)

    # Keep the merged file on the same backup pattern as the raw per-ticker
    # files so one download run is easy to trace on disk.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = (
        loader.config.backup_data_dir
        / timestamp
        / "news"
        / "merged_seed_news.json"
    )
    backup_path.parent.mkdir(parents=True, exist_ok=True)

    with backup_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def get_sector_name(sector_info: pd.DataFrame) -> str | None:
    """Extract the sector name from a one-row sector DataFrame."""

    if sector_info.empty:
        return None

    value = sector_info.iloc[0].get("sector")
    if pd.isna(value):
        return None

    return str(value)


def main() -> int:
    """Run the seed downloader from the command line."""

    parser = build_parser()
    args = parser.parse_args()

    stock_tickers = parse_ticker_argument(args.tickers)
    news_tickers = parse_ticker_argument(args.news_tickers)
    loader = MarketDataLoader()

    try:
        return download_seed_data(
            loader=loader,
            stock_tickers=stock_tickers,
            news_tickers=news_tickers,
            period=args.period,
            news_limit=args.news_limit,
            news_delay_seconds=args.news_delay_seconds,
            refresh=args.refresh,
            skip_news=args.skip_news,
            only_news=args.only_news,
        )
    except DataLoaderError as exc:
        print(f"Seed download failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
