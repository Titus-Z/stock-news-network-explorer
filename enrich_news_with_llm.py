"""Run offline LLM enrichment for local cached news data."""

from __future__ import annotations

import argparse
from pathlib import Path

from cli import filter_news_tables_by_tickers
import pandas as pd

from llm_enricher import LLMEnricherError, OpenAINewsEnricher
from local_data_store import LocalDataStore, LocalDataStoreError
from news_processor import NewsProcessor, NewsProcessorError
from streamlit_app import parse_ticker_text


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for LLM news enrichment."""

    parser = argparse.ArgumentParser(
        description="Enrich cached news with structured OpenAI impact labels."
    )
    parser.add_argument(
        "--tickers",
        default="AAPL,MSFT,NVDA,GOOGL,META,ORCL,AMZN,TSLA,JPM,BAC,UNH,JNJ,XOM,WMT,DIS",
        help="Optional comma-separated ticker filter for local analysis.",
    )
    parser.add_argument(
        "--news-file",
        default="merged_seed_news.json",
        help=(
            "News file name inside the configured data root. "
            "Default: merged_seed_news.json"
        ),
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=10,
        help="Maximum number of articles to enrich in one run.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional OpenAI model override. Default uses config or env.",
    )
    parser.add_argument(
        "--published-after",
        default=None,
        help="Optional inclusive lower timestamp/date bound for article selection.",
    )
    parser.add_argument(
        "--published-before",
        default=None,
        help="Optional inclusive upper timestamp/date bound for article selection.",
    )
    return parser


def build_output_stem(
    news_file: str,
    tickers: list[str] | None,
    max_articles: int,
    published_after: str | None = None,
    published_before: str | None = None,
) -> str:
    """Build one stable output file stem for the enrichment run."""

    news_stem = Path(news_file).stem
    ticker_part = "all_tickers" if not tickers else "_".join(tickers)
    window_part = _build_window_stem(
        published_after=published_after,
        published_before=published_before,
    )
    return f"{news_stem}_{ticker_part}{window_part}_top_{max_articles}"


def main() -> int:
    """Run offline OpenAI enrichment and save the resulting CSV tables."""

    parser = build_parser()
    args = parser.parse_args()

    tickers = parse_ticker_text(args.tickers)
    if args.max_articles <= 0:
        print("LLM enrichment failed: --max-articles must be a positive integer.")
        return 1

    store = LocalDataStore()
    processor = NewsProcessor()

    try:
        news_payload = store.load_news_payload(file_name=args.news_file)
        news_tables = processor.process_news_payload(news_payload)
        filtered_news_tables = filter_news_tables_by_tickers(news_tables, tickers)
        filtered_news_tables = _filter_articles_by_date_window(
            news_tables=filtered_news_tables,
            published_after=args.published_after,
            published_before=args.published_before,
        )
        sector_info = store.load_sector_info(tickers=tickers)

        enricher = OpenAINewsEnricher(model=args.model)
        tables = enricher.enrich_news_tables(
            articles=filtered_news_tables["articles"],
            article_tickers=filtered_news_tables["article_tickers"],
            sector_info=sector_info,
            max_articles=args.max_articles,
            published_after=args.published_after,
            published_before=args.published_before,
        )
        file_stem = build_output_stem(
            news_file=args.news_file,
            tickers=tickers,
            max_articles=args.max_articles,
            published_after=args.published_after,
            published_before=args.published_before,
        )
        enricher.write_output_tables(tables, file_stem=file_stem)
    except (LocalDataStoreError, NewsProcessorError, LLMEnricherError) as exc:
        print(f"LLM enrichment failed: {exc}")
        return 1

    print("LLM enrichment complete.")
    print(f"- articles enriched: {len(tables['article_llm_summary'])}")
    print(f"- sector impact rows: {len(tables['article_sector_impacts'])}")
    print(f"- stock impact rows: {len(tables['article_stock_impacts'])}")
    print(
        "- output directory: "
        f"{(store.config.raw_data_dir / 'llm_enriched').resolve()}"
    )
    return 0

def _filter_articles_by_date_window(
    news_tables: dict[str, pd.DataFrame],
    published_after: str | None,
    published_before: str | None,
) -> dict[str, pd.DataFrame]:
    """Filter article-linked tables to one inclusive publication window."""

    if not published_after and not published_before:
        return news_tables

    articles = news_tables["articles"].copy()
    articles["published_at"] = pd.to_datetime(articles["published_at"], errors="coerce")

    if published_after:
        after_timestamp = _coerce_cli_timestamp(published_after, end_of_day=False)
        if pd.isna(after_timestamp):
            raise LLMEnricherError("published_after must be a valid timestamp or date.")
        articles = articles[articles["published_at"] >= after_timestamp].copy()

    if published_before:
        before_timestamp = _coerce_cli_timestamp(published_before, end_of_day=True)
        if pd.isna(before_timestamp):
            raise LLMEnricherError("published_before must be a valid timestamp or date.")
        articles = articles[articles["published_at"] <= before_timestamp].copy()

    selected_ids = set(articles["article_id"].astype(str))
    filtered_tables = dict(news_tables)
    filtered_tables["articles"] = articles.reset_index(drop=True)
    filtered_tables["article_tickers"] = news_tables["article_tickers"][
        news_tables["article_tickers"]["article_id"].astype(str).isin(selected_ids)
    ].reset_index(drop=True)
    return filtered_tables


def _build_window_stem(
    published_after: str | None,
    published_before: str | None,
) -> str:
    """Build a compact file-stem suffix for the article time window."""

    if not published_after and not published_before:
        return ""

    after_part = _format_window_part(published_after, fallback="start")
    before_part = _format_window_part(published_before, fallback="end")
    return f"_window_{after_part}_to_{before_part}"


def _format_window_part(value: str | None, fallback: str) -> str:
    """Format one timestamp/date string into a compact stem token."""

    if value is None:
        return fallback
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise LLMEnricherError("Date window arguments must be valid timestamps or dates.")
    return timestamp.strftime("%Y%m%d")


def _coerce_cli_timestamp(value: str, end_of_day: bool) -> pd.Timestamp:
    """Parse one CLI date or timestamp with intuitive date-only bounds."""

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return timestamp
    if end_of_day and len(value.strip()) == 10:
        return timestamp + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return timestamp


if __name__ == "__main__":
    raise SystemExit(main())
