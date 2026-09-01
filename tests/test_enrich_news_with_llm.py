"""Tests for the offline LLM enrichment CLI helpers."""

from __future__ import annotations

import pandas as pd

from enrich_news_with_llm import _filter_articles_by_date_window, build_output_stem


def sample_news_tables() -> dict[str, pd.DataFrame]:
    """Build a tiny processed-news bundle for helper tests."""

    return {
        "articles": pd.DataFrame(
            [
                {
                    "article_id": "a1",
                    "published_at": pd.Timestamp("2026-03-30 12:00:00"),
                },
                {
                    "article_id": "a2",
                    "published_at": pd.Timestamp("2026-03-29 12:00:00"),
                },
            ]
        ),
        "article_tickers": pd.DataFrame(
            [
                {"article_id": "a1", "ticker": "AAPL"},
                {"article_id": "a2", "ticker": "MSFT"},
            ]
        ),
        "topic_stock": pd.DataFrame(
            [
                {"ticker": "AAPL", "topic": "technology"},
                {"ticker": "MSFT", "topic": "technology"},
            ]
        ),
    }


def test_filter_articles_by_date_window_keeps_only_selected_articles() -> None:
    """Date filtering should keep linked tables aligned to the filtered articles."""

    filtered = _filter_articles_by_date_window(
        news_tables=sample_news_tables(),
        published_after="2026-03-30",
        published_before=None,
    )

    assert len(filtered["articles"]) == 1
    assert filtered["articles"].loc[0, "article_id"] == "a1"
    assert filtered["article_tickers"].loc[0, "article_id"] == "a1"


def test_build_output_stem_includes_window_suffix_when_requested() -> None:
    """Windowed LLM runs should include a readable date suffix in the stem."""

    stem = build_output_stem(
        news_file="merged_seed_news.json",
        tickers=["AAPL", "MSFT"],
        max_articles=20,
        published_after="2026-03-01",
        published_before="2026-03-31",
    )

    assert stem == "merged_seed_news_AAPL_MSFT_window_20260301_to_20260331_top_20"
