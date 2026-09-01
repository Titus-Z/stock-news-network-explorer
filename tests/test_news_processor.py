"""Tests for Phase 2 news processing."""

from __future__ import annotations

import pandas as pd
import pytest

from news_processor import NewsProcessor, NewsProcessorError


def sample_payload() -> dict:
    """Build a small but realistic Alpha Vantage news payload."""

    return {
        "items": "2",
        "feed": [
            {
                "title": "Apple and Microsoft expand AI investment",
                "url": "https://example.com/article-1",
                "time_published": "20260320T120000",
                "source": "Reuters",
                "summary": "Apple and Microsoft both announced new AI spending.",
                "overall_sentiment_score": "0.10",
                "overall_sentiment_label": "Neutral",
                "topics": [
                    {"topic": "technology", "relevance_score": "0.80"},
                    {"topic": "financial_markets", "relevance_score": "0.60"},
                ],
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": "0.90",
                        "ticker_sentiment_score": "0.20",
                        "ticker_sentiment_label": "Somewhat-Bullish",
                    },
                    {
                        "ticker": "MSFT",
                        "relevance_score": "0.40",
                        "ticker_sentiment_score": "0.10",
                        "ticker_sentiment_label": "Neutral",
                    },
                ],
            },
            {
                "title": "Apple earnings face pressure",
                "url": "https://example.com/article-2",
                "time_published": "20260319T083000",
                "source": "Bloomberg",
                "summary": "Apple may face margin pressure this quarter.",
                "overall_sentiment_score": "-0.20",
                "overall_sentiment_label": "Bearish",
                "topics": [
                    {"topic": "technology", "relevance_score": "0.70"},
                    {"topic": "earnings", "relevance_score": "0.50"},
                ],
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": "0.80",
                        "ticker_sentiment_score": "-0.10",
                        "ticker_sentiment_label": "Neutral",
                    }
                ],
            },
        ],
    }


def test_build_articles_table_extracts_required_fields() -> None:
    """The article table should contain the Phase 2 article-level columns."""

    processor = NewsProcessor()
    articles = processor.build_articles_table(sample_payload())

    assert list(articles.columns) == NewsProcessor.ARTICLE_COLUMNS
    assert len(articles) == 2
    assert articles.loc[0, "title"] == "Apple and Microsoft expand AI investment"
    assert pd.api.types.is_datetime64_any_dtype(articles["published_at"])
    assert articles.loc[0, "overall_sentiment_score"] == pytest.approx(0.10)


def test_build_article_tickers_table_extracts_ticker_sentiment() -> None:
    """The article-ticker table should expand ticker_sentiment rows."""

    processor = NewsProcessor()
    article_tickers = processor.build_article_tickers_table(sample_payload())

    assert list(article_tickers.columns) == NewsProcessor.ARTICLE_TICKER_COLUMNS
    assert len(article_tickers) == 3
    assert set(article_tickers["ticker"]) == {"AAPL", "MSFT"}

    aapl_rows = article_tickers[article_tickers["ticker"] == "AAPL"]
    assert len(aapl_rows) == 2
    assert aapl_rows["ticker_sentiment_score"].mean() == pytest.approx(0.05)


def test_build_topic_stock_table_aggregates_topic_stock_metrics() -> None:
    """The topic-stock table should aggregate article, ticker, and topic data."""

    processor = NewsProcessor()
    topic_stock = processor.build_topic_stock_table(sample_payload())

    assert list(topic_stock.columns) == NewsProcessor.TOPIC_STOCK_COLUMNS

    aapl_technology = topic_stock[
        (topic_stock["ticker"] == "AAPL")
        & (topic_stock["topic"] == "technology")
    ].iloc[0]

    assert aapl_technology["article_count"] == 2
    assert aapl_technology["avg_topic_relevance"] == pytest.approx(0.75)
    assert aapl_technology["avg_ticker_relevance"] == pytest.approx(0.85)
    assert aapl_technology["avg_ticker_sentiment"] == pytest.approx(0.05)
    assert aapl_technology["avg_overall_sentiment"] == pytest.approx(-0.05)


def test_build_article_topics_table_extracts_topic_rows() -> None:
    """The article-topic table should preserve per-article topic relevance rows."""

    processor = NewsProcessor()
    article_topics = processor.build_article_topics_table(sample_payload())

    assert list(article_topics.columns) == NewsProcessor.ARTICLE_TOPIC_COLUMNS
    assert len(article_topics) == 4
    assert set(article_topics["topic"]) == {"technology", "financial_markets", "earnings"}


def test_process_news_payload_returns_all_phase2_tables() -> None:
    """The combined processor method should return the three required tables."""

    processor = NewsProcessor()
    tables = processor.process_news_payload(sample_payload())

    assert set(tables.keys()) == {"articles", "article_tickers", "topic_stock"}
    assert len(tables["articles"]) == 2
    assert len(tables["article_tickers"]) == 3
    assert not tables["topic_stock"].empty


def test_empty_feed_returns_empty_phase2_tables() -> None:
    """An empty feed should return empty DataFrames with the expected columns."""

    processor = NewsProcessor()
    tables = processor.process_news_payload({"items": "0", "feed": []})

    assert list(tables["articles"].columns) == NewsProcessor.ARTICLE_COLUMNS
    assert list(tables["article_tickers"].columns) == NewsProcessor.ARTICLE_TICKER_COLUMNS
    assert list(tables["topic_stock"].columns) == NewsProcessor.TOPIC_STOCK_COLUMNS
    assert tables["articles"].empty
    assert tables["article_tickers"].empty
    assert tables["topic_stock"].empty


def test_missing_feed_raises_clear_error() -> None:
    """A malformed payload should raise a clear processing error."""

    processor = NewsProcessor()

    with pytest.raises(NewsProcessorError, match="missing the 'feed' field"):
        processor.process_news_payload({"items": "0"})
