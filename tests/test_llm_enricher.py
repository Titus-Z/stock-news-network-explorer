"""Tests for offline LLM enrichment helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from config import AppConfig
from llm_enricher import (
    ArticleImpactAssessment,
    OpenAINewsEnricher,
    SectorImpact,
    StockImpact,
    select_articles_for_enrichment,
)


class FakeResponsesAPI:
    """Small fake Responses API object for deterministic enrichment tests."""

    def __init__(self, parsed_output: ArticleImpactAssessment) -> None:
        self.parsed_output = parsed_output

    def parse(self, **kwargs):
        """Return an object that mimics the SDK parsed response."""

        return SimpleNamespace(output_parsed=self.parsed_output)


class FakeClient:
    """Small fake OpenAI client wrapper used in unit tests."""

    def __init__(self, parsed_output: ArticleImpactAssessment) -> None:
        self.responses = FakeResponsesAPI(parsed_output)


def build_test_config(tmp_path) -> AppConfig:
    """Build a config object with isolated local directories."""

    data_dir = tmp_path / "data"
    return AppConfig(
        alpha_vantage_api_key="demo-alpha",
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


def sample_articles() -> pd.DataFrame:
    """Build a minimal article table for enrichment tests."""

    return pd.DataFrame(
        [
            {
                "article_id": "a1",
                "title": "Apple expands AI spending",
                "url": "https://example.com/a1",
                "published_at": pd.Timestamp("2026-03-20 12:00:00"),
                "source": "Reuters",
                "summary": "Apple announced a new AI investment plan.",
                "overall_sentiment_score": 0.2,
                "overall_sentiment_label": "Somewhat-Bullish",
            },
            {
                "article_id": "a2",
                "title": "Microsoft expands cloud AI",
                "url": "https://example.com/a2",
                "published_at": pd.Timestamp("2026-03-19 10:00:00"),
                "source": "Bloomberg",
                "summary": "Microsoft announced a cloud AI update.",
                "overall_sentiment_score": 0.1,
                "overall_sentiment_label": "Neutral",
            },
        ]
    )


def sample_article_tickers() -> pd.DataFrame:
    """Build a minimal article-ticker table for enrichment tests."""

    return pd.DataFrame(
        [
            {
                "article_id": "a1",
                "ticker": "AAPL",
                "relevance_score": 0.9,
                "ticker_sentiment_score": 0.2,
                "ticker_sentiment_label": "Somewhat-Bullish",
            },
            {
                "article_id": "a1",
                "ticker": "MSFT",
                "relevance_score": 0.3,
                "ticker_sentiment_score": 0.1,
                "ticker_sentiment_label": "Neutral",
            },
        ]
    )


def sample_sector_info() -> pd.DataFrame:
    """Build a minimal sector table for enrichment tests."""

    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "source": "yfinance",
            },
            {
                "ticker": "XOM",
                "company_name": "Exxon Mobil",
                "sector": "Energy",
                "industry": "Oil & Gas",
                "source": "yfinance",
            },
        ]
    )


def sample_assessment() -> ArticleImpactAssessment:
    """Build one parsed LLM assessment object for tests."""

    return ArticleImpactAssessment(
        article_id="ignored-by-sanitizer",
        event_summary="Apple announced new AI spending with sector implications.",
        primary_event_type="product",
        scope="mixed",
        overall_market_relevance="medium",
        sector_impacts=[
            SectorImpact(
                sector="Technology",
                impact_direction="positive",
                impact_strength="medium",
                confidence=0.8,
                rationale="The article describes new AI investment by a major tech firm.",
            ),
            SectorImpact(
                sector="Healthcare",
                impact_direction="uncertain",
                impact_strength="low",
                confidence=0.2,
                rationale="Should be dropped because it is outside the allowed list.",
            ),
        ],
        stock_impacts=[
            StockImpact(
                ticker="AAPL",
                impact_direction="positive",
                impact_strength="medium",
                confidence=0.85,
                rationale="Apple is the direct subject of the announcement.",
            ),
            StockImpact(
                ticker="TSLA",
                impact_direction="uncertain",
                impact_strength="low",
                confidence=0.1,
                rationale="Should be dropped because it is outside the allowed list.",
            ),
        ],
    )


def test_enrich_news_tables_builds_expected_output_frames(tmp_path) -> None:
    """Enrichment should return the three output tables with expected rows."""

    enricher = OpenAINewsEnricher(
        config=build_test_config(tmp_path),
        client=FakeClient(sample_assessment()),
    )

    tables = enricher.enrich_news_tables(
        articles=sample_articles(),
        article_tickers=sample_article_tickers(),
        sector_info=sample_sector_info(),
        max_articles=1,
    )

    assert len(tables["article_llm_summary"]) == 1
    assert len(tables["article_sector_impacts"]) == 1
    assert len(tables["article_stock_impacts"]) == 1
    assert tables["article_sector_impacts"].loc[0, "sector"] == "Technology"
    assert tables["article_stock_impacts"].loc[0, "ticker"] == "AAPL"


def test_write_output_tables_writes_raw_files(tmp_path) -> None:
    """Output tables should be written to the raw and backup directories."""

    enricher = OpenAINewsEnricher(
        config=build_test_config(tmp_path),
        client=FakeClient(sample_assessment()),
    )
    tables = enricher.enrich_news_tables(
        articles=sample_articles(),
        article_tickers=sample_article_tickers(),
        sector_info=sample_sector_info(),
        max_articles=1,
    )

    enricher.write_output_tables(tables, file_stem="demo_run")

    raw_dir = tmp_path / "data" / "raw" / "llm_enriched"
    assert (raw_dir / "demo_run_article_llm_summary.csv").exists()
    assert (raw_dir / "demo_run_article_sector_impacts.csv").exists()
    assert (raw_dir / "demo_run_article_stock_impacts.csv").exists()


def test_select_articles_for_enrichment_applies_recency_and_max_articles() -> None:
    """Article selection should sort by recency and respect max_articles."""

    selected = select_articles_for_enrichment(
        sample_articles(),
        max_articles=1,
    )

    assert len(selected) == 1
    assert selected.loc[0, "article_id"] == "a1"


def test_select_articles_for_enrichment_filters_by_published_after() -> None:
    """Date filters should keep only articles inside the requested window."""

    selected = select_articles_for_enrichment(
        sample_articles(),
        published_after="2026-03-20",
    )

    assert len(selected) == 1
    assert selected.loc[0, "article_id"] == "a1"
