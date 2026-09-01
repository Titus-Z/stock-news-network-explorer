"""Tests for Phase 3 graph construction."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from network_builder import NetworkBuilder, NetworkBuilderError


def price_frame(prices: list[float]) -> pd.DataFrame:
    """Build a simple stock price table for graph tests."""

    dates = pd.date_range("2026-01-01", periods=len(prices), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "adjusted_close": prices,
        }
    )


def sample_price_tables() -> dict[str, pd.DataFrame]:
    """Create small synthetic price tables with clear correlation structure."""

    return {
        "AAPL": price_frame([100.0, 101.0, 104.03, 101.95, 106.03, 107.09]),
        "MSFT": price_frame([100.0, 101.1, 104.03, 102.16, 106.14, 107.41]),
        "JPM": price_frame([100.0, 100.9, 104.03, 102.05, 106.24, 107.30]),
        "XOM": price_frame([100.0, 97.0, 97.97, 101.89, 99.85, 99.85]),
    }


def sample_sector_info() -> pd.DataFrame:
    """Create a small sector table for graph tests."""

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
                "ticker": "MSFT",
                "company_name": "Microsoft Corporation",
                "sector": "Technology",
                "industry": "Software",
                "source": "yfinance",
            },
            {
                "ticker": "JPM",
                "company_name": "JPMorgan Chase & Co.",
                "sector": "Financial Services",
                "industry": "Banks",
                "source": "yfinance",
            },
            {
                "ticker": "XOM",
                "company_name": "Exxon Mobil Corporation",
                "sector": "Energy",
                "industry": "Oil & Gas",
                "source": "yfinance",
            },
        ]
    )


def sample_topic_stock() -> pd.DataFrame:
    """Create a small topic-stock aggregation table for graph tests."""

    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "topic": "technology",
                "article_count": 10,
                "avg_topic_relevance": 0.85,
                "avg_ticker_relevance": 0.90,
                "avg_ticker_sentiment": 0.15,
                "avg_overall_sentiment": 0.10,
            },
            {
                "ticker": "MSFT",
                "topic": "technology",
                "article_count": 8,
                "avg_topic_relevance": 0.82,
                "avg_ticker_relevance": 0.88,
                "avg_ticker_sentiment": 0.12,
                "avg_overall_sentiment": 0.08,
            },
            {
                "ticker": "JPM",
                "topic": "finance",
                "article_count": 5,
                "avg_topic_relevance": 0.75,
                "avg_ticker_relevance": 0.80,
                "avg_ticker_sentiment": 0.05,
                "avg_overall_sentiment": 0.03,
            },
        ]
    )


def test_build_graph_adds_nodes_and_core_edge_types() -> None:
    """The builder should create stock, sector, topic, and derived sector edges."""

    builder = NetworkBuilder(correlation_threshold=0.95)
    graph = builder.build_graph(
        price_tables=sample_price_tables(),
        sector_info=sample_sector_info(),
        topic_stock=sample_topic_stock(),
    )

    assert isinstance(graph, nx.Graph)
    assert graph.nodes["stock:AAPL"]["node_type"] == "stock"
    assert graph.nodes["sector:Technology"]["node_type"] == "sector"
    assert graph.nodes["topic:technology"]["node_type"] == "topic"

    assert graph.edges["stock:AAPL", "stock:MSFT"]["edge_type"] == "stock_stock"
    assert graph.edges["stock:AAPL", "sector:Technology"]["edge_type"] == "stock_sector"
    assert graph.edges["stock:AAPL", "topic:technology"]["edge_type"] == "stock_topic"
    assert (
        graph.edges["sector:Financial Services", "sector:Technology"]["edge_type"]
        == "sector_sector"
    )

    assert graph.edges["stock:AAPL", "topic:technology"]["weight"] == pytest.approx(10.0)


def test_top_k_neighbor_mode_limits_stock_stock_edges() -> None:
    """top_k mode should keep a small set of strongest stock-stock edges."""

    builder = NetworkBuilder(correlation_threshold=0.0, top_k_neighbors=1)
    graph = builder.build_graph(
        price_tables=sample_price_tables(),
        sector_info=sample_sector_info(),
        topic_stock=sample_topic_stock(),
    )

    stock_stock_edges = [
        edge for edge in graph.edges(data=True) if edge[2].get("edge_type") == "stock_stock"
    ]

    assert 1 <= len(stock_stock_edges) <= 3


def test_missing_sector_info_skips_stock_sector_edge() -> None:
    """A stock with missing sector data should still exist without a sector edge."""

    sector_info = sample_sector_info()
    sector_info.loc[sector_info["ticker"] == "XOM", "sector"] = pd.NA

    builder = NetworkBuilder(correlation_threshold=0.95)
    graph = builder.build_graph(
        price_tables=sample_price_tables(),
        sector_info=sector_info,
        topic_stock=sample_topic_stock(),
    )

    assert "stock:XOM" in graph
    assert "sector:Energy" not in graph
    assert not graph.has_edge("stock:XOM", "sector:Energy")


def test_insufficient_price_history_skips_stock_stock_edges() -> None:
    """Stocks with too little history should not create correlation edges."""

    price_tables = {
        "AAPL": price_frame([100.0, 101.0, 102.0]),
        "MSFT": price_frame([50.0]),
    }
    sector_info = pd.DataFrame(
        [
            {"ticker": "AAPL", "sector": "Technology"},
            {"ticker": "MSFT", "sector": "Technology"},
        ]
    )
    topic_stock = pd.DataFrame(columns=[
        "ticker",
        "topic",
        "article_count",
        "avg_topic_relevance",
        "avg_ticker_relevance",
        "avg_ticker_sentiment",
        "avg_overall_sentiment",
    ])

    builder = NetworkBuilder(correlation_threshold=0.5)
    graph = builder.build_graph(price_tables=price_tables, sector_info=sector_info, topic_stock=topic_stock)

    stock_stock_edges = [
        edge for edge in graph.edges(data=True) if edge[2].get("edge_type") == "stock_stock"
    ]

    assert "stock:AAPL" in graph
    assert "stock:MSFT" in graph
    assert stock_stock_edges == []


def test_invalid_topic_weight_column_raises_clear_error() -> None:
    """Invalid stock-topic weight columns should fail fast."""

    with pytest.raises(NetworkBuilderError, match="Unsupported stock-topic weight column"):
        NetworkBuilder(stock_topic_weight_column="combined_score")


def test_topic_rows_outside_price_universe_are_ignored() -> None:
    """Topic-stock rows for unknown tickers should not create extra stock nodes."""

    topic_stock = pd.concat(
        [
            sample_topic_stock(),
            pd.DataFrame(
                [
                    {
                        "ticker": "TSLA",
                        "topic": "technology",
                        "article_count": 4,
                        "avg_topic_relevance": 0.7,
                        "avg_ticker_relevance": 0.7,
                        "avg_ticker_sentiment": 0.1,
                        "avg_overall_sentiment": 0.1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    builder = NetworkBuilder(correlation_threshold=0.95)
    graph = builder.build_graph(
        price_tables=sample_price_tables(),
        sector_info=sample_sector_info(),
        topic_stock=topic_stock,
    )

    assert "stock:TSLA" not in graph
    assert graph.nodes["topic:technology"]["article_count"] == 18
