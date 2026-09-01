"""Tests for Phase 4 graph analysis."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from network_analyzer import NetworkAnalyzer
from network_builder import NetworkBuilder


def price_frame(prices: list[float]) -> pd.DataFrame:
    """Build a simple stock price table for analyzer tests."""

    dates = pd.date_range("2026-01-01", periods=len(prices), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "adjusted_close": prices,
        }
    )


def sample_graph() -> nx.Graph:
    """Build one small but connected graph for analyzer tests."""

    price_tables = {
        "AAPL": price_frame([100.0, 101.0, 104.03, 101.95, 106.03, 107.09]),
        "MSFT": price_frame([100.0, 101.1, 104.03, 102.16, 106.14, 107.41]),
        "JPM": price_frame([100.0, 100.9, 104.03, 102.05, 106.24, 107.30]),
        "XOM": price_frame([100.0, 97.0, 97.97, 101.89, 99.85, 99.85]),
    }
    sector_info = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
            },
            {
                "ticker": "MSFT",
                "company_name": "Microsoft Corporation",
                "sector": "Technology",
                "industry": "Software",
            },
            {
                "ticker": "JPM",
                "company_name": "JPMorgan Chase & Co.",
                "sector": "Financial Services",
                "industry": "Banks",
            },
            {
                "ticker": "XOM",
                "company_name": "Exxon Mobil Corporation",
                "sector": "Energy",
                "industry": "Oil & Gas",
            },
        ]
    )
    topic_stock = pd.DataFrame(
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

    builder = NetworkBuilder(correlation_threshold=0.95)
    return builder.build_graph(
        price_tables=price_tables,
        sector_info=sector_info,
        topic_stock=topic_stock,
    )


def test_get_stock_info_returns_topics_and_neighbors() -> None:
    """Stock info should include sector, related topics, and stock neighbors."""

    analyzer = NetworkAnalyzer(sample_graph())
    stock_info = analyzer.get_stock_info("AAPL")

    assert stock_info["ticker"] == "AAPL"
    assert stock_info["sector"] == "Technology"
    assert stock_info["related_topics"][0]["topic"] == "technology"
    assert any(neighbor["ticker"] == "MSFT" for neighbor in stock_info["top_neighbors"])


def test_get_sector_info_returns_stocks_and_connected_sectors() -> None:
    """Sector info should include member stocks and connected sectors."""

    analyzer = NetworkAnalyzer(sample_graph())
    sector_info = analyzer.get_sector_info("Technology")

    assert sector_info["sector"] == "Technology"
    assert len(sector_info["stocks"]) == 2
    assert sector_info["top_connected_sectors"][0]["sector"] == "Financial Services"


def test_compare_stocks_reports_common_topics_and_direct_edge() -> None:
    """Stock comparison should report direct edges and shared topics."""

    analyzer = NetworkAnalyzer(sample_graph())
    comparison = analyzer.compare_stocks("AAPL", "MSFT")

    assert comparison["same_sector"] is True
    assert comparison["direct_edge"]["edge_type"] == "stock_stock"
    assert "technology" in comparison["common_topics"]
    assert comparison["shortest_path"]["path_found"] is True


def test_find_shortest_path_returns_no_path_for_disconnected_nodes() -> None:
    """The analyzer should handle no-path cases gracefully."""

    graph = nx.Graph()
    graph.add_node("stock:AAPL", node_type="stock", ticker="AAPL", label="AAPL")
    graph.add_node("topic:energy", node_type="topic", topic="energy", label="energy")

    analyzer = NetworkAnalyzer(graph)
    path_result = analyzer.find_shortest_path("AAPL", "energy")

    assert path_result["path_found"] is False
    assert path_result["path_nodes"] == []
    assert path_result["edge_types"] == []


def test_get_neighbors_returns_neighbor_metadata() -> None:
    """Neighbor lookup should include node and edge context."""

    analyzer = NetworkAnalyzer(sample_graph())
    neighbors = analyzer.get_neighbors("AAPL")

    assert neighbors
    assert any(item["edge_type"] == "stock_topic" for item in neighbors)
    assert any(item["edge_type"] == "stock_stock" for item in neighbors)


def test_compute_metrics_return_non_empty_tables() -> None:
    """Degree and centrality metrics should return sorted DataFrames."""

    analyzer = NetworkAnalyzer(sample_graph())
    degree_metrics = analyzer.compute_degree_metrics()
    centrality_metrics = analyzer.compute_centrality_metrics()

    assert not degree_metrics.empty
    assert not centrality_metrics.empty
    assert "degree" in degree_metrics.columns
    assert "betweenness_centrality" in centrality_metrics.columns
