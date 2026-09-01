"""Tests for Phase 5 CLI helpers."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from cli import (
    filter_news_tables_by_tickers,
    format_graph_summary,
    format_path_result,
    format_stock_comparison,
    format_stock_info,
    format_top_central_nodes,
)


def test_format_graph_summary_includes_node_and_edge_counts() -> None:
    """The graph summary formatter should show the main count groups."""

    summary = {
        "node_counts": Counter({"stock": 5, "topic": 10}),
        "edge_counts": Counter({"stock_topic": 20, "stock_stock": 4}),
    }
    context = {
        "price_table_count": 5,
        "sector_row_count": 5,
        "article_count": 100,
        "topic_stock_row_count": 20,
    }

    text = format_graph_summary(summary, context)

    assert "Local analysis summary" in text
    assert "- stock: 5" in text
    assert "- stock_topic: 20" in text


def test_format_stock_info_lists_neighbors_and_topics() -> None:
    """The stock formatter should include the core stock sections."""

    stock_info = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "degree": 12,
        "degree_centrality": 0.42,
        "top_neighbors": [
            {"ticker": "MSFT", "correlation": 0.91, "overlap_days": 200}
        ],
        "related_topics": [
            {"topic": "technology", "weight": 50, "article_count": 50}
        ],
    }

    text = format_stock_info(stock_info)

    assert "Stock: AAPL" in text
    assert "MSFT" in text
    assert "technology" in text


def test_format_stock_comparison_handles_missing_direct_edge() -> None:
    """Comparison formatting should gracefully handle no direct edge."""

    comparison = {
        "ticker1": "AAPL",
        "ticker2": "XOM",
        "same_sector": False,
        "common_topics": [],
        "degree_difference": 3,
        "direct_edge": None,
        "shortest_path": {
            "path_found": False,
            "path_nodes": [],
            "edge_types": [],
        },
    }

    text = format_stock_comparison(comparison)

    assert "same sector: False" in text
    assert "direct edge: none" in text
    assert "shortest path: none" in text


def test_format_path_result_handles_no_path_case() -> None:
    """Path formatting should show a clean message when no path exists."""

    text = format_path_result(
        {
            "path_found": False,
            "path_nodes": [],
            "edge_types": [],
            "path_length": None,
        }
    )

    assert text == "No path found."


def test_format_top_central_nodes_uses_dataframe_rows() -> None:
    """The centrality formatter should print ranked node rows."""

    centrality_metrics = pd.DataFrame(
        [
            {
                "label": "AAPL",
                "node_type": "stock",
                "betweenness_centrality": 0.2,
                "degree_centrality": 0.5,
                "closeness_centrality": 0.4,
            }
        ]
    )

    text = format_top_central_nodes(centrality_metrics)

    assert "Top central nodes:" in text
    assert "AAPL (stock)" in text


def test_filter_news_tables_by_tickers_keeps_matching_articles() -> None:
    """Ticker filtering should also update the article subset used in summaries."""

    news_tables = {
        "articles": pd.DataFrame(
            [
                {"article_id": "a1", "title": "One"},
                {"article_id": "a2", "title": "Two"},
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
                {"ticker": "AAPL", "topic": "technology", "article_count": 1},
                {"ticker": "MSFT", "topic": "software", "article_count": 1},
            ]
        ),
    }

    filtered = filter_news_tables_by_tickers(news_tables, ["AAPL"])

    assert filtered["articles"]["article_id"].tolist() == ["a1"]
    assert filtered["article_tickers"]["ticker"].tolist() == ["AAPL"]
    assert filtered["topic_stock"]["ticker"].tolist() == ["AAPL"]
