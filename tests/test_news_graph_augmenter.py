"""Tests for synthetic graph updates built from one new-news assessment."""

from __future__ import annotations

import networkx as nx

from llm_enricher import SectorImpact, StockImpact
from llm_news_impact_analyzer import NewNewsImpactAssessment, TopicMatch
from news_graph_augmenter import NewsGraphAugmenter


def build_preview_graph() -> nx.Graph:
    """Build a small stock-topic graph for augmentation tests."""

    graph = nx.Graph()
    graph.add_node(
        "stock:AAPL",
        node_type="stock",
        label="AAPL",
        ticker="AAPL",
        sector="Technology",
    )
    graph.add_node(
        "stock:MSFT",
        node_type="stock",
        label="MSFT",
        ticker="MSFT",
        sector="Technology",
    )
    graph.add_node(
        "topic:technology",
        node_type="topic",
        label="technology",
        topic="technology",
        article_count=12,
    )
    graph.add_node(
        "topic:earnings",
        node_type="topic",
        label="earnings",
        topic="earnings",
        article_count=8,
    )
    graph.add_edge(
        "stock:AAPL",
        "topic:technology",
        edge_type="stock_topic",
        weight=10.0,
        article_count=10,
        avg_topic_relevance=0.8,
        avg_ticker_relevance=0.9,
        avg_ticker_sentiment=0.2,
        avg_overall_sentiment=0.1,
    )
    return graph


def build_assessment() -> NewNewsImpactAssessment:
    """Build one deterministic new-news assessment."""

    return NewNewsImpactAssessment(
        event_summary="Large enterprise AI rollout.",
        primary_event_type="product",
        overall_market_relevance="high",
        related_topics=[
            TopicMatch(
                topic="technology",
                confidence=0.9,
                rationale="The article is clearly about AI technology.",
            ),
            TopicMatch(
                topic="earnings",
                confidence=0.4,
                rationale="The article may affect future earnings expectations.",
            ),
        ],
        sector_impacts=[
            SectorImpact(
                sector="Technology",
                impact_direction="positive",
                impact_strength="high",
                confidence=0.8,
                rationale="The article directly supports technology demand.",
            )
        ],
        stock_impacts=[
            StockImpact(
                ticker="AAPL",
                impact_direction="positive",
                impact_strength="high",
                confidence=0.9,
                rationale="Apple is directly involved.",
            ),
            StockImpact(
                ticker="MSFT",
                impact_direction="positive",
                impact_strength="low",
                confidence=0.3,
                rationale="Microsoft is indirectly exposed.",
            ),
        ],
        overall_rationale="The article suggests a meaningful AI-related corporate move.",
    )


def test_build_synthetic_topic_stock_rows_respects_impact_strength() -> None:
    """Synthetic topic-stock rows should encode stronger news with larger scores."""

    augmenter = NewsGraphAugmenter()

    preview_rows = augmenter.build_synthetic_topic_stock_rows(
        build_assessment(),
        build_preview_graph(),
    )

    assert len(preview_rows) == 4

    aapl_technology = preview_rows[
        (preview_rows["ticker"] == "AAPL")
        & (preview_rows["topic"] == "technology")
    ].iloc[0]
    msft_earnings = preview_rows[
        (preview_rows["ticker"] == "MSFT")
        & (preview_rows["topic"] == "earnings")
    ].iloc[0]

    assert aapl_technology["impact_score"] > msft_earnings["impact_score"]
    assert aapl_technology["avg_ticker_sentiment"] > msft_earnings["avg_ticker_sentiment"]
    assert aapl_technology["sector"] == "Technology"


def test_build_preview_marks_new_and_strengthened_edges() -> None:
    """Preview output should distinguish new edges from strengthened edges."""

    augmenter = NewsGraphAugmenter()
    preview = augmenter.build_preview(build_assessment(), build_preview_graph())

    update_frame = preview["stock_topic_updates"]
    summary = preview["summary"]
    augmented_graph = preview["augmented_graph"]

    assert summary["synthetic_row_count"] == 4
    assert summary["new_edges"] == 3
    assert summary["strengthened_edges"] == 1
    assert summary["base_stock_topic_edges"] == 1
    assert summary["augmented_stock_topic_edges"] == 4

    strengthened_row = update_frame[
        (update_frame["ticker"] == "AAPL")
        & (update_frame["topic"] == "technology")
    ].iloc[0]
    assert strengthened_row["edge_status"] == "strengthened"
    assert strengthened_row["projected_article_count"] == 11

    assert augmented_graph.has_edge("stock:MSFT", "topic:technology")
    assert (
        augmented_graph.edges["stock:AAPL", "topic:technology"]["preview_edge_status"]
        == "strengthened"
    )
    assert augmented_graph.edges["stock:MSFT", "topic:technology"]["llm_augmented"] is True
