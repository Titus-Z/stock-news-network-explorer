"""Tests for one-shot LLM new-news impact analysis helpers."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import pandas as pd

from llm_news_impact_analyzer import (
    LLMNewsImpactAnalyzer,
    NewNewsImpactAssessment,
    TopicMatch,
    build_assessment_frames,
    build_graph_grounding_frames,
)
from llm_enricher import SectorImpact, StockImpact
from network_analyzer import NetworkAnalyzer


class FakeResponsesAPI:
    """Small fake Responses API wrapper for deterministic tests."""

    def __init__(self, parsed_output: NewNewsImpactAssessment) -> None:
        self.parsed_output = parsed_output

    def parse(self, **kwargs):
        """Return an object that mimics the SDK parsed response."""

        return SimpleNamespace(output_parsed=self.parsed_output)


class FakeClient:
    """Small fake OpenAI client wrapper used in unit tests."""

    def __init__(self, parsed_output: NewNewsImpactAssessment) -> None:
        self.responses = FakeResponsesAPI(parsed_output)


def sample_assessment() -> NewNewsImpactAssessment:
    """Build one parsed LLM assessment object for tests."""

    return NewNewsImpactAssessment(
        event_summary="Apple announced a large AI partnership.",
        primary_event_type="product",
        overall_market_relevance="high",
        related_topics=[
            TopicMatch(
                topic="technology",
                confidence=0.9,
                rationale="The article focuses on AI products.",
            ),
            TopicMatch(
                topic="energy",
                confidence=0.2,
                rationale="Should be dropped outside the allowed topic set.",
            ),
        ],
        sector_impacts=[
            SectorImpact(
                sector="Technology",
                impact_direction="positive",
                impact_strength="high",
                confidence=0.8,
                rationale="The news directly supports tech spending.",
            ),
            SectorImpact(
                sector="Healthcare",
                impact_direction="uncertain",
                impact_strength="low",
                confidence=0.2,
                rationale="Should be dropped outside the allowed sector set.",
            ),
        ],
        stock_impacts=[
            StockImpact(
                ticker="AAPL",
                impact_direction="positive",
                impact_strength="high",
                confidence=0.85,
                rationale="Apple is directly involved.",
            ),
            StockImpact(
                ticker="TSLA",
                impact_direction="uncertain",
                impact_strength="low",
                confidence=0.2,
                rationale="Should be dropped outside the allowed ticker set.",
            ),
        ],
        overall_rationale="The article describes a large AI-related corporate action.",
    )


def build_graph() -> NetworkAnalyzer:
    """Build a tiny graph for graph-grounding tests."""

    graph = nx.Graph()
    graph.add_node(
        "stock:AAPL",
        node_type="stock",
        label="AAPL",
        ticker="AAPL",
        sector="Technology",
        industry="Consumer Electronics",
        company_name="Apple Inc.",
    )
    graph.add_node(
        "stock:MSFT",
        node_type="stock",
        label="MSFT",
        ticker="MSFT",
        sector="Technology",
        industry="Software",
        company_name="Microsoft Corporation",
    )
    graph.add_node(
        "sector:Technology",
        node_type="sector",
        label="Technology",
        sector="Technology",
        stock_count=2,
    )
    graph.add_node(
        "topic:technology",
        node_type="topic",
        label="technology",
        topic="technology",
        article_count=12,
    )
    graph.add_edge("stock:AAPL", "sector:Technology", edge_type="stock_sector", weight=1.0)
    graph.add_edge("stock:MSFT", "sector:Technology", edge_type="stock_sector", weight=1.0)
    graph.add_edge(
        "stock:AAPL",
        "topic:technology",
        edge_type="stock_topic",
        weight=10.0,
        article_count=10,
        avg_ticker_sentiment=0.2,
    )
    graph.add_edge(
        "stock:MSFT",
        "topic:technology",
        edge_type="stock_topic",
        weight=8.0,
        article_count=8,
        avg_ticker_sentiment=0.1,
    )
    graph.add_edge(
        "stock:AAPL",
        "stock:MSFT",
        edge_type="stock_stock",
        correlation=0.7,
        overlap_days=200,
        weight=0.7,
    )
    return NetworkAnalyzer(graph)


def test_assess_news_filters_to_allowed_universe() -> None:
    """Assessment results should be clipped to the current graph universe."""

    analyzer = LLMNewsImpactAnalyzer(client=FakeClient(sample_assessment()))

    assessment = analyzer.assess_news(
        title="Apple expands AI",
        summary="Apple announced a large AI partnership.",
        allowed_topics=["technology"],
        allowed_sectors=["Technology"],
        allowed_tickers=["AAPL", "MSFT"],
    )

    assert [item.topic for item in assessment.related_topics] == ["technology"]
    assert [item.sector for item in assessment.sector_impacts] == ["Technology"]
    assert [item.ticker for item in assessment.stock_impacts] == ["AAPL"]


def test_build_graph_grounding_frames_returns_graph_context_rows() -> None:
    """Graph grounding should expose stock, sector, and topic context tables."""

    assessment = NewNewsImpactAssessment(
        event_summary="Apple announced a large AI partnership.",
        primary_event_type="product",
        overall_market_relevance="high",
        related_topics=[
            TopicMatch(
                topic="technology",
                confidence=0.9,
                rationale="The article focuses on AI products.",
            )
        ],
        sector_impacts=[
            SectorImpact(
                sector="Technology",
                impact_direction="positive",
                impact_strength="high",
                confidence=0.8,
                rationale="The news directly supports tech spending.",
            )
        ],
        stock_impacts=[
            StockImpact(
                ticker="AAPL",
                impact_direction="positive",
                impact_strength="high",
                confidence=0.85,
                rationale="Apple is directly involved.",
            )
        ],
        overall_rationale="The article describes a large AI-related corporate action.",
    )

    grounding = build_graph_grounding_frames(assessment, build_graph())
    assessment_frames = build_assessment_frames(assessment)

    assert assessment_frames["topics"].loc[0, "topic"] == "technology"
    assert grounding["topic_grounding"].loc[0, "graph_stock_count"] == 2
    assert grounding["sector_grounding"].loc[0, "graph_stock_count"] == 2
    assert grounding["stock_grounding"].loc[0, "ticker"] == "AAPL"
    assert grounding["stock_grounding"].loc[0, "related_topic_overlap"] == "technology"
