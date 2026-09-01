"""LLM assessment for one user-provided news item plus graph-grounded context."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from config import AppConfig, load_config
from llm_enricher import (
    EventType,
    ImpactDirection,
    ImpactStrength,
    LLMEnricherError,
    MarketRelevance,
    OpenAI,
    SectorImpact,
    StockImpact,
)
from network_analyzer import NetworkAnalyzer, NetworkAnalyzerError


class LLMNewsImpactAnalyzerError(Exception):
    """Raised when one-shot LLM news assessment cannot run safely."""


class TopicMatch(BaseModel):
    """Structured topic match for a user-provided news item."""

    topic: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class NewNewsImpactAssessment(BaseModel):
    """Structured LLM output for one new article entered by the user."""

    event_summary: str
    primary_event_type: EventType
    overall_market_relevance: MarketRelevance
    related_topics: list[TopicMatch]
    sector_impacts: list[SectorImpact]
    stock_impacts: list[StockImpact]
    overall_rationale: str


class LLMNewsImpactAnalyzer:
    """Assess one new news item against the current graph universe."""

    def __init__(
        self,
        config: AppConfig | None = None,
        client: Any | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the analyzer with config and an optional OpenAI client."""

        self.config = config or load_config()
        self.model = model or self.config.default_openai_model
        self.client = client or self._build_openai_client()

    def assess_news(
        self,
        title: str,
        summary: str,
        allowed_topics: list[str],
        allowed_sectors: list[str],
        allowed_tickers: list[str],
    ) -> NewNewsImpactAssessment:
        """Assess one new news item inside the current graph universe."""

        clean_title = str(title).strip()
        clean_summary = str(summary).strip()
        if not clean_title and not clean_summary:
            raise LLMNewsImpactAnalyzerError(
                "Provide at least a title or a summary for the new news item."
            )

        instructions = self._build_instructions(
            allowed_topics=allowed_topics,
            allowed_sectors=allowed_sectors,
            allowed_tickers=allowed_tickers,
        )
        payload = self._build_payload(title=clean_title, summary=clean_summary)

        try:
            response = self.client.responses.parse(
                model=self.model,
                instructions=instructions,
                input=payload,
                text_format=NewNewsImpactAssessment,
                temperature=0,
            )
        except Exception as exc:  # pragma: no cover - third-party runtime safeguard
            raise LLMNewsImpactAnalyzerError(
                f"OpenAI news assessment failed: {exc}"
            ) from exc

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise LLMNewsImpactAnalyzerError(
                "OpenAI returned no parsed output for the new news item."
            )

        return self._sanitize_assessment(
            assessment=parsed,
            allowed_topics=allowed_topics,
            allowed_sectors=allowed_sectors,
            allowed_tickers=allowed_tickers,
        )

    def _build_openai_client(self) -> Any:
        """Build an OpenAI client from project config."""

        if OpenAI is None:
            raise LLMNewsImpactAnalyzerError(
                "openai is not installed. Run 'pip install -r requirements.txt'."
            )

        if not self.config.openai_api_key:
            raise LLMNewsImpactAnalyzerError(
                "Missing OpenAI API key. Set the OPENAI_API_KEY environment variable."
            )

        return OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
        )

    def _build_instructions(
        self,
        allowed_topics: list[str],
        allowed_sectors: list[str],
        allowed_tickers: list[str],
    ) -> str:
        """Build the system instructions for one-shot impact assessment."""

        topics_text = ", ".join(allowed_topics) or "none"
        sectors_text = ", ".join(allowed_sectors) or "none"
        tickers_text = ", ".join(allowed_tickers) or "none"

        return (
            "You are assessing one new financial news item for a graph-based market "
            "explorer. Return only structured output that matches the schema. "
            "Do not predict prices, returns, or give investment advice. "
            "Instead, estimate which allowed topics, sectors, and stocks are most "
            "likely to be impacted by the news item, along with direction, strength, "
            "confidence, and short evidence-based rationales. "
            "Only use topics from this allowed list: "
            f"{topics_text}. "
            "Only use sectors from this allowed list: "
            f"{sectors_text}. "
            "Only use tickers from this allowed list: "
            f"{tickers_text}. "
            "If evidence is weak, omit the item or mark the direction as 'uncertain'. "
            "Keep outputs concise and grounded in the provided text."
        )

    def _build_payload(self, title: str, summary: str) -> list[dict[str, Any]]:
        """Build one Responses API payload for the user-provided news item."""

        content = {
            "title": title,
            "summary": summary,
        }
        return [
            {
                "role": "user",
                "content": json.dumps(content, ensure_ascii=True),
            }
        ]

    def _sanitize_assessment(
        self,
        assessment: NewNewsImpactAssessment,
        allowed_topics: list[str],
        allowed_sectors: list[str],
        allowed_tickers: list[str],
    ) -> NewNewsImpactAssessment:
        """Restrict outputs to the current graph universe and trim list sizes."""

        allowed_topic_set = set(allowed_topics)
        allowed_sector_set = set(allowed_sectors)
        allowed_ticker_set = set(allowed_tickers)

        filtered_topics = [
            item for item in assessment.related_topics if item.topic in allowed_topic_set
        ][:5]
        filtered_sector_impacts = [
            item
            for item in assessment.sector_impacts
            if item.sector in allowed_sector_set
        ][:5]
        filtered_stock_impacts = [
            item
            for item in assessment.stock_impacts
            if item.ticker in allowed_ticker_set
        ][:8]

        return NewNewsImpactAssessment(
            event_summary=assessment.event_summary,
            primary_event_type=assessment.primary_event_type,
            overall_market_relevance=assessment.overall_market_relevance,
            related_topics=filtered_topics,
            sector_impacts=filtered_sector_impacts,
            stock_impacts=filtered_stock_impacts,
            overall_rationale=assessment.overall_rationale,
        )


def build_assessment_frames(
    assessment: NewNewsImpactAssessment,
) -> dict[str, pd.DataFrame]:
    """Convert one assessment object into display-ready tables."""

    topic_frame = pd.DataFrame(
        [
            {
                "topic": item.topic,
                "confidence": item.confidence,
                "rationale": item.rationale,
            }
            for item in assessment.related_topics
        ]
    )
    sector_frame = pd.DataFrame(
        [
            {
                "sector": item.sector,
                "impact_direction": item.impact_direction,
                "impact_strength": item.impact_strength,
                "confidence": item.confidence,
                "rationale": item.rationale,
            }
            for item in assessment.sector_impacts
        ]
    )
    stock_frame = pd.DataFrame(
        [
            {
                "ticker": item.ticker,
                "impact_direction": item.impact_direction,
                "impact_strength": item.impact_strength,
                "confidence": item.confidence,
                "rationale": item.rationale,
            }
            for item in assessment.stock_impacts
        ]
    )

    return {
        "topics": topic_frame,
        "sector_impacts": sector_frame,
        "stock_impacts": stock_frame,
    }


def build_graph_grounding_frames(
    assessment: NewNewsImpactAssessment,
    analyzer: NetworkAnalyzer,
) -> dict[str, pd.DataFrame]:
    """Build deterministic graph-grounded context for one assessment."""

    selected_topics = {item.topic for item in assessment.related_topics}

    topic_rows: list[dict[str, Any]] = []
    for item in assessment.related_topics:
        node_id = f"topic:{item.topic}"
        if node_id not in analyzer.graph:
            continue
        stock_neighbors = sorted(
            [
                analyzer.graph.nodes[neighbor_id].get("ticker")
                for neighbor_id in analyzer.graph.neighbors(node_id)
                if analyzer.graph.nodes[neighbor_id].get("node_type") == "stock"
            ]
        )
        topic_rows.append(
            {
                "topic": item.topic,
                "confidence": item.confidence,
                "graph_stock_count": len(stock_neighbors),
                "sample_stocks": ", ".join(stock_neighbors[:5]),
            }
        )

    sector_rows: list[dict[str, Any]] = []
    for item in assessment.sector_impacts:
        try:
            sector_info = analyzer.get_sector_info(item.sector)
        except NetworkAnalyzerError:
            continue

        sample_stocks = [row.get("ticker") for row in sector_info.get("stocks", [])[:5]]
        connected_sectors = [
            row.get("sector")
            for row in sector_info.get("top_connected_sectors", [])[:3]
        ]
        sector_rows.append(
            {
                "sector": item.sector,
                "impact_direction": item.impact_direction,
                "impact_strength": item.impact_strength,
                "confidence": item.confidence,
                "graph_stock_count": sector_info.get("stock_count"),
                "sample_stocks": ", ".join([stock for stock in sample_stocks if stock]),
                "top_connected_sectors": ", ".join(
                    [sector for sector in connected_sectors if sector]
                ),
            }
        )

    stock_rows: list[dict[str, Any]] = []
    for item in assessment.stock_impacts:
        try:
            stock_info = analyzer.get_stock_info(item.ticker)
        except NetworkAnalyzerError:
            continue

        overlapping_topics = [
            row.get("topic")
            for row in stock_info.get("related_topics", [])
            if row.get("topic") in selected_topics
        ]
        top_neighbors = [
            row.get("ticker") for row in stock_info.get("top_neighbors", [])[:3]
        ]
        stock_rows.append(
            {
                "ticker": item.ticker,
                "sector": stock_info.get("sector"),
                "impact_direction": item.impact_direction,
                "impact_strength": item.impact_strength,
                "confidence": item.confidence,
                "degree": stock_info.get("degree"),
                "degree_centrality": stock_info.get("degree_centrality"),
                "related_topic_overlap": ", ".join(
                    [topic for topic in overlapping_topics if topic]
                ),
                "top_neighbors": ", ".join(
                    [ticker for ticker in top_neighbors if ticker]
                ),
            }
        )

    return {
        "topic_grounding": pd.DataFrame(topic_rows),
        "sector_grounding": pd.DataFrame(sector_rows),
        "stock_grounding": pd.DataFrame(stock_rows),
    }


def extract_graph_universe(analyzer: NetworkAnalyzer) -> dict[str, list[str]]:
    """Extract allowed stock, sector, and topic labels from the current graph."""

    topics = sorted(
        [
            data.get("topic")
            for _, data in analyzer.graph.nodes(data=True)
            if data.get("node_type") == "topic" and data.get("topic")
        ]
    )
    sectors = sorted(
        [
            data.get("sector")
            for _, data in analyzer.graph.nodes(data=True)
            if data.get("node_type") == "sector" and data.get("sector")
        ]
    )
    tickers = sorted(
        [
            data.get("ticker")
            for _, data in analyzer.graph.nodes(data=True)
            if data.get("node_type") == "stock" and data.get("ticker")
        ]
    )
    return {
        "topics": topics,
        "sectors": sectors,
        "tickers": tickers,
    }
