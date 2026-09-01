"""Convert one new-news LLM assessment into graph-compatible preview updates."""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd

from llm_news_impact_analyzer import NewNewsImpactAssessment


IMPACT_DIRECTION_SIGNS = {
    "positive": 1.0,
    "negative": -1.0,
    "mixed": 0.0,
    "uncertain": 0.0,
}
IMPACT_STRENGTH_WEIGHTS = {
    "low": 1.0,
    "medium": 2.0,
    "high": 3.0,
}
MARKET_RELEVANCE_WEIGHTS = {
    "low": 1.0,
    "medium": 2.0,
    "high": 3.0,
}


class NewsGraphAugmenterError(Exception):
    """Raised when one-shot news graph augmentation cannot be computed safely."""


class NewsGraphAugmenter:
    """Build a preview graph update from one LLM-assessed news item."""

    SYNTHETIC_TOPIC_STOCK_COLUMNS = [
        "ticker",
        "topic",
        "article_count",
        "avg_topic_relevance",
        "avg_ticker_relevance",
        "avg_ticker_sentiment",
        "avg_overall_sentiment",
        "impact_score",
        "sector",
        "impact_direction",
        "impact_strength",
        "topic_confidence",
        "stock_confidence",
        "market_relevance",
        "sector_support",
    ]
    STOCK_TOPIC_UPDATE_COLUMNS = SYNTHETIC_TOPIC_STOCK_COLUMNS + [
        "edge_status",
        "existing_article_count",
        "projected_article_count",
        "existing_weight",
    ]

    def build_preview(
        self,
        assessment: NewNewsImpactAssessment,
        graph: nx.Graph,
    ) -> dict[str, Any]:
        """Build synthetic rows, edge updates, and an augmented preview graph."""

        synthetic_rows = self.build_synthetic_topic_stock_rows(assessment, graph)
        update_frame = self.build_stock_topic_update_frame(graph, synthetic_rows)
        augmented_graph = self.augment_graph(graph, update_frame)

        return {
            "synthetic_topic_stock": synthetic_rows,
            "stock_topic_updates": update_frame,
            "sector_exposure": self.build_sector_exposure_frame(update_frame),
            "topic_exposure": self.build_topic_exposure_frame(update_frame),
            "summary": self.build_preview_summary(graph, augmented_graph, update_frame),
            "augmented_graph": augmented_graph,
        }

    def build_synthetic_topic_stock_rows(
        self,
        assessment: NewNewsImpactAssessment,
        graph: nx.Graph,
    ) -> pd.DataFrame:
        """Convert one new-news assessment into topic-stock rows."""

        if not assessment.related_topics or not assessment.stock_impacts:
            return pd.DataFrame(columns=self.SYNTHETIC_TOPIC_STOCK_COLUMNS)

        stock_sector_lookup = self._build_stock_sector_lookup(graph)
        sector_support_lookup = self._build_sector_support_lookup(assessment)
        market_weight = MARKET_RELEVANCE_WEIGHTS.get(
            assessment.overall_market_relevance,
            1.0,
        )

        rows: list[dict[str, Any]] = []
        for stock_impact in assessment.stock_impacts:
            ticker = str(stock_impact.ticker).upper()
            if ticker not in stock_sector_lookup:
                continue

            stock_strength_weight = IMPACT_STRENGTH_WEIGHTS.get(
                stock_impact.impact_strength,
                1.0,
            )
            direction_sign = IMPACT_DIRECTION_SIGNS.get(
                stock_impact.impact_direction,
                0.0,
            )
            sector_name = stock_sector_lookup[ticker]
            sector_support = sector_support_lookup.get(sector_name, 0.0)

            for topic_match in assessment.related_topics:
                impact_score = (
                    float(topic_match.confidence)
                    * float(stock_impact.confidence)
                    * stock_strength_weight
                    * market_weight
                    * (1.0 + sector_support)
                )
                rows.append(
                    {
                        "ticker": ticker,
                        "topic": topic_match.topic,
                        "article_count": 1,
                        "avg_topic_relevance": round(float(topic_match.confidence), 4),
                        "avg_ticker_relevance": round(float(stock_impact.confidence), 4),
                        "avg_ticker_sentiment": round(
                            direction_sign
                            * min(
                                1.0,
                                (stock_strength_weight / 3.0)
                                * float(stock_impact.confidence),
                            ),
                            4,
                        ),
                        "avg_overall_sentiment": round(
                            direction_sign * min(1.0, market_weight / 3.0),
                            4,
                        ),
                        "impact_score": round(float(impact_score), 4),
                        "sector": sector_name,
                        "impact_direction": stock_impact.impact_direction,
                        "impact_strength": stock_impact.impact_strength,
                        "topic_confidence": round(float(topic_match.confidence), 4),
                        "stock_confidence": round(float(stock_impact.confidence), 4),
                        "market_relevance": assessment.overall_market_relevance,
                        "sector_support": round(float(sector_support), 4),
                    }
                )

        synthetic_rows = pd.DataFrame(rows, columns=self.SYNTHETIC_TOPIC_STOCK_COLUMNS)
        if synthetic_rows.empty:
            return synthetic_rows

        return synthetic_rows.sort_values(
            ["impact_score", "ticker", "topic"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    def build_stock_topic_update_frame(
        self,
        graph: nx.Graph,
        synthetic_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach edge status and projected counts to synthetic topic-stock rows."""

        if synthetic_rows.empty:
            return pd.DataFrame(columns=self.STOCK_TOPIC_UPDATE_COLUMNS)

        rows: list[dict[str, Any]] = []
        for _, row in synthetic_rows.iterrows():
            stock_node = self._stock_node_id(str(row["ticker"]))
            topic_node = self._topic_node_id(str(row["topic"]))
            edge_exists = graph.has_edge(stock_node, topic_node)
            existing_edge = graph.edges[stock_node, topic_node] if edge_exists else {}
            existing_article_count = int(existing_edge.get("article_count") or 0)
            rows.append(
                {
                    **row.to_dict(),
                    "edge_status": "strengthened" if edge_exists else "new",
                    "existing_article_count": existing_article_count,
                    "projected_article_count": existing_article_count
                    + int(row["article_count"]),
                    "existing_weight": float(existing_edge.get("weight") or 0.0),
                }
            )

        return pd.DataFrame(rows, columns=self.STOCK_TOPIC_UPDATE_COLUMNS).sort_values(
            ["impact_score", "ticker", "topic"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    def augment_graph(
        self,
        graph: nx.Graph,
        update_frame: pd.DataFrame,
    ) -> nx.Graph:
        """Overlay one synthetic topic-stock update frame onto the current graph."""

        augmented_graph = graph.copy()
        if update_frame.empty:
            return augmented_graph

        for _, row in update_frame.iterrows():
            stock_node = self._stock_node_id(str(row["ticker"]))
            topic_node = self._topic_node_id(str(row["topic"]))
            if stock_node not in augmented_graph:
                continue

            if topic_node not in augmented_graph:
                augmented_graph.add_node(
                    topic_node,
                    node_type="topic",
                    topic=str(row["topic"]),
                    label=str(row["topic"]),
                    article_count=0,
                )

            topic_data = augmented_graph.nodes[topic_node]
            topic_data["article_count"] = int(topic_data.get("article_count") or 0) + int(
                row["article_count"]
            )
            topic_data["preview_article_delta"] = int(
                topic_data.get("preview_article_delta") or 0
            ) + int(row["article_count"])

            if augmented_graph.has_edge(stock_node, topic_node):
                edge_data = augmented_graph.edges[stock_node, topic_node]
                existing_article_count = int(edge_data.get("article_count") or 0)
                added_article_count = int(row["article_count"])
                combined_article_count = existing_article_count + added_article_count
                edge_data["article_count"] = combined_article_count
                edge_data["avg_topic_relevance"] = self._weighted_average(
                    edge_data.get("avg_topic_relevance"),
                    existing_article_count,
                    row["avg_topic_relevance"],
                    added_article_count,
                )
                edge_data["avg_ticker_relevance"] = self._weighted_average(
                    edge_data.get("avg_ticker_relevance"),
                    existing_article_count,
                    row["avg_ticker_relevance"],
                    added_article_count,
                )
                edge_data["avg_ticker_sentiment"] = self._weighted_average(
                    edge_data.get("avg_ticker_sentiment"),
                    existing_article_count,
                    row["avg_ticker_sentiment"],
                    added_article_count,
                )
                edge_data["avg_overall_sentiment"] = self._weighted_average(
                    edge_data.get("avg_overall_sentiment"),
                    existing_article_count,
                    row["avg_overall_sentiment"],
                    added_article_count,
                )
                edge_data["preview_impact_score"] = round(
                    float(edge_data.get("preview_impact_score") or 0.0)
                    + float(row["impact_score"]),
                    4,
                )
                edge_data["preview_edge_status"] = "strengthened"
                edge_data["llm_augmented"] = True
            else:
                augmented_graph.add_edge(
                    stock_node,
                    topic_node,
                    edge_type="stock_topic",
                    weight=float(row["impact_score"]),
                    article_count=int(row["article_count"]),
                    avg_topic_relevance=float(row["avg_topic_relevance"]),
                    avg_ticker_relevance=float(row["avg_ticker_relevance"]),
                    avg_ticker_sentiment=float(row["avg_ticker_sentiment"]),
                    avg_overall_sentiment=float(row["avg_overall_sentiment"]),
                    preview_impact_score=float(row["impact_score"]),
                    preview_edge_status="new",
                    llm_augmented=True,
                )

        return augmented_graph

    def build_sector_exposure_frame(self, update_frame: pd.DataFrame) -> pd.DataFrame:
        """Aggregate synthetic updates into a sector exposure preview."""

        if update_frame.empty:
            return pd.DataFrame(
                columns=[
                    "sector",
                    "impacted_stocks",
                    "impacted_topics",
                    "new_edges",
                    "strengthened_edges",
                    "total_impact_score",
                ]
            )

        sector_frame = (
            update_frame.groupby("sector", as_index=False)
            .agg(
                impacted_stocks=("ticker", "nunique"),
                impacted_topics=("topic", "nunique"),
                total_impact_score=("impact_score", "sum"),
                avg_stock_confidence=("stock_confidence", "mean"),
            )
            .reset_index(drop=True)
        )

        status_counts = (
            update_frame.groupby(["sector", "edge_status"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        sector_frame = sector_frame.merge(status_counts, on="sector", how="left")
        for column in ["new", "strengthened"]:
            if column not in sector_frame:
                sector_frame[column] = 0

        sector_frame = sector_frame.rename(
            columns={
                "new": "new_edges",
                "strengthened": "strengthened_edges",
            }
        )
        sector_frame["total_impact_score"] = sector_frame["total_impact_score"].round(4)
        sector_frame["avg_stock_confidence"] = sector_frame[
            "avg_stock_confidence"
        ].round(4)

        return sector_frame.sort_values(
            ["total_impact_score", "sector"],
            ascending=[False, True],
        ).reset_index(drop=True)

    def build_topic_exposure_frame(self, update_frame: pd.DataFrame) -> pd.DataFrame:
        """Aggregate synthetic updates into a topic exposure preview."""

        if update_frame.empty:
            return pd.DataFrame(
                columns=[
                    "topic",
                    "impacted_stocks",
                    "impacted_sectors",
                    "total_impact_score",
                    "avg_topic_confidence",
                ]
            )

        topic_frame = (
            update_frame.groupby("topic", as_index=False)
            .agg(
                impacted_stocks=("ticker", "nunique"),
                impacted_sectors=("sector", "nunique"),
                total_impact_score=("impact_score", "sum"),
                avg_topic_confidence=("topic_confidence", "mean"),
            )
            .reset_index(drop=True)
        )
        topic_frame["total_impact_score"] = topic_frame["total_impact_score"].round(4)
        topic_frame["avg_topic_confidence"] = topic_frame[
            "avg_topic_confidence"
        ].round(4)

        return topic_frame.sort_values(
            ["total_impact_score", "topic"],
            ascending=[False, True],
        ).reset_index(drop=True)

    def build_preview_summary(
        self,
        base_graph: nx.Graph,
        augmented_graph: nx.Graph,
        update_frame: pd.DataFrame,
    ) -> dict[str, Any]:
        """Summarize the structural effect of one synthetic news insertion."""

        base_stock_topic_edges = self._count_edges_by_type(base_graph, "stock_topic")
        augmented_stock_topic_edges = self._count_edges_by_type(
            augmented_graph,
            "stock_topic",
        )

        return {
            "synthetic_row_count": len(update_frame),
            "new_edges": int((update_frame["edge_status"] == "new").sum())
            if not update_frame.empty
            else 0,
            "strengthened_edges": int(
                (update_frame["edge_status"] == "strengthened").sum()
            )
            if not update_frame.empty
            else 0,
            "impacted_stocks": int(update_frame["ticker"].nunique())
            if not update_frame.empty
            else 0,
            "impacted_sectors": int(update_frame["sector"].nunique())
            if not update_frame.empty
            else 0,
            "impacted_topics": int(update_frame["topic"].nunique())
            if not update_frame.empty
            else 0,
            "total_impact_score": round(
                float(update_frame["impact_score"].sum()) if not update_frame.empty else 0.0,
                4,
            ),
            "base_stock_topic_edges": base_stock_topic_edges,
            "augmented_stock_topic_edges": augmented_stock_topic_edges,
        }

    def _build_stock_sector_lookup(self, graph: nx.Graph) -> dict[str, str]:
        """Map ticker symbols to sector names from the current graph."""

        return {
            data["ticker"]: str(data.get("sector") or "Unknown")
            for _, data in graph.nodes(data=True)
            if data.get("node_type") == "stock" and data.get("ticker")
        }

    def _build_sector_support_lookup(
        self,
        assessment: NewNewsImpactAssessment,
    ) -> dict[str, float]:
        """Build a soft sector support multiplier from sector impacts."""

        support_lookup: dict[str, float] = {}
        for item in assessment.sector_impacts:
            strength_weight = IMPACT_STRENGTH_WEIGHTS.get(item.impact_strength, 1.0)
            support_lookup[item.sector] = round(
                (strength_weight * float(item.confidence)) / 6.0,
                4,
            )
        return support_lookup

    def _count_edges_by_type(self, graph: nx.Graph, edge_type: str) -> int:
        """Count graph edges of one type."""

        return sum(
            1
            for _, _, data in graph.edges(data=True)
            if data.get("edge_type") == edge_type
        )

    def _weighted_average(
        self,
        existing_value: Any,
        existing_count: int,
        added_value: Any,
        added_count: int,
    ) -> float:
        """Compute a simple weighted average for one projected edge attribute."""

        clean_existing = float(existing_value or 0.0)
        clean_added = float(added_value or 0.0)
        total_count = existing_count + added_count
        if total_count <= 0:
            return 0.0
        return round(
            ((clean_existing * existing_count) + (clean_added * added_count))
            / total_count,
            4,
        )

    def _stock_node_id(self, ticker: str) -> str:
        """Build the canonical stock node id."""

        return f"stock:{ticker.upper()}"

    def _topic_node_id(self, topic: str) -> str:
        """Build the canonical topic node id."""

        return f"topic:{topic}"
