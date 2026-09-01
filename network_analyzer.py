"""Graph analysis utilities for the market exploration network."""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd


class NetworkAnalyzerError(Exception):
    """Raised when graph analysis inputs are invalid or unresolved."""


class NetworkAnalyzer:
    """Run graph queries and metrics on the market network."""

    def __init__(self, graph: nx.Graph) -> None:
        """Initialize the analyzer with a built NetworkX graph."""

        self.graph = graph

    def get_stock_info(self, ticker: str) -> dict[str, Any]:
        """Return stock-level graph context for one ticker."""

        node_id = self._resolve_node_id(ticker, expected_type="stock")
        node_data = self.graph.nodes[node_id]
        centrality = nx.degree_centrality(self.graph)

        topic_neighbors = []
        stock_neighbors = []
        # Split neighbor collection by node type so the CLI can present
        # "related topics" and "similar stocks" as separate sections.
        for neighbor_id, edge_data in self.graph[node_id].items():
            neighbor_data = self.graph.nodes[neighbor_id]
            if neighbor_data.get("node_type") == "topic":
                topic_neighbors.append(
                    {
                        "node_id": neighbor_id,
                        "topic": neighbor_data.get("topic"),
                        "weight": edge_data.get("weight"),
                        "article_count": edge_data.get("article_count"),
                        "avg_ticker_sentiment": edge_data.get(
                            "avg_ticker_sentiment"
                        ),
                    }
                )
            elif neighbor_data.get("node_type") == "stock" and edge_data.get(
                "edge_type"
            ) == "stock_stock":
                stock_neighbors.append(
                    {
                        "node_id": neighbor_id,
                        "ticker": neighbor_data.get("ticker"),
                        "correlation": edge_data.get("correlation"),
                        "overlap_days": edge_data.get("overlap_days"),
                    }
                )

        topic_neighbors.sort(
            key=lambda item: (item.get("weight") or 0, item.get("article_count") or 0),
            reverse=True,
        )
        stock_neighbors.sort(
            key=lambda item: item.get("correlation") or 0,
            reverse=True,
        )

        return {
            "node_id": node_id,
            "ticker": node_data.get("ticker"),
            "company_name": node_data.get("company_name"),
            "sector": node_data.get("sector"),
            "industry": node_data.get("industry"),
            "degree": self.graph.degree(node_id),
            "degree_centrality": centrality.get(node_id),
            "related_topics": topic_neighbors,
            "top_neighbors": stock_neighbors,
        }

    def get_sector_info(self, sector_name: str) -> dict[str, Any]:
        """Return sector-level graph context and connected nodes."""

        node_id = self._resolve_node_id(sector_name, expected_type="sector")
        node_data = self.graph.nodes[node_id]

        stocks = []
        connected_sectors = []
        for neighbor_id, edge_data in self.graph[node_id].items():
            neighbor_data = self.graph.nodes[neighbor_id]
            if neighbor_data.get("node_type") == "stock":
                stocks.append(
                    {
                        "node_id": neighbor_id,
                        "ticker": neighbor_data.get("ticker"),
                        "company_name": neighbor_data.get("company_name"),
                        "degree": self.graph.degree(neighbor_id),
                    }
                )
            elif neighbor_data.get("node_type") == "sector":
                connected_sectors.append(
                    {
                        "node_id": neighbor_id,
                        "sector": neighbor_data.get("sector"),
                        "average_correlation": edge_data.get("average_correlation"),
                        "stock_edge_count": edge_data.get("stock_edge_count"),
                    }
                )

        stocks.sort(key=lambda item: item["degree"], reverse=True)
        connected_sectors.sort(
            key=lambda item: item.get("average_correlation") or 0,
            reverse=True,
        )

        return {
            "node_id": node_id,
            "sector": node_data.get("sector"),
            "stock_count": node_data.get("stock_count", len(stocks)),
            "stocks": stocks,
            "top_connected_sectors": connected_sectors,
        }

    def compare_stocks(self, ticker1: str, ticker2: str) -> dict[str, Any]:
        """Compare two stock nodes in the graph."""

        node_one = self._resolve_node_id(ticker1, expected_type="stock")
        node_two = self._resolve_node_id(ticker2, expected_type="stock")
        data_one = self.graph.nodes[node_one]
        data_two = self.graph.nodes[node_two]

        direct_edge = self.graph.get_edge_data(node_one, node_two)
        topics_one = self._neighbor_name_set(node_one, neighbor_type="topic")
        topics_two = self._neighbor_name_set(node_two, neighbor_type="topic")
        common_topics = sorted(topics_one & topics_two)

        shortest_path = self.find_shortest_path(node_one, node_two)
        centrality = nx.degree_centrality(self.graph)

        return {
            "ticker1": data_one.get("ticker"),
            "ticker2": data_two.get("ticker"),
            "same_sector": data_one.get("sector") == data_two.get("sector"),
            "direct_edge": direct_edge,
            "common_topics": common_topics,
            "shortest_path": shortest_path,
            "degree_difference": abs(
                self.graph.degree(node_one) - self.graph.degree(node_two)
            ),
            "degree_centrality": {
                data_one.get("ticker"): centrality.get(node_one),
                data_two.get("ticker"): centrality.get(node_two),
            },
        }

    def find_shortest_path(self, node_a: str, node_b: str) -> dict[str, Any]:
        """Return the shortest path, if any, between two graph nodes."""

        # Users can type raw tickers / sector names / topic names, so resolve
        # them before asking NetworkX for the actual graph path.
        source = self._resolve_node_id(node_a)
        target = self._resolve_node_id(node_b)

        try:
            path = nx.shortest_path(self.graph, source=source, target=target)
        except nx.NetworkXNoPath:
            return {
                "path_found": False,
                "path_nodes": [],
                "edge_types": [],
                "path_length": None,
            }

        edge_types = []
        for index in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[index], path[index + 1], {})
            edge_types.append(edge_data.get("edge_type"))

        return {
            "path_found": True,
            "path_nodes": path,
            "edge_types": edge_types,
            "path_length": len(path) - 1,
        }

    def get_neighbors(self, node_id: str) -> list[dict[str, Any]]:
        """Return all neighbors for one node with edge metadata."""

        resolved_node = self._resolve_node_id(node_id)
        neighbors = []

        for neighbor_id, edge_data in self.graph[resolved_node].items():
            neighbor_data = self.graph.nodes[neighbor_id]
            neighbors.append(
                {
                    "node_id": neighbor_id,
                    "node_type": neighbor_data.get("node_type"),
                    "label": neighbor_data.get("label"),
                    "edge_type": edge_data.get("edge_type"),
                    "weight": edge_data.get("weight"),
                }
            )

        neighbors.sort(
            key=lambda item: (
                item.get("edge_type") or "",
                -(item.get("weight") or 0),
                item.get("label") or "",
            )
        )
        return neighbors

    def compute_degree_metrics(self) -> pd.DataFrame:
        """Compute degree counts for every node in the graph."""

        rows = []
        for node_id, node_data in self.graph.nodes(data=True):
            rows.append(
                {
                    "node_id": node_id,
                    "node_type": node_data.get("node_type"),
                    "label": node_data.get("label"),
                    "degree": self.graph.degree(node_id),
                }
            )

        degree_metrics = pd.DataFrame(rows)
        if degree_metrics.empty:
            return degree_metrics

        return degree_metrics.sort_values(
            ["degree", "node_type", "label"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    def compute_centrality_metrics(self) -> pd.DataFrame:
        """Compute common graph centrality metrics for every node."""

        if self.graph.number_of_nodes() == 0:
            return pd.DataFrame(
                columns=[
                    "node_id",
                    "node_type",
                    "label",
                    "degree_centrality",
                    "betweenness_centrality",
                    "closeness_centrality",
                ]
            )

        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        closeness_centrality = nx.closeness_centrality(self.graph)

        rows = []
        for node_id, node_data in self.graph.nodes(data=True):
            rows.append(
                {
                    "node_id": node_id,
                    "node_type": node_data.get("node_type"),
                    "label": node_data.get("label"),
                    "degree_centrality": degree_centrality.get(node_id),
                    "betweenness_centrality": betweenness_centrality.get(node_id),
                    "closeness_centrality": closeness_centrality.get(node_id),
                }
            )

        centrality_metrics = pd.DataFrame(rows)
        return centrality_metrics.sort_values(
            ["betweenness_centrality", "degree_centrality", "label"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    def _resolve_node_id(
        self,
        value: str,
        expected_type: str | None = None,
    ) -> str:
        """Resolve a user-facing stock, sector, or topic name to a graph node id."""

        if not isinstance(value, str):
            raise NetworkAnalyzerError("Node identifiers must be strings.")

        cleaned = value.strip()
        if not cleaned:
            raise NetworkAnalyzerError("Node identifiers cannot be empty.")

        if cleaned in self.graph:
            if expected_type and self.graph.nodes[cleaned].get("node_type") != expected_type:
                raise NetworkAnalyzerError(
                    f"Node '{cleaned}' is not a {expected_type} node."
                )
            return cleaned

        # Most users will enter a bare ticker such as "AAPL" instead of the
        # internal node id "stock:AAPL".
        stock_candidate = f"stock:{cleaned.upper()}"
        if stock_candidate in self.graph:
            return self._validate_expected_type(stock_candidate, expected_type)

        # Sector and topic lookups stay case-insensitive to make the CLI more
        # forgiving during interactive use.
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get("node_type")
            if expected_type and node_type != expected_type:
                continue

            if node_type == "sector" and str(node_data.get("sector", "")).lower() == cleaned.lower():
                return node_id
            if node_type == "topic" and str(node_data.get("topic", "")).lower() == cleaned.lower():
                return node_id

        raise NetworkAnalyzerError(f"Could not resolve node '{value}'.")

    def _validate_expected_type(
        self,
        node_id: str,
        expected_type: str | None,
    ) -> str:
        """Validate a resolved node type when the caller expects one."""

        if expected_type and self.graph.nodes[node_id].get("node_type") != expected_type:
            raise NetworkAnalyzerError(
                f"Node '{node_id}' is not a {expected_type} node."
            )
        return node_id

    def _neighbor_name_set(self, node_id: str, neighbor_type: str) -> set[str]:
        """Collect one set of neighbor labels for a specific node type."""

        values: set[str] = set()
        for neighbor_id in self.graph.neighbors(node_id):
            neighbor_data = self.graph.nodes[neighbor_id]
            if neighbor_data.get("node_type") != neighbor_type:
                continue

            if neighbor_type == "topic":
                values.add(str(neighbor_data.get("topic")))
            elif neighbor_type == "sector":
                values.add(str(neighbor_data.get("sector")))
            else:
                values.add(str(neighbor_data.get("label")))

        return values
