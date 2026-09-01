"""Graph construction utilities for the market exploration network."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import networkx as nx
import pandas as pd


class NetworkBuilderError(Exception):
    """Raised when graph inputs or builder configuration are invalid."""


class NetworkBuilder:
    """Build the stock, sector, and topic graph for the project."""

    VALID_TOPIC_WEIGHT_COLUMNS = {
        "article_count",
        "avg_topic_relevance",
        "avg_ticker_relevance",
        "avg_ticker_sentiment",
        "avg_overall_sentiment",
    }

    def __init__(
        self,
        correlation_threshold: float = 0.6,
        top_k_neighbors: int | None = None,
        stock_topic_weight_column: str = "article_count",
        min_correlation_periods: int = 3,
    ) -> None:
        """Initialize graph-building configuration."""

        if top_k_neighbors is not None and top_k_neighbors <= 0:
            raise NetworkBuilderError("top_k_neighbors must be a positive integer.")

        if stock_topic_weight_column not in self.VALID_TOPIC_WEIGHT_COLUMNS:
            raise NetworkBuilderError(
                f"Unsupported stock-topic weight column "
                f"'{stock_topic_weight_column}'."
            )

        if min_correlation_periods < 2:
            raise NetworkBuilderError(
                "min_correlation_periods must be at least 2."
            )

        self.correlation_threshold = correlation_threshold
        self.top_k_neighbors = top_k_neighbors
        self.stock_topic_weight_column = stock_topic_weight_column
        self.min_correlation_periods = min_correlation_periods

    def build_graph(
        self,
        price_tables: dict[str, pd.DataFrame],
        sector_info: pd.DataFrame,
        topic_stock: pd.DataFrame,
    ) -> nx.Graph:
        """Build the full market graph from price, sector, and topic data."""

        graph = nx.Graph()
        graph.graph["builder"] = "NetworkBuilder"
        graph.graph["stock_topic_weight_column"] = self.stock_topic_weight_column
        graph.graph["correlation_threshold"] = self.correlation_threshold
        graph.graph["top_k_neighbors"] = self.top_k_neighbors

        sector_lookup = self._build_sector_lookup(sector_info)
        allowed_tickers = {
            self._normalize_ticker(ticker) for ticker in price_tables.keys()
        }

        # Add entity nodes first so later edge builders can attach metadata
        # without repeatedly re-creating the same records.
        self._add_stock_nodes(graph, price_tables, sector_lookup)
        self._add_sector_nodes_and_edges(graph, sector_info, allowed_tickers)
        self._add_topic_nodes_and_edges(
            graph,
            topic_stock,
            sector_lookup,
            allowed_tickers,
        )

        returns_table = self.build_returns_table(price_tables)
        self._add_stock_stock_edges(graph, returns_table)
        self._add_sector_sector_edges(graph)

        return graph

    def build_returns_table(self, price_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build a daily returns table from per-ticker price DataFrames."""

        return_series: dict[str, pd.Series] = {}

        for ticker, prices in price_tables.items():
            symbol = self._normalize_ticker(ticker)
            returns = self._build_single_return_series(symbol, prices)
            if not returns.empty:
                return_series[symbol] = returns

        if not return_series:
            return pd.DataFrame()

        returns_table = pd.concat(return_series, axis=1).sort_index()
        returns_table.index.name = "date"
        return returns_table

    def _build_single_return_series(
        self,
        ticker: str,
        prices: pd.DataFrame,
    ) -> pd.Series:
        """Build one daily returns series from a stock price table."""

        if prices.empty:
            return pd.Series(dtype=float, name=ticker)

        required_columns = {"date", "adjusted_close"}
        missing_columns = required_columns - set(prices.columns)
        if missing_columns:
            raise NetworkBuilderError(
                f"Price table for '{ticker}' is missing columns: "
                f"{sorted(missing_columns)}."
            )

        ordered_prices = prices.copy()
        ordered_prices["date"] = pd.to_datetime(ordered_prices["date"], errors="coerce")
        ordered_prices["adjusted_close"] = pd.to_numeric(
            ordered_prices["adjusted_close"],
            errors="coerce",
        )
        ordered_prices = ordered_prices.sort_values("date")

        returns = ordered_prices.set_index("date")["adjusted_close"].pct_change()
        returns = returns.dropna()
        returns.name = ticker

        return returns

    def _add_stock_nodes(
        self,
        graph: nx.Graph,
        price_tables: dict[str, pd.DataFrame],
        sector_lookup: dict[str, dict[str, Any]],
    ) -> None:
        """Add stock nodes with basic metadata."""

        for ticker, prices in price_tables.items():
            symbol = self._normalize_ticker(ticker)
            node_id = self._stock_node_id(symbol)
            sector_record = sector_lookup.get(symbol, {})

            graph.add_node(
                node_id,
                node_type="stock",
                ticker=symbol,
                label=symbol,
                sector=sector_record.get("sector"),
                industry=sector_record.get("industry"),
                company_name=sector_record.get("company_name"),
                price_points=len(prices),
            )

    def _add_sector_nodes_and_edges(
        self,
        graph: nx.Graph,
        sector_info: pd.DataFrame,
        allowed_tickers: set[str],
    ) -> None:
        """Add sector nodes and stock-sector membership edges."""

        if sector_info.empty:
            return

        clean_sector_info = sector_info.copy()
        clean_sector_info["ticker"] = clean_sector_info["ticker"].astype(str).str.upper()

        for _, row in clean_sector_info.iterrows():
            ticker = row.get("ticker")
            sector = row.get("sector")
            if pd.isna(ticker) or pd.isna(sector):
                continue
            if str(ticker).upper() not in allowed_tickers:
                continue

            stock_node = self._stock_node_id(str(ticker))
            sector_name = str(sector)
            sector_node = self._sector_node_id(sector_name)
            if stock_node not in graph:
                continue

            graph.nodes[stock_node]["sector"] = sector_name
            if pd.notna(row.get("industry")):
                graph.nodes[stock_node]["industry"] = row.get("industry")
            if pd.notna(row.get("company_name")):
                graph.nodes[stock_node]["company_name"] = row.get("company_name")

            if sector_node not in graph:
                graph.add_node(
                    sector_node,
                    node_type="sector",
                    sector=sector_name,
                    label=sector_name,
                    stock_count=0,
                )

            graph.nodes[sector_node]["stock_count"] += 1
            graph.add_edge(
                stock_node,
                sector_node,
                edge_type="stock_sector",
                weight=1.0,
            )

    def _add_topic_nodes_and_edges(
        self,
        graph: nx.Graph,
        topic_stock: pd.DataFrame,
        sector_lookup: dict[str, dict[str, Any]],
        allowed_tickers: set[str],
    ) -> None:
        """Add topic nodes and stock-topic edges from the aggregated table."""

        if topic_stock.empty:
            return

        clean_topic_stock = topic_stock.copy()
        clean_topic_stock["ticker"] = clean_topic_stock["ticker"].astype(str).str.upper()
        clean_topic_stock = clean_topic_stock.dropna(subset=["ticker", "topic"])
        clean_topic_stock = clean_topic_stock[
            clean_topic_stock["ticker"].isin(sorted(allowed_tickers))
        ].reset_index(drop=True)
        if clean_topic_stock.empty:
            return

        topic_summary = (
            clean_topic_stock.groupby("topic", as_index=False)
            .agg(article_count=("article_count", "sum"))
            .reset_index(drop=True)
        )

        for _, row in topic_summary.iterrows():
            topic_name = str(row["topic"])
            topic_node = self._topic_node_id(topic_name)
            graph.add_node(
                topic_node,
                node_type="topic",
                topic=topic_name,
                label=topic_name,
                article_count=int(row["article_count"]),
            )

        for _, row in clean_topic_stock.iterrows():
            ticker = str(row["ticker"])
            topic_name = str(row["topic"])
            stock_node = self._stock_node_id(ticker)
            topic_node = self._topic_node_id(topic_name)
            sector_record = sector_lookup.get(ticker, {})
            if stock_node not in graph:
                continue

            weight = row.get(self.stock_topic_weight_column)
            graph.add_edge(
                stock_node,
                topic_node,
                edge_type="stock_topic",
                weight=float(weight) if pd.notna(weight) else 0.0,
                article_count=int(row["article_count"]),
                avg_topic_relevance=self._safe_float(row["avg_topic_relevance"]),
                avg_ticker_relevance=self._safe_float(row["avg_ticker_relevance"]),
                avg_ticker_sentiment=self._safe_float(row["avg_ticker_sentiment"]),
                avg_overall_sentiment=self._safe_float(row["avg_overall_sentiment"]),
            )

    def _add_stock_stock_edges(
        self,
        graph: nx.Graph,
        returns_table: pd.DataFrame,
    ) -> None:
        """Add stock-stock correlation edges to the graph."""

        if returns_table.shape[1] < 2:
            return

        correlation_matrix = returns_table.corr(
            method="pearson",
            min_periods=self.min_correlation_periods,
        )

        if self.top_k_neighbors is None:
            # Threshold mode keeps every pair above one explicit cutoff.
            for ticker_one, ticker_two in combinations(correlation_matrix.columns, 2):
                correlation = correlation_matrix.loc[ticker_one, ticker_two]
                if pd.isna(correlation) or correlation < self.correlation_threshold:
                    continue

                overlap_days = len(
                    returns_table[[ticker_one, ticker_two]].dropna()
                )
                self._add_stock_stock_edge(
                    graph=graph,
                    ticker_one=ticker_one,
                    ticker_two=ticker_two,
                    correlation=float(correlation),
                    overlap_days=overlap_days,
                )
            return

        # Top-k mode guarantees each stock can keep a few strongest neighbors
        # even when the global threshold would be too strict.
        for ticker in correlation_matrix.columns:
            candidate_series = correlation_matrix.loc[ticker].drop(labels=[ticker])
            candidate_series = candidate_series.dropna().sort_values(ascending=False)
            candidate_series = candidate_series.head(self.top_k_neighbors)

            for neighbor, correlation in candidate_series.items():
                if correlation < 0:
                    continue

                overlap_days = len(returns_table[[ticker, neighbor]].dropna())
                self._add_stock_stock_edge(
                    graph=graph,
                    ticker_one=ticker,
                    ticker_two=neighbor,
                    correlation=float(correlation),
                    overlap_days=overlap_days,
                )

    def _add_stock_stock_edge(
        self,
        graph: nx.Graph,
        ticker_one: str,
        ticker_two: str,
        correlation: float,
        overlap_days: int,
    ) -> None:
        """Add one stock-stock edge with consistent metadata."""

        node_one = self._stock_node_id(ticker_one)
        node_two = self._stock_node_id(ticker_two)
        graph.add_edge(
            node_one,
            node_two,
            edge_type="stock_stock",
            weight=correlation,
            correlation=correlation,
            overlap_days=overlap_days,
        )

    def _add_sector_sector_edges(self, graph: nx.Graph) -> None:
        """Derive sector-sector edges from cross-sector stock-stock edges."""

        sector_pairs: dict[tuple[str, str], list[float]] = {}

        # Sector links are not downloaded directly. They are summarized from
        # strong stock-stock relationships that cross sector boundaries.
        for node_one, node_two, edge_data in graph.edges(data=True):
            if edge_data.get("edge_type") != "stock_stock":
                continue

            sector_one = graph.nodes[node_one].get("sector")
            sector_two = graph.nodes[node_two].get("sector")
            if not sector_one or not sector_two or sector_one == sector_two:
                continue

            pair_key = tuple(sorted((str(sector_one), str(sector_two))))
            sector_pairs.setdefault(pair_key, []).append(
                float(edge_data.get("correlation", 0.0))
            )

        for (sector_one, sector_two), correlations in sector_pairs.items():
            sector_node_one = self._sector_node_id(sector_one)
            sector_node_two = self._sector_node_id(sector_two)
            avg_correlation = sum(correlations) / len(correlations)

            if sector_node_one not in graph:
                graph.add_node(
                    sector_node_one,
                    node_type="sector",
                    sector=sector_one,
                    label=sector_one,
                    stock_count=0,
                )

            if sector_node_two not in graph:
                graph.add_node(
                    sector_node_two,
                    node_type="sector",
                    sector=sector_two,
                    label=sector_two,
                    stock_count=0,
                )

            graph.add_edge(
                sector_node_one,
                sector_node_two,
                edge_type="sector_sector",
                weight=avg_correlation,
                average_correlation=avg_correlation,
                stock_edge_count=len(correlations),
            )

    def _build_sector_lookup(
        self,
        sector_info: pd.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        """Build a ticker-indexed lookup from the sector table."""

        if sector_info.empty:
            return {}

        lookup: dict[str, dict[str, Any]] = {}
        for _, row in sector_info.iterrows():
            ticker = row.get("ticker")
            if pd.isna(ticker):
                continue

            lookup[str(ticker).upper()] = {
                "company_name": row.get("company_name"),
                "sector": row.get("sector"),
                "industry": row.get("industry"),
            }

        return lookup

    def _normalize_ticker(self, ticker: str) -> str:
        """Validate and normalize a stock ticker."""

        if not isinstance(ticker, str):
            raise NetworkBuilderError("Ticker keys must be strings.")

        cleaned = ticker.strip().upper()
        if not cleaned:
            raise NetworkBuilderError("Ticker keys cannot be empty.")

        return cleaned

    def _safe_float(self, value: Any) -> float | None:
        """Convert a scalar value to float when possible."""

        if pd.isna(value):
            return None

        return float(value)

    def _stock_node_id(self, ticker: str) -> str:
        """Build a stock node identifier."""

        return f"stock:{ticker}"

    def _sector_node_id(self, sector_name: str) -> str:
        """Build a sector node identifier."""

        return f"sector:{sector_name}"

    def _topic_node_id(self, topic_name: str) -> str:
        """Build a topic node identifier."""

        return f"topic:{topic_name}"
