"""Simple command-line interface for graph exploration."""

from __future__ import annotations

from collections import Counter
from typing import Any

from local_data_store import LocalDataStore, LocalDataStoreError
from network_analyzer import NetworkAnalyzer, NetworkAnalyzerError
from network_builder import NetworkBuilder, NetworkBuilderError
from news_processor import NewsProcessor, NewsProcessorError


class MarketExplorerCLI:
    """Simple CLI wrapper around the analysis layer."""

    def __init__(self, analyzer: NetworkAnalyzer) -> None:
        """Initialize the CLI with one ready-to-use analyzer."""

        self.analyzer = analyzer

    def run(self) -> None:
        """Run the interactive menu loop."""

        while True:
            self._print_menu()
            choice = input("Choose an option: ").strip()
            print()

            if choice == "1":
                self._handle_stock_lookup()
            elif choice == "2":
                self._handle_sector_lookup()
            elif choice == "3":
                self._handle_stock_compare()
            elif choice == "4":
                self._handle_path_lookup()
            elif choice == "5":
                self._handle_top_nodes()
            elif choice == "6":
                print("Goodbye.")
                return
            else:
                print("Invalid choice. Please enter a number from 1 to 6.")

            print()

    def _print_menu(self) -> None:
        """Print the main menu."""

        print("Market Network Explorer")
        print("1. Search for a stock")
        print("2. Explore a sector")
        print("3. Compare two stocks")
        print("4. Find a path between nodes")
        print("5. Show top central nodes")
        print("6. Exit")

    def _handle_stock_lookup(self) -> None:
        """Handle the stock lookup menu action."""

        ticker = input("Enter ticker: ").strip()
        if not ticker:
            print("Ticker cannot be empty.")
            return

        try:
            stock_info = self.analyzer.get_stock_info(ticker)
            print(format_stock_info(stock_info))
        except NetworkAnalyzerError as exc:
            print(f"Stock lookup failed: {exc}")

    def _handle_sector_lookup(self) -> None:
        """Handle the sector lookup menu action."""

        sector_name = input("Enter sector name: ").strip()
        if not sector_name:
            print("Sector name cannot be empty.")
            return

        try:
            sector_info = self.analyzer.get_sector_info(sector_name)
            print(format_sector_info(sector_info))
        except NetworkAnalyzerError as exc:
            print(f"Sector lookup failed: {exc}")

    def _handle_stock_compare(self) -> None:
        """Handle the stock comparison menu action."""

        ticker_one = input("Enter first ticker: ").strip()
        ticker_two = input("Enter second ticker: ").strip()
        if not ticker_one or not ticker_two:
            print("Both ticker inputs are required.")
            return

        try:
            comparison = self.analyzer.compare_stocks(ticker_one, ticker_two)
            print(format_stock_comparison(comparison))
        except NetworkAnalyzerError as exc:
            print(f"Stock comparison failed: {exc}")

    def _handle_path_lookup(self) -> None:
        """Handle the shortest-path menu action."""

        node_a = input("Enter first node (ticker, sector, topic, or full id): ").strip()
        node_b = input("Enter second node: ").strip()
        if not node_a or not node_b:
            print("Both node inputs are required.")
            return

        try:
            path_result = self.analyzer.find_shortest_path(node_a, node_b)
            print(format_path_result(path_result))
        except NetworkAnalyzerError as exc:
            print(f"Path lookup failed: {exc}")

    def _handle_top_nodes(self) -> None:
        """Handle the centrality summary menu action."""

        centrality_metrics = self.analyzer.compute_centrality_metrics()
        print(format_top_central_nodes(centrality_metrics))


def build_analyzer_from_local_data(
    tickers: list[str] | None,
    news_file: str,
    correlation_threshold: float,
    top_k: int | None,
    topic_weight: str,
) -> tuple[NetworkAnalyzer, dict[str, Counter], dict[str, Any]]:
    """Load local files, build the graph, and return one analyzer."""

    store = LocalDataStore()
    # The CLI never reaches out to APIs directly. It only reads the cached
    # project dataset so that interactive exploration stays fast and repeatable.
    prices = store.load_price_tables(tickers=tickers)
    sector_info = store.load_sector_info(tickers=tickers)
    news_payload = store.load_news_payload(file_name=news_file)

    processor = NewsProcessor()
    news_tables = processor.process_news_payload(news_payload)
    filtered_news_tables = filter_news_tables_by_tickers(news_tables, tickers)
    topic_stock = filtered_news_tables["topic_stock"]

    builder = NetworkBuilder(
        correlation_threshold=correlation_threshold,
        top_k_neighbors=top_k,
        stock_topic_weight_column=topic_weight,
    )
    graph = builder.build_graph(
        price_tables=prices,
        sector_info=sector_info,
        topic_stock=topic_stock,
    )

    summary = build_graph_summary(graph)
    context = {
        "price_table_count": len(prices),
        "sector_row_count": len(sector_info),
        "article_count": len(filtered_news_tables["articles"]),
        "topic_stock_row_count": len(topic_stock),
    }

    return NetworkAnalyzer(graph), summary, context


def filter_news_tables_by_tickers(
    news_tables: dict[str, Any],
    tickers: list[str] | None,
) -> dict[str, Any]:
    """Filter processed news tables to the selected ticker subset."""

    if tickers is None:
        return news_tables

    articles = news_tables["articles"]
    article_tickers = news_tables["article_tickers"]
    topic_stock = news_tables["topic_stock"]

    # Keep only ticker-level rows that belong to the selected stock subset,
    # then use their article ids to derive the matching article count summary.
    filtered_article_tickers = article_tickers[
        article_tickers["ticker"].isin(tickers)
    ].reset_index(drop=True)
    selected_article_ids = set(filtered_article_tickers["article_id"].tolist())

    filtered_articles = articles[articles["article_id"].isin(selected_article_ids)]
    filtered_articles = filtered_articles.reset_index(drop=True)

    filtered_topic_stock = topic_stock[topic_stock["ticker"].isin(tickers)]
    filtered_topic_stock = filtered_topic_stock.reset_index(drop=True)

    return {
        "articles": filtered_articles,
        "article_tickers": filtered_article_tickers,
        "topic_stock": filtered_topic_stock,
    }


def build_graph_summary(graph) -> dict[str, Counter]:
    """Build node and edge summaries grouped by type."""

    # These summaries are meant for startup sanity checks, not deep analysis.
    node_counts = Counter(
        data.get("node_type", "unknown") for _, data in graph.nodes(data=True)
    )
    edge_counts = Counter(
        data.get("edge_type", "unknown") for _, _, data in graph.edges(data=True)
    )
    return {
        "node_counts": node_counts,
        "edge_counts": edge_counts,
    }


def format_graph_summary(summary: dict[str, Counter], context: dict[str, Any]) -> str:
    """Format a short graph summary for display."""

    lines = [
        "Local analysis summary",
        f"- price tables loaded: {context['price_table_count']}",
        f"- sector rows loaded: {context['sector_row_count']}",
        f"- news articles loaded: {context['article_count']}",
        f"- topic_stock rows loaded: {context['topic_stock_row_count']}",
        "",
        "Node types:",
    ]

    for node_type, count in sorted(summary["node_counts"].items()):
        lines.append(f"- {node_type}: {count}")

    lines.append("")
    lines.append("Edge types:")
    for edge_type, count in sorted(summary["edge_counts"].items()):
        lines.append(f"- {edge_type}: {count}")

    return "\n".join(lines)


def format_stock_info(stock_info: dict[str, Any]) -> str:
    """Format stock-level analysis results for display."""

    lines = [
        f"Stock: {stock_info.get('ticker')}",
        f"- company: {stock_info.get('company_name') or 'Unknown'}",
        f"- sector: {stock_info.get('sector') or 'Unknown'}",
        f"- industry: {stock_info.get('industry') or 'Unknown'}",
        f"- degree: {stock_info.get('degree')}",
        f"- degree centrality: {stock_info.get('degree_centrality'):.4f}",
        "- top stock neighbors:",
    ]

    neighbors = stock_info.get("top_neighbors", [])
    if neighbors:
        for neighbor in neighbors[:5]:
            lines.append(
                f"  {neighbor['ticker']} "
                f"(corr={neighbor.get('correlation'):.4f}, "
                f"overlap_days={neighbor.get('overlap_days')})"
            )
    else:
        lines.append("  none")

    lines.append("- related topics:")
    topics = stock_info.get("related_topics", [])
    if topics:
        for topic in topics[:5]:
            lines.append(
                f"  {topic['topic']} "
                f"(weight={topic.get('weight')}, "
                f"article_count={topic.get('article_count')})"
            )
    else:
        lines.append("  none")

    return "\n".join(lines)


def format_sector_info(sector_info: dict[str, Any]) -> str:
    """Format sector-level analysis results for display."""

    lines = [
        f"Sector: {sector_info.get('sector')}",
        f"- stock count: {sector_info.get('stock_count')}",
        "- stocks:",
    ]

    stocks = sector_info.get("stocks", [])
    if stocks:
        for stock in stocks[:10]:
            lines.append(
                f"  {stock['ticker']} "
                f"(company={stock.get('company_name') or 'Unknown'}, "
                f"degree={stock.get('degree')})"
            )
    else:
        lines.append("  none")

    lines.append("- connected sectors:")
    connected_sectors = sector_info.get("top_connected_sectors", [])
    if connected_sectors:
        for sector in connected_sectors[:5]:
            lines.append(
                f"  {sector['sector']} "
                f"(avg_corr={sector.get('average_correlation'):.4f}, "
                f"stock_edge_count={sector.get('stock_edge_count')})"
            )
    else:
        lines.append("  none")

    return "\n".join(lines)


def format_stock_comparison(comparison: dict[str, Any]) -> str:
    """Format stock comparison results for display."""

    direct_edge = comparison.get("direct_edge")
    lines = [
        f"Compare: {comparison.get('ticker1')} vs {comparison.get('ticker2')}",
        f"- same sector: {comparison.get('same_sector')}",
        f"- common topics: {', '.join(comparison.get('common_topics', [])) or 'none'}",
        f"- degree difference: {comparison.get('degree_difference')}",
    ]

    if direct_edge:
        lines.append(
            f"- direct edge: {direct_edge.get('edge_type')} "
            f"(correlation={direct_edge.get('correlation')})"
        )
    else:
        lines.append("- direct edge: none")

    path_result = comparison.get("shortest_path", {})
    if path_result.get("path_found"):
        lines.append(
            f"- shortest path: {' -> '.join(path_result.get('path_nodes', []))}"
        )
        lines.append(
            f"- edge types: {', '.join(path_result.get('edge_types', []))}"
        )
    else:
        lines.append("- shortest path: none")

    return "\n".join(lines)


def format_path_result(path_result: dict[str, Any]) -> str:
    """Format shortest-path results for display."""

    if not path_result.get("path_found"):
        return "No path found."

    return "\n".join(
        [
            f"Path length: {path_result.get('path_length')}",
            f"Nodes: {' -> '.join(path_result.get('path_nodes', []))}",
            f"Edge types: {', '.join(path_result.get('edge_types', []))}",
        ]
    )


def format_top_central_nodes(centrality_metrics) -> str:
    """Format the top central nodes table for display."""

    if centrality_metrics.empty:
        return "No centrality data available."

    top_rows = centrality_metrics.head(10)
    lines = ["Top central nodes:"]
    for _, row in top_rows.iterrows():
        lines.append(
            f"- {row['label']} ({row['node_type']}) "
            f"betweenness={row['betweenness_centrality']:.4f}, "
            f"degree={row['degree_centrality']:.4f}, "
            f"closeness={row['closeness_centrality']:.4f}"
        )

    return "\n".join(lines)


def build_cli(
    tickers: list[str] | None,
    news_file: str,
    correlation_threshold: float,
    top_k: int | None,
    topic_weight: str,
) -> tuple[MarketExplorerCLI, str]:
    """Create one CLI instance and a graph summary string."""

    analyzer, summary, context = build_analyzer_from_local_data(
        tickers=tickers,
        news_file=news_file,
        correlation_threshold=correlation_threshold,
        top_k=top_k,
        topic_weight=topic_weight,
    )
    summary_text = format_graph_summary(summary, context)
    return MarketExplorerCLI(analyzer), summary_text


def load_cli_or_raise(
    tickers: list[str] | None,
    news_file: str,
    correlation_threshold: float,
    top_k: int | None,
    topic_weight: str,
) -> tuple[MarketExplorerCLI, str]:
    """Build the CLI and normalize all setup-time error types."""

    try:
        return build_cli(
            tickers=tickers,
            news_file=news_file,
            correlation_threshold=correlation_threshold,
            top_k=top_k,
            topic_weight=topic_weight,
        )
    except (LocalDataStoreError, NewsProcessorError, NetworkBuilderError) as exc:
        # Main only needs one user-facing setup error type during startup.
        raise RuntimeError(str(exc)) from exc
