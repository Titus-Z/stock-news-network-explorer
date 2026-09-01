"""Verify that the checked-in synthetic dataset builds a queryable graph."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cli import build_analyzer_from_local_data  # noqa: E402


def main() -> int:
    """Build the default demo graph and enforce its public data contract."""

    analyzer, summary, context = build_analyzer_from_local_data(
        tickers=["AAPL", "MSFT", "NVDA"],
        news_file="merged_seed_news.json",
        correlation_threshold=0.7,
        top_k=1,
        topic_weight="article_count",
    )

    expected_context = {
        "price_table_count": 3,
        "sector_row_count": 3,
        "article_count": 4,
    }
    for field, expected in expected_context.items():
        actual = context[field]
        if actual != expected:
            raise RuntimeError(f"Demo {field} changed: expected {expected}, got {actual}")

    node_counts = summary["node_counts"]
    if node_counts != {"stock": 3, "sector": 1, "topic": 5}:
        raise RuntimeError(f"Unexpected demo node counts: {dict(node_counts)}")

    stock_info = analyzer.get_stock_info("AAPL")
    if stock_info["company_name"] != "Synthetic Apple Example":
        raise RuntimeError("Synthetic AAPL record could not be queried.")

    print(
        "Demo graph verified: "
        f"{analyzer.graph.number_of_nodes()} nodes, "
        f"{analyzer.graph.number_of_edges()} edges, "
        f"{context['article_count']} synthetic articles."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
