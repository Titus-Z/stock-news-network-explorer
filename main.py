"""Main local analysis entry point for the Stock and News Network Explorer."""

from __future__ import annotations

import argparse

from cli import load_cli_or_raise


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the local CLI."""

    parser = argparse.ArgumentParser(
        description="Launch the local graph exploration CLI."
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Optional comma-separated ticker filter for local analysis.",
    )
    parser.add_argument(
        "--news-file",
        default="merged_seed_news.json",
        help=(
            "News file name inside the configured data root. "
            "Default: merged_seed_news.json"
        ),
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.6,
        help="Minimum stock correlation to keep when top-k mode is not used.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k stock neighbor mode for stock-stock edges.",
    )
    parser.add_argument(
        "--topic-weight",
        default="article_count",
        help="Column used as the stock-topic edge weight. Default: article_count",
    )
    return parser


def parse_ticker_argument(raw_value: str | None) -> list[str] | None:
    """Parse an optional comma-separated ticker string into a list."""

    if raw_value is None:
        return None

    items = [item.strip().upper() for item in raw_value.split(",")]
    clean_items = [item for item in items if item]
    return clean_items or None


def main() -> int:
    """Build the local graph, print a summary, and launch the CLI."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        # Data download happens in seed_data.py. main.py is intentionally kept
        # focused on local graph assembly and interactive exploration.
        cli, summary_text = load_cli_or_raise(
            tickers=parse_ticker_argument(args.tickers),
            news_file=args.news_file,
            correlation_threshold=args.correlation_threshold,
            top_k=args.top_k,
            topic_weight=args.topic_weight,
        )
        print(summary_text)
        print()
        cli.run()
        return 0
    except RuntimeError as exc:
        print(f"Local analysis failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
