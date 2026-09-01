"""News processing utilities for Alpha Vantage NEWS_SENTIMENT payloads."""

from __future__ import annotations

from hashlib import sha1
from typing import Any

import pandas as pd


class NewsProcessorError(Exception):
    """Raised when a news payload is missing required structure."""


class NewsProcessor:
    """Convert raw news payloads into tabular data for graph building."""

    ARTICLE_COLUMNS = [
        "article_id",
        "title",
        "url",
        "published_at",
        "source",
        "summary",
        "overall_sentiment_score",
        "overall_sentiment_label",
    ]
    ARTICLE_TICKER_COLUMNS = [
        "article_id",
        "ticker",
        "relevance_score",
        "ticker_sentiment_score",
        "ticker_sentiment_label",
    ]
    ARTICLE_TOPIC_COLUMNS = [
        "article_id",
        "topic",
        "topic_relevance_score",
    ]
    TOPIC_STOCK_COLUMNS = [
        "ticker",
        "topic",
        "article_count",
        "avg_topic_relevance",
        "avg_ticker_relevance",
        "avg_ticker_sentiment",
        "avg_overall_sentiment",
    ]

    def process_news_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, pd.DataFrame]:
        """Build all Phase 2 news tables from one raw payload."""

        # Keep the extraction step separate from DataFrame creation so the
        # parsing logic stays easy to test and reason about.
        article_rows, article_ticker_rows, article_topic_rows = self._extract_rows(
            payload
        )

        articles = self._build_articles_dataframe(article_rows)
        article_tickers = self._build_article_tickers_dataframe(article_ticker_rows)
        article_topics = self._build_article_topics_dataframe(article_topic_rows)
        topic_stock = self._build_topic_stock_dataframe(
            articles=articles,
            article_tickers=article_tickers,
            article_topics=article_topics,
        )

        return {
            "articles": articles,
            "article_tickers": article_tickers,
            "topic_stock": topic_stock,
        }

    def build_articles_table(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Build the article-level table from a raw news payload."""

        return self.process_news_payload(payload)["articles"]

    def build_article_tickers_table(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Build the article-ticker sentiment table from a raw news payload."""

        return self.process_news_payload(payload)["article_tickers"]

    def build_topic_stock_table(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Build the aggregated topic-stock table from a raw news payload."""

        return self.process_news_payload(payload)["topic_stock"]

    def build_article_topics_table(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Build the article-topic relevance table from a raw news payload."""

        _, _, article_topic_rows = self._extract_rows(payload)
        return self._build_article_topics_dataframe(article_topic_rows)

    def _extract_rows(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract article, ticker, and topic rows from the payload feed."""

        feed = self._validate_feed(payload)
        article_rows: list[dict[str, Any]] = []
        article_ticker_rows: list[dict[str, Any]] = []
        article_topic_rows: list[dict[str, Any]] = []

        for article in feed:
            article_id = self._build_article_id(article)
            article_rows.append(
                {
                    "article_id": article_id,
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "published_at": article.get("time_published"),
                    "source": article.get("source"),
                    "summary": article.get("summary"),
                    "overall_sentiment_score": article.get(
                        "overall_sentiment_score"
                    ),
                    "overall_sentiment_label": article.get(
                        "overall_sentiment_label"
                    ),
                }
            )

            ticker_items = article.get("ticker_sentiment") or []
            for ticker_item in ticker_items:
                # This table preserves article-to-stock relationships before
                # later topic aggregation collapses them.
                article_ticker_rows.append(
                    {
                        "article_id": article_id,
                        "ticker": ticker_item.get("ticker"),
                        "relevance_score": ticker_item.get("relevance_score"),
                        "ticker_sentiment_score": ticker_item.get(
                            "ticker_sentiment_score"
                        ),
                        "ticker_sentiment_label": ticker_item.get(
                            "ticker_sentiment_label"
                        ),
                    }
                )

            topic_items = article.get("topics") or []
            for topic_item in topic_items:
                # Topics are extracted separately so we can join them to ticker
                # sentiment rows through article_id later.
                article_topic_rows.append(
                    {
                        "article_id": article_id,
                        "topic": topic_item.get("topic"),
                        "topic_relevance_score": topic_item.get("relevance_score"),
                    }
                )

        return article_rows, article_ticker_rows, article_topic_rows

    def _build_articles_dataframe(
        self,
        article_rows: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Create the article-level DataFrame with stable column types."""

        articles = pd.DataFrame(article_rows, columns=self.ARTICLE_COLUMNS)
        if articles.empty:
            return articles

        articles = articles.drop_duplicates(subset=["article_id"]).reset_index(
            drop=True
        )
        articles["published_at"] = pd.to_datetime(
            articles["published_at"],
            format="%Y%m%dT%H%M%S",
            errors="coerce",
        )
        articles["overall_sentiment_score"] = pd.to_numeric(
            articles["overall_sentiment_score"],
            errors="coerce",
        )

        return articles.sort_values("published_at", ascending=False).reset_index(
            drop=True
        )

    def _build_article_tickers_dataframe(
        self,
        article_ticker_rows: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Create the article-ticker sentiment DataFrame."""

        article_tickers = pd.DataFrame(
            article_ticker_rows,
            columns=self.ARTICLE_TICKER_COLUMNS,
        )
        if article_tickers.empty:
            return article_tickers

        article_tickers = article_tickers.drop_duplicates(
            subset=["article_id", "ticker"]
        ).reset_index(drop=True)
        article_tickers["relevance_score"] = pd.to_numeric(
            article_tickers["relevance_score"],
            errors="coerce",
        )
        article_tickers["ticker_sentiment_score"] = pd.to_numeric(
            article_tickers["ticker_sentiment_score"],
            errors="coerce",
        )

        return article_tickers.sort_values(
            ["ticker", "article_id"]
        ).reset_index(drop=True)

    def _build_article_topics_dataframe(
        self,
        article_topic_rows: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Create the article-topic DataFrame used for aggregation."""

        article_topics = pd.DataFrame(
            article_topic_rows,
            columns=self.ARTICLE_TOPIC_COLUMNS,
        )
        if article_topics.empty:
            return article_topics

        article_topics = article_topics.drop_duplicates(
            subset=["article_id", "topic"]
        ).reset_index(drop=True)
        article_topics["topic_relevance_score"] = pd.to_numeric(
            article_topics["topic_relevance_score"],
            errors="coerce",
        )

        return article_topics.sort_values(
            ["topic", "article_id"]
        ).reset_index(drop=True)

    def _build_topic_stock_dataframe(
        self,
        articles: pd.DataFrame,
        article_tickers: pd.DataFrame,
        article_topics: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate article, ticker, and topic rows into topic-stock metrics."""

        if articles.empty or article_tickers.empty or article_topics.empty:
            return pd.DataFrame(columns=self.TOPIC_STOCK_COLUMNS)

        merged = article_tickers.merge(article_topics, on="article_id", how="inner")
        merged = merged.merge(
            articles[["article_id", "overall_sentiment_score"]],
            on="article_id",
            how="left",
        )

        if merged.empty:
            return pd.DataFrame(columns=self.TOPIC_STOCK_COLUMNS)

        # The topic-stock table is the main graph input for stock-topic edges,
        # so the aggregation stays explicit and easy to explain in the report.
        topic_stock = (
            merged.groupby(["ticker", "topic"], as_index=False)
            .agg(
                article_count=("article_id", "nunique"),
                avg_topic_relevance=("topic_relevance_score", "mean"),
                avg_ticker_relevance=("relevance_score", "mean"),
                avg_ticker_sentiment=("ticker_sentiment_score", "mean"),
                avg_overall_sentiment=("overall_sentiment_score", "mean"),
            )
            .sort_values(
                ["article_count", "avg_topic_relevance", "ticker", "topic"],
                ascending=[False, False, True, True],
            )
            .reset_index(drop=True)
        )

        return topic_stock

    def _validate_feed(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Validate and return the feed list from a raw payload."""

        if not isinstance(payload, dict):
            raise NewsProcessorError("News payload must be a dictionary.")

        if "feed" not in payload:
            raise NewsProcessorError("News payload is missing the 'feed' field.")

        feed = payload["feed"]
        if not isinstance(feed, list):
            raise NewsProcessorError("News payload field 'feed' must be a list.")

        return feed

    def _build_article_id(self, article: dict[str, Any]) -> str:
        """Build a stable article identifier from URL or title/timestamp."""

        url = str(article.get("url") or "").strip()
        if url:
            seed = url
        else:
            # Fall back to title + timestamp when a URL is missing so joins can
            # still work across the intermediate news tables.
            title = str(article.get("title") or "").strip()
            published_at = str(article.get("time_published") or "").strip()
            seed = f"{title}|{published_at}"

        return sha1(seed.encode("utf-8")).hexdigest()
