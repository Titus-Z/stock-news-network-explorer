"""LLM-based enrichment for article-to-sector and article-to-stock impacts."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from config import AppConfig, load_config

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None


ImpactDirection = Literal["positive", "negative", "mixed", "uncertain"]
ImpactStrength = Literal["low", "medium", "high"]
EventType = Literal[
    "earnings",
    "guidance",
    "product",
    "regulation",
    "macro",
    "merger_acquisition",
    "litigation",
    "operations",
    "analyst",
    "other",
]
ImpactScope = Literal["company_specific", "sector_wide", "macro", "mixed"]
MarketRelevance = Literal["low", "medium", "high"]


class LLMEnricherError(Exception):
    """Raised when LLM enrichment cannot run or cannot be parsed safely."""


class SectorImpact(BaseModel):
    """Structured impact record for one sector."""

    sector: str
    impact_direction: ImpactDirection
    impact_strength: ImpactStrength
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class StockImpact(BaseModel):
    """Structured impact record for one ticker."""

    ticker: str
    impact_direction: ImpactDirection
    impact_strength: ImpactStrength
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class ArticleImpactAssessment(BaseModel):
    """Structured LLM output for one article."""

    article_id: str
    event_summary: str
    primary_event_type: EventType
    scope: ImpactScope
    overall_market_relevance: MarketRelevance
    sector_impacts: list[SectorImpact]
    stock_impacts: list[StockImpact]


class OpenAINewsEnricher:
    """Use the OpenAI API to add structured impact labels to news articles."""

    SUMMARY_COLUMNS = [
        "article_id",
        "event_summary",
        "primary_event_type",
        "scope",
        "overall_market_relevance",
        "sector_impact_count",
        "stock_impact_count",
        "model",
        "enriched_at",
    ]
    SECTOR_IMPACT_COLUMNS = [
        "article_id",
        "sector",
        "impact_direction",
        "impact_strength",
        "confidence",
        "rationale",
        "model",
        "enriched_at",
    ]
    STOCK_IMPACT_COLUMNS = [
        "article_id",
        "ticker",
        "impact_direction",
        "impact_strength",
        "confidence",
        "rationale",
        "model",
        "enriched_at",
    ]

    def __init__(
        self,
        config: AppConfig | None = None,
        client: Any | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the enricher with config and an optional OpenAI client."""

        self.config = config or load_config()
        self.model = model or self.config.default_openai_model
        self.client = client or self._build_openai_client()

    def enrich_news_tables(
        self,
        articles: pd.DataFrame,
        article_tickers: pd.DataFrame,
        sector_info: pd.DataFrame,
        max_articles: int | None = None,
        published_after: str | pd.Timestamp | None = None,
        published_before: str | pd.Timestamp | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Enrich processed news tables and return structured impact tables."""

        if articles.empty:
            return self._empty_output_tables()

        required_article_columns = {"article_id", "title", "summary", "source"}
        missing_article_columns = required_article_columns - set(articles.columns)
        if missing_article_columns:
            raise LLMEnricherError(
                f"Articles table is missing columns: {sorted(missing_article_columns)}."
            )

        required_ticker_columns = {"article_id", "ticker"}
        missing_ticker_columns = required_ticker_columns - set(article_tickers.columns)
        if missing_ticker_columns:
            raise LLMEnricherError(
                "Article ticker table is missing columns: "
                f"{sorted(missing_ticker_columns)}."
            )

        allowed_sectors = self._allowed_sectors(sector_info)
        allowed_tickers = self._allowed_tickers(sector_info, article_tickers)
        selected_articles = select_articles_for_enrichment(
            articles=articles,
            max_articles=max_articles,
            published_after=published_after,
            published_before=published_before,
        )

        summary_rows: list[dict[str, Any]] = []
        sector_rows: list[dict[str, Any]] = []
        stock_rows: list[dict[str, Any]] = []

        for _, article_row in selected_articles.iterrows():
            article_id = str(article_row["article_id"])
            matching_tickers = article_tickers[
                article_tickers["article_id"] == article_id
            ].reset_index(drop=True)

            assessment = self.enrich_article(
                article_row=article_row,
                article_tickers=matching_tickers,
                allowed_sectors=allowed_sectors,
                allowed_tickers=allowed_tickers,
            )
            summary_row, article_sector_rows, article_stock_rows = (
                self._assessment_to_rows(assessment)
            )
            summary_rows.append(summary_row)
            sector_rows.extend(article_sector_rows)
            stock_rows.extend(article_stock_rows)

        return {
            "article_llm_summary": pd.DataFrame(
                summary_rows,
                columns=self.SUMMARY_COLUMNS,
            ),
            "article_sector_impacts": pd.DataFrame(
                sector_rows,
                columns=self.SECTOR_IMPACT_COLUMNS,
            ),
            "article_stock_impacts": pd.DataFrame(
                stock_rows,
                columns=self.STOCK_IMPACT_COLUMNS,
            ),
        }

    def enrich_article(
        self,
        article_row: pd.Series,
        article_tickers: pd.DataFrame,
        allowed_sectors: list[str],
        allowed_tickers: list[str],
    ) -> ArticleImpactAssessment:
        """Enrich one article row into a structured impact assessment."""

        article_payload = self._build_article_payload(article_row, article_tickers)
        instructions = self._build_instructions(allowed_sectors, allowed_tickers)

        try:
            response = self.client.responses.parse(
                model=self.model,
                instructions=instructions,
                input=article_payload,
                text_format=ArticleImpactAssessment,
                temperature=0,
            )
        except Exception as exc:  # pragma: no cover - third-party runtime safeguard
            raise LLMEnricherError(
                f"OpenAI enrichment failed for article "
                f"'{article_row.get('article_id')}': {exc}"
            ) from exc

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise LLMEnricherError(
                f"OpenAI returned no parsed output for article "
                f"'{article_row.get('article_id')}'."
            )

        return self._sanitize_assessment(
            assessment=parsed,
            article_id=str(article_row["article_id"]),
            allowed_sectors=allowed_sectors,
            allowed_tickers=allowed_tickers,
        )

    def write_output_tables(
        self,
        tables: dict[str, pd.DataFrame],
        file_stem: str,
    ) -> None:
        """Write enrichment output tables to raw and backup directories."""

        output_dir = self.config.raw_data_dir / "llm_enriched"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.config.backup_data_dir / timestamp / "llm_enriched"
        backup_dir.mkdir(parents=True, exist_ok=True)

        for table_name, data_frame in tables.items():
            file_name = f"{file_stem}_{table_name}.csv"
            raw_path = output_dir / file_name
            backup_path = backup_dir / file_name
            data_frame.to_csv(raw_path, index=False)
            data_frame.to_csv(backup_path, index=False)

    def _build_openai_client(self) -> Any:
        """Build an OpenAI client from config."""

        if OpenAI is None:
            raise LLMEnricherError(
                "openai is not installed. Run 'pip install -r requirements.txt'."
            )

        if not self.config.openai_api_key:
            raise LLMEnricherError(
                "Missing OpenAI API key. Set the OPENAI_API_KEY environment variable."
            )

        return OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
        )
    def _build_instructions(
        self,
        allowed_sectors: list[str],
        allowed_tickers: list[str],
    ) -> str:
        """Build the system instructions for the structured extraction call."""

        sectors_text = ", ".join(allowed_sectors) or "none"
        tickers_text = ", ".join(allowed_tickers) or "none"

        return (
            "You are labeling financial news for a graph-based market explorer. "
            "Return only structured output that matches the schema. "
            "Do not predict stock prices. Infer likely impact relationships from the "
            "article text and provided ticker context. "
            "Only use sectors from this allowed list: "
            f"{sectors_text}. "
            "Only use tickers from this allowed list: "
            f"{tickers_text}. "
            "If the article does not support a confident link, return no impact item "
            "or mark the direction as 'uncertain'. "
            "Keep rationales short and evidence-based."
        )

    def _build_article_payload(
        self,
        article_row: pd.Series,
        article_tickers: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Build one Responses API input payload for a single article."""

        ticker_context = article_tickers.fillna("").to_dict(orient="records")
        article_content = {
            "article_id": str(article_row.get("article_id")),
            "title": str(article_row.get("title") or ""),
            "source": str(article_row.get("source") or ""),
            "published_at": self._serialize_timestamp(article_row.get("published_at")),
            "summary": str(article_row.get("summary") or ""),
            "overall_sentiment_score": self._safe_scalar(article_row.get("overall_sentiment_score")),
            "overall_sentiment_label": str(
                article_row.get("overall_sentiment_label") or ""
            ),
            "ticker_context": ticker_context,
        }

        return [
            {
                "role": "user",
                "content": json.dumps(article_content, ensure_ascii=True),
            }
        ]

    def _sanitize_assessment(
        self,
        assessment: ArticleImpactAssessment,
        article_id: str,
        allowed_sectors: list[str],
        allowed_tickers: list[str],
    ) -> ArticleImpactAssessment:
        """Drop any sector or ticker values outside the allowed project universe."""

        allowed_sector_set = set(allowed_sectors)
        allowed_ticker_set = set(allowed_tickers)

        filtered_sector_impacts = [
            item
            for item in assessment.sector_impacts
            if item.sector in allowed_sector_set
        ][:3]
        filtered_stock_impacts = [
            item
            for item in assessment.stock_impacts
            if item.ticker in allowed_ticker_set
        ][:5]

        return ArticleImpactAssessment(
            article_id=article_id,
            event_summary=assessment.event_summary,
            primary_event_type=assessment.primary_event_type,
            scope=assessment.scope,
            overall_market_relevance=assessment.overall_market_relevance,
            sector_impacts=filtered_sector_impacts,
            stock_impacts=filtered_stock_impacts,
        )

    def _assessment_to_rows(
        self,
        assessment: ArticleImpactAssessment,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        """Convert a parsed assessment object into flat table rows."""

        enriched_at = datetime.now().isoformat(timespec="seconds")
        summary_row = {
            "article_id": assessment.article_id,
            "event_summary": assessment.event_summary,
            "primary_event_type": assessment.primary_event_type,
            "scope": assessment.scope,
            "overall_market_relevance": assessment.overall_market_relevance,
            "sector_impact_count": len(assessment.sector_impacts),
            "stock_impact_count": len(assessment.stock_impacts),
            "model": self.model,
            "enriched_at": enriched_at,
        }

        sector_rows = [
            {
                "article_id": assessment.article_id,
                "sector": item.sector,
                "impact_direction": item.impact_direction,
                "impact_strength": item.impact_strength,
                "confidence": item.confidence,
                "rationale": item.rationale,
                "model": self.model,
                "enriched_at": enriched_at,
            }
            for item in assessment.sector_impacts
        ]

        stock_rows = [
            {
                "article_id": assessment.article_id,
                "ticker": item.ticker,
                "impact_direction": item.impact_direction,
                "impact_strength": item.impact_strength,
                "confidence": item.confidence,
                "rationale": item.rationale,
                "model": self.model,
                "enriched_at": enriched_at,
            }
            for item in assessment.stock_impacts
        ]

        return summary_row, sector_rows, stock_rows

    def _allowed_sectors(self, sector_info: pd.DataFrame) -> list[str]:
        """Build the allowed sector list from local sector metadata."""

        if sector_info.empty or "sector" not in sector_info.columns:
            return []

        values = (
            sector_info["sector"]
            .dropna()
            .astype(str)
            .map(str.strip)
        )
        return sorted({value for value in values if value})

    def _allowed_tickers(
        self,
        sector_info: pd.DataFrame,
        article_tickers: pd.DataFrame,
    ) -> list[str]:
        """Build the allowed ticker list from local sector metadata and articles."""

        ticker_values: set[str] = set()

        if not sector_info.empty and "ticker" in sector_info.columns:
            ticker_values.update(
                sector_info["ticker"]
                .dropna()
                .astype(str)
                .map(str.upper)
                .tolist()
            )

        if not article_tickers.empty and "ticker" in article_tickers.columns:
            ticker_values.update(
                article_tickers["ticker"]
                .dropna()
                .astype(str)
                .map(str.upper)
                .tolist()
            )

        return sorted(ticker_values)

    def _empty_output_tables(self) -> dict[str, pd.DataFrame]:
        """Return empty enrichment tables with stable schemas."""

        return {
            "article_llm_summary": pd.DataFrame(columns=self.SUMMARY_COLUMNS),
            "article_sector_impacts": pd.DataFrame(
                columns=self.SECTOR_IMPACT_COLUMNS
            ),
            "article_stock_impacts": pd.DataFrame(columns=self.STOCK_IMPACT_COLUMNS),
        }

    def _serialize_timestamp(self, value: Any) -> str | None:
        """Convert timestamps into ISO strings for prompt serialization."""

        if value is None or pd.isna(value):
            return None
        timestamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(timestamp):
            return None
        return timestamp.isoformat()

    def _safe_scalar(self, value: Any) -> Any:
        """Convert pandas scalars into JSON-safe prompt values."""

        if value is None or pd.isna(value):
            return None
        if isinstance(value, (int, float, str, bool)):
            return value
        return str(value)


def select_articles_for_enrichment(
    articles: pd.DataFrame,
    max_articles: int | None = None,
    published_after: str | pd.Timestamp | None = None,
    published_before: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return one stable article subset for an enrichment run."""

    if articles.empty:
        return articles.copy()

    if max_articles is not None and max_articles <= 0:
        raise LLMEnricherError("max_articles must be a positive integer.")

    selected_articles = articles.copy()
    if "published_at" in selected_articles.columns:
        selected_articles["published_at"] = pd.to_datetime(
            selected_articles["published_at"],
            errors="coerce",
        )

        if published_after is not None:
            after_timestamp = _coerce_window_timestamp(
                published_after,
                end_of_day=False,
            )
            if pd.isna(after_timestamp):
                raise LLMEnricherError("published_after must be a valid timestamp or date.")
            selected_articles = selected_articles[
                selected_articles["published_at"] >= after_timestamp
            ].copy()

        if published_before is not None:
            before_timestamp = _coerce_window_timestamp(
                published_before,
                end_of_day=True,
            )
            if pd.isna(before_timestamp):
                raise LLMEnricherError(
                    "published_before must be a valid timestamp or date."
                )
            selected_articles = selected_articles[
                selected_articles["published_at"] <= before_timestamp
            ].copy()

        selected_articles = selected_articles.sort_values(
            "published_at",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    if max_articles is not None:
        selected_articles = selected_articles.head(max_articles).reset_index(drop=True)

    return selected_articles


def _coerce_window_timestamp(
    value: str | pd.Timestamp,
    end_of_day: bool,
) -> pd.Timestamp:
    """Parse one date or timestamp string for article-window filtering."""

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return timestamp

    text = str(value).strip()
    if end_of_day and len(text) == 10:
        return timestamp + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return timestamp
