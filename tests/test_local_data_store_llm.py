"""Tests for local LLM impact file helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import AppConfig
from local_data_store import LocalDataStore


def build_config(tmp_path: Path) -> AppConfig:
    """Build a small config rooted in a temporary data directory."""

    data_dir = tmp_path / "data"
    return AppConfig(
        alpha_vantage_api_key=None,
        openai_api_key=None,
        alpha_vantage_base_url="https://example.com/query",
        openai_base_url="https://example.com/openai",
        default_openai_model="gpt-4o-mini",
        request_timeout=30,
        default_news_limit=50,
        default_price_period="1y",
        data_dir=data_dir,
        raw_data_dir=data_dir / "raw",
        backup_data_dir=data_dir / "backups",
    )


def test_list_and_load_llm_impact_tables() -> None:
    """The local store should discover and load one complete LLM run."""

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        config = build_config(tmp_path)
        llm_dir = config.raw_data_dir / "llm_enriched"
        llm_dir.mkdir(parents=True, exist_ok=True)

        stem = "demo_run"
        pd.DataFrame([{"article_id": "a1"}]).to_csv(
            llm_dir / f"{stem}_article_llm_summary.csv",
            index=False,
        )
        pd.DataFrame([{"article_id": "a1", "sector": "Technology"}]).to_csv(
            llm_dir / f"{stem}_article_sector_impacts.csv",
            index=False,
        )
        pd.DataFrame([{"article_id": "a1", "ticker": "AAPL"}]).to_csv(
            llm_dir / f"{stem}_article_stock_impacts.csv",
            index=False,
        )

        store = LocalDataStore(config=config)

        assert store.list_llm_impact_runs() == [stem]
        tables = store.load_llm_impact_tables(stem)
        assert tables["article_llm_summary"].loc[0, "article_id"] == "a1"
        assert tables["article_sector_impacts"].loc[0, "sector"] == "Technology"
        assert tables["article_stock_impacts"].loc[0, "ticker"] == "AAPL"
