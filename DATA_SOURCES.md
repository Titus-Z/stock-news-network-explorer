# Data sources and redistribution policy

## Checked-in demo

Everything under `data/demo/` was created for this repository. Prices,
headlines, summaries, sentiment scores, and sector records are synthetic. Real
ticker symbols are used only as familiar graph labels. The sample is suitable
for checking program flow, not for financial analysis.

## Optional local downloads

The downloader can request price history and company metadata through
`yfinance`, plus news sentiment payloads through Alpha Vantage. Those services
have their own access and redistribution terms. Users are responsible for
reviewing them. This repository excludes cached provider responses and
downloaded market data.

Set `MARKET_EXPLORER_DATA_DIR=data/local` before downloading. `data/local/` is
ignored by Git so a routine commit does not publish the cache.

## LLM enrichment

LLM outputs may reproduce or transform user-supplied article text. Do not
commit enrichment runs unless you have permission to share every input and
output. The default ignored path is `data/**/llm_enriched/`.
