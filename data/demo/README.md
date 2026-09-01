# Synthetic demo data

This directory contains a deliberately small, deterministic, synthetic dataset.
Real ticker symbols are labels only. No values or articles were downloaded from
a market-data or news provider.

To use private local data, set `MARKET_EXPLORER_DATA_DIR=data/local` and run
`seed_data.py`. Never replace this directory with provider data in a commit.
