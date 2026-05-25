# Market Data Foundation Research Smoke Test

Issue: #166

## Purpose

This note documents the research-side smoke test for the market data foundation.

The test verifies that the research track can:

1. fetch daily OHLCV data through SDU_DataScienceTool where possible,
2. fall back to yfinance if the local SDU package interface differs,
3. write CSV artifacts,
4. write Parquet artifacts,
5. write DuckDB tables,
6. reload data from DuckDB,
7. create a FinRL-compatible dataframe.

No RL training is started by this smoke test.

## Files

```text
research/requirements-market-data.txt
research/src/stockinvestmentdss_research/data/__init__.py
research/src/stockinvestmentdss_research/data/market_data_loader.py
research/experiments/finrl_yfinance_duckdb_smoke.py
```

## Run from repo root

```powershell
pip install -r research/requirements-market-data.txt
python research/experiments/finrl_yfinance_duckdb_smoke.py
```

## Expected artifacts

```text
research/experiments/artifacts/market_data_smoke/market_prices_daily.csv
research/experiments/artifacts/market_data_smoke/market_prices_daily.parquet
research/experiments/artifacts/market_data_smoke/finrl_prices.csv
system/runtime-data/market_research.duckdb
```

## Expected FinRL-compatible columns

```text
date
open
high
low
close
volume
tic
day
```

## Boundary

This smoke test belongs to the research track. It does not run when Docker Compose starts and does not start RL training.
